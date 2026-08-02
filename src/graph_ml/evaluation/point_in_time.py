"""Point-in-time label availability and rolling-origin evaluation folds."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelAvailability:
    """Binary labels paired with the first timestamp at which they are knowable.

    ``valid_mask`` is false when the stored final label cannot be assigned a
    defensible availability timestamp. Such rows must remain unsupervised.
    """

    labels: np.ndarray
    available_at: pd.Series
    valid_mask: np.ndarray

    def known_by(self, as_of: str | pd.Timestamp) -> np.ndarray:
        """Return labels that are valid and observable on or before ``as_of``."""

        timestamp = _normalize_timestamp(as_of, "as_of")
        return self.valid_mask & self.available_at.le(timestamp).to_numpy()

    def known_before(self, boundary: str | pd.Timestamp) -> np.ndarray:
        """Return labels observable before an end-exclusive fold boundary."""

        timestamp = _normalize_timestamp(boundary, "boundary")
        return self.valid_mask & self.available_at.lt(timestamp).to_numpy()


@dataclass(frozen=True)
class RollingOriginFoldSpec:
    """Boundaries for one train → validation → test chronology."""

    train_end: str | pd.Timestamp
    validation_end: str | pd.Timestamp
    test_end: str | pd.Timestamp


@dataclass(frozen=True)
class PointInTimeFold:
    """Row-aligned masks whose labels are knowable at each decision boundary."""

    train_end: pd.Timestamp
    validation_end: pd.Timestamp
    test_end: pd.Timestamp
    train_mask: np.ndarray
    validation_mask: np.ndarray
    refit_mask: np.ndarray
    test_mask: np.ndarray

    def summary(self) -> dict[str, int | str]:
        """Return boundaries and supervised cohort sizes for reporting."""

        return {
            "train_end": self.train_end.date().isoformat(),
            "validation_end": self.validation_end.date().isoformat(),
            "test_end": self.test_end.date().isoformat(),
            "train_rows": int(self.train_mask.sum()),
            "validation_rows": int(self.validation_mask.sum()),
            "refit_rows": int(self.refit_mask.sum()),
            "test_rows": int(self.test_mask.sum()),
        }


def build_horizon_label_availability(
    frame: pd.DataFrame,
    *,
    target_column: str,
    due_date_column: str = "due_date",
    horizon_days: int,
) -> LabelAvailability:
    """Assign conservative availability for a due-date-plus-horizon target.

    A p90 label, for example, is treated as knowable at ``due_date + 90 days``
    for both classes. A positive may become operationally evident slightly
    earlier depending on the exact definition, but the shared horizon is simple,
    conservative, and prevents final-snapshot labels entering premature folds.
    """

    if horizon_days < 1:
        raise ValueError("horizon_days must be positive")
    _require_columns(frame, target_column, due_date_column)
    labels, label_valid = _binary_labels(frame[target_column], target_column)
    due_dates = pd.to_datetime(frame[due_date_column], errors="coerce")
    available_at = due_dates + timedelta(days=horizon_days)
    valid = label_valid & due_dates.notna().to_numpy()
    return LabelAvailability(labels, available_at, valid)


def build_event_label_availability(
    frame: pd.DataFrame,
    *,
    target_column: str,
    positive_event_date_column: str,
    negative_resolution_date_column: str,
) -> LabelAvailability:
    """Assign availability from explicit positive-event/negative-resolution dates.

    Positive labels become known at their event timestamp. Negative labels become
    known only when resolution establishes that the event will not occur. A final
    label with no applicable timestamp is retained in ``labels`` but marked
    invalid, so no fold can use it for supervision.
    """

    _require_columns(
        frame,
        target_column,
        positive_event_date_column,
        negative_resolution_date_column,
    )
    labels, label_valid = _binary_labels(frame[target_column], target_column)
    positive_dates = pd.to_datetime(
        frame[positive_event_date_column], errors="coerce"
    )
    negative_dates = pd.to_datetime(
        frame[negative_resolution_date_column], errors="coerce"
    )
    available_at = positive_dates.where(labels, negative_dates)
    valid = label_valid & available_at.notna().to_numpy()
    return LabelAvailability(labels, available_at, valid)


def build_point_in_time_fold(
    prediction_times: pd.Series,
    availability: LabelAvailability,
    spec: RollingOriginFoldSpec,
) -> PointInTimeFold:
    """Build one rolling-origin fold without using labels before availability.

    Training labels must be known by ``train_end``. Validation labels originate
    in the next window and must be known by ``validation_end``. ``refit_mask`` is
    the history legally available when refitting after validation. Test labels
    originate in the final window and are evaluated only when known by
    ``test_end``.
    """

    times = pd.to_datetime(prediction_times, errors="coerce")
    if times.isna().any():
        raise ValueError("prediction_times must contain valid timestamps")
    size = len(times)
    _validate_availability(availability, size)

    train_end = _normalize_timestamp(spec.train_end, "train_end")
    validation_end = _normalize_timestamp(spec.validation_end, "validation_end")
    test_end = _normalize_timestamp(spec.test_end, "test_end")
    if not train_end < validation_end < test_end:
        raise ValueError("Fold boundaries must satisfy train_end < validation_end < test_end")

    before_train = times.lt(train_end).to_numpy()
    validation_origin = times.ge(train_end).to_numpy() & times.lt(
        validation_end
    ).to_numpy()
    before_validation = times.lt(validation_end).to_numpy()
    test_origin = times.ge(validation_end).to_numpy() & times.lt(test_end).to_numpy()

    train = before_train & availability.known_before(train_end)
    validation = validation_origin & availability.known_before(validation_end)
    refit = before_validation & availability.known_before(validation_end)
    test = test_origin & availability.known_before(test_end)
    return PointInTimeFold(
        train_end=train_end,
        validation_end=validation_end,
        test_end=test_end,
        train_mask=train,
        validation_mask=validation,
        refit_mask=refit,
        test_mask=test,
    )


def _binary_labels(series: pd.Series, name: str) -> tuple[np.ndarray, np.ndarray]:
    valid = series.notna().to_numpy()
    observed = set(series.loc[valid].unique().tolist())
    if not observed.issubset({False, True, 0, 1}):
        raise ValueError(f"{name} must contain binary labels")
    labels = series.fillna(False).astype(bool).to_numpy()
    return labels, valid


def _require_columns(frame: pd.DataFrame, *columns: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _validate_availability(availability: LabelAvailability, size: int) -> None:
    if availability.labels.shape != (size,):
        raise ValueError("availability labels must align with prediction_times")
    if availability.valid_mask.shape != (size,):
        raise ValueError("availability valid_mask must align with prediction_times")
    if len(availability.available_at) != size:
        raise ValueError("availability timestamps must align with prediction_times")


def _normalize_timestamp(value: str | pd.Timestamp, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"{name} must be a valid timestamp")
    if timestamp.tz is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.normalize()
