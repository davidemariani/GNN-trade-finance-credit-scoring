"""Convert legacy pickle datasets to Parquet.

See wiki/this-project/data-availability.md "Storage format & policy" for why: pickle executes
arbitrary code on load and is brittle across library versions, while Parquet (zstd) is columnar,
compressed (measured 7-14% of the original pickle size on this project's data), and safe.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_SKIP = frozenset({"04_network_snapshots.pkl"})


def convert_file(
    pickle_path: Path, parquet_path: Path, compression: str = "zstd"
) -> None:
    """Read a single pickled DataFrame and write it out as Parquet."""
    df = pd.read_pickle(pickle_path)
    df.to_parquet(parquet_path, compression=compression)


def convert_directory(
    data_dir: Path,
    compression: str = "zstd",
    skip: frozenset[str] = DEFAULT_SKIP,
) -> list[Path]:
    """Convert every `*.pkl` file in `data_dir` to a sibling `*.parquet` file.

    `skip` names are left untouched — by default the large temporal snapshot file, whose
    conversion is deferred until the temporal-graph phase (see the roadmap).
    """
    converted = []
    for pickle_path in sorted(data_dir.glob("*.pkl")):
        if pickle_path.name in skip:
            continue
        parquet_path = pickle_path.with_suffix(".parquet")
        convert_file(pickle_path, parquet_path, compression=compression)
        converted.append(parquet_path)
    return converted


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, nargs="?", default=Path("data"))
    parser.add_argument("--compression", default="zstd")
    parser.add_argument(
        "--skip",
        nargs="*",
        default=list(DEFAULT_SKIP),
        help="Filenames to leave unconverted (default: the large temporal snapshot file).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    converted = convert_directory(
        args.data_dir, compression=args.compression, skip=frozenset(args.skip)
    )
    for path in converted:
        print(path)


if __name__ == "__main__":
    main()
