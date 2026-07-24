import pandas as pd

from graph_ml.data.convert import convert_directory, convert_file


def test_convert_file_round_trips_a_dataframe(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.5, None, 3.5]})
    pickle_path = tmp_path / "sample.pkl"
    df.to_pickle(pickle_path)

    parquet_path = tmp_path / "sample.parquet"
    convert_file(pickle_path, parquet_path)

    pd.testing.assert_frame_equal(df, pd.read_parquet(parquet_path))


def test_convert_directory_converts_every_pkl_except_skipped(tmp_path):
    pd.DataFrame({"a": [1]}).to_pickle(tmp_path / "keep.pkl")
    pd.DataFrame({"b": [2]}).to_pickle(tmp_path / "04_network_snapshots.pkl")

    converted = convert_directory(tmp_path)

    assert [p.name for p in converted] == ["keep.parquet"]
    assert (tmp_path / "keep.parquet").exists()
    assert not (tmp_path / "04_network_snapshots.parquet").exists()


def test_convert_directory_returns_empty_list_when_no_pkl_files(tmp_path):
    assert convert_directory(tmp_path) == []
