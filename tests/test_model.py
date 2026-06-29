from src.data import clean_dataset, generate_demo_dataset


def test_clean_dataset_keeps_required_target_values():
    df = clean_dataset(generate_demo_dataset(rows=50))
    assert set(df["target"].unique()).issubset({0, 1})
    assert df.isna().sum().sum() == 0

