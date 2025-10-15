# train/test_train.py
import train.train as train


def test_read_dataframe():
    """Ensure the diabetes dataset loads correctly."""
    df = train.read_dataframe()
    assert not df.empty
    assert "target" in df.columns
