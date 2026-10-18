# data_loader.py
import pandas as pd

from config import DATA_PATH


def load_data(path=DATA_PATH):
    """Read the heart.csv file into a DataFrame."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ File not found at: {path}") from None

    print(f"✅ Data loaded successfully! Shape: {df.shape}")
    return df
