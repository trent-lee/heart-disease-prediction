# preprocessing.py
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import CATEGORICAL_COLS, FEATURE_COLS, NUMERIC_COLS, RANDOM_STATE, TARGET


def clean_data(df):
    """Drop missing values and duplicate rows."""
    df = df.copy()

    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        df = df.dropna()
        print(f"🧹 Dropped rows with missing values: {missing_count}")
    else:
        print("✅ No missing values found")

    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates()
        print(f"🧹 Dropped duplicate rows: {duplicate_count}")
    else:
        print("✅ No duplicates found")

    if TARGET not in df.columns:
        raise ValueError(f"❌ '{TARGET}' column not found in dataset")

    return df


def build_preprocessor():
    """Scale numeric columns and one-hot encode categorical columns.

    Living inside the model pipeline means the dashboard can pass raw
    values (e.g. Sex='M') and get exactly the same transformation as training.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ],
        verbose_feature_names_out=False,
    )


def preprocess_and_split(df, test_size=0.2):
    """Clean the data and return a stratified train/test split of raw features."""
    df = clean_data(df)

    X = df[FEATURE_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test
