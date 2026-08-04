import pandas as pd


def clean_dataframe(df: pd.DataFrame, preprocessing_info: dict) -> pd.DataFrame:

    df = df.copy()

    numeric_cols = preprocessing_info["numeric_columns"]
    categorical_cols = preprocessing_info["categorical_columns"]
    medians = preprocessing_info["numeric_medians"]
    fill_value = preprocessing_info["categorical_fill"]

    for col in numeric_cols:
        if col not in df.columns:
            df[col] = medians.get(col, 0.0)
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(medians.get(col, df[col].median()))

    for col in categorical_cols:
        if col not in df.columns:
            df[col] = fill_value
        df[col] = df[col].fillna(fill_value).astype(str)

    return df


def encode_dataframe(df: pd.DataFrame, encoders: dict, preprocessing_info: dict) -> pd.DataFrame:
    
    df = df.copy()
    fill_value = preprocessing_info["categorical_fill"]

    for col in preprocessing_info["categorical_columns"]:
        encoder = encoders[col]
        class_to_code = {cls: i for i, cls in enumerate(encoder.classes_)}
        fallback = class_to_code.get(fill_value, 0)
        df[col] = df[col].map(lambda v: class_to_code.get(v, fallback))

    return df


def prepare_features(
    df: pd.DataFrame,
    encoders: dict,
    preprocessing_info: dict,
    feature_names: list,
) -> pd.DataFrame:
    """Full inference-time pipeline: clean -> encode -> select/order columns
    exactly as the model expects."""
    df = clean_dataframe(df, preprocessing_info)
    df = encode_dataframe(df, encoders, preprocessing_info)
    return df[feature_names]
