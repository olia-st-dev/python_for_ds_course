"""
Data preprocessing pipeline for the churn prediction dataset.

Splits raw training data into train/validation sets, encodes categorical
features (binary gender mapping + one-hot for multi-category columns),
optionally scales numeric features with StandardScaler, and optionally
oversamples the minority class in the training set (SMOTE or random).

Two entry points:
  - preprocess_data(raw_df, ...): fits everything from scratch on training data
    and returns train/validation splits plus the fitted scaler/encoder.
  - preprocess_new_data(new_df, scaler, encoder, ...): applies an already
    fitted scaler/encoder to new data (e.g. test.csv or production data).
"""

from typing import Optional

import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = ["Exited"]
ID_COL = ["id", "CustomerId", "Surname"]
GENDER_COL = "Gender"
GENDER_CODE_COL = "GenderCode"
GENDER_CODES = {"Male": 0, "Female": 1}
MULTI_CATEGORY_COLS = ["Geography"]
OVERSAMPLE_METHODS = ("smote", "random")


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file (e.g. train.csv or test.csv) into a dataframe."""
    return pd.read_csv(path)


def get_input_cols(df: pd.DataFrame, target_col: list[str]) -> list[str]:
    """Return all columns of df except the target column(s)."""
    return list(set(df.columns).difference(target_col))


def split_inputs_targets(
    df: pd.DataFrame, input_cols: list[str], target_col: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into an inputs dataframe and a targets dataframe."""
    inputs = df[input_cols]
    targets = df[target_col]
    return inputs, targets


def train_val_split(
    inputs: pd.DataFrame,
    targets: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/validation split on the targets."""
    train_inputs, val_inputs, train_targets, val_targets = train_test_split(
        inputs, targets, test_size=test_size, random_state=random_state,
        stratify=targets,
    )
    return train_inputs, val_inputs, train_targets, val_targets


def get_categorical_cols(df: pd.DataFrame) -> list[str]:
    """Return the list of object (categorical) dtype columns."""
    return df.select_dtypes(include="object").columns.to_list()


def get_numeric_cols(
    df: pd.DataFrame, target_col: list[str], exclude_cols: Optional[list[str]] = None
) -> list[str]:
    """Return numeric columns, excluding the target and any extra columns (e.g. id)."""
    exclude_cols = exclude_cols or []
    numeric_cols = df.select_dtypes(include="number").columns.difference(target_col).to_list()
    return list(set(numeric_cols).difference(exclude_cols))


def add_gender_code_col(
    inputs: pd.DataFrame,
    gender_col: str = GENDER_COL,
    gender_code_col: str = GENDER_CODE_COL,
    gender_codes: Optional[dict[str, int]] = None,
) -> pd.DataFrame:
    """Map a binary gender column to a numeric code column on a single dataframe."""
    gender_codes = gender_codes or GENDER_CODES
    inputs = inputs.copy()
    inputs[gender_code_col] = inputs[gender_col].map(gender_codes)
    return inputs


def fit_onehot_encoder(train_inputs: pd.DataFrame, cols_to_encode: list[str]) -> OneHotEncoder:
    """Fit a OneHotEncoder on the given columns of the training inputs."""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(train_inputs[cols_to_encode])
    return encoder


def apply_onehot_encoder(
    inputs: pd.DataFrame, encoder: OneHotEncoder, cols_to_encode: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Apply a fitted OneHotEncoder to a dataframe, adding the encoded columns."""
    inputs = inputs.copy()
    encoded_cols = list(encoder.get_feature_names_out(cols_to_encode))
    inputs[encoded_cols] = encoder.transform(inputs[cols_to_encode])
    return inputs, encoded_cols


def fit_scaler(train_inputs: pd.DataFrame, numeric_cols: list[str]) -> StandardScaler:
    """Fit a StandardScaler on the numeric columns of the training inputs."""
    scaler = StandardScaler()
    scaler.fit(train_inputs[numeric_cols])
    return scaler


def apply_scaler(
    inputs: pd.DataFrame, scaler: StandardScaler, numeric_cols: list[str]
) -> pd.DataFrame:
    """Apply a fitted StandardScaler to the numeric columns of a dataframe."""
    inputs = inputs.copy()
    inputs[numeric_cols] = scaler.transform(inputs[numeric_cols])
    return inputs


def oversample_train_data(
    X_train: pd.DataFrame,
    train_targets: pd.DataFrame,
    method: str = "smote",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Oversample the minority class in the training data only.

    Never apply this to validation or test data — doing so would leak
    synthetic/duplicated information into your evaluation and produce an
    overly optimistic, misleading score.

    Args:
        X_train: Training feature matrix.
        train_targets: Training targets (single-column DataFrame or Series).
        method: "smote" (synthetic minority samples, generally preferred for
            tree-based models since it avoids exact duplicates) or "random"
            (plain duplication of minority rows).
        random_state: Random state for reproducibility.

    Returns:
        (X_train_resampled, train_targets_resampled) with the minority class
        upsampled to match the majority class count.
    """
    if method not in OVERSAMPLE_METHODS:
        raise ValueError(f"method must be one of {OVERSAMPLE_METHODS}, got {method!r}")

    y = train_targets.squeeze()
    sampler = SMOTE(random_state=random_state) if method == "smote" else RandomOverSampler(random_state=random_state)
    X_resampled, y_resampled = sampler.fit_resample(X_train, y)

    target_name = train_targets.columns[0] if hasattr(train_targets, "columns") else "target"
    train_targets_resampled = y_resampled.to_frame(name=target_name)

    return X_resampled, train_targets_resampled


def preprocess_data(
    raw_df: pd.DataFrame,
    target_col: list[str] = TARGET_COL,
    id_col: list[str] = ID_COL,
    gender_col: str = GENDER_COL,
    gender_code_col: str = GENDER_CODE_COL,
    multi_category_cols: Optional[list[str]] = None,
    scale_numeric: bool = True,
    oversample: bool = False,
    oversample_method: str = "smote",
) -> dict:
    """
    Run the full preprocessing pipeline on a raw training dataframe.

    Steps:
      1. Split raw_df into train/validation inputs and targets (stratified).
      2. Encode the binary Gender column into a numeric GenderCode column.
      3. One-hot encode multi-category columns (e.g. Geography).
      4. Optionally scale numeric columns with StandardScaler.
      5. Assemble final feature matrices X_train / X_val.

    Args:
        raw_df: The full raw training dataframe, including the target column.
        target_col: List with the name of the target column.
        id_col: Identifier column to exclude from numeric scaling/features.
        gender_col: Name of the raw binary gender column.
        gender_code_col: Name of the derived numeric gender column.
        multi_category_cols: Columns to one-hot encode.
        scale_numeric: Whether to fit and apply a StandardScaler to numeric
            columns. If False, numeric columns are left unscaled and
            "scaler" in the returned dict will be None.
        oversample: Whether to oversample the minority class in X_train /
            train_targets only (never applied to validation data). Useful
            for imbalanced targets, e.g. with tree-based models.
        oversample_method: "smote" (default, generates synthetic minority
            samples) or "random" (duplicates existing minority rows).

    Returns:
        A dictionary with keys:
            "X_train" (pd.DataFrame): Training feature matrix.
            "train_targets" (pd.DataFrame): Training targets.
            "X_val" (pd.DataFrame): Validation feature matrix.
            "val_targets" (pd.DataFrame): Validation targets.
            "input_cols" (list[str]): Input columns before encoding/scaling.
            "numeric_cols" (list[str]): Numeric columns used as features.
            "feature_cols" (list[str]): Final ordered list of feature columns.
            "scaler" (Optional[StandardScaler]): Fitted scaler, or None if
                scale_numeric was False.
            "encoder" (OneHotEncoder): Fitted OneHotEncoder instance.
    """
    multi_category_cols = multi_category_cols or MULTI_CATEGORY_COLS

    # 1. Split raw data into inputs/targets, then train/validation sets
    input_cols = get_input_cols(raw_df, target_col)
    inputs, targets = split_inputs_targets(raw_df, input_cols, target_col)
    train_inputs, val_inputs, train_targets, val_targets = train_val_split(inputs, targets)

    # Identify numeric columns (based on full raw_df, matching original notebook logic)
    numeric_cols = get_numeric_cols(raw_df, target_col, exclude_cols=id_col)

    # 2. Encode binary gender column
    train_inputs = add_gender_code_col(train_inputs, gender_col, gender_code_col)
    val_inputs = add_gender_code_col(val_inputs, gender_col, gender_code_col)

    # 3. One-hot encode multi-category columns
    encoder = fit_onehot_encoder(train_inputs, multi_category_cols)
    train_inputs, encoded_cols = apply_onehot_encoder(train_inputs, encoder, multi_category_cols)
    val_inputs, _ = apply_onehot_encoder(val_inputs, encoder, multi_category_cols)

    # 4. Optionally scale numeric columns
    scaler: Optional[StandardScaler] = None
    if scale_numeric:
        scaler = fit_scaler(train_inputs, numeric_cols)
        train_inputs = apply_scaler(train_inputs, scaler, numeric_cols)
        val_inputs = apply_scaler(val_inputs, scaler, numeric_cols)

    # 5. Assemble final feature matrices
    feature_cols = numeric_cols + [gender_code_col] + encoded_cols
    X_train = train_inputs[feature_cols]
    X_val = val_inputs[feature_cols]

    # 6. Optionally oversample the minority class (X_train only, never X_val)
    if oversample:
        X_train, train_targets = oversample_train_data(X_train, train_targets, method=oversample_method)

    return {
        "X_train": X_train,
        "train_targets": train_targets,
        "X_val": X_val,
        "val_targets": val_targets,
        "input_cols": input_cols,
        "numeric_cols": numeric_cols,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "encoder": encoder,
    }


def preprocess_new_data(
    new_df: pd.DataFrame,
    encoder: OneHotEncoder,
    feature_cols: list[str],
    numeric_cols: list[str],
    scaler: Optional[StandardScaler] = None,
    target_col: list[str] = TARGET_COL,
    gender_col: str = GENDER_COL,
    gender_code_col: str = GENDER_CODE_COL,
    multi_category_cols: Optional[list[str]] = None,
    scale_numeric: bool = True,
) -> dict:
    """
    Apply an already-fitted preprocessing pipeline to new data (e.g. test.csv
    or production data), using the encoder/scaler produced by preprocess_data.

    Args:
        new_df: New raw dataframe to preprocess. May or may not include the
            target column.
        encoder: OneHotEncoder already fitted on training data.
        feature_cols: Final ordered list of feature columns, as returned by
            preprocess_data (its "feature_cols" key). Ensures new data ends up
            with exactly the same columns, in the same order, as the model
            was trained on.
        numeric_cols: Numeric columns to scale, as returned by preprocess_data
            (its "numeric_cols" key).
        scaler: StandardScaler already fitted on training data. Required if
            scale_numeric is True.
        target_col: List with the name of the target column.
        gender_col: Name of the raw binary gender column.
        gender_code_col: Name of the derived numeric gender column.
        multi_category_cols: Columns to one-hot encode.
        scale_numeric: Whether to apply the fitted scaler to numeric columns.
            Must match the scale_numeric setting used in preprocess_data.

    Returns:
        A dictionary with keys:
            "X_new" (pd.DataFrame): Preprocessed feature matrix for new_df.
            "new_targets" (Optional[pd.DataFrame]): Targets from new_df, or
                None if new_df does not contain the target column.
    """
    if scale_numeric and scaler is None:
        raise ValueError("scaler must be provided when scale_numeric is True")

    multi_category_cols = multi_category_cols or MULTI_CATEGORY_COLS

    has_target = all(col in new_df.columns for col in target_col)
    new_targets = new_df[target_col] if has_target else None

    inputs = new_df.drop(columns=target_col) if has_target else new_df.copy()

    inputs = add_gender_code_col(inputs, gender_col, gender_code_col)
    inputs, _ = apply_onehot_encoder(inputs, encoder, multi_category_cols)

    if scale_numeric:
        inputs = apply_scaler(inputs, scaler, numeric_cols)

    X_new = inputs[feature_cols]

    return {
        "X_new": X_new,
        "new_targets": new_targets,
    }


if __name__ == "__main__":
    # Example usage:
    train_raw_df = load_csv("train.csv")
    result = preprocess_data(train_raw_df, scale_numeric=True)
    print("X_train shape:", result["X_train"].shape)
    print("X_val shape:", result["X_val"].shape)

    test_raw_df = load_csv("test.csv")
    test_result = preprocess_new_data(
        test_raw_df,
        encoder=result["encoder"],
        feature_cols=result["feature_cols"],
        numeric_cols=result["numeric_cols"],
        scaler=result["scaler"],
        scale_numeric=True,
    )
    print("X_new shape:", test_result["X_new"].shape)