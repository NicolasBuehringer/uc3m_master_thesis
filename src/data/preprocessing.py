import numpy as np
import pandas as pd
from typing import Tuple, Dict
from numpy.lib.stride_tricks import sliding_window_view

def preprocess_stock_interval(
    df_raw: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    interval_minutes: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocess the input DataFrame by calculating log returns and realized volatility,
    split it into training and testing sets.
    
    Ported from Volatility_Transformer_Code.ipynb
    """

    # 1. Validation
    minutes_per_day = 380
    if minutes_per_day % interval_minutes != 0:
        raise ValueError(f"Interval {interval_minutes} must divide 380 evenly.")

    bins_per_day = minutes_per_day // interval_minutes

    minutely = df_raw.copy()
    minutely["Date Time"] = pd.to_datetime(minutely["Date Time"])
    minutely = minutely.sort_values("Date Time") # Strict sort required for diff

    # 2. Log Returns
    minutely["log_close"] = np.log(minutely["Close"].astype(float))

    # LEAKAGE PROTECTION:
    # We use a simple diff(), but we MUST drop the specific times that bridge days.
    minutely["ret_1m"] = minutely["log_close"].diff()

    # Remove 09:35. This removes the row containing the "Overnight Return"
    # (09:35 today - 15:55 yesterday).
    # This ensures Day N data depends ONLY on Day N prices.
    mask_valid_time = minutely["Date Time"].dt.time != pd.to_datetime("09:35").time()
    minutely = minutely.loc[mask_valid_time].reset_index(drop=True)

    # 3. Strict Shape Check (Prevents Partial Day Leakage)
    n_rows = len(minutely)
    if n_rows % minutes_per_day != 0:
        # If we have leftover rows, truncate them to ensure no partial days enter the train set
        # Partial days would result in a lower Daily RV target, confusing the model.
        n_days = n_rows // minutes_per_day
        minutely = minutely.iloc[:n_days * minutes_per_day]
        print(f"Warning: Dropped {n_rows % minutes_per_day} rows to enforce full days.")
    else:
        n_days = n_rows // minutes_per_day

    # 4. Reshape & Calculate (The Fast Part)
    # Shape: (Days, Intervals, Minutes)
    returns_array = minutely["ret_1m"].values.reshape(n_days, bins_per_day, interval_minutes)

    # Interval RV (Feature)
    rv_interval_values = np.sqrt(np.sum(returns_array**2, axis=2))

    # Daily RV (Target) - Summing over axes 1 and 2 covers the whole day
    rv_daily_values = np.sqrt(np.sum(returns_array**2, axis=(1, 2)))

    # 5. Reconstruct
    # We take the date from the first interval of every day
    dates = minutely["Date Time"].dt.date.values[::minutes_per_day]

    # Flatten features to 1D arrays
    flat_interval_rv = rv_interval_values.flatten()

    # Repeat daily targets (once per interval)
    expanded_daily_rv = np.repeat(rv_daily_values, bins_per_day)
    expanded_dates = np.repeat(dates, bins_per_day)

    # Interval Counter (0 to 37)
    interval_indices = np.tile(np.arange(bins_per_day), n_days)

    df_result = pd.DataFrame({
        "day": expanded_dates,
        "interval_idx": interval_indices,
        f"rv_feature": flat_interval_rv,
        "daily_rv": expanded_daily_rv
    })

    # Log Transforms
    df_result[f"log_rv_feature"] = np.log(df_result[f"rv_feature"].replace(0.0, 1e-12))
    df_result["log_daily_rv"] = np.log(df_result["daily_rv"].replace(0.0, 1e-12))

    # 6. Split
    # Split strictly by day index to avoid leaking info from 2pm to 10am of the same day (if shuffling)
    # Though sequential splitting usually handles this, defining by rows_per_day is safest.
    total_rows = len(df_result)
    rows_per_day_out = bins_per_day

    train_idx = int(np.floor(total_rows * train_frac / rows_per_day_out) * rows_per_day_out)
    val_idx = int(np.floor(total_rows * val_frac / rows_per_day_out) * rows_per_day_out)

    df_train = df_result.iloc[:train_idx]
    df_val = df_result.iloc[train_idx:train_idx+val_idx]
    df_test = df_result.iloc[train_idx+val_idx:]

    return df_train, df_val, df_test


def create_sequences_interval(
    df: pd.DataFrame,
    n_days: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized generation of rolling subsequences.
    Automatically detects 'rows_per_day' from the input dataframe.

    Parameters:
    df : pd.DataFrame
        Must contain columns: 'day', 'log_rv_feature', 'log_daily_rv'
    n_days : int
        Context window size in days

    Returns:
    X : shape (num_sequences, rows_per_day * n_days, 1)
    y_residual : shape (num_sequences,)
    y_true_RV : shape (num_sequences,)
    har_feature : shape (num_sequences,)
    """
    # 1. Infer rows per day dynamically
    # We count how many rows belong to the first day in the dataframe
    first_day = df["day"].iloc[0]
    rows_per_day = len(df[df["day"] == first_day])

    # 2. Prepare Data as NumPy arrays (Memory Efficient)
    # We use the generic column name 'log_rv_feature' established in Part 1
    rv_feature_arr = df["log_rv_feature"].to_numpy()
    daily_rv_arr = df["log_daily_rv"].to_numpy()

    # 3. Create X (Features) using Stride Tricks
    # We want a sliding window of length (n_days * rows_per_day)
    # But we only want to slide it forward by 1 DAY (rows_per_day) at a time, not 1 minute.

    window_size = n_days * rows_per_day

    # Generate ALL windows (sliding by 1 row)
    # Shape: (Total_Rows - Window_Size + 1, Window_Size)
    all_windows = sliding_window_view(rv_feature_arr, window_shape=window_size)

    # Slice to keep only windows that start at the beginning of a day
    # We step by `rows_per_day`
    # We strictly slice up to the point where we still have a "Next Day" target available
    valid_sequences_count = (len(df) // rows_per_day) - n_days

    X = all_windows[::rows_per_day]

    # Truncate X to match the number of valid targets available
    # (Sometimes sliding_window_view includes the very last window which has no "next day" target)
    X = X[:valid_sequences_count]

    # Reshape X to (Samples, TimeSteps, Features)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # 4. Create y (Targets) using Vectorized Indexing
    # We need the daily_rv of the day AFTER each sequence.
    # If sequence `i` covers days [0...9], target is Day 10.
    # The index for the start of Day 10 is: (n_days) * rows_per_day

    # Create indices for the start of every target day
    # Start: n_days * rows_per_day
    # Step: rows_per_day
    target_indices = np.arange(
        start=n_days * rows_per_day,
        stop=n_days * rows_per_day + (valid_sequences_count * rows_per_day),
        step=rows_per_day
    )

    # Get True Target (Day T)
    y_true_RV = daily_rv_arr[target_indices]

    # Get HAR Feature (Day T-1)
    # Since daily_rv is repeated for every row in the day,
    # the value at `target_index - 1` is the last row of Day T-1.
    har_feature = daily_rv_arr[target_indices - 1]

    # Calculate Residual
    y_residual = y_true_RV - har_feature

    return X, y_residual, y_true_RV, har_feature


def scaler(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test_dic: Dict[str, np.ndarray],
    d: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Standardize train, val, and test along the feature dimension.
    Learn parameters from train set only.
    Test data has to be handled separate since the arrays can't be stacked into
    one large dataset like train and val
    Only scaled first d features in case extra input features are tested like calendar data
    """

    # only scaled last d features, was used when eg calendar features were tested
    mu = X_train[:, :, :d].mean(axis=(0, 1), keepdims=True)
    sd = X_train[:, :, :d].std(axis=(0, 1), keepdims=True) + 1e-8

    def scale(X):
        X_scaled = X.copy()
        X_scaled[:, :, :d] = (X_scaled[:, :, :d] - mu) / sd
        return X_scaled

    X_train_scaled = scale(X_train)
    X_val_scaled   = scale(X_val)

    # loop through test dict
    X_test_scaled = {stock: scale(arr) for stock, arr in X_test_dic.items()}

    return X_train_scaled, X_val_scaled, X_test_scaled
