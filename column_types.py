import pandas as pd
import warnings


def classify_column(series: pd.Series, col_name: str) -> str:
    n = len(series)
    n_unique = series.nunique()
    dtype = series.dtype

    # 1. Already a real datetime dtype
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"

    # 2. True unique identifier -> strip before modeling, no analytical value
    if n_unique >= 0.95 * n and n > 20:
        return "id"

    # 3. Numeric dtype
    if pd.api.types.is_numeric_dtype(dtype):
        # a numeric column that repeats a lot is a label/foreign key, not a measurement
        if n_unique <= 5 or "id" in col_name.lower():
            return "categorical"
        return "numeric"

    # 4. Text column that's secretly a date
    if dtype == "object":
        sample = series.dropna().astype(str).head(20)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                pd.to_datetime(sample, errors="raise")
                return "datetime"
            except (ValueError, TypeError):
                pass

    # 5. Text: few unique values -> categorical, many -> free text
    if n_unique <= max(20, 0.05 * n):
        return "categorical"

    return "text"
