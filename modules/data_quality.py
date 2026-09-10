import pandas as pd


def detect_missing(df):
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    return pd.DataFrame({"missing_count": missing_count, "missing_pct": missing_pct})


def detect_duplicates(df):
    return int(df.duplicated().sum())


def detect_outliers(df, column_types):
    outlier_report = {}
    for col, ctype in column_types.items():
        if ctype == "numeric":
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = df[(df[col] < lower) | (df[col] > upper)]
            outlier_report[col] = int(len(outliers))
    return outlier_report


def run_data_detective(df, column_types):
    missing_df = detect_missing(df)
    total_missing_pct = missing_df["missing_pct"].mean()

    dup_count = detect_duplicates(df)
    dup_pct = dup_count / len(df) * 100

    outliers = detect_outliers(df, column_types)
    total_outlier_count = sum(outliers.values())
    total_outlier_pct = (total_outlier_count / len(df) * 100) if len(df) else 0

    score = 100
    score -= total_missing_pct * 1.5
    score -= dup_pct * 2
    score -= total_outlier_pct * 0.5
    score = max(0, min(100, round(score, 1)))

    return {
        "quality_score": score,
        "missing_pct_avg": round(total_missing_pct, 2),
        "missing_by_column": missing_df.to_dict(orient="index"),
        "duplicate_rows": dup_count,
        "duplicate_pct": round(dup_pct, 2),
        "outliers_by_column": outliers,
        "total_outlier_count": total_outlier_count,
    }
