import pandas as pd


def recommend_missing_fix(df, column_types):
    recommendations = []
    n = len(df)

    for col, ctype in column_types.items():
        missing_count = df[col].isnull().sum()
        if missing_count == 0:
            continue
        missing_pct = round((missing_count / n) * 100, 3)

        if ctype == "numeric":
            if missing_pct <= 5:
                value = df[col].median()
                action = "impute_median"
                reasoning = f"Only {missing_pct}% missing — filling with the median ({value:.2f}) is safe and won't distort the distribution."
            elif missing_pct <= 15:
                value = None
                action = "impute_knn"
                reasoning = f"{missing_pct}% missing is moderate — KNN imputation (estimating from similar rows) tends to beat a flat median here."
            else:
                value = None
                action = "manual"
                reasoning = f"{missing_pct}% missing is too much to safely auto-fill — recommend manual review or dropping the column."

        elif ctype == "categorical":
            if missing_pct <= 5:
                mode_val = df[col].mode()
                value = mode_val[0] if not mode_val.empty else None
                action = "impute_mode"
                reasoning = f"Only {missing_pct}% missing — filling with the most common value ('{value}') is a reasonable default."
            else:
                value = None
                action = "manual"
                reasoning = f"{missing_pct}% missing is too much to guess confidently — recommend manual review."

        else:  # id, datetime, text
            value = None
            action = "manual"
            reasoning = f"'{ctype}' columns shouldn't be auto-filled — a guessed value here could be actively wrong. Recommend manual review."

        recommendations.append({
            "column": col, "column_type": ctype, "missing_count": int(missing_count),
            "missing_pct": missing_pct, "action": action, "suggested_value": value, "reasoning": reasoning,
        })

    return recommendations


def recommend_duplicate_fix(df):
    dup_count = int(df.duplicated().sum())
    if dup_count == 0:
        return None
    return {
        "issue": "duplicate_rows", "count": dup_count, "action": "drop_duplicates",
        "reasoning": f"{dup_count} exact duplicate rows add no new information and can bias counts/averages — safe to drop.",
    }


def recommend_outlier_fix(df, outlier_report):
    recommendations = []
    n = len(df)

    for col, count in outlier_report.items():
        if count == 0:
            continue
        pct = round((count / n) * 100, 3)

        if pct <= 0.1:
            reasoning = f"Only {count} outliers ({pct}%) — small enough to review individually rather than guess a fix."
        else:
            reasoning = (
                f"{count} outliers ({pct}%) flagged by the IQR rule. This share is high enough that it may reflect "
                f"a genuinely skewed distribution (e.g. premium items priced much higher) rather than errors — "
                f"don't auto-remove; review the actual value distribution before deciding."
            )

        recommendations.append({
            "column": col, "issue": "outliers", "outlier_count": count,
            "outlier_pct": pct, "action": "review_manual", "reasoning": reasoning,
        })

    return recommendations


def run_cleaning_assistant(df, column_types, quality_report):
    return {
        "missing_recommendations": recommend_missing_fix(df, column_types),
        "duplicate_recommendation": recommend_duplicate_fix(df),
        "outlier_recommendations": recommend_outlier_fix(df, quality_report["outliers_by_column"]),
    }
