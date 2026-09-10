import pandas as pd


def apply_recommendations(df, missing_recs, duplicate_rec, manual_values=None):
    """
    manual_values: dict like {"column_name": value} — only used for
    recommendations marked 'manual' that the user chose to fill in.
    """
    manual_values = manual_values or {}
    cleaned_df = df.copy()
    applied_log = []

    for rec in missing_recs:
        col, action = rec["column"], rec["action"]

        if action == "impute_median":
            val = cleaned_df[col].median()
            cleaned_df[col] = cleaned_df[col].fillna(val)
            applied_log.append({"column": col, "action": action, "value_used": val})

        elif action == "impute_mode":
            mode_vals = cleaned_df[col].mode()
            if not mode_vals.empty:
                val = mode_vals[0]
                cleaned_df[col] = cleaned_df[col].fillna(val)
                applied_log.append({"column": col, "action": action, "value_used": val})

        elif action == "impute_knn":
            val = cleaned_df[col].median()  # fallback until real KNN is wired in
            cleaned_df[col] = cleaned_df[col].fillna(val)
            applied_log.append({"column": col, "action": "impute_knn (fallback: median)", "value_used": val})

        elif action == "manual":
            if col in manual_values:
                cleaned_df[col] = cleaned_df[col].fillna(manual_values[col])
                applied_log.append({"column": col, "action": "manual_fill", "value_used": manual_values[col]})
            else:
                applied_log.append({"column": col, "action": "skipped (manual, no value given)", "value_used": None})

    if duplicate_rec is not None:
        before = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = before - len(cleaned_df)
        applied_log.append({"column": "ALL", "action": "drop_duplicates", "value_used": f"{removed} rows removed"})

    return cleaned_df, applied_log


def verify_cleaning(original_report, new_report):
    return {
        "quality_score_before": original_report["quality_score"],
        "quality_score_after": new_report["quality_score"],
        "improvement": round(new_report["quality_score"] - original_report["quality_score"], 2),
        "missing_pct_before": original_report["missing_pct_avg"],
        "missing_pct_after": new_report["missing_pct_avg"],
        "duplicates_before": original_report["duplicate_rows"],
        "duplicates_after": new_report["duplicate_rows"],
    }
