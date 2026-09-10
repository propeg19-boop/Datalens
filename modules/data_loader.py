import pandas as pd
import os


def load_data(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(file_path)
        else:
            return None, f"Unsupported file format: {ext}"
    except Exception as e:
        return None, f"Could not read file: {e}"

    if df.empty:
        return None, "The file loaded but contains no data."

    return df, None
