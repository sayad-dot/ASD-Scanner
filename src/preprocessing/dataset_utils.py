import pandas as pd

NUMERICS = ["int16", "int32", "int64", "float16", "float32", "float64"]

def unify_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case and strip spaces from column names."""
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace("-", "_")
    )
    return df

def auto_cast(df: pd.DataFrame) -> pd.DataFrame:
    """Down-cast numerics for memory and convert booleans where applicable."""
    for col in df.columns:
        if df[col].dtype == object:
            unique = df[col].dropna().unique()
            if set(unique) <= {"yes", "no", "y", "n", "true", "false"}:
                df[col] = df[col].str.lower().map({"yes": 1, "y": 1, "true": 1,
                                                   "no": 0, "n": 0, "false": 0})
        elif df[col].dtype in NUMERICS:
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df
