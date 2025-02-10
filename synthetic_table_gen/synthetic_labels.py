# synthetic_labels.py
# -----------------------------------------------------------------------------
# A helper module providing a function to generate a synthetic table, 
# then print a combined or "merged" table plus separate JSON. 
# -----------------------------------------------------------------------------

"""
Demonstrates usage of 'generate_random_table' from main.py with 
some optional merging of value+unit columns for display, then 
printing separate JSON output. 
"""

import json
import pandas as pd
from synthetic_table_gen.main import generate_random_table

def _map_column_to_json_key(col_name: str) -> str:
    """
    Convert a column name to a JSON-friendly key. 
    Certain column headers are mapped directly.

    :param col_name: Original column name.
    :return: JSON key name (snake_case).
    """
    direct_map = {
        "Matrix Name": "matrix_name",
        "Filler Name": "filler_name",
        "Filler Loading": "composition_amount",
        "Weibull Shape Param (β)": "properties_shape parameter_value",
        "Weibull Scale Param (α)": "properties_scale parameter_value",
    }
    if col_name in direct_map:
        return direct_map[col_name]

    key = col_name.lower()
    for ch in ["(", ")", "[", "]", "%", ",", "/", "-", "+"]:
        key = key.replace(ch, "")
    key = key.strip().replace(" ", "_")
    return key


def _dataframe_to_jsonified_labels(df: pd.DataFrame):
    """
    Convert each row of df into a dictionary 
    using the _map_column_to_json_key function.

    :param df: Input DataFrame.
    :return: List of dictionaries (one per row).
    """
    json_rows = []
    for i, row in df.iterrows():
        row_dict = {"sample_id": i + 1}
        for col in df.columns:
            val = row[col]
            if pd.isna(val) or val is None:
                continue
            json_key = _map_column_to_json_key(col)
            row_dict[json_key] = val
        json_rows.append(row_dict)
    return json_rows


def _print_jsonified_labels(json_rows):
    """
    Print each dictionary in json_rows in a JSON-formatted manner.

    :param json_rows: A list of dicts.
    """
    for item in json_rows:
        print("### Jsonified Table: ")
        print(json.dumps(item, indent=4))
        print()


def _create_display_df(original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge columns that contain Value and Unit for more compact display.

    :param original_df: DataFrame with potential <Value> and <Unit> columns.
    :return: A new DataFrame with some columns combined.
    """
    df = original_df.copy()

    if "Interfacial Tension" in df.columns and "IT Unit" in df.columns:
        combined = []
        for val, unit in zip(df["Interfacial Tension"], df["IT Unit"]):
            if pd.isna(val):
                combined.append(None)
            else:
                combined.append(f"{val} {unit}")
        df["Interfacial Tension"] = combined
        df.drop(columns=["IT Unit"], inplace=True)

    if "Compression Stress at Break" in df.columns and "CS Unit" in df.columns:
        combined = []
        for val, unit in zip(df["Compression Stress at Break"], df["CS Unit"]):
            if pd.isna(val):
                combined.append(None)
            else:
                combined.append(f"{val} {unit}")
        df["Compression Stress at Break"] = combined
        df.drop(columns=["CS Unit"], inplace=True)

    if "Poissons Ratio" in df.columns and "PR Unit" in df.columns:
        combined = []
        for val, unit in zip(df["Poissons Ratio"], df["PR Unit"]):
            if pd.isna(val):
                combined.append(None)
            else:
                if str(unit).lower() == "dimensionless":
                    combined.append(str(val))
                else:
                    combined.append(f"{val} {unit}")
        df["Poissons Ratio"] = combined
        df.drop(columns=["PR Unit"], inplace=True)

    return df


def generate_synthetic_table_and_json(table_name: str, n_rows: int = 5, rng_seed=None):
    """
    Generate a synthetic table, print the merged version, 
    then print JSON with separate columns.

    :param table_name: Key in the TABLE_SCHEMAS dictionary.
    :param n_rows: Number of rows to generate.
    :param rng_seed: Optional random seed for deterministic output.
    """
    if rng_seed is not None:
        import numpy as np
        rng = np.random.default_rng(seed=rng_seed)
    else:
        rng = None

    # 1) Generate original
    original_df = generate_random_table(table_name, n_rows=n_rows, rng=rng)

    # 2) Create merged display
    display_df = _create_display_df(original_df)

    # 3) Print table
    print("=== Synthetic Table (DataFrame) ===")
    print(display_df.to_string(index=False))

    # 4) Convert to JSON structures
    json_rows = _dataframe_to_jsonified_labels(original_df)

    # 5) Print the JSON
    print("\n=== Corresponding 'Jsonified' Labels ===\n")
    _print_jsonified_labels(json_rows)


if __name__ == "__main__":
    generate_synthetic_table_and_json(
        table_name="composite_fixed_props",
        n_rows=3,
        rng_seed=42
    )

