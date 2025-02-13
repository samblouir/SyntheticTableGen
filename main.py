# main.py
# ----------------------------------------------------------------------
# This example shows how to:
#   1) Generate synthetic tables from schemas.py (with random data).
#   2) Randomly merge certain columns (value+unit) or drop the unit.
#   3) Randomly set alignment, case, delimiter for the final displayed table
#      (but not print them as a separate line).
#   4) Rename any "IT Unit", "CS Unit", "PR Unit" columns to
#      "interfacial_tension_unit", "compression_stress_at_break_unit", and
#      "poissons_ratio_unit" in the final JSON (so we never see "it_unit" or "cs_unit").
# ----------------------------------------------------------------------

import pandas as pd
import numpy as np
import json

from schemas import TABLE_SCHEMAS, ensure_schema_title


########################
# STEP 1) Random Format Helpers
########################

def random_format_options(rng: np.random.Generator):
    """
    Return a random combo of (alignment, text_case, delimiter),
    chosen from small sets. Using the provided rng ensures
    deterministic behavior if the rng seed is fixed.
    """
    align_choices = ["left", "right", "center"]
    case_choices = ["lower", "normal", "upper"]
    delim_choices = ["comma", "tab", "space"]

    alignment = rng.choice(align_choices)
    text_case = rng.choice(case_choices)
    delimiter = rng.choice(delim_choices)
    return alignment, text_case, delimiter


def format_table_display(df: pd.DataFrame, alignment="left", text_case="normal", delimiter="space") -> str:
    """
    Converts each row of df into a formatted string, applying:
      - alignment: 'left', 'right', or 'center'
      - text_case: 'lower', 'normal', or 'upper'
      - delimiter: 'comma', 'tab', or 'space'
    """
    # Convert data to string form
    rows_str = df.astype(str).values.tolist()  # list of lists
    columns_str = list(df.columns.astype(str))

    # Determine column widths
    col_widths = []
    for col_idx in range(len(columns_str)):
        items_in_col = [columns_str[col_idx]] + [row[col_idx] for row in rows_str]
        max_len = max(len(x) for x in items_in_col)
        col_widths.append(max_len)

    def align_text(txt: str, width: int, mode: str):
        if mode == "left":
            return txt.ljust(width)
        elif mode == "right":
            return txt.rjust(width)
        elif mode == "center":
            return txt.center(width)
        return txt  # fallback

    # Choose delimiter char
    if delimiter == "comma":
        delim_char = ","
    elif delimiter == "tab":
        delim_char = "\t"
    else:
        delim_char = " "

    # Build the header line
    header_cells = []
    for i, col_name in enumerate(columns_str):
        header_cells.append(align_text(col_name, col_widths[i], alignment))
    header_line = delim_char.join(header_cells)

    # Build row lines
    row_lines = []
    for row_vals in rows_str:
        line_cells = []
        for col_idx, val in enumerate(row_vals):
            cell_text = align_text(val, col_widths[col_idx], alignment)
            line_cells.append(cell_text)
        row_lines.append(delim_char.join(line_cells))

    # text case transform
    def transform_case(s: str, mode: str):
        if mode == "upper":
            return s.upper()
        elif mode == "lower":
            return s.lower()
        else:
            return s

    header_line = transform_case(header_line, text_case)
    row_lines = [transform_case(line, text_case) for line in row_lines]

    # Combine everything
    output_lines = [header_line] + row_lines
    return "\n".join(output_lines)


########################
# STEP 2) Generating Data
########################

def generate_random_table(table_name: str, n_rows=5, rng=None) -> pd.DataFrame:
    """
    Build a synthetic DataFrame for 'table_name' using the generator_func
    from TABLE_SCHEMAS, for n_rows. We pass 'rng' so data is deterministic.
    """
    if table_name not in TABLE_SCHEMAS:
        raise ValueError(f"Unknown table_name: {table_name}")

    if rng is None:
        rng = np.random.default_rng()  # fallback if user omits

    schema = TABLE_SCHEMAS[table_name]
    generator_func = schema["generator_func"]
    columns = schema["columns"]

    rows_data = []
    for _ in range(n_rows):
        row = generator_func(rng)  # e.g. [val1, val2, val3...]
        # slice or pad if needed
        if len(row) < len(columns):
            row += [None] * (len(columns) - len(row))
        else:
            row = row[:len(columns)]
        rows_data.append(row)

    df = pd.DataFrame(rows_data, columns=columns)
    return df


########################
# STEP 3) Merging Columns with RNG
########################

def create_display_df(original_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Merge columns like (value + unit). 
    Always omit "dimensionless".
    Sometimes randomly drop the actual unit so only the numeric is stored.
    Also fix the FutureWarning by forcing merged columns to object dtype.
    """
    df = original_df.copy()

    def merge_randomly(value_col: str, unit_col: str, new_col: str):
        # Same code as above
        if value_col in df.columns and unit_col in df.columns:
            # Ensure new_col is object dtype
            if new_col not in df.columns:
                df[new_col] = pd.Series([None]*len(df), dtype="object")
            else:
                df[new_col] = df[new_col].astype("object")

            for i, (val, u) in enumerate(zip(df[value_col], df[unit_col])):
                if pd.isna(val):
                    df.loc[i, new_col] = None
                    continue

                str_unit = str(u).lower() if not pd.isna(u) else ""

                # If the unit is dimensionless, skip it
                if str_unit == "dimensionless":
                    df.loc[i, new_col] = str(val)
                else:
                    # 50% chance to keep the unit, 50% to skip
                    if rng.random() < 0.5 and str_unit:
                        df.loc[i, new_col] = f"{val} {u}"
                    else:
                        df.loc[i, new_col] = str(val)

            df.drop(columns=[unit_col], inplace=True)

    merge_randomly("Interfacial Tension", "IT Unit", "Interfacial Tension")
    merge_randomly("Compression Stress at Break", "CS Unit", "Compression Stress at Break")
    merge_randomly("Poissons Ratio", "PR Unit", "Poissons Ratio")

    return df


########################
# STEP 4) JSON conversion with renamed columns
########################

def dataframe_to_jsonified_labels(df: pd.DataFrame):
    """
    Convert each row to a dictionary. 
    We rename certain "Unit" columns so we never see abbreviations 
    like 'cs_unit' or 'it_unit'.

    E.g. "IT Unit" -> "interfacial_tension_unit".
    But since we already dropped them from the display (via merging),
    typically we'll see them if they still exist in the data. 
    """
    # For a case where you want to rename any leftover "X Unit" 
    # columns to something else, do so here. 
    # But after merging, you might not have them in DF anymore.

    # If you do want to rename columns in the DF, do so. 
    # But in your code, 
    # 'IT Unit' was dropped from the DF, 
    # so let's handle e.g. "Weibull Scale Param (α) Unit" if existed, etc.

    # We'll also add logic for any leftover columns that might 
    # not have been merged:

    # Build a rename map
    rename_map = {
        "IT Unit": "interfacial_tension_unit",
        "CS Unit": "compression_stress_at_break_unit",
        "PR Unit": "poissons_ratio_unit",
        # etc. If there's any other "X Unit" 
        # you'd do it here
    }

    # But presumably, we've dropped them. If not, rename them:
    df_renamed = df.rename(columns=rename_map, errors="ignore")

    # Now convert to JSON rows
    json_rows = []
    for i, row in df_renamed.iterrows():
        row_dict = {"sample_id": i + 1}
        for col in df_renamed.columns:
            val = row[col]
            if pd.isna(val):
                continue
            # naive snake-case
            key = col.lower().replace(" ", "_")
            row_dict[key] = val
        json_rows.append(row_dict)

    return json_rows


########################
# STEP 5) The main table-with-JSON function
########################

def generate_table_with_json_labels(table_name, n_rows=5, rng=None):
    """
    1) Ensure the table has a 'title' (with random #) if not present
    2) Generate DataFrame
    3) Merge columns (value+unit) sometimes dropping the unit
    4) Randomly pick alignment/case/delimiter
    5) Format + produce final output (including JSON)
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1) Possibly fill in random table # if no "title"
    ensure_schema_title(table_name, rng)
    schema = TABLE_SCHEMAS[table_name]
    title_str = schema["title"]

    # 2) Generate data
    original_df = generate_random_table(table_name, n_rows, rng)

    # 3) Merge columns with random unit keep/drop
    display_df = create_display_df(original_df, rng)

    # 4) Random alignment/case/delimiter
    alignment, text_case, delim = random_format_options(rng)

    # Format the table
    table_str = format_table_display(display_df, alignment=alignment, text_case=text_case, delimiter=delim)

    # Build output lines (no mention of alignment/case/delim in the final print)
    out_lines = []
    out_lines.append(title_str)
    out_lines.append(table_str)
    out_lines.append("\n---\n### JSON Output\n")

    # 5) Convert to JSON
    json_rows = dataframe_to_jsonified_labels(original_df)
    out_lines.append(json.dumps(json_rows, indent=4))

    return "\n".join(out_lines)


def main():
    # Create a single RNG so everything is deterministic
    rng = np.random.default_rng(seed=42)

    # Example usage: generate 'weibull_epoxy_tio2' with 4 rows
    result1 = generate_table_with_json_labels("weibull_epoxy_tio2", n_rows=4, rng=rng)
    print(result1)

    print("\n" + "="*80 + "\n")

    # Another example: 'composite_fixed_props' with 3 rows
    result2 = generate_table_with_json_labels("composite_fixed_props", n_rows=3, rng=rng)
    print(result2)


if __name__ == "__main__":
    main()
