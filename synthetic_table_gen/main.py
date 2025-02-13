# main.py
# -----------------------------------------------------------------------------
# Demonstrates generating tables via schema definitions, optionally merging
# columns for display, randomizing text formatting, and outputting JSON.
# Now also supporting dict-based columns with "table_header" & "json_key".
# -----------------------------------------------------------------------------


import pandas as pd
import numpy as np
import json
from typing import Union

from synthetic_table_gen.schemas import TABLE_SCHEMAS, ensure_schema_title


def random_format_options(rng: np.random.Generator):
    align_choices = ["left", "right", "center"]
    case_choices = ["lower", "normal", "upper"]
    delim_choices = ["comma", "tab", "space"]
    alignment = rng.choice(align_choices)
    text_case = rng.choice(case_choices)
    delimiter = rng.choice(delim_choices)
    return alignment, text_case, delimiter


def _get_column_header(column_def: Union[str, dict]) -> str:
    """
    If column_def is a string, return it.
    If column_def is a dict with `table_header`, return that.
    """
    if isinstance(column_def, dict):
        return column_def["table_header"]
    return column_def


def _get_json_key(column_def: Union[str, dict]) -> str:
    """
    Return the JSON key for this column. 
    If it's a dict with 'json_key', return that; 
    else if it's a dict with 'table_header', transform it 
    else transform the string.
    """
    if isinstance(column_def, dict):
        if "json_key" in column_def:
            return column_def["json_key"]
        else:
            # fallback to table_header transformed
            raw = column_def["table_header"]
    else:
        raw = column_def
    # transform raw -> snake_case
    key = raw.lower()
    for ch in ["(", ")", "[", "]", "%", ",", "/", "-", "+"]:
        key = key.replace(ch, "")
    key = key.strip().replace(" ", "_")
    return key


def generate_random_table(table_name: str, n_rows=5, rng=None) -> pd.DataFrame:
    """
    Build a DF from the table schema, which now can have either strings or dicts
    for columns.
    """
    if table_name not in TABLE_SCHEMAS:
        raise ValueError(f"Unknown table_name: {table_name}")

    if rng is None:
        rng = np.random.default_rng()

    schema = TABLE_SCHEMAS[table_name]
    generator_func = schema["generator_func"]
    columns_def = schema["columns"]

    # We'll build two parallel lists:
    # 1) table_headers (the displayed column names)
    # 2) We'll store the rest for reference to build the final DF
    table_headers = [_get_column_header(c) for c in columns_def]

    rows_data = []
    for _ in range(n_rows):
        row = generator_func(rng)
        # ensure length
        if len(row) < len(columns_def):
            row += [None]*(len(columns_def)-len(row))
        else:
            row = row[:len(columns_def)]
        rows_data.append(row)

    df = pd.DataFrame(rows_data, columns=table_headers)
    return df


def create_display_df(original_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Merge 'Value' + 'Unit' columns if present, randomly. 
    This is optional. 
    """
    df = original_df.copy()
    columns_at_start = list(df.columns)

    def merge_randomly(value_col: str, unit_col: str):
        for i, (val, u) in enumerate(zip(df[value_col], df[unit_col])):
            if pd.isna(val):
                df.at[i, value_col] = None
                continue
            str_unit = str(u).lower() if not pd.isna(u) else ""
            if str_unit == "dimensionless":
                df.at[i, value_col] = str(val)
            else:
                if rng.random() < 0.25 and str_unit:
                    df.at[i, value_col] = f"{val} {u}"
                else:
                    df.at[i, value_col] = str(val)

    pairs_to_merge = []
    for col in columns_at_start:
        if col.lower().endswith("unit"):
            base_name = col.rsplit("Unit", 1)[0].rstrip()
            if base_name in df.columns:
                pairs_to_merge.append((base_name, col))

    for (value_col, unit_col) in pairs_to_merge:
        merge_randomly(value_col, unit_col)
        df.drop(columns=[unit_col], inplace=True)

    return df


def dataframe_to_jsonified_labels(table_name: str, df: pd.DataFrame):
    """
    Convert DataFrame rows to JSON dictionaries, 
    plus inject any schema['_metadata'] into each row as well.
    """
    schema = TABLE_SCHEMAS[table_name]
    columns_def = schema["columns"]

    # Convert columns to the final JSON keys
    def get_table_header(col_def: Union[str, dict]) -> str:
        if isinstance(col_def, dict):
            return col_def["table_header"]
        return col_def

    def get_json_key(col_def: Union[str, dict]) -> str:
        if isinstance(col_def, dict) and "json_key" in col_def:
            return col_def["json_key"]
        if isinstance(col_def, dict):
            raw = col_def["table_header"]
        else:
            raw = col_def
        key = raw.lower()
        for ch in ["(", ")", "[", "]", "%", ",", "/", "-", "+"]:
            key = key.replace(ch, "")
        key = key.strip().replace(" ", "_")
        return key

    json_keys = []
    table_headers = []
    for cdef in columns_def:
        table_headers.append(get_table_header(cdef))
        json_keys.append(get_json_key(cdef))

    # Now let's build the JSON rows
    out_rows = []
    # Check if there's extra metadata
    meta = schema.get("_metadata", {})

    for i, row in df.iterrows():
        row_dict = {"sample_id": i + 1}
        # Insert the metadata here, so each row gets it
        # e.g. "base_material": "Nylon 6"
        for k, v in meta.items():
            row_dict[k] = v

        for (header, jkey) in zip(table_headers, json_keys):
            if header in df.columns:
                val = row[header]
                if pd.isna(val):
                    continue
                row_dict[jkey] = val
        out_rows.append(row_dict)

    return out_rows


def generate_table_with_json_labels(table_name: str, n_rows=5, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    ensure_schema_title(table_name, rng) 
    schema = TABLE_SCHEMAS[table_name]
    title_str = schema["title"]

    # ... create the DataFrame
    # For demonstration, let's do a minimal version:

    # Actually call your row-generator function
    generator_func = schema["generator_func"]
    columns_def = schema["columns"]
    
    # Build table headers
    def get_table_header(c):
        if isinstance(c, dict):
            return c["table_header"]
        return c

    table_headers = [get_table_header(c) for c in columns_def]

    rows = []
    for _ in range(n_rows):
        r = generator_func(rng)
        # ensure length
        if len(r) < len(columns_def):
            r += [None]*(len(columns_def)-len(r))
        rows.append(r)

    df = pd.DataFrame(rows, columns=table_headers)

    # JSON
    json_rows = dataframe_to_jsonified_labels(table_name, df)

    # Minimal text table
    table_str = df.to_string(index=False)
    
    # Combine it all
    lines = []
    lines.append(title_str)
    lines.append(table_str)
    lines.append("\n---\n### JSON Output\n")
    lines.append(json.dumps(json_rows, indent=4))
    return "\n".join(lines)




def _format_table(df: pd.DataFrame, alignment, text_case, delimiter) -> str:
    rows_str = df.astype(str).values.tolist()
    columns_str = list(df.columns.astype(str))

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
        return txt

    if delimiter == "comma":
        delim_char = ","
    elif delimiter == "tab":
        delim_char = "\t"
    else:
        delim_char = " "

    header_cells = []
    for i, col_name in enumerate(columns_str):
        header_cells.append(align_text(col_name, col_widths[i], alignment))
    header_line = delim_char.join(header_cells)

    row_lines = []
    for row_vals in rows_str:
        line_cells = []
        for col_idx, val in enumerate(row_vals):
            cell_text = align_text(val, col_widths[col_idx], alignment)
            line_cells.append(cell_text)
        row_lines.append(delim_char.join(line_cells))

    def transform_case(s: str, mode: str):
        if mode == "upper":
            return s.upper()
        elif mode == "lower":
            return s.lower()
        return s

    header_line = transform_case(header_line, text_case)
    row_lines = [transform_case(line, text_case) for line in row_lines]

    output_lines = [header_line] + row_lines
    return "\n".join(output_lines)


def main():
    rng = np.random.default_rng(seed=42)

    result1 = generate_table_with_json_labels("weibull_epoxy_tio2", n_rows=4, rng=rng)
    print(result1)

    print("\n" + "="*80 + "\n")

    result2 = generate_table_with_json_labels("composite_fixed_props", n_rows=3, rng=rng)
    print(result2)


if __name__ == "__main__":
    main()
