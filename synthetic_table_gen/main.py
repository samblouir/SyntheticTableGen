# main.py
# -----------------------------------------------------------------------------
# Demonstrates generating tables via schema definitions, optionally merging
# columns for display, randomizing text formatting, and outputting JSON.
#
# Provides functions that:
#  1) Generate synthetic tables from a given schema,
#  2) Merge value+unit columns (optionally),
#  3) Randomly apply different alignments, cases, or delimiters,
#  4) Convert final DataFrame to JSON.
# -----------------------------------------------------------------------------


import pandas as pd
import numpy as np
import json

from synthetic_table_gen.schemas import TABLE_SCHEMAS, ensure_schema_title


def random_format_options(rng: np.random.Generator):
    """
    Return a random (alignment, text_case, delimiter).

    :param rng: A numpy Random Generator instance.
    :return: A tuple (alignment, text_case, delimiter).
    """
    align_choices = ["left", "right", "center"]
    case_choices = ["lower", "normal", "upper"]
    delim_choices = ["comma", "tab", "space"]

    alignment = rng.choice(align_choices)
    text_case = rng.choice(case_choices)
    delimiter = rng.choice(delim_choices)
    return alignment, text_case, delimiter


def format_table_display(df: pd.DataFrame, alignment="left",
                         text_case="normal", delimiter="space") -> str:
    """
    Convert a DataFrame to a formatted string. Adjust alignment, 
    text case, and delimiter for demonstration variety.

    :param df: The DataFrame to format.
    :param alignment: 'left', 'right', or 'center' alignment.
    :param text_case: 'lower', 'normal', or 'upper'.
    :param delimiter: 'comma', 'tab', or 'space'.
    :return: A single string containing the formatted table.
    """
    rows_str = df.astype(str).values.tolist()
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
        return txt

    if delimiter == "comma":
        delim_char = ","
    elif delimiter == "tab":
        delim_char = "\t"
    else:
        delim_char = " "

    # Header line
    header_cells = []
    for i, col_name in enumerate(columns_str):
        header_cells.append(align_text(col_name, col_widths[i], alignment))
    header_line = delim_char.join(header_cells)

    # Row lines
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


def generate_random_table(table_name: str, n_rows=5, rng=None) -> pd.DataFrame:
    """
    Build a synthetic DataFrame for 'table_name' using the generator_func 
    from TABLE_SCHEMAS.

    :param table_name: Key in TABLE_SCHEMAS.
    :param n_rows: Number of rows to generate.
    :param rng: Optional numpy Random Generator.
    :return: A DataFrame containing the random rows.
    :raises ValueError: If table_name not recognized.
    """
    if table_name not in TABLE_SCHEMAS:
        raise ValueError(f"Unknown table_name: {table_name}")

    if rng is None:
        rng = np.random.default_rng()

    schema = TABLE_SCHEMAS[table_name]
    generator_func = schema["generator_func"]
    columns = schema["columns"]

    rows_data = []
    for _ in range(n_rows):
        row = generator_func(rng)
        # Adjust length of row if needed
        if len(row) < len(columns):
            row += [None]*(len(columns)-len(row))
        else:
            row = row[:len(columns)]
        rows_data.append(row)

    df = pd.DataFrame(rows_data, columns=columns)
    return df


def create_display_df(original_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Merge columns like (value + unit) in a random manner for display.

    :param original_df: DataFrame with separate columns for Value / Unit.
    :param rng: A numpy Random Generator.
    :return: A new DataFrame with some columns merged.
    """
    df = original_df.copy()

    def merge_randomly(value_col: str, unit_col: str, new_col: str):
        if value_col in df.columns and unit_col in df.columns:
            df[new_col] = df[new_col].astype("object")
            for i, (val, u) in enumerate(zip(df[value_col], df[unit_col])):
                if pd.isna(val):
                    df.loc[i, new_col] = None
                    continue
                str_unit = str(u).lower() if not pd.isna(u) else ""
                if str_unit == "dimensionless":
                    df.loc[i, new_col] = str(val)
                else:
                    # 50% chance to keep the unit
                    if rng.random() < 0.5 and str_unit:
                        df.loc[i, new_col] = f"{val} {u}"
                    else:
                        df.loc[i, new_col] = str(val)
            df.drop(columns=[unit_col], inplace=True)

    merge_randomly("Interfacial Tension", "IT Unit", "Interfacial Tension")
    merge_randomly("Compression Stress at Break", "CS Unit", "Compression Stress at Break")
    merge_randomly("Poissons Ratio", "PR Unit", "Poissons Ratio")

    return df


def dataframe_to_jsonified_labels(df: pd.DataFrame):
    """
    Convert DataFrame rows to JSON dictionaries, 
    renaming leftover 'X Unit' columns if they exist.

    :param df: The DataFrame to convert.
    :return: A list of dictionaries (one per row).
    """
    rename_map = {
        "IT Unit": "interfacial_tension_unit",
        "CS Unit": "compression_stress_at_break_unit",
        "PR Unit": "poissons_ratio_unit",
    }
    df_renamed = df.rename(columns=rename_map, errors="ignore")

    json_rows = []
    for i, row in df_renamed.iterrows():
        row_dict = {"sample_id": i + 1}
        for col in df_renamed.columns:
            val = row[col]
            if pd.isna(val):
                continue
            key = col.lower().replace(" ", "_")
            row_dict[key] = val
        json_rows.append(row_dict)
    return json_rows


def generate_table_with_json_labels(table_name, n_rows=5, rng=None):
    """
    1) Ensure the schema has a 'title',
    2) Generate data,
    3) Merge columns,
    4) Randomly format the display,
    5) Convert to JSON.

    :param table_name: The schema key name.
    :param n_rows: Number of rows to generate.
    :param rng: Optional numpy Random Generator.
    :return: A string containing the table and JSON.
    """
    if rng is None:
        rng = np.random.default_rng()

    ensure_schema_title(table_name, rng)
    schema = TABLE_SCHEMAS[table_name]
    title_str = schema["title"]

    original_df = generate_random_table(table_name, n_rows, rng)
    display_df = create_display_df(original_df, rng)
    alignment, text_case, delim = random_format_options(rng)
    table_str = format_table_display(display_df, alignment=alignment, 
                                     text_case=text_case, delimiter=delim)

    out_lines = []
    out_lines.append(title_str)
    out_lines.append(table_str)
    out_lines.append("\n---\n### JSON Output\n")

    json_rows = dataframe_to_jsonified_labels(original_df)
    out_lines.append(json.dumps(json_rows, indent=4))

    return "\n".join(out_lines)


def main():
    """
    An example script that prints two different tables (Weibull Tio2 and 
    composite_fixed_props) with random formatting, plus JSON output.
    """
    rng = np.random.default_rng(seed=42)

    result1 = generate_table_with_json_labels("weibull_epoxy_tio2", n_rows=4, rng=rng)
    print(result1)

    print("\n" + "="*80 + "\n")

    result2 = generate_table_with_json_labels("composite_fixed_props", n_rows=3, rng=rng)
    print(result2)


if __name__ == "__main__":
    main()

