#!/usr/bin/env python3
# example.py
# -----------------------------------------------------------------------------
# Demonstration script that imports the 'synthetic_table_gen' package
# and shows a short example of EVERY table schema defined in TABLE_SCHEMAS.
# -----------------------------------------------------------------------------

"""
example.py

This script imports the 'synthetic_table_gen' package and loops over 
all table schemas. For each schema, it generates a few rows of data 
and prints the table plus JSON output to stdout.

Usage:
  python example.py

Adjust the number of rows (n_rows) as needed to reduce output size.
"""

import numpy as np

# Import the dictionary of table schemas, which contains the generator function
# references and column definitions:
from synthetic_table_gen.schemas import TABLE_SCHEMAS

# Import the main function that generates and prints a table + JSON:
from synthetic_table_gen.main import generate_table_with_json_labels


def main():
    """
    Loop over all schemas in TABLE_SCHEMAS, generate 3 rows from each,
    and print the resulting table and JSON output.
    """
    # Create a random generator with a fixed seed so examples are reproducible:
    rng = np.random.default_rng(seed=12345)

    # We iterate over each table name in TABLE_SCHEMAS.
    # For each schema, we generate 3 rows, then print them out.
    for table_name in TABLE_SCHEMAS.keys():
        print(f"=== Demonstrating Schema: {table_name} ===\n")

        # Generate 3 rows for this schema
        result_str = generate_table_with_json_labels(table_name, n_rows=3, rng=rng)

        # Print the output, which includes:
        # 1) A randomly chosen table title
        # 2) A formatted ASCII table
        # 3) A JSON block
        print(result_str)

        print("\n" + "-"*80 + "\n")  # A separator between tables


if __name__ == "__main__":
    main()

