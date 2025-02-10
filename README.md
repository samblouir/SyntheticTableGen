# Composite Synthesis Package

A Python package for generating **synthetic data tables** related to 
composite materials. This package includes:

- Large lists of **fillers**, **matrices**, **properties**, and **units**  
- Random row generators producing domain-specific synthetic data  
- A **schema-based** architecture for easy addition of new table types  
- Tools for formatting **ASCII tables** and **JSON output**  
- Utilities for optionally merging or omitting certain fields/units  

---

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Usage Examples](#usage-examples)
4. [Generating a Distribution Package](#generating-a-distribution-package)
5. [Uploading to PyPI](#uploading-to-pypi)
6. [Project Structure](#project-structure)
7. [Contributing](#contributing)
8. [License](#license)

---

## Features

- **Rich Data Lists**: Definitions of historical/trade synonyms for fillers, matrices, properties, and units
- **Random Generators**: Schemas and row-generating functions for consistent, randomized test data
- **Flexible Output**: Display tables with random alignment/case/delimiters or export to JSON with custom key naming
- **Schema-based**: Each table type is described by a schema that lists columns and references a row-generation function

---

## Installation

### Local (Editable) Install

```bash
git clone https://github.com/samblouir/synthetic_table_gen.git
cd synthetic_table_gen
pip install -e .
```

This installs the package in “editable” mode so changes to the code 
immediately reflect in the installed package.

### Standard Local Install

```bash
git clone https://github.com/samblouir/synthetic_table_gen.git
cd synthetic_table_gen
pip install .
```

---

## Usage Examples

After installation, you can import and use:

```python
import numpy as np
from synthetic_table_gen.main import generate_table_with_json_labels

rng = np.random.default_rng(seed=42)
output_str = generate_table_with_json_labels("composite_fixed_props", n_rows=5, rng=rng)
print(output_str)
```

You can also run:

```bash
python -m synthetic_table_gen.main
```

to execute the built-in `main()` function demonstration.

