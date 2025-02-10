# Synthetic Tables Example

Below is an example of a generated table and its JSON output, shown in a
simple MarkDown file for demonstration..

## Example Table

| Matrix Name | Filler Name       | Filler Loading | Weibull Shape Param (β) | Weibull Scale Param (α) | Interfacial Tension | IT Unit | ... |
|-------------|-------------------|----------------|-------------------------|-------------------------|---------------------|---------|-----|
| Nylon 11    | Graphite oxide    | 41.40 wt%      | 23.952                  | 27.73                   | 970698.024          | dyne/cm | ... |


## Corresponding JSON

```json
[
    {
        "sample_id": 1,
        "matrix_name": "Nylon 11",
        "filler_name": "Graphite oxide",
        "composition_amount": "41.40 wt%",
        "properties_shape parameter_value": 23.952,
        "properties_scale parameter_value": 27.73,
        "interfacial_tension": 970698.024,
        "it_unit": "dyne/cm"
    }
    ...
]

