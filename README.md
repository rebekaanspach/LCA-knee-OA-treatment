# LCA-knee-OA-treatment

This repository is linked to the project ''High tibial osteotomy and additive manufacture can significantly reduce climate impact of surgically treating  knee osteoarthritis'' by R. L. Anspach, H. S. Gill, V. Dhokia and R. C. Lupton  which investigates the effect that additive manufacturing surgical devices for the treatment of knee osteoarthritis has on the environmental burden associated with two types of interventions: high tibial osteotomy (HTO) and unicompartmental knee replacement (UKR).


## Inventory data

The foreground inventory databases are managed using Brightway2 and Activity Browser, and can be restored from `inventory/Foreground_inventory.xlsx`. Primary data from manufacturers and a hospital is treated as confidential.

## LCA calculations

Two main notebooks do the LCA calculations:

- `Comparative LCA.ipynb` does Monte Carlo comparative LCA calculations, for GWP only, writing the results to `results/samples_comparative_gwp_contributions.csv`.

- `Contribution analysis.ipynb` does LCA calculations for all impact categories (but not including uncertainty), writing the results to `results/all_impact_category_contributions.csv`.

The median, minimum, maximum, 25th percentile, 75th percentile results in Table 4 and 5 of the article are calculated in:
    
- `Statistical description of results.ipynb` writing the results to `results/stats_totals.csv`, `results/stats_totals.csv and results/stats_metals.csv`.

- `Statistical description of results-ratios.ipynb` writing the results to `results/stats_totals_ratios.csv`. 

## Figures for the paper

Figures are plotted in `Figures.ipynb`, which reads the two input files created above, and saves the generated figures within `figures`.

## Installation

Using `conda`: run `conda env create` to create an environment called `LCA-knee-OA-treatment` with the required packages (listed in `environment.yml`)

Activate this environment with `conda activate LCA-knee-OA-treatment` before running the commands below.

## License

 High tibial osteotomy and additive manufacture can significantly reduce climate impact of surgically treating  knee osteoarthritis © 2024 by by R. L. Anspach, H. S. Gill, V. Dhokia and R. C. Lupton is licensed under CC BY 4.0. To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/
```python

```
