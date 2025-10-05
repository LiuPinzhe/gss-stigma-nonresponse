# Data Directory

## Required Data File

This analysis requires the GSS cumulative data file:

**File:** `gss7224_r1.dta`  
**Source:** [NORC GSS Data Explorer](https://gssdataexplorer.norc.org/)  
**Description:** General Social Survey Cumulative File 1972-2024  
**Size:** ~565 MB

## How to Obtain the Data

1. Visit the [GSS Data Explorer](https://gssdataexplorer.norc.org/)
2. Download the complete cumulative dataset (1972-2024)
3. Choose Stata format (.dta)
4. Place the file `gss7224_r1.dta` in this directory

## Data Structure

The analysis expects the following variables (lowercase):
- **Predictors:** age, educ, sex, race, region, year, relig, attend, polviews, income, marital
- **Sensitive Items:** sexornt, premarsx, xmarsex, homosex, gaymarry

## Note

The data file is not included in this repository due to size limitations (>100MB). Please download it separately from the official GSS website.