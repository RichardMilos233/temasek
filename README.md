# Temasek Quant Strat Associate Interview Repository

Welcome to the repository for the Temasek Quantitative Strategies Associate Interview. This repository contains Python scripts and data files designed to evaluate candidates' skills in quantitative analysis, hedging strategies, and data manipulation. Candidates are required to complete a mini-project as indicated in `QS_case_study.xlsx` within 12 hours.

## Repository Structure

The repository is organized as follows:

```
quant_strat_interview/
    q1_analysis.py          # Analysis script for Question 1
    q1_pricing.csv          # Pricing data for Question 1
    q1_pricing.py           # Pricing model for Question 1
    q1_rolling_hedge.py     # Rolling hedge implementation for Question 1
    q1_vix_adjusted.csv     # Adjusted VIX data for Question 1
    q1_vix_adjustor.py      # VIX adjustor script for Question 1
    q1_workflow.py          # Workflow script for Question 1
    q2_classifier.py        # Classifier implementation for Question 2
    q2_hedging.py           # Hedging strategy for Question 2
    q2.csv                  # Data file for Question 2
    q3.py                   # Script for Question 3
    q1_pnl/                 # Output directory for Question 1 PnL results
    q2_pnl/                 # Output directory for Question 2 PnL results
```

## Environment Configuration

To ensure compatibility and reproducibility, please follow the steps below to configure the Python environment for this repository.

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).
2. Open a terminal and create the Conda environment from `environment.yml`:
   ```bash
   conda env create -f environment.yml
   ```
3. Activate the environment:
   ```bash
   conda activate temasek
   ```
4. If the environment already exists and you need to refresh dependencies:
   ```bash
   conda env update -f environment.yml --prune
   ```


## Notes

- The `environment.yml` file contains all required dependencies for this repository.
- For any issues or questions, please contact the repository maintainer.

---

Good luck with your interview preparation!