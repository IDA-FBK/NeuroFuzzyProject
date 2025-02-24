## Folder's contents
This folder contains the results on the dataset maternal heart rate of some **previous experiments**.

## Files
- `ReadMe.md`: This file

- `global_results_w.csv`: csv file with results of an expertiment with the evolutionary algorithm, in particular the mutation occurs on the weights of the fuzzy system
- `global_results_V.csv`: csv file with results of an expertiment with the evolutionary algorithm, in particular the mutation occurs on the V matrix of the fuzzy system

- `no_evoruns_results.csv`: csv file with the results of the experiments with the standard version of the algorithm

- `Precision_as_fitness`: folder in which there are the results of the experiments with the the usage of precision as fitness:
    - `precision_calc_standard_maternal`: contains the results derived from the standard version of the algorithm (a folder for each seed, `mean_std_results.csv`and `runs_results.csv`)
    - `precision_calculation_maternal`: folder with the results of the experiments with the evolutionary algorithm
        - `fuzzy_rules.txt`: txt file with the fuzzy rules 
        - `global_results_0.csv`: csv file with the results of the experiments with the evolutionary algorithm

    - `summery_results_maternal_micro`:  csv file with the summary of the results of  experiments present in the `runs_results.csv` and `global_results_0.csv`

