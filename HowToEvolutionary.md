## Project structure
```
NeuroFuzzyProject
├── data
│   ├── datasets
│   │   ├── issues (not used)
│   │   ├── sepsis
│   │   │   ├── sepsis_survival_primary_cohort.csv
│   │   │   ├── sepsis_survival_study_cohort.csv
│   │   │   └── sepsis_survival_validation_cohort.csv
│   │   ├── diabetes.csv
│   │   ├── maternal_health_risk.csv
│   │   ├── ... (other datasets not used)
│   │   └──  obesity.csv
│   └── data.py
├── experiments
│   ├── configurations
│   │   ├── diabetes
│   │   │   └── json file with configurations
│   │   ├── maternal_hr
│   │   │   └── json file with configurations
│   │   ├── obesity
│   │   │   └── json file with configurations
│   │   ├── sepsis
│   │   │   └── json file with configurations
│   │   ├── .... (other datasets not used)
│   │   ├── conf_general_V.json
│   │   ├── conf_general_weights.json
│   │   └──  configurations.py
│   ├── results
│   │   ├── diabetes
│   │   │   └── results of the experiments
│   │   ├── maternal_hr
│   │   │   └── results of the experiments (also with micro precision as fitness)
│   │   ├── sepsis
│   │   │   └── results of the experiments
│   │   ├── previous_experiments
│   │   │   ├── diabetes
│   │   │   │   └── csv file with preliminar results 
│   │   │   ├── maternal_hr
│   │   │   │   └── csv file with preliminar results 
│   │   │   ├── sepsis
│   │   │   │   └── csv file with preliminar results 
│   │   │   └── sepsis_evo_weights
│   │   │       └── csv file with preliminar results 
│   │   ├── summery_results
│   │   │   ├── show_results_diabets.ipynb
│   │   │   ├── show_results_maternal_hr.ipynb
│   │   │   ├── show_results_sepsis.ipynb
│   │   │   ├── summary_results_diabetes.csv
│   │   │   ├── summary_results_maternal.csv
│   │   │   ├── summary_results_sepsis.csv
│   │   │   ├── summary_results_maternal_micro.csv
│   │   │   └── table_confrontation.py
│   │   ├── no_evo_diabete.css
│   │   ├── no_evo_maternal.csv
│   │   ├── no_evo_sepsis.csv
│   │   ├── res_w_diabetes.csv
│   │   ├── res_w_maternal.csv
│   │   └── res_w_sepsis.csv
│   ├── calculate.py
│   ├── evolution.py
│   ├── plots.py
│   └── utils.py
├── models
│   ├── crossover.py
│   ├── models.py
│   ├── operators.py
│   └── selection.py
├── env2.yml
├── environment.yml
├── README.md
├── baseline-1.py
├── main_evol_ind.py
├── main.py
└── show_results.ipynb
```

## Evolution operations
The evolution operations are implemented in the `models` directory. 
- in `models.py`, the creation of individuals for the population and the calculus of the fitness were added
- in `selection.py`, the implementation of selection of the best individuals from the population (eventually with mutation) is present
- in `crossover.py` there is the implementation of crossover between two individuals

## Setting configuration file

For running correcly the evolutionary version of the project, the configuration file should be present in the `experiments/configurations/<dataset>/` directory. This file should contain this info:
1. number of seeds
2. neuron types (AND, OR)
3. number of membership functions (MFs)
4. which genes update, in particular 
    - V 
    - weights of neurons
5. activation function
6. optimizer
7. data encoding
8. prediction method
9. how to calculate the fitness function (accuracy, f1, ...)
10. parameters for mutation rate (general)
11. parameters for mutation rate of each individual
12. parameters for crossover rate
13. max number of generations
14. max number for patience (early stopping)
15. initial population size
16. number of individuals to generated (offspings)
17. selection strategy
    - plus 
    - comma
18. path for storing the results

## Pipeline of the project
An experiment for each possible configuration is performed.
The pipeline of the project is the following:
1. load the dataset
2. run the experiment
3. save the results
    - *local* results, with detailed results of a single experiment (they are as many as combiantion of configurations)
    - *global* results, with the summary of all the experiments

In the `run_experiment` function, the following steps are performed:
1. initialize the **population**, that is a List[FNNModel]
    - an individual is created with the class `FNNModel`
    - the individual is initilized and added to the population

2. for each generation (from 1 to *max_gen* or until *patient != 0* ), the following steps are performed:
    - **selection** of the <ins>best individuals</ins> from the population, eventually with mutation
    - for EACH individual, the **fitness function** is calculated and saved for the train, validation and test set
        - if the individual is the best, according to the fitness function on the val set, it is saved

    - the population performances (mean, std, max and min fitness) are saved in the *local* results for each set (train, val, test)
    - if there was no improvement in individuals, the patient counter is decreased


3. only the fitness in the train, validation and test set of the best individual is return (and saved in the *global* results)

## SELECTION STRATEGY 
The selection strategy is implemented in the `selection.py` file.

For compute the selection, the number of parents (pop size) has to be *lower* that number of offsprings (children) to generate.

The one implemented is a **tournament selection**: for each offspring (child) to be generated, the best individual is chosen from a random subset of the population.
Also, a **plus strategy** is developed: 

In the file, the following steps are done:
1. check if the number of parents is lower than the number of offsprings
2. calculate the fitness of individual, if not already done
3. offsprings are generated:
    - tournament selection is performed
    - crossover and mutation are performed
    - the new individual is added to the offsprings set

4. for each offspring, the fitness is calculated and saved

5. if the selection strategy is equal to *plus*, offsprings are added to the initial population 

6. the new population created is sorted according to the fitness

7. the population truncated: the same number of individuals as the initial population is kept and returned

## CROSSOVER
The crossover operation is implemented in the `crossover.py` file.

The crossover operation is performed between two individuals. This works by **swapping** some parameters of the two fuzzy neural networks.

## MUTATION
The mutation operation is implemented in the `models.py` file.

An individual is mutated by by adding random noise: first the fitness is resetted, then the mutation is performed.