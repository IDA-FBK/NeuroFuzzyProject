# Neuro-Fuzzy project
This repo contains the code for running a standard and evolutionary version of a Neuro-Fuzzy Inference System.
We explore the development and application of Evolutionary Algorithms to evolve a Neuro-Fuzzy network in different medical fields.

## Directories

- **data/**: Contains everything related to data.
  - This directory contains the script for loading the datasets. 

- **experiments/**: Contains everything related to experiments.
  - This directory includes configurations for each dataset, results, and evaluation scripts.
    - **results/**: Contains results for each dataset used, also subdivided into folders, and an overall summary
    

- **models/**: Contains models, operators, selection and crossover
  - They includes respectively
    - models used, the creation of individuals for the population and the calculus of the fitness
    - implementation of AND and OR neurons and operations on those values
    - selection of the best individuals from the population, eventually with mutation.
    - performation of crossover between two individuals

## Running the Code

### 1. Set up the Project Environment

- First, ensure you have Conda installed. If not, follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
- Then, create the project environment using the provided environment.yml file:
  ```bash
  conda env create -f environment.yml
  ```

- And activate it:

  ```bash
  conda activate neurofuzzy
  ```

### 2. Set up Configuration

- Ensure you have a configuration file located in the `experiments/configurations/<dataset>/` directory.

- For running the _standard_ version, this file should contain experiment settings such as number of seeds, neuron types, number of membership functions (MFs), activation function, and optimizer (see conf file inside `experiments/configurations/iris/` ).

- For running the _evolutionary_ version, this file should contain experiment settings such as the number of seeds, neuron types, fitness fn, parameters for mutation and crossover etc, and the path for storing the results.

### 3. Command-line Arguments

- The script accepts command-line arguments to specify dataset, path to configuration file, and directory to store results.

- Use the following command-line arguments:
  - `-dataset`: Specify the dataset to use (default is "iris").
  - `-path_to_conf`: Provide path to configuration file (default is `./experiments/configurations/iris/conf-00.json`).
  - `-path_to_results`: Define directory where results will be saved (default is `./experiments/results/iris/`). This argument in not mandatory for running the evolutionary version (since it is present in the configuration file)

### 4. Run the Script

- Run the main script using Python:
  ```bash
  python main.py -dataset <dataset> -path_to_conf ./experiments/configurations/<dataset>/<name_of_conf>.json -path_to_results ./experiments/results/<dataset>/
  ```

### 5. Results 

Results are stored in `./results/<dataset>/` directory. This directory can contains:

- Plots, fuzzy rules and axioms generated during each experiment.
- `runs_results.csv`: A CSV file storing the results of each run (standard version).
- `mean_std_results.csv`: A CSV file storing the mean and standard deviation of grouped runs (by NeuronType and MFS) (standard version).
- `local_results_{all_parameters_used}`: A CSV file containing the results of the current experiment, each row is a new generation (evo version).
- `global_results.csv`: A CSV file combaining the fitness values of training, evaluation and test sets for each configuration (evo version).

## Authors
- Paulo Vitor De Campos Souza: pdecampossouza@fbk.eu
- Gianluca Apriceno: apriceno@fbk.eu
- Mauro Dragoni: dragoni@fbk.eu
## License
For open source projects, say how it is licensed.
