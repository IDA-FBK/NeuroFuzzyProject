# A Trustworthy Evolutionary Fuzzy Neural Network Framework for Maternal Health Risk Classification
In this work, we propose a novel evolutionary fuzzy neural network framework that incorporates a genetic algorithm to enhance classification capabilities while preserving
model transparency. Our architecture integrates evolutionary optimization into
the parameter update process of an existing Fuzzy Neural Network. An extensive
validation on the Maternal Health Risk dataset demonstrates the framework’s effectiveness in achieving a balance between predictive accuracy and explainability.

## Directories

- **data/**: Contains everything related to data.
  - This directory contains the script for loading the datasets. 

- **experiments/**: Contains everything related to experiments.
  - This directory includes configurations for each dataset, results, and evaluation scripts.
    - **results/**: Contains results for each dataset used, also subdivided into folders, and an overall summary.
    

- **models/**: Contains models, operators, selection and crossover
  - They includes respectively:
    - models used, the creation of individuals for the population and the calculus of the fitness.
    - implementation of AND and OR neurons and operations on those values.
    - selection of the best individuals from the population, eventually with mutation.
    - performation of crossover between two individuals.

## Running the Code

### 1. Set up the Project Environment

- First, ensure you have Conda installed. If not, follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
- Then, create the project environment using the provided environment.yml file:
  ```bash
  conda env create -f environment.yml
  ```

- And activate it:

  ```bash
  conda activate evoFNN
  ```
> **⚠️ Important**
> 
> The environment has been created and tested on Ubuntu 20.04.6 LTS. If you encounter any issues or need assistance, please don't hesitate to contact us.
> 
### 2. Set up Configuration

- Ensure you have a configuration file located in the `experiments/configurations/<dataset>/` directory.

- For running the _standard_ version, this file should contain experiment settings such as number of seeds, neuron types, number of membership functions (MFs), activation function, and optimizer.

- For running the _evolutionary_ version, this file should contain experiment settings such as the number of seeds, neuron types, fitness fn, parameters for mutation and crossover etc, and the path for storing the results.

### 3. Command-line Arguments

- The script accepts command-line arguments to specify dataset, path to configuration file, and directory to store results.

- Use the following command-line arguments:
  - `-dataset`: Specify the dataset to use.
  - `-path_to_conf`: Provide path to configuration file.
  - `-path_to_results`: Define directory where results will be saved. This argument in not mandatory for running the evolutionary version (since it is present in the configuration file)

### 4. Run the Script

- Run the FNN without evo:
  ```bash
  python main.py -dataset <dataset> -path_to_conf ./experiments/configurations/<dataset>/<name_of_conf>.json -path_to_results ./experiments/results/<dataset>/
  ```
  For **Maternal Health Risk**:
  ```bash
  python main.py -dataset maternal-hr -path_to_conf ./experiments/configurations/maternal_hr/conf_w.json -path_to_results ./experiments/results/maternal_hr/
  ```
- Run the evo-FNN:
  ```bash
  python main_evol_ind.py -dataset <dataset> -path_to_conf ./experiments/configurations/<dataset>/<name_of_conf>.json -path_to_results ./experiments/results/<dataset>/
  ```

  For **Maternal Health Risk**:
  ```bash
  python main_evol_ind.py -dataset maternal-hr -path_to_conf ./experiments/configurations/maternal_hr/conf_w.json -path_to_results ./experiments/results/maternal_hr/
  ```

### 5. Results 

Results are stored in `./results/<dataset>/` directory. You can see the results on the **Maternal Health Risk** dataset [here](https://github.com/IDA-FBK/NeuroFuzzyProject/tree/main/experiments/results/maternal_hr) (for information about its content, refer to the corresponding [README.md](https://github.com/IDA-FBK/NeuroFuzzyProject/blob/main/experiments/results/maternal_hr/README.md). 

## Authors
- Paulo Vitor De Campos Souza: pdecampossouza@fbk.eu
- Gianluca Apriceno: apriceno@fbk.eu
- Mauro Dragoni: dragoni@fbk.eu
## License
For open source projects, say how it is licensed.
