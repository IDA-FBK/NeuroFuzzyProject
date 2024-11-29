import argparse
import os.path

import numpy as np
import pandas as pd
#from tqdm import tqdm

from data.data import get_data
from experiments.configurations.configurations import get_configuration
from experiments.evaluation import evaluate_interpretability
from experiments.calculate import calculate_avg_results
from experiments.utils import save_list_in_a_file
from experiments.plots import plot_class_confusion_matrix
from models.models import FNNModel
from models.selection import selection
import copy


def initialize_population(pop_size, num_mfs, neuron_type, fuzzy_interpretation, activation, optimizer, x_train, mutation_ind_rate, rng_seed):
    population = []
    
    for _ in range(pop_size):
        individuo = FNNModel(num_mfs=num_mfs, neuron_type=neuron_type, interpretation=fuzzy_interpretation, activation=activation, optimizer=optimizer, visualizeMF=False, mutation_ind_rate=mutation_ind_rate, rng_seed=rng_seed)
        individuo.initialize_individual(x_train)
        population.append(individuo)
        
    return population


def get_train_eval_split(data, percentage_train=0.8): #Potremmo usare sklearn.model_selection.train_test_split; ma dobbiamo aggiungere la libreria
    x_data, y_data = data[0], data[1]
    
    indice_split = int(percentage_train*len(x_data))
    
    x_train, x_eval = x_data[:indice_split], x_data[indice_split:]
    y_train, y_eval = y_data[:indice_split], y_data[indice_split:]    
    
    return (x_train, y_train), (x_eval, y_eval)

def run_experiment(
    train_data,
    test_data,
    data_encoding,
    pred_method,
    fitness_function,
    mutation_rate,
    map_class_dict,
    neuron_type,
    num_mfs,
    activation,
    optimizer,
    i_seed,
    rng_seed,
    results_df,
    path_to_results,
):
    """
    Run an experiment with the given configuration and save the results.

    Parameters:
    - train_data (tuple): Tuple containing features and labels for the training set.
    - test_data (tuple): Tuple containing features and labels for the testing set.
    - data_encoding (str) : Specifies the data encoding method to be used. This parameter affects how data
      is processed within the model. Example: 'no-encoding', 'one-hot-encoding'.
    - pred_method (str): The prediction method to be used by the FNN model (e.g., argmax).
    - map_class_dict (dict): A dictionary that maps the predicted class values (used internally by the model)
      to their original dataset class values.
    - neuron_type (str): Type of neuron to use in the FNN model.
    - num_mfs (int): Number of membership functions for each input dimension.
    - activation (str): Activation function to use in the FNN model.
    - optimizer (str): Optimizer algorithm to use for training the FNN model.
    - i_seed (int): Seed for the experiment.
    - rng_seed (int): Seed for random number generation in the FNN model.
    - results_df (DataFrame): DataFrame to store the results of each experiment.
    - path_to_results (str): Path to the directory where experiment results will be saved.

    Returns:
    None
    """

    current_neuron_type, fuzzy_interpretation = neuron_type.split("_")

    exp_str = f"/exp-seed_{i_seed}_neurontype_{current_neuron_type}_interp_{fuzzy_interpretation}_nummfs_{num_mfs}_activation_{activation}/"
    path_to_exp_results = path_to_results + exp_str

    if not os.path.exists(path_to_exp_results):
        os.makedirs(path_to_exp_results, exist_ok=True)

    #x_train, y_train = train_data[0], train_data[1]
    x_test, y_test = test_data[0], test_data[1]

    #Population
    max_generations = 50
    #mutation_rate = 0.1 #Probabilità di mutazione di un gene

    mu = 10 # = pop_size
    lambda_ = 20 #Numero di figli generati ad ogni iterazione (da cui si prendono #mu individui per la successiva generazione)
    selection_strategy = "plus" #otherwise "comma"
    mutation_ind_rate = 0.5 #Probabilità di mutazione di un individuo
    
    (x_train, y_train), (x_eval, y_eval) = get_train_eval_split(train_data, percentage_train=0.8)
    
    
    population = initialize_population(mu, num_mfs, current_neuron_type, fuzzy_interpretation, activation, optimizer, x_train, mutation_ind_rate, rng_seed)
    
    generation = 0
    best_fitness = 0
    best_guy = None
    max_patience = 20
    patience = max_patience
    epoch_improoved = False
    #pbar = tqdm(range(1,max_generations))

    for generation in range(max_generations):
        if patience == 0:
            break
        
        population = selection(population, mu, lambda_, mutation_rate,
                                        fitness_function, x_train, y_train, data_encoding, pred_method,
                                        map_class_dict, selection_strategy)
        
        epoch_improoved = False
        
        for individuo in population:
            fitness_eval = individuo.calculate_fitness(fitness_function, x_eval, y_eval, data_encoding, pred_method, map_class_dict, update_fitness = False)
        
            if fitness_eval > best_fitness: #Save the best individuo
                best_fitness = fitness_eval
                best_guy = copy.deepcopy(individuo)
                epoch_improoved = True

        if epoch_improoved:
            patience = max_patience
        else:
            patience -= 1
        
        generation += 1
        
        
        #Il best individuo è quello con la migliore performance sul validation set
        print("best individuo (results on train set): ", best_guy.fitness)
        print("best individuo (results on val set): ", best_guy.calculate_fitness(fitness_function, x_eval, y_eval, data_encoding, pred_method, map_class_dict, update_fitness = False))
        print("best individuo (results on test set): ", best_guy.calculate_fitness(fitness_function, x_test, y_test, data_encoding, pred_method, map_class_dict, update_fitness = False))
        print("patient: ", patience)
        

    print("\n\nEvolution part done:") #Il best individuo è quello con la migliore performance sul validation set
    print("Best individuo: ", best_guy.fitness)
    print("Best individuo eval set: ", best_guy.calculate_fitness(fitness_function, x_eval, y_eval, data_encoding, pred_method, map_class_dict, update_fitness = False))
    print("Best individuo test set: ", best_guy.calculate_fitness(fitness_function, x_test, y_test, data_encoding, pred_method, map_class_dict, update_fitness = False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-dataset", type=str, default="liver", help="specify the dataset to use"
    )
    parser.add_argument(
        "-path_to_conf",
        type=str,
        default="./experiments/configurations/iris/conf-00.json",
        help="configuration file for the current experiment",
    )
    parser.add_argument(
        "-path_to_results",
        type=str,
        default="./experiments/results/liver3/",
        help="directory where to store the results",
    )

    args = parser.parse_args()

    dataset = args.dataset
    path_to_conf = args.path_to_conf
    path_to_results = args.path_to_results
    conf = get_configuration(path_to_conf)

    # experiment setting
    num_seeds = conf["num_seeds"]
    neuron_types= conf["neuron_types"]
    num_mfs_options = conf["num_mfs_options"]
    activation = conf["activation"]
    optimizer = conf["optimizer"]
    data_encoding = conf["data_encoding"]
    pred_method = conf["pred_method"]
    fitness_function = conf["fitness_function"]
    mutation_rate = conf["mutation_rate"]

    data_train, data_test, map_class_dict = get_data(dataset, data_encoding)

    # this store the results of each run
    results_df = pd.DataFrame(
        columns=[
            "Seed", "NeuronType", "MFs", "Train_Acc.", "Train_F1", "Train_Rec.", "Train_Prec.",
            "Train_Spec.", "Test_Acc.", "Test_F1", "Test_Rec.", "Test_Prec.", "Test_Spec.",
        ]
    )
    i_seed = 5 #np.random.default_rng(num_seeds)
    rng_seed = np.random.default_rng(i_seed)
    run_experiment(
                    data_train,
                    data_test,
                    data_encoding,
                    pred_method,
                    fitness_function,
                    mutation_rate,
                    map_class_dict,
                    "andneuron_prod-probsum",
                    2,
                    activation,
                    optimizer,
                    i_seed,
                    rng_seed,
                    results_df,
                    path_to_results
                )
    

    """ for i_seed in range(num_seeds):
        # run_rng
        rng_seed = np.random.default_rng(i_seed)
        for neuron_type in neuron_types:
            for num_mfs in num_mfs_options:
                run_experiment(
                    data_train,
                    data_test,
                    data_encoding,
                    pred_method,
                    map_class_dict,
                    neuron_type,
                    num_mfs,
                    activation,
                    optimizer,
                    i_seed,
                    rng_seed,
                    results_df,
                    path_to_results
                ) """

    # save results
    results_df.to_csv(path_to_results + "runs_results.csv")
    # compute mean and sd
    calculate_avg_results(results_df, path_to_results)


