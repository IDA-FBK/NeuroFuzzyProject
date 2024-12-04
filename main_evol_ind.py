import argparse
import os.path

import numpy as np
import pandas as pd
from tqdm import tqdm

from data.data import get_data
from experiments.configurations.configurations import get_configuration
from experiments.evaluation import evaluate_interpretability
from experiments.calculate import calculate_avg_results
from experiments.utils import save_list_in_a_file
from experiments.plots import plot_class_confusion_matrix
from models.models import FNNModel
from models.selection import selection
import copy
import pandas as pd 
import seaborn as sns
import time


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
    mutation_ind_rate,
    crossover_rate,
    max_gen,
    max_patience,
    mu, 
    lambda_,
    selection_strategy,
    map_class_dict,
    neuron_type,
    num_mfs,
    activation,
    optimizer,
    i_seed,
    rng_seed,
    local_results,
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
    - local_results (DataFrame): DataFrame to store the results of each experiment.
    - path_to_results (str): Path to the directory where experiment results will be saved.

    Returns:
    None
    """

    current_neuron_type, fuzzy_interpretation = neuron_type.split("_")

    exp_str = f"/exp-seed_{i_seed}_neurontype_{current_neuron_type}_interp_{fuzzy_interpretation}_nummfs_{num_mfs}_activation_{activation}/"
    path_to_exp_results = path_to_results + exp_str

    """ if not os.path.exists(path_to_exp_results):
        os.makedirs(path_to_exp_results, exist_ok=True) """

    #x_train, y_train = train_data[0], train_data[1]
    x_test, y_test = test_data[0], test_data[1]

    #Population
    # max_generations = 50
    # #mutation_rate = 0.1 #Probabilità di mutazione di un gene

    # mu = 10 # = pop_size
    # lambda_ = 20 #Numero di figli generati ad ogni iterazione (da cui si prendono #mu individui per la successiva generazione)
    # selection_strategy = "plus" #otherwise "comma"
    # mutation_ind_rate = 0.5 #Probabilità di mutazione di un individuo
    
    (x_train, y_train), (x_eval, y_eval) = get_train_eval_split(train_data, percentage_train=0.8)
    
    
    population = initialize_population(mu, num_mfs, current_neuron_type, fuzzy_interpretation, activation, optimizer, x_train, mutation_ind_rate, rng_seed)
    
    generation = 0
    best_fitness = 0
    best_guy = None
    # max_patience = 20
    patience = max_patience
    epoch_improoved = False
    #pbar = tqdm(range(1,max_generations))
    
    

    for generation in range(max_gen):
        if patience == 0:
            break
        
        population = selection(population, mu, lambda_, mutation_rate, crossover_rate,
                                        fitness_function, x_train, y_train, data_encoding, pred_method,
                                        map_class_dict, selection_strategy)
        
        epoch_improoved = False
        
        performance_train = []
        performance_eval = []
        performance_test = []
        
        for individuo in population:
            fitness_eval = individuo.calculate_fitness(fitness_function, x_eval, y_eval, data_encoding, pred_method, map_class_dict, update_fitness = False)
            performance_eval.append(fitness_eval)
            
            fitness_train = individuo.calculate_fitness(fitness_function, x_train, y_train, data_encoding, pred_method, map_class_dict, update_fitness = False)
            performance_train.append(fitness_train)
            
            fitness_test = individuo.calculate_fitness(fitness_function, x_test, y_test, data_encoding, pred_method, map_class_dict, update_fitness = False)
            performance_test.append(fitness_test)
        
            if fitness_eval > best_fitness: #Save the best individuo
                best_fitness = fitness_eval
                best_guy = copy.deepcopy(individuo)
                epoch_improoved = True

        mean_performance_train = np.mean(performance_train)
        std_performance_train = np.std(performance_train)
        max_performance_train = np.max(performance_train)
        min_performance_train = np.min(performance_train)
        
        mean_performance_eval = np.mean(performance_eval)
        std_performance_eval = np.std(performance_eval)
        max_performance_eval = np.max(performance_eval)
        min_performance_eval = np.min(performance_eval)
        
        mean_performance_test = np.mean(performance_test)
        std_performance_test = np.std(performance_test)
        max_performance_test = np.max(performance_test)
        min_performance_test = np.min(performance_test)
        
        new_row = pd.DataFrame({"Epoch": [generation], "Train_max_fitness": [max_performance_train], "Train_min_fitness": [min_performance_train], "Train_avg_fitness": [mean_performance_train], "Train_std_fitness": [std_performance_train], "Dev_max_fitness": [max_performance_eval], "Dev_min_fitness": [min_performance_eval], "Dev_avg_fitness": [mean_performance_eval], "Dev_std_fitness": [std_performance_eval], "Test_max_fitness": [max_performance_test], "Test_min_fitness": [min_performance_test], "Test_avg_fitness": [mean_performance_test], "Test_std_fitness": [std_performance_test]})
        local_results = pd.concat([local_results, new_row], ignore_index=True)

        if epoch_improoved:
            patience = max_patience
        else:
            patience -= 1
        
        generation += 1
        
        
        #Il best individuo è quello con la migliore performance sul validation set
        #print("best individuo (results on train set): ", best_guy.fitness)
        #print("best individuo (results on val set): ", best_guy.calculate_fitness(fitness_function, x_eval, y_eval, data_encoding, pred_method, map_class_dict, update_fitness = False))
        #print("best individuo (results on test set): ", best_guy.calculate_fitness(fitness_function, x_test, y_test, data_encoding, pred_method, map_class_dict, update_fitness = False))
        #print("patient: ", patience)
        

    #print("\n\nEvolution part done:") #Il best individuo è quello con la migliore performance sul validation set
    #print("Best individuo: ", best_guy.fitness)
    #print("Best individuo eval set: ", best_guy.calculate_fitness(fitness_function, x_eval, y_eval, data_encoding, pred_method, map_class_dict, update_fitness = False))
    #print("Best individuo test set: ", best_guy.calculate_fitness(fitness_function, x_test, y_test, data_encoding, pred_method, map_class_dict, update_fitness = False))
    
    
    result_train = best_guy.calculate_fitness(fitness_function, x_train, y_train, data_encoding, pred_method, map_class_dict, update_fitness = False)
    result_eval = best_guy.calculate_fitness(fitness_function, x_eval, y_eval, data_encoding, pred_method, map_class_dict, update_fitness = False)
    result_test = best_guy.calculate_fitness(fitness_function, x_test, y_test, data_encoding, pred_method, map_class_dict, update_fitness = False)
    
    return result_train, result_eval, result_test, local_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-dataset", type=str, default="liver", help="specify the dataset to use"
    )
    parser.add_argument(
        "-path_to_conf",
        type=str,
        default="./experiments/configurations/iris/conf-01.json",
        help="configuration file for the current experiment",
    )
    parser.add_argument(
        "-path_to_results",
        type=str,
        default="./experiments/results/",
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
    crossover_rate = conf["crossover_rate"]
    max_generations = conf["max_generations"]
    max_patience = conf["max_patience"]
    mu_values = conf["mu_values"]
    lambda_values = conf["lambda_values"]
    selection_strategies = conf["selection_strategies"]
    mutation_individual_rate = conf["mutation_individual_rate"]
    default_path_results = path_to_results + conf["path_to_results"]


    data_train, data_test, map_class_dict = get_data(dataset, data_encoding)

    # this store the results of each run
    """ results_df = pd.DataFrame(
        columns=[
            "Seed", "NeuronType", "MFs", "Train_Acc.", "Train_F1", "Train_Rec.", "Train_Prec.",
            "Train_Spec.", "Test_Acc.", "Test_F1", "Test_Rec.", "Test_Prec.", "Test_Spec.",
        ]
    ) """
    
    
    """ i_seed = 5 #np.random.default_rng(num_seeds)
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
                ) """
    
    global_results = pd.DataFrame(columns=["Seed", "NeuronType", "MFs", "mutation_rate", "mutation_individual_rate", "crossover_rate",  "max_generations", "max_patience", "mu", "lambda", "selection_strategy", "Train_Acc.", "Dev_Acc.", "Test_Acc.", "time"])

    for i_seed in range(num_seeds):
        rng_seed = np.random.default_rng(i_seed)
        for neuron_type in neuron_types:
            for num_mfs in num_mfs_options:
                for mut_rate in mutation_rate:
                    for mut_ind_rate in mutation_individual_rate:
                        for cross_rate in crossover_rate:
                            for max_gen in max_generations:
                                for max_pat in max_patience:
                                    for mu in mu_values:
                                        for lamb in lambda_values:
                                            for sel_str in selection_strategies:
                                                
                                                local_results = pd.DataFrame(columns=["Epoch", "Train_max_fitness", "Train_min_fitness", "Train_avg_fitness", "Train_std_fitness", "Dev_max_fitness", "Dev_min_fitness", "Dev_avg_fitness", "Dev_std_fitness", "Test_max_fitness", "Test_min_fitness", "Test_avg_fitness", "Test_std_fitness"])
                                                
                                                start_time = time.time()
                                                result_train, result_eval, result_test, local_results = run_experiment(
                                                    data_train,
                                                    data_test,
                                                    data_encoding,
                                                    pred_method,
                                                    fitness_function,
                                                    mut_rate,
                                                    mut_ind_rate,
                                                    cross_rate,
                                                    max_gen, 
                                                    max_pat,
                                                    mu, 
                                                    lamb,
                                                    sel_str,
                                                    map_class_dict,
                                                    neuron_type,
                                                    num_mfs,
                                                    activation,
                                                    optimizer,
                                                    i_seed,
                                                    rng_seed,
                                                    local_results,
                                                    path_to_results
                                                )
                                                end_time = time.time()
                                                elapsed_time = end_time - start_time
                                                
                                                local_results.to_csv(default_path_results + f"local_results_seed_{i_seed}_neurontype_{neuron_type}_nummfs_{num_mfs}_mutrate_{mut_rate}_mutindrate_{mut_ind_rate}_crossrate_{cross_rate}_maxgen_{max_gen}_maxpat_{max_pat}_mu_{mu}_lambda_{lamb}_selstr_{sel_str}.csv")
                                                
                                                
                                                new_result = pd.DataFrame({"Seed": [i_seed], "NeuronType": [neuron_type], "MFs": [num_mfs], "mutation_rate": [mut_rate], "mutation_individual_rate": [mut_ind_rate], "crossover_rate": [cross_rate],  "max_generations": [max_gen], "max_patience": [max_pat], "mu": [mu], "lambda": [lamb], "selection_strategy": [sel_str], "Train_Acc.": [result_train], "Dev_Acc.": [result_eval], "Test_Acc.": [result_test], "time": [elapsed_time]})
                                                global_results = pd.concat([global_results, new_result], ignore_index=True)
                    

    # save results
    base_filename = "global_result_"
    extension= ".csv"
    new_file = False
    complete_filename = ""
    counter = 0
    
    while not new_file: #Check if the file already exists, if so, generate a new filename
        id = str(counter)
        complete_filename = base_filename + id + extension
        
        if not os.path.exists(f"{default_path_results}{complete_filename}"):
            new_file = True
            
        counter+=1
    
    global_results.to_csv(default_path_results + complete_filename)
    
    # compute mean and sd
    #calculate_avg_results(results_df, path_to_results)


