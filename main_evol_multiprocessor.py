import argparse
import os

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
from sklearn.model_selection import train_test_split
import copy
import pandas as pd 
import seaborn as sns
import time
from multiprocessing import Lock, Process, Queue, current_process
import queue



def do_job(tasks_to_accomplish, data_train, data_test, data_encoding, pred_method, fitness_function, map_class_dict, activation, optimizer, filename_file, default_path_results, path_to_results, core_id):
    print("Core ID: ", core_id)
    global_results = pd.DataFrame(columns=["Seed", "NeuronType", "MFs", "update_gene", "mutation_rate", "mutation_individual_rate", "crossover_rate",  "max_generations", "max_patience", "mu", "lambda", "selection_strategy", "Train_Acc.", "Dev_Acc.", "Test_Acc.", "time"])

    while True:
        try:
            task = tasks_to_accomplish.get_nowait()

            i_seed = task["seed"]
            neuron_type = task["neuron_type"]
            num_mfs = task["num_mfs"]
            mut_rate = task["mut_rate"]
            mut_ind_rate = task["mut_ind_rate"]
            cross_rate = task["cross_rate"]
            max_gen = task["max_gen"]
            max_pat = task["max_pat"]
            mu = task["mu"]
            lamb = task["lambda"]
            sel_str = task["sel_str"]
            update_gene = task["update_gene"]

            rng_seed = np.random.default_rng(i_seed)
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
                update_gene,
                activation,
                optimizer,
                i_seed,
                rng_seed,
                local_results,
                path_to_results
            )
            end_time = time.time()
            elapsed_time = end_time - start_time

            os.makedirs(default_path_results, exist_ok=True)
            local_results.to_csv(default_path_results + f"local_results_seed_{i_seed}_neurontype_{neuron_type}_nummfs_{num_mfs}_updategene_{update_gene}_mutrate_{mut_rate}_mutindrate_{mut_ind_rate}_crossrate_{cross_rate}_maxgen_{max_gen}_maxpat_{max_pat}_mu_{mu}_lambda_{lamb}_selstr_{sel_str}.csv", index=False)

            new_result = pd.DataFrame({"Seed": [i_seed], "NeuronType": [neuron_type], "MFs": [num_mfs], "update_gene": [update_gene], "mutation_rate": [mut_rate], "mutation_individual_rate": [mut_ind_rate], "crossover_rate": [cross_rate],  "max_generations": [max_gen], "max_patience": [max_pat], "mu": [mu], "lambda": [lamb], "selection_strategy": [sel_str], "Train_Acc.": [result_train["accuracy"]], "Dev_Acc.": [result_eval["accuracy"]], "Test_Acc.": [result_test["accuracy"]], "Train_F1": [result_train["fscore"]], "Dev_F1": [result_eval["fscore"]], "Test_F1": [result_test["fscore"]], "time": [elapsed_time]})
            global_results = pd.concat([global_results, new_result], ignore_index=True)
            global_results.to_csv(filename_file, index=False)

        except queue.Empty:
            print("DONE: " + str(core_id))    
            break
    
    return True






def initialize_population(pop_size, num_mfs, update_gene, neuron_type, fuzzy_interpretation, activation, optimizer, x_train, mutation_ind_rate, data_encoding, rng_seed):
    population = []
    
    for _ in range(pop_size):
        individuo = FNNModel(num_mfs=num_mfs, update_gene=update_gene, neuron_type=neuron_type, interpretation=fuzzy_interpretation, activation=activation, optimizer=optimizer, visualizeMF=False, mutation_ind_rate=mutation_ind_rate, data_encoding=data_encoding, rng_seed=rng_seed)
        individuo.initialize_individual(x_train)
        population.append(individuo)
        
    return population



def get_population_performance(fitness_population):
    mean_fitness = np.mean(fitness_population)
    std_fitness = np.std(fitness_population)
    max_fitness = np.max(fitness_population)
    min_fitness = np.min(fitness_population)
    
    return mean_fitness, std_fitness, max_fitness, min_fitness


def get_train_eval_split(data, percentage_train=0.8): #Potremmo usare sklearn.model_selection.train_test_split; ma dobbiamo aggiungere la libreria
    x_data, y_data = data[0], data[1]
    x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_data, train_size=percentage_train, random_state=0, stratify=y_data)
    
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
    update_gene,
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
    
    
    population = initialize_population(mu, num_mfs, update_gene, current_neuron_type, fuzzy_interpretation, activation, optimizer, x_train, mutation_ind_rate, data_encoding, rng_seed)
    
    generation = 0
    best_fitness = 0
    best_guy = None
    # max_patience = 20
    patience = max_patience
    epoch_improoved = False
    #pbar = tqdm(range(1,max_generations))
    
    for generation in range(max_gen):
        print("Generation: ", generation)
        if patience == 0:
            break
        
        population = selection(population, mu, lambda_, mutation_rate, crossover_rate, rng_seed,
                                        fitness_function, x_train, y_train, data_encoding, pred_method,
                                        map_class_dict, selection_strategy)
        
        epoch_improoved = False
        
        performance_train = []
        performance_eval = []
        performance_test = []
        
        for individuo in population:
            fitness_train = individuo.fitness
            performance_train.append(fitness_train)
            
            fitness_eval = individuo.calculate_fitness(fitness_function, x_eval, y_eval, data_encoding, pred_method, map_class_dict, update_fitness = False)
            performance_eval.append(fitness_eval)
            
            fitness_test = individuo.calculate_fitness(fitness_function, x_test, y_test, data_encoding, pred_method, map_class_dict, update_fitness = False)
            performance_test.append(fitness_test)
        
            if fitness_eval > best_fitness or best_guy is None: #Save the best individuo
                best_fitness = fitness_eval
                best_guy = copy.deepcopy(individuo)
                epoch_improoved = True

        mean_performance_train, std_performance_train, max_performance_train, min_performance_train = get_population_performance(performance_train)
        mean_performance_eval, std_performance_eval, max_performance_eval, min_performance_eval = get_population_performance(performance_eval)
        mean_performance_test, std_performance_test, max_performance_test, min_performance_test = get_population_performance(performance_test)

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
    
    
    
    result_train = best_guy.evaluate_model(x_train, y_train, data_encoding, pred_method, map_class_dict)
    result_eval = best_guy.evaluate_model(x_eval, y_eval, data_encoding, pred_method, map_class_dict)
    result_test = best_guy.evaluate_model(x_test, y_test, data_encoding, pred_method, map_class_dict)
    
    return result_train, result_eval, result_test, local_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-dataset", type=str, default="mammography", help="specify the dataset to use"
    )
    parser.add_argument(
        "-path_to_conf",
        type=str,
        default="./experiments/configurations/mammography/conf-01_evo.json",
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
    mu_values = conf["mu"]
    lambda_values = conf["lambda_"]
    selection_strategies = conf["selection_strategies"]
    mutation_individual_rate = conf["mutation_individual_rate"]
    default_path_results = path_to_results + conf["path_to_results"]
    update_genes = conf["update_genes"]
    
    print("Configuration loaded")
    print("data_encoding: ", data_encoding) 
    print("pred_method: ", pred_method)


    data_train, data_test, map_class_dict = get_data(dataset, data_encoding)

    # used for debugging
    """ local_results = pd.DataFrame(columns=["Epoch", "Train_max_fitness", "Train_min_fitness", "Train_avg_fitness", "Train_std_fitness", "Dev_max_fitness", "Dev_min_fitness", "Dev_avg_fitness", "Dev_std_fitness", "Test_max_fitness", "Test_min_fitness", "Test_avg_fitness", "Test_std_fitness"])
    
    i_seed = 1 #np.random.default_rng(num_seeds)
    rng_seed = np.random.default_rng(i_seed)
    result_train, result_eval, result_test, local_results = run_experiment(
                    data_train,
                    data_test,
                    data_encoding,
                    pred_method,
                    fitness_function,
                    mutation_rate=0.9,
                    mutation_ind_rate=0.5,
                    crossover_rate=0.5,
                    max_gen = 20,
                    max_patience= 1000,
                    mu = 20, 
                    lambda_ = 60,
                    selection_strategy = "plus",
                    map_class_dict = map_class_dict,
                    neuron_type = "andneuron_prod-probsum",
                    num_mfs = 2,
                    activation = "linear",
                    optimizer = "moore-penrose",
                    i_seed = i_seed,
                    rng_seed = rng_seed,
                    local_results = local_results,
                    path_to_results = "")
    
    local_results.to_csv("demo_results.csv")
    exit(0) """
    
    global_results = pd.DataFrame(columns=["Seed", "NeuronType", "MFs", "mutation_rate", "mutation_individual_rate", "crossover_rate",  "max_generations", "max_patience", "mu", "lambda", "selection_strategy", "Train_Acc.", "Dev_Acc.", "Test_Acc.", "time"])


    # get the filename for the global results
    base_filename = "global_result_"
    extension= ".csv"
    new_file = False
    complete_filename = ""
    counter = 0
    filename_no_extension = ""
    
    while not new_file: #Check if the file already exists, if so, generate a new filename
        id = str(counter)
        complete_filename = base_filename + id + extension
        filename_no_extension = base_filename + id
        
        if not os.path.exists(f"{default_path_results}{complete_filename}"):
            new_file = True
            
        counter+=1

    tasks_pending = Queue()
    tasks_done = Queue()

    for i_seed in range(num_seeds):
        for neuron_type in neuron_types:
            for num_mfs in num_mfs_options:
                for update_gene in update_genes:
                    for mut_rate in mutation_rate:
                        for mut_ind_rate in mutation_individual_rate:
                            for cross_rate in crossover_rate:
                                for max_gen in max_generations:
                                    for max_pat in max_patience:
                                        if max_pat > max_gen:
                                            continue
                                        for mu in mu_values:
                                            for lamb in lambda_values:
                                                if mu > lamb:
                                                    continue
                                                for sel_str in selection_strategies:
                                                    
                                                    parametri = {"seed": i_seed, "neuron_type": neuron_type, "num_mfs": num_mfs, "update_gene":update_gene, "mut_rate": mut_rate, "mut_ind_rate": mut_ind_rate, "cross_rate": cross_rate, "max_gen": max_gen, "max_pat": max_pat, "mu": mu, "lambda": lamb, "sel_str": sel_str}
                                                    tasks_pending.put(parametri)

    #get the number of processors
    num_processors = os.cpu_count()
    print("Number of processors: ", num_processors)

    #create a process
    processes = []
    for core_id in range(num_processors):
        filename_file = f"{default_path_results}{filename_no_extension}_{core_id}{extension}" 
        process = Process(target=do_job, args=(tasks_pending, data_train, data_test, data_encoding, pred_method, fitness_function, map_class_dict, activation, optimizer, filename_file, default_path_results, path_to_results, core_id))
        processes.append(process)
        process.start()

    
    for process in processes:
        process.join()


