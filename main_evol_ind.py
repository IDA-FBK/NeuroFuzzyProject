import argparse
import os.path

import numpy as np
import pandas as pd

from data.data import get_data
from experiments.configurations.configurations import get_configuration
from experiments.evaluation import evaluate_interpretability
from experiments.calculate import calculate_avg_results
from experiments.utils import save_list_in_a_file
from experiments.plots import plot_class_confusion_matrix
from models.models import FNNModel


def run_experiment(
    train_data,
    test_data,
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

    x_train, y_train = train_data[0], train_data[1]
    x_test, y_test = test_data[0], test_data[1]


    print(f"\n---\nModel: {neuron_type} with {num_mfs} MFs")
    fnn_model = FNNModel(
        num_mfs=num_mfs,
        neuron_type=current_neuron_type,
        interpretation=fuzzy_interpretation,
        activation=activation,
        optimizer=optimizer,
        visualizeMF=False,
        rng_seed=rng_seed,
    )

    fnn_model.initialize_individual(x_train) #this call fuzzification_layer for initialize the individual
    
    fitness_type = 'accuracy'
    fitness = fnn_model.calculate_fitness(fitness_type, x_train, y_train, data_encoding, pred_method, map_class_dict)
    # print("\nSummary of Performance Metrics:")
    #fnn_model.train_model(x_train, y_train)
    # evaluation_metrics_train = fnn_model.evaluate_model(x_train, y_train, data_encoding, pred_method, map_class_dict)
    # evaluation_metrics_test = fnn_model.evaluate_model(x_test, y_test, data_encoding, pred_method, map_class_dict)
    print('fitness: ', fitness)
    
    print("V ", fnn_model.V)
    print("neuron_weights: ", fnn_model.neuron_weights)
    
    print("gaussians: ")
    for feature_index in range(len(fnn_model.mf_params)):
        for mf_index in range(fnn_model.num_mfs):
            print(fnn_model.mf_params[feature_index]["centers"][mf_index], fnn_model.mf_params[feature_index]["sigmas"][mf_index])

    #Early stopping da implementare
    #Mutation rate decrease?
    #In caso: weights and biases
    
    #Variazioni di individui solo tra AND e solo tra OR. No mix tra tipi di neuroni. (Almeno per il momento, poi faccio interagire i tipi diversi di neruoni e numero diverso di membership functions)
    #Rimetto il for basato sul .json, e faccio i test con le varie configurazioni separatamente.

    #Confronto con la versione precedente del codice.

    #In caso fase esplorativa: leggo il paper e ne cerco altri.

    # mutation
    mutation_rate = 0.1
    fnn_model.mutate(mutation_rate)
    print("\nV, after mutation: ", fnn_model.V)
    print("neuron_weights, after mutation: ", fnn_model.neuron_weights)
    print("gaussians, after mutations: ")
    for feature_index in range(len(fnn_model.mf_params)):
        for mf_index in range(fnn_model.num_mfs):
            print(fnn_model.mf_params[feature_index]["centers"][mf_index], fnn_model.mf_params[feature_index]["sigmas"][mf_index])


    mutation_rate = 0.9
    fnn_model.mutate(mutation_rate)
    print("\nV, after mutation 2: ", fnn_model.V)
    print("neuron_weights, after 2 mutation: ", fnn_model.neuron_weights)

    rules = fnn_model.generate_fuzzy_rules()

    # Save fuzzy rules to a file
    save_list_in_a_file(rules, path_to_exp_results + "fuzzy_rules.txt")

    fnn_model.generate_fuzzy_axioms()

    # Save axioms in a file
    # TODO: These axioms were related to the integration with LTN Framework (To me we can remove it)
    # save_list_in_a_file(fnn_model.axioms, path_to_exp_results + "fuzzy_axiom.txt")

    """ # Plot confusion matrix of class prediction
    plot_class_confusion_matrix("TRAIN", evaluation_metrics_train["cm"], evaluation_metrics_train["unique_labels"],
                                path_to_exp_results)
    plot_class_confusion_matrix("TEST", evaluation_metrics_test["cm"], evaluation_metrics_test["unique_labels"],
                                path_to_exp_results)


    # Append both train and test metrics to the DataFrame
    results_df.loc[len(results_df)] = [
        i_seed,
        neuron_type,
        num_mfs,
        evaluation_metrics_train["accuracy"],  # Training accuracy
        evaluation_metrics_train["fscore"],  # Training F-score
        evaluation_metrics_train["recall"],  # Training Recall
        evaluation_metrics_train["precision"],  # Training Precision
        evaluation_metrics_train["specificity"],  # Training Specificity
        evaluation_metrics_test["accuracy"],  # Testing accuracy
        evaluation_metrics_test["fscore"],  # Testing F-score
        evaluation_metrics_test["recall"],  # Testing Recall
        evaluation_metrics_test["precision"],  # Testing Precision
        evaluation_metrics_test["specificity"],  # Testing Specificity
    ] """

    # TODO: evaluate interpretabilty is in stand-by
    #evaluate_interpretability(fnn_model, x_test, path_to_exp_results)


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



