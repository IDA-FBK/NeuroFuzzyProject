import random
import copy 

SWAP4ALL_MFS = True # If True, swap all gaussian parameters of a feature, for all its membership functions.


def crossover(parent1, parent2, crossover_rate=0.5):
    """
    Perform crossover between two individuals.
    This works by swapping some parameters of the two fuzzy neural networks.
    The parameters we consider are parent.neuron_weights, parent.V and gaussian parameters (parent.mf_params).
    crossover_rate: float, probability of swapping each parameter
    """
    child = copy_individual(parent1)

    # Swap neuron weights
    for i in range(len(child.neuron_weights)):
        if random.random() < crossover_rate:
            child.neuron_weights[i] = parent2.neuron_weights[i]

    # Swap last layer weights
    if random.random() < crossover_rate:
        child.V = parent2.V
        
    # Swap gaussian parameters
    for feature_index in range(len(child.mf_params)):
        if SWAP4ALL_MFS:
            if random.random() < crossover_rate:
                child.mf_params[feature_index] = parent2.mf_params[feature_index]
        else:
            for mf_index in range(child.num_mfs):
                if random.random() < crossover_rate:
                    child.mf_params[feature_index]["centers"][mf_index] = parent2.mf_params[feature_index]["centers"][mf_index]
                    child.mf_params[feature_index]["sigmas"][mf_index] = parent2.mf_params[feature_index]["sigmas"][mf_index]
    return child


def copy_individual(individual):
    """
    Create a deep copy of an individual.
    """
    new_individual = copy.deepcopy(individual)
    new_individual.fitness = None
    return new_individual
