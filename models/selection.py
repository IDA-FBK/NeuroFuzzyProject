import random
from models.crossover import *


def selection(
        population,
        selection_mu,
        selection_lambda,
        mutation_rate,
        fitness_type,
        x_train,
        y_train,
        data_encoding,
        pred_method,
        map_class_dict,
        selection_strategy="comma"
        ):
    """
    Select the best individuals from the population, eventually with mutation.
    population: list of Individual
    selection_mu: int, number of individuals to select from the population
    selection_lambda: int, number of individuals to generate
    mutation_rate: float, probability of mutation
    selection_strategy: str, either "plus" or "comma"
    """
    if selection_mu > selection_lambda:
        raise ValueError("selection_mu must be greater than selection_lambda")
    if selection_strategy not in ["plus", "comma"]:
        raise ValueError("selection_strategy must be either 'plus' or 'comma'")
    
    # Calculate fitness of population
    for individual in population:
        if individual.fitness is None:
            individual.calculate_fitness(fitness_type, x_train, y_train, data_encoding, pred_method, map_class_dict)

    offspring = generate_offspring(population, selection_lambda, mutation_rate)
    
    # Calculate fitness of offspring
    for individual in offspring:
        if individual.fitness is None:
            individual.calculate_fitness(fitness_type, x_train, y_train, data_encoding, pred_method, map_class_dict)

    # Select the best individuals according to the selection strategy
    if selection_strategy == "plus":
        offspring += population
    offspring.sort(key=lambda x: x.fitness, reverse=True)
    
    # for i in range(len(offspring)):
    #     if offspring[i].fitness != 0.38:
    #         print(offspring[i].fitness)

    return offspring[:selection_mu]


def generate_offspring(parents, selection_lambda, mutation_rate):
    """
    Generate new individuals from the parents, with probability of parent selection proportional to fitness.
    parents: list of Individual
    selection_lambda: int, number of individuals to generate
    mutation_rate: float, probability of mutation
    """
    # Compute the probability of selecting each parent
    fitness_sum = sum(p.fitness for p in parents)
    parent_probabilities = [p.fitness / fitness_sum for p in parents]

    offspring = []
    for _ in range(selection_lambda):
        parent1 = random.choices(parents, weights=parent_probabilities)[0]
        parent2 = random.choices(parents, weights=parent_probabilities)[0]
        child = crossover(parent1, parent2)
        #Potremmo mutare solo con una probabilità
        child.mutate(mutation_rate=mutation_rate)
        child.fitness = None
        offspring.append(child)
    return offspring
