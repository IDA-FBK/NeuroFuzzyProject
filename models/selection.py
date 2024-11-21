import random
from models.crossover import *


def selection(population, selection_mu, selection_lambda, mutation_rate, selection_strategy="comma"):
    """
    Select the best individuals from the population, eventually with mutation.
    population: list of Individual
    selection_mu: int, number of individuals to select from the population
    selection_lambda: int, number of individuals to generate
    mutation_rate: float, probability of mutation
    selection_strategy: str, either "plus" or "comma"
    """
    # Generate new individuals
    offspring = generate_offspring(population, selection_lambda, mutation_rate)
    # Select the best individuals
    if selection_strategy == "plus":
        offspring += population
    elif selection_strategy != "comma":
        raise ValueError("Invalid selection strategy")
    offspring.sort(key=lambda x: x.fitness, reverse=True)
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
    # Generate offspring
    offspring = []
    for _ in range(selection_lambda):
        # Select two parents
        parent1 = random.choices(parents, weights=parent_probabilities)[0]
        parent2 = random.choices(parents, weights=parent_probabilities)[0]
        # Crossover
        child = crossover(parent1, parent2)
        # Mutation
        child.mutate(mutation_rate=mutation_rate)
        offspring.append(child)
    return offspring
