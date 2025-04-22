import time
from typing import List

from models.crossover import *
from models.models import FNNModel


def selection(
        population: List[FNNModel],
        selection_mu,
        selection_lambda,
        mutation_rate,
        crossover_rate,
        rng_seed,
        fitness_type,
        x,
        y,
        data_encoding,
        pred_method,
        map_class_dict,
        time_tracker,
        selection_strategy="comma",
        method="tournament",
        tournament_size=3
        ):
    """
    Select the best individuals from the population, eventually with mutation.
    population: list of Individual
    selection_mu: int, number of individuals to select from the population
    selection_lambda: int, number of individuals to generate
    mutation_rate: float, probability of mutation
    selection_strategy: str, either "plus" or "comma"
    x, y: input-output data for fitness calculation
    """
    pop_size = len(population)
    if selection_mu > selection_lambda:
        raise ValueError("selection_lambda must be greater than selection_mu")
    if selection_strategy not in ["plus", "comma"]:
        raise ValueError("selection_strategy must be either 'plus' or 'comma'")

    # Start timing the computation of the fitness on the population
    start_time_pop_fit = time.time()
    # Calculate fitness of population
    for individual in population:
        if individual.fitness is None:
            individual.calculate_fitness(fitness_type, x, y, data_encoding, pred_method, map_class_dict,fast=False)[fitness_type]
    # End timing the computation of the fitness on the population
    end_time_pop_fit = time.time()
    fitness_pop_time = round(end_time_pop_fit - start_time_pop_fit, 4)

    time_tracker["fitness_pop_time"] = fitness_pop_time
    print(f"[TIME] COMPUTATION OF THE FITNESS ON POPULATION TOOK {fitness_pop_time}s - avg time per individual: {fitness_pop_time/pop_size:.4f}s")

    # Start timing the generation of the offspring
    start_time_off_gen = time.time()
    offspring:List[FNNModel] = generate_offspring(population, selection_lambda, mutation_rate, crossover_rate, rng_seed, method=method, tournament_size=tournament_size)
    end_time_off_gen = time.time()

    off_gen_time = round(end_time_off_gen-start_time_off_gen, 4)
    time_tracker["off_gen_time"] = off_gen_time
    print(f"[TIME] GENERATION OF THE OFFSPRING TOOK {off_gen_time}s")

    time_tracker["off_params_time"] = 0
    time_tracker["off_fit_time"] = 0
    # Calculate fitness of offspring
    for individual in offspring:
        start_time_params_gen = time.time()
        individual.generate_parameters(y)
        end_time_params_gen= time.time()
        time_tracker["off_params_time"] += end_time_params_gen-start_time_params_gen

        start_time_fit = time.time()
        individual.calculate_fitness(fitness_type, x, y, data_encoding, pred_method, map_class_dict, fast=False)[fitness_type]
        end_time_fit = time.time()
        time_tracker["off_fit_time"] += end_time_fit-start_time_fit

    time_tracker["off_params_time"] = round(time_tracker["off_params_time"], 4)
    time_tracker["off_fit_time"] = round(time_tracker["off_fit_time"], 4)

    print(f'[TIME] GENERATION OF PARAMS FOR OFFSPRING TOOK {time_tracker["off_params_time"]}s - avg time per individual {time_tracker["off_params_time"]/pop_size:.4f}s')
    print(f'[TIME] COMPUTATION OF THE FITNESS ON THE OFFSPRING TOOK {time_tracker["off_fit_time"]}s - avg time per individual {time_tracker["off_fit_time"]/pop_size:.4f}s')
    # Select the best individuals according to the selection strategy
    if selection_strategy == "plus":
        offspring += population
    offspring.sort(key=lambda x: x.fitness, reverse=True)
    
    return offspring[:selection_mu]


def tournament(population, rng_seed, tournament_size):
    """
    Select the best individual from a random subset of the population.
    population: list of individuals
    tournament_size: int, number of individuals in the subset
    """
    subset = rng_seed.choice(population, tournament_size, replace=False)
    return max(subset, key=lambda x: x.fitness)


def generate_offspring(parents, selection_lambda, mutation_rate, crossover_rate, rng_seed, method="tournament", tournament_size=3):
    """
    Generate new individuals from the parents, with probability of parent selection proportional to fitness.
    parents: list of Individual
    selection_lambda: int, number of individuals to generate
    mutation_rate: float, probability of mutation
    method: str, either "tournament" or "proportional"
    """
    if method not in ["tournament", "proportional"]:
        raise ValueError("method must be either 'tournament' or 'proportional'")
    
    if method == "proportional":
        # Compute the probability of selecting each parent
        fitness_sum = sum(p.fitness for p in parents)
        parent_probabilities = [p.fitness / fitness_sum for p in parents]

    offspring = []
    for _ in range(selection_lambda):
        if method == "tournament":
            parent1 = tournament(parents, rng_seed, tournament_size=tournament_size)
            parent2 = tournament(parents, rng_seed, tournament_size=tournament_size)
        else:
            parent1 = rng_seed.choices(parents, weights=parent_probabilities)[0]
            parent2 = rng_seed.choices(parents, weights=parent_probabilities)[0]
        child = crossover(parent1, parent2, rng_seed, crossover_rate )
        child.mutate(mutation_rate=mutation_rate)
        offspring.append(child)
    return offspring
