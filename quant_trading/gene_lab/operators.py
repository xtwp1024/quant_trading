"""Genetic algorithm operators — absorbed from GeneTrader.

Provides:
- crossover(): single-point crossover with optional trading-pair exchange
- mutate(): typed gene mutation (Int/Decimal/Categorical/Boolean)
- select_tournament(): tournament selection
- calculate_genetic_distance(): normalized distance between individuals
- calculate_population_diversity(): average pairwise diversity
- select_with_diversity(): diversity-aware tournament selection
- maintain_diversity(): boost mutation when population converges
"""
import random
from typing import List, Tuple, Dict, Any, Optional

from quant_trading.gene_lab.individual import Individual


def crossover(
    parent1: Individual,
    parent2: Individual,
    with_pair: bool = True
) -> Tuple[Individual, Individual]:
    """Single-point crossover between two individuals.

    Args:
        parent1: First parent
        parent2: Second parent
        with_pair: Whether to also crossover trading pairs

    Returns:
        Tuple of two child individuals
    """
    if len(parent1.genes) < 2:
        return parent1.copy(), parent2.copy()

    point = random.randint(1, len(parent1.genes) - 1)
    child1_genes = parent1.genes[:point] + parent2.genes[point:]
    child2_genes = parent2.genes[:point] + parent1.genes[point:]

    if with_pair and parent1.trading_pairs and parent2.trading_pairs:
        all_pairs = list(set(parent1.trading_pairs + parent2.trading_pairs))
        random.shuffle(all_pairs)
        child1_pairs = all_pairs[:len(parent1.trading_pairs)]
        child2_pairs = all_pairs[:len(parent2.trading_pairs)]
        return (
            Individual(child1_genes, child1_pairs, parent1.param_types),
            Individual(child2_genes, child2_pairs, parent2.param_types),
        )

    return (
        Individual(child1_genes, parent1.trading_pairs.copy(), parent1.param_types),
        Individual(child2_genes, parent2.trading_pairs.copy(), parent2.param_types),
    )


def mutate(individual: Individual, mutation_rate: float) -> None:
    """Apply mutation to an individual's genes.

    Supports multiple mutation strategies:
    - noise:   Add Gaussian noise to numeric values
    - reset:   Reset to random value within range
    - scale:   Scale value by random factor [0.8, 1.2]

    Args:
        individual: Individual to mutate (modified in place)
        mutation_rate: Probability of mutating each gene
    """
    for i in range(len(individual.genes)):
        if random.random() >= mutation_rate:
            continue

        param_type = individual.param_types[i]

        if isinstance(param_type, dict) and "type" in param_type:
            _mutate_typed_gene(individual, i, param_type)
        elif isinstance(individual.genes[i], bool):
            individual.genes[i] = not individual.genes[i]
        elif isinstance(param_type, dict) and "options" in param_type:
            individual.genes[i] = random.choice(param_type["options"])


def _mutate_typed_gene(individual: Individual, index: int, param_type: Dict[str, Any]) -> None:
    """Apply mutation to a typed gene (Int, Decimal, Boolean, Categorical)."""
    gene_type = param_type["type"]

    if gene_type == "Boolean":
        individual.genes[index] = not individual.genes[index]
        return

    if gene_type == "Categorical":
        options = param_type.get("options", [])
        if options:
            individual.genes[index] = random.choice(options)
        return

    if gene_type not in ("Int", "Decimal"):
        return

    mutation_strategy = random.choice(["noise", "reset", "scale"])
    start = param_type.get("start", 0)
    end = param_type.get("end", 100)
    decimal_places = param_type.get("decimal_places", 2)

    if mutation_strategy == "noise":
        noise_scale = (end - start) * 0.1
        noise = random.gauss(0, noise_scale)
        new_value = individual.genes[index] + noise
    elif mutation_strategy == "reset":
        if gene_type == "Int":
            individual.genes[index] = random.randint(int(start), int(end))
            return
        else:
            new_value = random.uniform(start, end)
    else:  # scale
        scale_factor = random.uniform(0.8, 1.2)
        new_value = individual.genes[index] * scale_factor

    new_value = max(start, min(end, new_value))

    if gene_type == "Int":
        individual.genes[index] = int(round(new_value))
    else:
        individual.genes[index] = round(new_value, decimal_places)


def select_tournament(population: List[Individual], tournament_size: int) -> Individual:
    """Select an individual using tournament selection.

    Args:
        population: List of individuals to select from
        tournament_size: Number of individuals in tournament

    Returns:
        Individual with highest fitness from tournament
    """
    if not population:
        raise ValueError("Cannot select from empty population")

    tournament_size = min(tournament_size, len(population))
    tournament = random.sample(population, tournament_size)

    return max(
        tournament,
        key=lambda ind: ind.fitness if ind.fitness is not None else float("-inf"),
    )


def calculate_genetic_distance(ind1: Individual, ind2: Individual) -> float:
    """Calculate genetic distance between two individuals.

    Uses normalized Euclidean distance for numeric genes and
    Hamming distance for categorical genes.

    Returns:
        Distance value between 0 (identical) and 1 (maximum difference)
    """
    if len(ind1.genes) != len(ind2.genes):
        return 1.0

    total_distance = 0.0
    num_genes = len(ind1.genes)

    for i, (g1, g2) in enumerate(zip(ind1.genes, ind2.genes)):
        param_type = ind1.param_types[i] if i < len(ind1.param_types) else None

        if isinstance(g1, bool) and isinstance(g2, bool):
            total_distance += 0.0 if g1 == g2 else 1.0
        elif isinstance(g1, (int, float)) and isinstance(g2, (int, float)):
            if isinstance(param_type, dict):
                start = param_type.get("start", 0)
                end = param_type.get("end", 100)
                range_size = max(end - start, 1)
                total_distance += abs(g1 - g2) / range_size
            else:
                max_val = max(abs(g1), abs(g2), 1)
                total_distance += abs(g1 - g2) / max_val
        else:
            total_distance += 0.0 if g1 == g2 else 1.0

    return total_distance / num_genes


def calculate_population_diversity(population: List[Individual]) -> float:
    """Calculate average diversity of a population.

    Higher values indicate more diverse population.

    Returns:
        Average pairwise genetic distance (0 to 1)
    """
    if len(population) < 2:
        return 0.0

    total_distance = 0.0
    num_pairs = 0

    sample_size = min(len(population), 20)
    sample = random.sample(population, sample_size)

    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            total_distance += calculate_genetic_distance(sample[i], sample[j])
            num_pairs += 1

    return total_distance / num_pairs if num_pairs > 0 else 0.0


def select_with_diversity(
    population: List[Individual],
    tournament_size: int,
    diversity_weight: float = 0.3,
    reference_individual: Individual = None,
) -> Individual:
    """Select individual considering both fitness and diversity.

    Helps prevent premature convergence by favoring individuals that
    are both fit AND different from existing selections.

    Args:
        population: List of individuals to select from
        tournament_size: Number of individuals in tournament
        diversity_weight: Weight for diversity (0 to 1, default 0.3)
        reference_individual: Individual to measure diversity against

    Returns:
        Selected individual balancing fitness and diversity
    """
    if not population:
        raise ValueError("Cannot select from empty population")

    tournament_size = min(tournament_size, len(population))
    tournament = random.sample(population, tournament_size)

    def combined_score(ind: Individual) -> float:
        fitness = ind.fitness if ind.fitness is not None else float("-inf")

        if reference_individual is None or diversity_weight <= 0:
            return fitness

        distance = calculate_genetic_distance(ind, reference_individual)

        max_fitness = max(
            (i.fitness for i in tournament if i.fitness is not None), default=1.0
        )
        min_fitness = min(
            (i.fitness for i in tournament if i.fitness is not None), default=0.0
        )
        fitness_range = max_fitness - min_fitness if max_fitness > min_fitness else 1.0
        normalized_fitness = (fitness - min_fitness) / fitness_range

        return (1 - diversity_weight) * normalized_fitness + diversity_weight * distance

    return max(tournament, key=combined_score)


def maintain_diversity(
    population: List[Individual],
    min_diversity: float = 0.1,
    mutation_boost: float = 0.3,
) -> int:
    """Maintain population diversity by mutating similar individuals.

    When population diversity drops below threshold, applies additional
    mutations to restore diversity.

    Returns:
        Number of individuals that were mutated
    """
    current_diversity = calculate_population_diversity(population)

    if current_diversity >= min_diversity:
        return 0

    mutations_applied = 0

    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            distance = calculate_genetic_distance(population[i], population[j])

            if distance < min_diversity / 2:
                target_idx = i if (
                    population[i].fitness is None
                    or (
                        population[j].fitness is not None
                        and population[j].fitness > population[i].fitness
                    )
                ) else j

                mutate(population[target_idx], mutation_boost)
                mutations_applied += 1

                if mutations_applied >= len(population) // 4:
                    return mutations_applied

    return mutations_applied
