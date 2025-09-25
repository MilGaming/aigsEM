# %% qd.py
#   quality diversity exercises
# by: Noah Syrkis

# Imports
from email.mime import image
from math import e
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from typing import Tuple
from pcgym import PcgrlEnv
from pcgym.envs.helper import get_int_prob, get_string_map
from PIL import Image
import random


# %% n-dimensional function with a strange topology
@partial(np.vectorize, signature="(d)->()")
def griewank_function(pop):  # this is kind of our fitness function (except we a minimizing)
    return 1 + np.sum(pop**2) / 4000 - np.prod(np.cos(pop / np.sqrt(np.arange(1, pop.size + 1))))


@partial(np.vectorize, signature="(d)->(d)", excluded=[0])
def mutate(sigma, pop):  # What are we doing here?
    return pop + np.random.normal(0, sigma, pop.shape)


@partial(np.vectorize, signature="(d),(d)->(d)")
def crossover(x1, x2):  # TODO: think about what we are doing here. Is it smart?
    return x1 * np.random.rand() + x2 * (1 - np.random.rand())


def step(pop, cfg):
    loss = griewank_function(pop)
    idxs = np.argsort(loss)[: int(cfg.population * cfg.proportion)]  # select best
    best = np.tile(pop[idxs], (int(cfg.population * cfg.proportion), 1))  # cross over
    pop = crossover(best, best[np.random.permutation(best.shape[0])])  # mutate
    return mutate(cfg.sigma, pop), loss  # return new generation and loss

def mutateMap(sigma, pop, num_tile_types):
    pop_mut = pop.copy()
    num_tiles = pop.size
    n_mut = max(1, int(sigma * num_tiles))  # number of tiles to mutate

    # Flatten indices for mutation
    flat_indices = np.random.choice(num_tiles, n_mut, replace=False)
    # Mutate selected tiles
    pop_mut.flat[flat_indices] = np.random.randint(0, num_tile_types, n_mut)
    return pop_mut


# %% Setup
def main(cfg):
    """env, pop = init_pcgym(cfg)
    Image.fromarray(env.render()).save("map.png")
    map = get_string_map(env._rep._map, env._prob.get_tile_types())
    fitness = level_fitness(env, map)
    print("Level fitness:", fitness)
    #behavior = env._prob.get_stats(map)
    print(env._prob.get_stats(map))"""

    Archive = map_elites(cfg)

    if Archive:
        best_key = max(Archive, key=lambda k: Archive[k]["fitness"])
        best_entry = Archive[best_key]
        best_map = best_entry["solution"]

        env = PcgrlEnv(prob=cfg.game, rep=cfg.rep, render_mode="rgb_array")
        env.reset()
        env._rep._map = best_map

        stats = env._prob.get_stats(get_string_map(best_map, env._prob.get_tile_types()))
        print("Best map stats:", stats)
        print("Best map fitness:", level_fitness(env, get_string_map(best_map, env._prob.get_tile_types())))

        Image.fromarray(env.render()).save("map.png")
        print(f"Best elite descriptor: {best_key}, performance: {best_entry['fitness']}")
    
    
    #for sexy sexy both run this
    #best_individual, best_value = booth_opt(cfg)
    #print("Best Booth solution:", best_individual)
    #print("Booth function value:", best_value)
    #visualize_booth_function(best_individual)

    exit()


# %% Init population (maps)
def init_pcgym(cfg) -> Tuple[PcgrlEnv, np.ndarray]:
    env = PcgrlEnv(prob=cfg.game, rep=cfg.rep, render_mode="rgb_array")
    env.reset()
    pop = np.random.randint(0, env.get_num_tiles(), (cfg.n, *env._rep._map.shape))  # type: ignore
    return env, pop

#fit-diss dih function
def level_fitness(env, level_map):
    stats = env._prob.get_stats(level_map)
    dist_weight = 1000 if stats["dist-win"] == 0 else -200 * stats["dist-win"]  
    jumps_penalty = -10 * stats["jumps"]                
    enemies_penalty = -10 * stats["enemies"]              
    noise_penalty = -30 * stats["noise"]                  
    floor_penalty = -30 * stats["dist-floor"]
    empty_weight = 50 * stats["empty"]
 
    fitness = (
        dist_weight +
        jumps_penalty +
        enemies_penalty +
        noise_penalty +
        floor_penalty +
        empty_weight
    )

    return fitness

def get_key(b, resolution):
    # Discretize each feature into resolution bins
    return tuple(int(x * resolution) if x < 1 else (resolution - 1) for x in b)

def map_elites(cfg):
    # Stolen once again from utils
    nbudget = 2000 # needed more evals
    ninit = int(0.1 * nbudget)
    resolution = 10

    Archive = {}

    env, _ = init_pcgym(cfg)
    num_tile_types = env.get_num_tiles()

    for i in range(nbudget):
        if i < ninit:
            pop = np.random.randint(0, num_tile_types, env._rep._map.shape)
        else:
            if not Archive:
                pop = np.random.randint(0, num_tile_types, env._rep._map.shape)
            else:
                elite_key = random.choice(list(Archive.keys()))
                elite = Archive[elite_key]["solution"]
                pop = mutateMap(cfg.sigma, elite, num_tile_types)

        map_str = get_string_map(pop, env._prob.get_tile_types())
        stats = env._prob.get_stats(map_str)

        b = (stats["jumps"], stats["enemies"])
        p = level_fitness(env, map_str)

        # Stolen Plundered even from utils
        key = get_key(b, resolution)
        if key not in Archive or Archive[key]["fitness"] < p:
            Archive[key] = {
                "fitness": p,
                "behavior": b,
                "solution": pop.copy()
            }

    return Archive






# test function the Booth function
def booth_function(pop):
    x = pop[:, 0]
    y = pop[:, 1]
    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2


# evolutionary optmization code function thing
def booth_opt(cfg):
    pop = np.random.uniform(-10,10,(cfg.n,2))
    #stolen again tihi
    for gen in range(cfg.generation):
        loss = booth_function(pop)
        idxs = np.argsort(loss)[: int(cfg.population * cfg.proportion)]  # select best
        best = np.tile(pop[idxs], (int(cfg.population * cfg.proportion), 1))  # cross over
        pop = crossover(best, best[np.random.permutation(best.shape[0])])  # mutate
        pop = mutate(cfg.sigma,pop)
    
    final_loss = booth_function(pop)
    best_idx = np.argmin(final_loss)
    best_individual = pop[best_idx]
    best_value = final_loss[best_idx]
    return best_individual, best_value

def visualize_booth_function(best_individual=None):
    # Define plot
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-10, 10, 400)
    X, Y = np.meshgrid(x, y)
    Z = booth_function(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    # Make plot
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(cp)
    plt.title("Booth Function Visualization")
    plt.xlabel("x")
    plt.ylabel("y")

    # Plot
    if best_individual is not None:
        plt.plot(best_individual[0], best_individual[1], 'ro', markersize=10, label='Optimized')
        plt.legend()

    plt.show()