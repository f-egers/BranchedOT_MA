import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def eucl_dist(x, Y):
    dim_x = len(x)
    if len(x.shape) > 1:
        dim_x = x.shape[1]
    if Y.shape == (dim_x,):
        return np.sqrt(np.sum((Y - x) ** 2, axis=0))
    elif Y.shape[1] == dim_x:
        return np.sqrt(np.sum((Y - x) ** 2, axis=1))
    else:
        print("dim error")


def generate_random_bot_problem(num_sources, num_sinks, normalised_to=1, dim=2, max_length=1.):
    
    """
    create a random 2d problem inside the [0,1] x [0,1] square with non-integer demand and supplies.
    demand and supplies as well as the coordinates are uniformly distributed.
    input:
    - num_sources and sinks
    - normalised_to: number to which demand and supply are normalised
    - dim: spatial dimension of coordinates
    - max_length: size of the hyper-cube from which coordinates are sampled

    output:
    - return a bot_problem_dict in the usual form.
    """
    
    al = np.random.random()
    coords_sources = np.random.random((num_sources, dim)) * max_length
    coords_sinks = np.random.random((num_sinks, dim)) * max_length
    supply_arr = np.random.random(num_sources)
    supply_arr = normalised_to * supply_arr / np.sum(supply_arr)
    demand_arr = np.random.random(num_sinks)
    demand_arr = normalised_to * demand_arr / np.sum(demand_arr)

    bot_problem_dict = {
        "al": al,
        "coords_sources": coords_sources,
        "coords_sinks": coords_sinks,
        "supply_arr": supply_arr,
        "demand_arr": demand_arr
    }

    return bot_problem_dict