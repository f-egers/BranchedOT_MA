import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..geometry_optimization import BOT_optimize
from ..topology import generate_topologies


def brute_force(bot_problem_dict,prog_bar=False):

    al = bot_problem_dict["al"]
    coords_sources = bot_problem_dict["coords_sources"]
    coords_sinks = bot_problem_dict["coords_sinks"]
    supply_arr = bot_problem_dict["supply_arr"]
    demand_arr = bot_problem_dict["demand_arr"]

    n = len(supply_arr)+len(demand_arr)
    if n==3:
        topo, cost,coords,EW,_ = BOT_optimize(np.array([[0,1,2]],dtype=np.intc), supply_arr, demand_arr, coords_sources, coords_sinks, al,improv_threshold=1e-9)
        return topo,cost,coords,EW
    num_topos = np.prod(np.arange(1,2*n-4)[::2])

    topos = generate_topologies(num_topos,n)

    if prog_bar:
        topos = tqdm(generate_topologies)

    best_topo, best_cost, best_coords, best_EW, _ = BOT_optimize(next(topos), supply_arr, demand_arr, coords_sources, coords_sinks, al)
    for t in topos:
        topo, cost,coords,EW,_ = BOT_optimize(t, supply_arr, demand_arr, coords_sources, coords_sinks, al,improv_threshold=1e-9)
        if cost<best_cost:
            best_topo, best_cost, best_coords, best_EW  =  topo, cost, coords, EW

    return best_topo, best_cost, best_coords, best_EW

