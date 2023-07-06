import numpy as np
import networkx as nx

from ..geometry_optimization.fast_optimizer import BOT_optimize
from ..topology import topology
from . import DCUP,piecewise_linearization,MCCNFP_prior,downhill_climb,CBOT_piecewise_linearization
try:
    from ..utilities import CPLEX_MILP,MILP_Model
except:
    print("CPLEX could not be imported. ony HiGHs solver will be available")




def BBBOT_solver(bot_problem_dict,p,k,mip_time=600,local_search_iters=100,verbose=False,beta=1.0,DCUP_iter=None,DCUP_speed=None,cost_func="BOT"):


    # unpack parameters:
    coords_sources = bot_problem_dict["coords_sources"]
    coords_sinks = bot_problem_dict["coords_sinks"]
    supply_arr = bot_problem_dict["supply_arr"]
    demand_arr = bot_problem_dict["demand_arr"]
    al = bot_problem_dict["al"]

    coords = np.append(coords_sources,coords_sinks,axis=0)
    demands =  np.append(supply_arr,-demand_arr)
    n= len(coords)

    if DCUP_iter is None:
        DCUP_iter = 5*n

    if DCUP_speed is None:
        DCUP_speed = min(k/n,0.05)
    
    #get initial solution 
    init_sol1 = DCUP(bot_problem_dict,k=k,return_lp=True,max_iter=DCUP_iter,num_update=int(DCUP_speed*n),cost_func=cost_func,solver="CPLEX",init="OT")
    init_sol2 = DCUP(bot_problem_dict,k=k,return_lp=True,max_iter=DCUP_iter,num_update=int(DCUP_speed*n),cost_func=cost_func,solver="CPLEX",init="MST")
    #approximation parameter
    if cost_func == "BOT":
        fp,sig_p,nominal_pipe_flows = piecewise_linearization(al,p)
    elif cost_func == "CBOT":
        fp,sig_p,nominal_pipe_flows = CBOT_piecewise_linearization(al,p)

    #turn initial solutions into warm starts
    warm_starts = []
    for init_sol in [init_sol1,init_sol2]:
        best_pipes = np.clip(np.digitize(init_sol,nominal_pipe_flows)-1,0,p-1)
        cont_vars = np.zeros((len(init_sol),p))
        cont_vars[np.arange(len(best_pipes)),best_pipes] = init_sol
        bin_vars = np.zeros((len(init_sol),p))
        bin_vars[np.arange(len(best_pipes)),best_pipes] = 1.0*np.ceil(init_sol)
        warm_starts.append(np.append(cont_vars.reshape(-1),bin_vars.reshape(-1)))


    #solve MIP
    MCCNFP_sol,MCCNFP_coords,MCCNFP_cost,gap = MCCNFP_prior(bot_problem_dict,num_p=p,k=k,solver="CPLEX",warm_start=warm_starts,calc_cost=True,MILP_time_limit=mip_time,return_gap=True,verbose=verbose,beta=beta,cost_func=cost_func)

    #geo optimization
    topo, BOT_cost, coords, EW, _ = BOT_optimize(MCCNFP_sol, supply_arr, demand_arr, coords_sources, coords_sinks, al, beta=beta,cost_func="CBOT")

    #local search improvement
    final_topo,final_cost,final_coords,_ = downhill_climb(topo,bot_problem_dict,maxiter=local_search_iters,beta=beta,cost_func="CBOT")

    #calculate approx ratio
    #print(gap,MCCNFP_cost,final_cost)
    f = 2*(1+gap)-(MCCNFP_cost-final_cost)/MCCNFP_cost

    return final_topo,final_coords,final_cost,MCCNFP_coords,MCCNFP_cost,EW









