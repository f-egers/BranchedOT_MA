import ctypes
import numpy as np
from numpy import ctypeslib as npct

from ..topology import topology

#load shared C-library for optimizer
opt_lib = npct.load_library("fast_local_search","./BOT/heuristics/lib")

#define input and output types for C-function
array_2d_double = npct.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array_2d_int = npct.ndpointer(dtype=np.intc, ndim=2, flags='CONTIGUOUS')

opt_lib.downhill_climb.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, array_2d_int, array_2d_double, array_1d_double, array_1d_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_bool]
opt_lib.downhill_climb.restype = ctypes.c_double

#opt_lib.calculate_EW.argtypes = [ctypes.c_int,array_2d_double,array_2d_int,array_1d_double,ctypes.c_double]


def BOT_local_search(itopo: topology, supply_arr, demand_arr, coords_sources, coords_sinks, al, improv_threshold=1e-7, max_steps = 100, max_tries = 20, kernel = 0.3, beta=1.0, cost_func="BOT"):

    if isinstance(itopo,topology):
        adj = itopo.adj
    elif isinstance(itopo,np.ndarray):
        adj = itopo

    if cost_func=="BOT":
        CBOT = False 
    elif cost_func=="CBOT":
        CBOT = True
    else:
        raise ValueError("cost_func must be either BOT or CBOT!")

    #get basic parameters
    dim = len(coords_sources[0])
    nsites = len(coords_sinks)+len(coords_sources)

    #assign initial BP positions
    coords_arr = np.vstack((coords_sources, coords_sinks, np.random.rand(nsites-2,dim)))
    
    #construct data arrays for optimizer 
    EW = np.array(np.zeros((nsites-2,3))).astype(np.double)
    demands = np.append(supply_arr,-demand_arr)
    coords_arr = coords_arr.flatten()

    #additional output variables
    iter = ctypes.c_int(0)

    #run optimization
    cost = opt_lib.downhill_climb(ctypes.byref(iter),dim,nsites,adj,EW,demands,coords_arr,al,improv_threshold,max_steps,max_tries,kernel,beta,CBOT)
    #reshape coords array into original shape
    coords_arr = coords_arr.reshape((-1,dim))

    return topology(adj=adj), cost, coords_arr, EW, iter.value






    


    
