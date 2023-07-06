import ctypes
import numpy as np
from numpy import ctypeslib as npct

from ..topology import topology

#load shared C-library for optimizer
opt_lib = npct.load_library("fast_optimizer","./BOT/geometry_optimization/lib")

#define input and output types for C-function
array_2d_double = npct.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array_2d_int = npct.ndpointer(dtype=np.intc, ndim=2, flags='CONTIGUOUS')

opt_lib.iterations.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, array_2d_int, array_2d_double, array_1d_double, array_1d_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
opt_lib.iterations.restype = ctypes.c_double

opt_lib.calculate_EW.argtypes = [ctypes.c_int,array_2d_double,array_2d_int,array_1d_double,ctypes.c_double]


def BOT_optimize(itopo: topology, supply_arr, demand_arr, coords_sources, coords_sinks, al, improv_threshold=1e-7, beta=1.0, cost_func="BOT"):

    if isinstance(itopo,topology):
        adj = itopo.adj
    elif isinstance(itopo,np.ndarray):
        adj = itopo

    #get basic parameters
    dim = len(coords_sources[0])
    nsites = len(coords_sinks)+len(coords_sources)

    #assign initial BP positions
    coords_arr = np.vstack((coords_sources, coords_sinks, np.random.rand(nsites-2,dim)))
    
    #construct data arrays for optimizer 
    EW = np.array(np.zeros((nsites-2,3))).astype(np.double)
    demands = np.append(supply_arr,-demand_arr)
    opt_lib.calculate_EW(nsites,EW,adj,demands,al)
    if cost_func == "CBOT":
        EW = np.abs(EW)*(1-np.abs(EW))
    coords_arr = coords_arr.flatten()

    #additional output variables
    iter = ctypes.c_int(0)

    #run optimization
    cost = opt_lib.iterations(ctypes.byref(iter),dim,nsites,adj,EW,demands,coords_arr,al,improv_threshold,beta)

    #reshape coords array into original shape
    coords_arr = coords_arr.reshape((-1,dim))

    return itopo, cost, coords_arr, EW, iter.value


def CBOT_optimize(itopo: topology, coords, demand_matrix, al, improv_threshold=1e-7, beta=1.0):

    if isinstance(itopo,topology):
        adj = itopo.adj
    elif isinstance(itopo,np.ndarray):
        adj = itopo

    #get basic parameters
    dim = len(coords[0])
    nsites = len(coords)

    #assign initial BP positions
    coords_arr = np.vstack((coords, np.random.rand(nsites-2,dim)))
    
    #construct data arrays for optimizer 
    EW_cum = np.array(np.zeros((nsites-2,3))).astype(np.double)
    demands = np.zeros(len(coords))
    for i,dem in enumerate(demand_matrix):
        dem[i] = 0
        dem = dem/np.sum(dem)
        dem[i] = -1
        EW = np.array(np.zeros((nsites-2,3))).astype(np.double)
        opt_lib.calculate_EW(nsites,EW,adj,dem,al)
        EW_cum = EW_cum+np.abs(EW)
    EW_cum = EW_cum/nsites
    #EW_cum = EW_cum*(1-EW_cum)
    coords_arr = coords_arr.flatten()

    #additional output variables
    iter = ctypes.c_int(0)

    #run optimization
    cost = opt_lib.iterations(ctypes.byref(iter),dim,nsites,adj,EW_cum,demands,coords_arr,al,improv_threshold,beta)

    #reshape coords array into original shape
    coords_arr = coords_arr.reshape((-1,dim))

    return itopo, cost, coords_arr, EW_cum, iter.value



    


    
