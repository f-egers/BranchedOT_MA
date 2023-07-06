import numpy as np
import networkx as nx
from tqdm import tqdm

from ..topology import topology,generate_topologies
from ..geometry_optimization import BOT_optimize, CBOT_optimize
from time import time

def generate_neighbors(topo,coords=None,num_reconnect=np.inf):

    if isinstance(topo,topology):
        adj = topo.adj
    elif isinstance(topo,np.ndarray):
        adj = topo

    n=adj.shape[0]+2
    n_bp = n-2

    #get ordering of BPs according to distance to closest BP (catch couple BPs first)
    if coords is not None:
        bp_coords = coords[n:]
        bp_dists = np.linalg.norm(bp_coords[None,:,:]-bp_coords[:,None,:],axis=-1)
        np.fill_diagonal(bp_dists,np.inf)
        bp_min_dists = np.min(bp_dists,axis=1)
        bp_range = n+np.argsort(bp_min_dists)
    else:
        bp_range = range(n,n+n_bp)

    #iterate BPs
    for bp in bp_range:
        #iterate over neighbors of bp
        for nhb in adj[bp-n]:
            if nhb<n or bp>nhb: #avoid dubble counting
                #(bp,nhb) is the edge we have decided to cut
                #iterate over who is connector
                for e1,e2 in [(bp,nhb),(nhb,bp)]:
                    if e1>=n:
                        #find relevant edges to connect to
                        roots=[(node,e1) for node in adj[e1-n] if node>=n and node!=e2]
                        edges_of_interest = []#[(e1,node) for node in adj[e1-n] if e1>node and node!=e2]
                        while roots:
                            r,p=roots.pop()
                            edges_of_interest.extend([[r,node] for node in adj[r-n] if r>node])
                            roots.extend([(node,r) for node in adj[r-n] if node>=n and node!=p])
                        if not edges_of_interest:
                            continue

                        #order edges with their distance to the connector node
                        if coords is not None:
                            edges_of_interest = np.array(edges_of_interest,ndmin=2)
                            edges_of_interest = np.array(edges_of_interest)[point_edge_dist_ordering(coords[e2],coords[edges_of_interest[:,0]],coords[edges_of_interest[:,1]])]

                        #make the moves
                        for edge in edges_of_interest[:int(np.min([len(edges_of_interest),num_reconnect]))]:
                            if e1 not in edge: #more elegant way of getting rid of this check?
                                adj_new = np.copy(adj)
                                a,b = adj[e1-n][adj[e1-n]!=e2]
                                if a>=n: adj_new[a-n][adj_new[a-n]==e1]=b
                                if b>=n: adj_new[b-n][adj_new[b-n]==e1]=a
                                adj_new[e1-n]=np.array([edge[0],edge[1],e2])
                                if edge[0]>=n: adj_new[edge[0]-n][adj_new[edge[0]-n]==edge[1]]=e1
                                if edge[1]>=n: adj_new[edge[1]-n][adj_new[edge[1]-n]==edge[0]]=e1

                                if isinstance(topo,topology):
                                    yield topology(adj=adj_new)
                                elif isinstance(topo,np.ndarray):
                                    yield adj_new
                            


def build_nbh_graph(n):
    G = nx.Graph()
    num_topos = np.prod(np.arange(1,2*n-3)[::2])
    topologies = []
    for topo in generate_topologies(num_topos,n):
        topologies.append(topo)

    for topo in topologies:
        for nb in generate_neighbors(topo):
            for t in topologies:
                if np.array_equal(t.vec_rep,nb.vec_rep):
                    G.add_edge(topo,t)
            
    return G




def downhill_climb(topo,bot_problem_dict,maxiter=10000,num_tries=10,beta=1.0,max_time=np.inf,cost_func="BOT",verbose=False):

    if maxiter==np.inf:
        num_try = np.inf
    else:
        num_try = num_tries

    if isinstance(topo,topology):
        adj = topo.adj
    elif isinstance(topo,np.ndarray):
        adj = topo

    #unpack bot_problem_dict
    al = bot_problem_dict["al"]
    coords_sources = bot_problem_dict["coords_sources"]
    coords_sinks = bot_problem_dict["coords_sinks"]
    supply_arr = bot_problem_dict["supply_arr"]
    demand_arr = bot_problem_dict["demand_arr"]

    time0 = time()

    #iteratively look for improvements
    best_topo, best_cost, best_coords, best_EW, _ = BOT_optimize(adj, supply_arr, demand_arr, coords_sources, coords_sinks, al, beta=beta, cost_func=cost_func)
    init_cost = best_cost
    niter = 0
    improv=True
    while improv and maxiter>0:
        improv=False
        for t in generate_neighbors(best_topo,coords=None,num_reconnect=num_try):
            niter+=1

            if verbose: print(niter," topologies checked ,",best_cost/init_cost," of initial cost",end="\r")

            top,cost,coords,EW,_ = BOT_optimize(t,supply_arr, demand_arr, coords_sources, coords_sinks, al, beta=beta, cost_func=cost_func)
            if cost<best_cost:
                best_topo,best_cost,best_coords,best_EW = top,cost,coords,EW
                improv=True
                break
            

            if niter>maxiter or time()-time0 > max_time:
                return topology(adj=best_topo),best_cost,best_coords,best_EW

    return topology(adj=best_topo),best_cost,best_coords,best_EW





def downhill_climb_CBOT(topo,al,coords_nodes,demand_matrix,maxiter=10000,num_tries=10,beta=1.0,max_time=np.inf,verbose=False):

    if maxiter==np.inf:
        num_try = np.inf
    else:
        num_try = num_tries

    if isinstance(topo,topology):
        adj = topo.adj
    elif isinstance(topo,np.ndarray):
        adj = topo

    time0 = time()

    #iteratively look for improvements
    best_topo, best_cost, best_coords, best_EW, _ = CBOT_optimize(adj, coords_nodes, demand_matrix, al, beta=beta)
    init_cost = best_cost
    niter = 0
    improv=True
    while improv:
        improv=False
        for t in generate_neighbors(best_topo,coords=best_coords,num_reconnect=num_try):
            niter+=1

            if verbose: print(niter," topologies checked ,",best_cost/init_cost," of initial cost",end="\r")

            top,cost,coords,EW,_ = CBOT_optimize(t, coords_nodes, demand_matrix, al, beta=beta)
            if cost<best_cost:
                best_topo,best_cost,best_coords,best_EW = top,cost,coords,EW
                improv=True
                break
            

            if niter>maxiter or time()-time0 > max_time:
                return topology(adj=best_topo),best_cost,best_coords,best_EW

    return topology(adj=best_topo),best_cost,best_coords,best_EW





def point_edge_dist_ordering(p, a, b):
    """
    p - point to project
    a - array of starting points of the segments
    b - array of end points of the segments
    """
    diff = b-a
    lam = np.sum((diff) * (p - a), axis=1) / (np.linalg.norm(diff,axis=1)+1e-12)
    p_proj = a + lam[:,None] * (diff)

    p_proj[lam <= 0] = a[lam <= 0]
    p_proj[lam >= 1] = b[lam >= 1]

    dist_arr = np.linalg.norm(p[None,:]-p_proj,axis=1)
    return np.argsort(dist_arr)









