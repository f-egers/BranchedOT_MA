import numpy as np
import networkx as nx
import ot
from scipy.optimize import linprog
import scipy.sparse as sparse
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt

from ..geometry_optimization.fast_optimizer import BOT_optimize
from ..topology import topology
try:
    from ..utilities import CPLEX_MILP,MILP_Model
except:
    print("CPLEX could not be imported. ony HiGHs solver will be available")


# return a topo where all terminals have degree 1 that is a beta-interpolation between the MST and OT edges.
def IMST_prior(bot_problem_dict, beta=None, calc_cost=False):

    # unpack parameters:
    coords_sources = bot_problem_dict["coords_sources"]
    coords_sinks = bot_problem_dict["coords_sinks"]
    supply_arr = bot_problem_dict["supply_arr"]
    demand_arr = bot_problem_dict["demand_arr"]
    al = bot_problem_dict["al"]

    if beta is None:
        beta = 1-al

    # use OT package to solve optimal transport
    M = ot.dist(coords_sources, coords_sinks, 'euclidean')
    flow_mat = ot.emd(supply_arr, demand_arr, M)  #solve optimal transport

    # setup OT edges and their cost:
    num_edges = len(supply_arr) + len(demand_arr) - 1
    OT_edges_arr = np.zeros((num_edges, 2))
    OT_edges_dist = np.zeros(num_edges)
    count = 0
    for i, flow in enumerate(flow_mat.reshape(-1)):
        if flow > 1e-15:
            ind_pair = np.array(np.unravel_index(i, M.shape))
            OT_edges_dist[count] = M[ind_pair[0], ind_pair[1]]
            ind_pair[1] += len(supply_arr)
            OT_edges_arr[count] = ind_pair
            count += 1
    #OT_topo.add_edges_from(OT_edges_arr)

    # find the MST and get edges and cost of edges there:
    coords = np.vstack((coords_sources, coords_sinks))

    # construct all index pairs
    edges_arr = np.zeros((int(len(coords) * (len(coords) - 1) / 2), 2), dtype=int)
    count = 0
    for i in range(len(coords)):
        for j in range(i):
            edges_arr[count] = np.array([i, j])
            count += 1

    # fast way to calculate distances:
    weight_arr = np.sqrt(np.sum((coords[edges_arr[:, 0]] - coords[edges_arr[:, 1]]) ** 2, axis=1))

    MST_topo, MST_edges_arr, MST_edges_dist = get_MST(edges_arr, weight_arr, coords)

    # consider the joint set of edges with reduces edge cost for OT edges:
    edges_arr_joint = np.vstack((MST_edges_arr, OT_edges_arr))
    dist_arr_joint = np.append(MST_edges_dist, OT_edges_dist * beta)

    MST_joint, _, _ = get_MST(edges_arr_joint, dist_arr_joint, coords)

    return angular_stress_reduction(MST_joint,bot_problem_dict,calc_cost=calc_cost)




def get_MST(edges_arr, weight_arr, coords):
    MST = nx.Graph()
    accepted_edges_idx = []
    for n in range(len(coords)):
        MST.add_node(n, pos=coords[n])
    weight_arr_c = np.copy(weight_arr)

    while True:
        argmin = np.argmin(weight_arr_c)
        index_pair = edges_arr[argmin]
        weight_arr_c[argmin] = np.inf

        # add edge if no cycle
        if not nx.has_path(MST, index_pair[0], index_pair[1]):
            MST.add_edge(index_pair[0], index_pair[1])
            accepted_edges_idx.append(argmin)

        if len(list(nx.connected_components(MST))) == 1:
            break

    return MST, edges_arr[accepted_edges_idx], weight_arr[accepted_edges_idx]



def nx_to_adj(nx_graph):
    N=nx.number_of_nodes(nx_graph)
    n=int(N/2+1)

    adj=[]

    for bp in range(n,N):
        adj.append(list(nx.neighbors(nx_graph, bp)))

    try:
        adj = np.array(adj,dtype=np.intc)
    except:
        pass

    return adj



def piecewise_linearization(al,num_p,rescale_point_pos=True):

    p=num_p
    if rescale_point_pos:
        nominal_pipe_flows = np.linspace(0,1,p+1)**(1/np.clip(al,0.001,1))
    else:
        nominal_pipe_flows = np.linspace(0,1,p+1)

    if al == 0.0:
        fp = np.zeros(num_p)
        sig_p = np.ones(num_p)

    fp = (nominal_pipe_flows[:-1]**al-nominal_pipe_flows[1:]**al)/(nominal_pipe_flows[:-1]-nominal_pipe_flows[1:])
    sig_p = nominal_pipe_flows[:-1]**al-fp*nominal_pipe_flows[:-1]

    return fp,sig_p,nominal_pipe_flows


def CBOT_piecewise_linearization(al,num_p,rescale_point_pos=False):

    p=num_p
    if rescale_point_pos:
        if p%2 == 0:
            lower =  np.linspace(0,0.5,int(p/2)+1)**(1/np.clip(al,0.001,1.0))
            lower[-1] = 0.5
            nominal_pipe_flows = np.concatenate((lower,(1-lower)[::-1][1:]))
        else:
            lower =  np.linspace(0,0.5,int((p+1)/2)+1)[:-1]**(1/np.clip(al,0.001,1.0))
            nominal_pipe_flows = np.concatenate((lower,(1-lower)[::-1]))
    else:
        nominal_pipe_flows = np.linspace(0,1,p+1)


    CBOT_flows = nominal_pipe_flows-nominal_pipe_flows**al

    fp = (CBOT_flows[:-1]-CBOT_flows[1:])/(nominal_pipe_flows[:-1]-nominal_pipe_flows[1:])
    sig_p = CBOT_flows[:-1]-fp*nominal_pipe_flows[:-1]
    sig_p[-1] += 1e-5

    return fp,sig_p,nominal_pipe_flows



def unpack_parameters(bot_problem_dict):

    # unpack parameters:
    coords_sources = bot_problem_dict["coords_sources"]
    coords_sinks = bot_problem_dict["coords_sinks"]
    supply_arr = bot_problem_dict["supply_arr"]
    demand_arr = bot_problem_dict["demand_arr"]
    al = bot_problem_dict["al"]

    coords = np.append(coords_sources,coords_sinks,axis=0)
    demands =  np.append(supply_arr,-demand_arr)
    n= len(coords)

    return coords_sinks,coords_sources,supply_arr,demand_arr,al,coords,demands,n


def build_adj_matrix(bot_problem_dict,k,include_ot=True):

    coords_sinks,coords_sources,supply_arr,demand_arr,al,coords,demands,n = unpack_parameters(bot_problem_dict)

    #OT
    M = ot.dist(coords_sources, coords_sinks, 'euclidean')
    ot_flow = ot.emd(supply_arr, demand_arr, M)
    ot_adj = (np.block([[np.zeros((len(supply_arr),len(supply_arr))),ot_flow],[ot_flow.T,np.zeros((len(demand_arr),len(demand_arr)))]])!=0).astype(int)
    #MST
    dist = np.linalg.norm(coords[None,:,:]-coords[:,None,:],axis=-1)
    mst_adj = nx.to_numpy_array(nx.minimum_spanning_tree(nx.from_numpy_array(dist)),weight=None).astype(int)
    #mst_adj = (scics.minimum_spanning_tree(dist).todense() != 0).astype(int)
    #kNN
    kNN_adj = kneighbors_graph(coords,max(1,min(k,n-1))).todense().astype(int)

    adj_mat = (include_ot*ot_adj+mst_adj+k*(kNN_adj+kNN_adj.T))!=0

    return adj_mat,dist





def MCCNFP_prior(bot_problem_dict,num_p=3,k=5,integral=True,calc_cost=False,verbose=True,MILP_time_limit=600,solver="highs",warm_start=None,return_gap=False,beta=1.0,ensure_tree=True,cost_func="BOT"):

    # unpack parameters:
    coords_sinks,coords_sources,supply_arr,demand_arr,al,coords,demands,n = unpack_parameters(bot_problem_dict)

    #define pipes
    p=num_p
    if cost_func == "BOT":
        fp,sig_p,nominal_pipe_flows = piecewise_linearization(al,num_p)
    elif cost_func == "CBOT":
        fp,sig_p,nominal_pipe_flows = CBOT_piecewise_linearization(al,num_p)


    ###define graph
    adj_mat,dist = build_adj_matrix(bot_problem_dict,k,include_ot=(cost_func != "CBOT"))
    num_edges = np.sum(adj_mat)

    #build objective vector
    #dist = dist[~np.eye(n).astype(bool)].reshape(n,n-1)
    c = np.append((dist[:,:,None]**beta*fp[None,None,:])[adj_mat],(dist[:,:,None]**beta*sig_p[None,None,:])[adj_mat])

    #flow conserv constraints
    A_eq = np.zeros((n,len(c)))#np.zeros((n+int(n*(n-1)/2)*p,len(c)))
    b_eq = np.array(demands)
    for i in range(n):
        mat = np.zeros((n,n,p))
        mat[i,:,:] = 1
        mat[:,i,:] = -1
        mat = np.append(mat[adj_mat],np.zeros(num_edges*p))
        A_eq[i,:] = mat


    #build inequal constraints
    A_ineq = np.zeros((num_edges*(p+1),len(c)))
    #m_i <= x_i
    for i in range(num_edges*p):
        mat = np.zeros(len(c))
        mat[i]=1
        mat[i+num_edges*p]=-1
        A_ineq[i] = mat 
    #sum_i x_i <= 1
    for i in range(num_edges):
        mat = np.zeros(len(c))
        mat[(num_edges*p)+i*p:(num_edges*p)+(i+1)*p] = 1
        A_ineq[p*num_edges+i] = mat
    #m_i in range given by selected pipe
    # for i in range(num_edges):
    #     mat = np.zeros(len(c))
    #     mat[i*p:(i+1)*p] = 1
    #     mat[(num_edges*p)+i*p:(num_edges*p)+(i+1)*p] = -nominal_pipe_flows[1:]
    #     A_ineq[(p+1)*num_edges+i] = mat
    # for i in range(num_edges):
    #     mat = np.zeros(len(c))
    #     mat[i*p:(i+1)*p] = -1
    #     mat[(num_edges*p)+i*p:(num_edges*p)+(i+1)*p] = nominal_pipe_flows[:-1]
    #     A_ineq[(p+2)*num_edges+i] = mat
    b_ineq = np.concatenate((np.zeros(num_edges*p),np.ones(num_edges),np.zeros(0*num_edges)))

    bounds=(0,1)#[[-1,1] if i<n*(n-1)*p else [0,1] for i in range(len(c))]

    if integral:
        integrality = np.append(np.zeros(num_edges*p),np.ones(num_edges*p))
    else:
        integrality = np.zeros(2*num_edges*p)
    #print(len(integrality))

    #solve LP
    gap = None
    if solver == "highs":
        options = {'disp': verbose, 
                'time_limit': MILP_time_limit,
                'primal_feasibility_tolerance': 1e-01,
                'dual_feasibility_tolerance': 1e-01}
        lp_res = linprog(c,A_ineq,b_ineq,A_eq,b_eq,bounds,integrality=integrality,options=options).x
    elif solver == "CPLEX":
        lp_res,gap = CPLEX_MILP(c,sparse.coo_array(A_eq),b_eq,sparse.coo_array(A_ineq),b_ineq,bounds[1],bounds[0],integrality,time_limit=MILP_time_limit,verbose=verbose,warm_start=warm_start)

    #get adj. matrix of topology
    flows = lp_res[:num_edges*p].reshape(num_edges,p)
    flows = np.sum(flows,axis=-1)
    arr = np.zeros((n,n))
    arr[adj_mat] = flows
    flows = arr
    #flows = np.sum(lp_res[:n*(n-1)*p].reshape(n,n-1,p),axis=-1)

    #remove potential loops
    t = nx.from_numpy_array(-np.abs(flows+flows.T),parallel_edges=False)
    if ensure_tree:
        t = nx.minimum_spanning_tree(t)

    if return_gap:
        return *angular_stress_reduction(t,bot_problem_dict,calc_cost=calc_cost,cfunc=cost_func),gap

    # return topology(adj=nx_to_adj(t))
    return angular_stress_reduction(t,bot_problem_dict,calc_cost=calc_cost,cfunc=cost_func)



def generalized_MCCNFP_prior(coords,demand_matrix,al,num_p=3,k=5,calc_cost=False,verbose=True,MILP_time_limit=600,solver="CPLEX",warm_start=None,return_gap=False,beta=1.0):

    # unpack parameters:
    n = len(coords)
    num_layers = len(demand_matrix)

    #define pipes
    p=num_p
    fp,sig_p,nominal_pipe_flows = piecewise_linearization(al,num_p)

    ###define graph
    print("defining graph...",end="")
    #MST
    dist = np.linalg.norm(coords[None,:,:]-coords[:,None,:],axis=-1)
    mst_adj = nx.to_numpy_array(nx.minimum_spanning_tree(nx.from_numpy_array(dist)),weight=None).astype(int)
    #kNN
    kNN_adj = kneighbors_graph(coords,max(1,min(k,n-1))).todense().astype(int)

    adj_mat = (mst_adj+k*(kNN_adj+kNN_adj.T))!=0
    num_edges = np.sum(adj_mat)
    print("done")

    #build objective vector
    c = np.append((dist[:,:,None]**beta*fp[None,None,:])[adj_mat],(dist[:,:,None]**beta*sig_p[None,None,:])[adj_mat])
    c = np.append(np.zeros(num_layers*num_edges),c)

    print("building matrix constraints...",end="")
    #flow conserv constraints
    A_eq = np.zeros((num_layers*n+num_edges+p*num_edges,len(c)))
    b_eq = np.concatenate((np.array(demand_matrix).reshape(-1),np.zeros(num_edges),np.zeros(p*num_edges)))
    for i in range(num_layers):
        for j in range(n):
            mat = np.zeros((n,n))
            mat[j,:] = 1
            mat[:,j] = -1
            mat = mat[adj_mat]
            A_eq[n*i+j,num_edges*i:(i+1)*num_edges] = mat
    #sum constraints
    for i in range(num_edges):
        mat = np.zeros(len(c))
        mat[i:num_layers*num_edges:num_edges] = 1
        mat[num_layers*num_edges+i*p:num_layers*num_edges+(i+1)*p] = -1
        A_eq[num_layers*n+i] = mat 
    #link parallel pipes
    ind_e = np.nonzero(adj_mat.reshape(-1))[-1]
    opp_e = np.ravel_multi_index(np.array(np.unravel_index(ind_e,adj_mat.shape)),adj_mat.shape)
    for i in range(num_edges):
        o = np.where(ind_e[i]==opp_e)[0][0]
        if i < o:
            for j in range(p):
                A_eq[num_layers*n+num_edges+i*p+j,num_layers*num_edges+num_edges*p+i*p+j]=1
                A_eq[num_layers*n+num_edges+i*p+j,num_layers*num_edges+num_edges*p+o*p+j]=-1

    #build inequal constraints
    A_ineq = np.zeros((num_edges*p+num_edges,len(c)))
    pos=0
    for i in range(num_edges*p):
        mat = np.zeros(len(c))
        mat[num_layers*num_edges+i]=1
        mat[num_layers*num_edges+i+num_edges*p]=-1
        A_ineq[pos] = mat 
        pos+=1
    for i in range(num_edges):
        mat = np.zeros(len(c))
        mat[(num_layers+p)*num_edges+i*p:(num_layers+p)*num_edges+(i+1)*p] = 1
        A_ineq[p*num_edges+i] = mat 
    b_ineq = np.concatenate((np.zeros(num_edges*p),np.ones(num_edges)))
    print("done")

    #upper and lower bounds
    bounds=(0,1)

    #define variable types; 0: cont. , 1: bin.
    integrality = np.append(np.zeros(num_edges*(p+num_layers)),np.ones(num_edges*p))

    #solve LP
    gap = None
    if solver == "highs":
        options = {'disp': verbose, 
                'time_limit': MILP_time_limit,
                'primal_feasibility_tolerance': 1e-01,
                'dual_feasibility_tolerance': 1e-01}
        lp_res = linprog(c,A_ineq,b_ineq,A_eq,b_eq,bounds,integrality=integrality,options=options).x
    elif solver == "CPLEX":
        lp_res,gap = CPLEX_MILP(c,A_eq,b_eq,A_ineq,b_ineq,bounds[1],bounds[0],integrality,time_limit=MILP_time_limit,verbose=verbose,warm_start=warm_start)

    #get adj. matrix of topology
    flows = lp_res[num_layers*num_edges:(num_layers+p)*num_edges].reshape(num_edges,p)
    flows = np.sum(flows,axis=-1)
    arr = np.zeros((n,n))
    arr[adj_mat] = flows
    flows = arr

    #neglect very small flows
    flows[flows<1e-8]=0

    #remove potential loops
    t = nx.from_numpy_array(flows,create_using=nx.DiGraph)
    #print(t.edges(data=True))

    if return_gap:
        return *generalized_angular_stress_reduction(t,demand_matrix,coords),gap

    # return topology(adj=nx_to_adj(t))
    return generalized_angular_stress_reduction(t,demand_matrix,coords)



def DCUP(bot_problem_dict,num_p=None,k=5,calc_cost=False,verbose=True,MILP_time_limit=600,solver="CPLEX",return_lp=False,beta=1.0,max_iter=1000,num_update=None,cost_func="BOT",init="OT"):

    # unpack parameters:
    coords_sinks,coords_sources,supply_arr,demand_arr,al,coords,demands,n = unpack_parameters(bot_problem_dict)

    if num_update is None:
        num_update = k
        
    #define pipes
    if num_p is not None:
        p=num_p
    else:
        p=3*n
    if cost_func == "BOT":
        fp,sig_p,nominal_pipe_flows = piecewise_linearization(al,p,rescale_point_pos=False)
    elif cost_func == "CBOT":
        fp,sig_p,nominal_pipe_flows = CBOT_piecewise_linearization(al,p)

    ###define graph
    adj_mat,dist = build_adj_matrix(bot_problem_dict,k,include_ot=(cost_func != "CBOT"))
    num_edges = np.sum(adj_mat)

    num_update = min(num_edges-1,num_update)

    #build objective vector
    #dist = dist[~np.eye(n).astype(bool)].reshape(n,n-1)
    c = (dist[:,:,None]**beta*fp[None,None,:])[adj_mat].reshape(-1)
    s = (dist[:,:,None]**beta*sig_p[None,None,:])[adj_mat].reshape(-1)

    c_curr = dist[adj_mat].reshape(-1)

    #flow conserv constraints
    A_eq = np.zeros((n,len(c_curr)))
    b_eq = np.array(demands)
    for i in range(n):
        mat = np.zeros((n,n))
        mat[i,:] = 1
        mat[:,i] = -1
        mat = mat[adj_mat].reshape(-1)
        A_eq[i,:] = mat

    bounds=(0,1)

    if solver == "CPLEX":
        m = MILP_Model(c_curr,A_eq=sparse.coo_array(A_eq),b_eq=b_eq,lo=bounds[0],up=bounds[1],integer=np.zeros(len(c_curr)))

    #initialzation
    if init=="OT":
        y_old = np.zeros((num_edges,p))
        y_old[:,0] = 1.0
        y_old = y_old.reshape(-1)
    elif init=="MST":
        mst_adj = nx.to_numpy_array(nx.minimum_spanning_tree(nx.from_numpy_array(dist)),weight=None).astype(int)
        c_curr = 100.0-99.9*mst_adj[adj_mat]
        
        if solver == "highs":
            lp_res = linprog(c_curr,A_eq=A_eq,b_eq=b_eq,bounds=bounds).x
        elif solver == "CPLEX":
            m.set_objective(c_curr)
            lp_res = m.solve()
        best_pipes = np.clip(np.digitize(lp_res,nominal_pipe_flows)-1,0,p-1)
        y_old = np.zeros((num_edges,p))
        y_old[np.arange(num_edges),best_pipes] = 1.0 
        y_old = y_old.reshape(-1)
    else:
        raise ValueError("invalid init parameter. Must be one of OT,MST")
    

    #solve iterative LP
    iter_count = 0
    y_new = np.copy(y_old)
    #change = 1.0
    while True:
        iter_count += 1
        y_old,y_new = y_new,y_old
        c_curr = np.sum(np.reshape(c*y_old,(-1,p)),axis=-1)
        if solver == "highs":
            lp_res = linprog(c_curr,A_eq=A_eq,b_eq=b_eq,bounds=bounds).x
        elif solver == "CPLEX":
            m.set_objective(c_curr)
            lp_res = m.solve()
        x = np.copy(lp_res)

        upper_flows = nominal_pipe_flows[1:][np.argmax(y_old.reshape(num_edges,p),axis=1)]
        overflow = lp_res-upper_flows
        # if cost_func == "CBOT":
        #     overflow = overflow*(1-overflow) 
        if np.any(overflow > 0):
            best_pipes = np.clip(np.digitize(lp_res,nominal_pipe_flows)-1,0,p-1)
            y_new = np.copy(y_old).reshape(num_edges,p)
            ind = np.argpartition(-overflow*(1/dist[adj_mat].reshape(-1)),kth=num_update)[:num_update]#np.argmax(overflow)
            y_new[ind] = np.zeros(p)
            y_new[ind,best_pipes[ind]] = 1.0 
            y_new = y_new.reshape(-1)
            if verbose: print(f"number of LP iterations: {iter_count}/{max_iter}",end="\r",flush=True)
        else:
            if verbose: print(f"number of LP iterations: {iter_count}/{max_iter}",flush=True)
            break

        if iter_count==max_iter:
            if verbose: print(f"number of LP iterations: {iter_count}/{max_iter}",flush=True)
            break

    if  return_lp:
        return lp_res

    #get adj. matrix of topology
    flows = x
    arr = np.zeros((n,n))
    arr[adj_mat] = flows
    flows = arr

    #remove potential loops
    t = nx.from_numpy_array(-np.abs(flows+flows.T),parallel_edges=False)
    t = nx.minimum_spanning_tree(t)

    # return topology(adj=nx_to_adj(t))
    return angular_stress_reduction(t,bot_problem_dict,calc_cost=calc_cost,cfunc=cost_func)


def DSCUP(bot_problem_dict,num_p=None,k=5,calc_cost=False,verbose=True,MILP_time_limit=600,solver="CPLEX",beta=1.0,max_iter=1000,num_update=None,cost_func="BOT"):
    
    sol1,coords1,cost1 = DCUP(bot_problem_dict,num_p,k,calc_cost=True,verbose=verbose,MILP_time_limit=MILP_time_limit,solver=solver,return_lp=False,beta=beta,max_iter=max_iter,num_update=num_update,cost_func=cost_func,init="OT")
    sol2,coords2,cost2 = DCUP(bot_problem_dict,num_p,k,calc_cost=True,verbose=verbose,MILP_time_limit=MILP_time_limit,solver=solver,return_lp=False,beta=beta,max_iter=max_iter,num_update=num_update,cost_func=cost_func,init="MST")
    
    if calc_cost:
        if cost1>cost2:
            return sol2,coords2,cost2
        else:
            return sol1,coords1,cost1
    else:
        if cost1>cost2:
            return sol2,coords2
        else:
            return sol1,coords1


def generalized_DCUP(coords,demand_matrix,al,num_p=3,k=5,calc_cost=False,verbose=True,MILP_time_limit=600,solver="CPLEX",return_lp=False,max_iter=100,num_update=None):

    if num_update is None:
        num_update = k

    # unpack parameters:
    n = len(coords)
    num_layers = len(demand_matrix)

    #define pipes
    p=num_p
    fp,sig_p,nominal_pipe_flows = piecewise_linearization(al,num_p,rescale_point_pos=False)

    ###define graph
    print("defining graph...",end="")
    #MST
    dist = np.linalg.norm(coords[None,:,:]-coords[:,None,:],axis=-1)
    mst_adj = nx.to_numpy_array(nx.minimum_spanning_tree(nx.from_numpy_array(dist)),weight=None).astype(int)
    #kNN
    kNN_adj = kneighbors_graph(coords,max(1,min(k,n-1))).todense().astype(int)

    adj_mat = (mst_adj+k*(kNN_adj+kNN_adj.T))!=0
    num_edges = np.sum(adj_mat)
    print("done")

    num_update = min(num_edges-1,num_update)

    #build objective vector
    c = (dist[:,:,None]*fp[None,None,:])[adj_mat].reshape(-1)
    s = (dist[:,:,None]*sig_p[None,None,:])[adj_mat].reshape(-1)
    #print(c,s)
    c_curr = np.append(np.zeros((num_layers*num_edges)),dist[adj_mat].reshape(-1))

    print("building matrix constraints...",end="")
    #flow conserv constraints
    A_eq = np.zeros((num_layers*n+num_edges,len(c_curr)))
    b_eq = np.concatenate((np.array(demand_matrix).reshape(-1),np.zeros(num_edges)))
    for i in range(num_layers):
        for j in range(n):
            mat = np.zeros((n,n))
            mat[j,:] = 1
            mat[:,j] = -1
            mat = mat[adj_mat]
            A_eq[n*i+j,num_edges*i:(i+1)*num_edges] = mat
    #sum constraints
    for i in range(num_edges):
        mat = np.zeros(len(c_curr))
        mat[i:num_layers*num_edges:num_edges] = 1
        mat[num_layers*num_edges+i] = -1
        A_eq[num_layers*n+i] = mat 
    ind_e = np.nonzero(adj_mat.reshape(-1))[-1]
    opp_e = np.ravel_multi_index(np.array(np.unravel_index(ind_e,adj_mat.shape)),adj_mat.shape)

    print("done")

    bounds=(0,1)

    if solver == "CPLEX":
        m = MILP_Model(c_curr,A_eq=sparse.coo_array(A_eq),b_eq=b_eq,lo=bounds[0],up=bounds[1],integer=np.zeros(len(c_curr)))

    #initialzation
    y_old = np.zeros((num_edges,p))
    y_old[:,0] = 1.0
    y_old = y_old.reshape(-1)
    

    #solve iterative LP
    iter_count = 0
    y_new = np.copy(y_old)
    while True:
        iter_count += 1
        y_old,y_new = y_new,y_old
        c_curr[num_edges*num_layers:] = np.sum(np.reshape(c*y_old,(-1,p)),axis=-1)
        if solver == "highs":
            lp_res = linprog(c_curr,A_eq=A_eq,b_eq=b_eq,bounds=bounds).x
        elif solver == "CPLEX":
            m.set_objective(c_curr)
            lp_res = m.solve()
        x = np.copy(lp_res)

        upper_flows = nominal_pipe_flows[1:][np.argmax(y_old.reshape(num_edges,p),axis=1)]
        overflow = lp_res[-num_edges:]-upper_flows
        if np.any(overflow > 0):
            best_pipes = np.clip(np.digitize(lp_res[-num_edges:],nominal_pipe_flows)-1,0,p-1)#np.digitize(lp_res[-num_edges:],nominal_pipe_flows)-1
            for i in range(num_edges):
                j=np.where(ind_e[i]==opp_e)[0][0]
                best_pipes[i] = best_pipes[j] = max(best_pipes[i],best_pipes[j])
            y_new = np.copy(y_old).reshape(num_edges,p)
            ind = np.argpartition(-overflow*(1/dist[adj_mat].reshape(-1)),kth=num_update)[:num_update]
            y_new[ind] = np.zeros(p)
            y_new[ind,best_pipes[ind]] = 1.0 
            y_new = y_new.reshape(-1)
            print(f"number of LP iterations: {iter_count}",end="\r",flush=True)
        else:
            print(f"number of LP iterations: {iter_count}",flush=True)
            break

        if iter_count==max_iter:
            print(f"number of LP iterations: {iter_count}",flush=True)
            break

    if  return_lp:
        return lp_res

    #get adj. matrix of topology
    flows = x[-num_edges:]/num_layers
    arr = np.zeros((n,n))
    arr[adj_mat] = flows
    flows = arr

    #to nx graph
    t = nx.from_numpy_array(flows,create_using=nx.DiGraph)

    return generalized_angular_stress_reduction(t,demand_matrix,coords)



def angular_stress_reduction(t,bot_problem_dict,calc_cost=False,cfunc="BOT"):

    # unpack parameters:
    coords_sinks,coords_sources,supply_arr,demand_arr,al,coords,demands,n = unpack_parameters(bot_problem_dict)

    coords = np.append(coords_sources,coords_sinks,axis=0)
    coords_real = np.append(coords_sources,coords_sinks,axis=0)

    label_max = len(coords) - 1
    for node in list(t.nodes()):
        if t.degree(node) == 1:
            continue
        else:
            #connect all neighbors plus node itself via high degree BP
            coords_bp = coords[node] 
            label_max += 1
            n0 = label_max
            coords = np.append(coords,coords_bp[None,:],axis=0)
            coords_real = np.append(coords_real,coords_bp[None,:],axis=0)
            neighbours = list(nx.neighbors(t, node))
            neighbours = [int(nbh) for nbh in neighbours]
            t.remove_node(node)
            t.add_edge(label_max,node)
            for nd in neighbours:
                t.add_edge(label_max,nd)
            while len(neighbours)>2:
                coords_nhbs = coords[neighbours] - coords[node][None,:]
                norms = np.linalg.norm(coords_nhbs,axis=-1)
                angles = np.arccos(np.clip(np.sum(coords_nhbs[None,:,:]*coords_nhbs[:,None,:],axis=-1)/(norms[None,:]*norms[:,None]),-1,1))
                np.fill_diagonal(angles,10)
                na,nb = np.unravel_index(np.argmin(angles, axis=None), angles.shape)
                na = neighbours[na]
                nb = neighbours[nb]
                label_max += 1
                coords = np.append(coords,((coords[na]+coords[nb])/2)[None,:],axis=0)
                coords_real = np.append(coords_real,coords_bp[None,:],axis=0)
                t.remove_edges_from([(n0,na),(n0,nb)])
                t.add_edges_from([(n0,label_max),(label_max,na),(label_max,nb)])
                neighbours.remove(na)
                neighbours.remove(nb)
                neighbours.append(label_max)

    if calc_cost:
        topo = topology(adj=nx_to_adj(t))
        _,_,_,EW,_ = BOT_optimize(topo,supply_arr,demand_arr,coords_sources,coords_sinks,al,cost_func=cfunc)
        length = np.linalg.norm(coords_real[np.arange(n,2*n-2)][:,None,:]-coords_real[topo.adj],axis=-1)
        edge_costs = np.abs(EW)**al*length
        cost = np.sum(edge_costs[topo.adj<np.arange(n,2*n-2)[:,None]])

        return topo,coords_real,cost

    return topology(adj=nx_to_adj(t)),coords_real


def generalized_angular_stress_reduction(t,demands,coords):

    n = len(coords)

    coords_real = np.copy(coords)
    #nx.draw(t)
    #plt.show()
    label_max = len(coords) - 1
    for node in list(t.nodes()):
        #get list of neighbor nodes
        neighbours = list(nx.all_neighbors(t, node))
        neighbours = [int(nbh) for nbh in neighbours]
        neighbours = np.unique(neighbours).tolist()
        if len(neighbours) > 1:
            ###connect all neighbors plus node itself via high degree BP
            #BP coords and labels
            coords_bp = coords[node] 
            label_max += 1
            n0 = label_max
            coords = np.append(coords,coords_bp[None,:],axis=0)
            coords_real = np.append(coords_real,coords_bp[None,:],axis=0)

            #get list of in and out flows
            in_flows = [t.get_edge_data(nbh,node,default={"weight": 0})["weight"] for nbh in neighbours]
            out_flows = [t.get_edge_data(node,nbh,default={"weight": 0})["weight"] for nbh in neighbours]

            #connect node and its neighbors to BP
            t.remove_node(node)
            t.add_edge(label_max,node,weight=np.sum(in_flows))
            t.add_edge(node,label_max,weight=np.sum(out_flows))
            for i,nd in enumerate(neighbours):
                t.add_edge(label_max,nd,weight=out_flows[i])
                t.add_edge(nd,label_max,weight=in_flows[i])

            ###iteratively reduce neigbors by introducing new BPs
            while len(neighbours)>2:
                # find neighbors na,nb with smallest angle
                coords_nhbs = coords[neighbours] - coords[node][None,:]
                norms = np.linalg.norm(coords_nhbs,axis=-1)
                angles = np.arccos(np.clip(np.sum(coords_nhbs[None,:,:]*coords_nhbs[:,None,:],axis=-1)/(norms[None,:]*norms[:,None]),-1,1))
                np.fill_diagonal(angles,10)
                ind_na,ind_nb = np.unravel_index(np.argmin(angles, axis=None), angles.shape)

                #store flows and labels
                in_a = in_flows[ind_na]
                in_b = in_flows[ind_nb]
                out_a = out_flows[ind_na]
                out_b = out_flows[ind_nb] 
                na = neighbours[ind_na]
                nb = neighbours[ind_nb]
                label_max += 1

                #get real and heuristically optimal position of new BP
                coords = np.append(coords,((coords[na]+coords[nb])/2)[None,:],axis=0)
                coords_real = np.append(coords_real,coords_bp[None,:],axis=0)

                #connect nodes via new BP
                t.remove_edges_from([(n0,na),(na,n0),(n0,nb),(nb,n0)])
                t.add_weighted_edges_from([(n0,label_max,out_a+out_b),(label_max,n0,in_a+in_b),(label_max,na,out_a),(na,label_max,in_a),(label_max,nb,out_b),(nb,label_max,in_b)])
                neighbours.remove(na)
                neighbours.remove(nb)
                neighbours.append(label_max)
                in_flows.pop(max(ind_na,ind_nb))
                in_flows.pop(min(ind_na,ind_nb))
                in_flows.append(in_a+in_b)
                out_flows.pop(max(ind_na,ind_nb))
                out_flows.pop(min(ind_na,ind_nb))
                out_flows.append(out_a+out_b)

    #nx.draw(t)
    #plt.show()
    ### turn into undirected graph by adding flow weights
    graph = nx.Graph()
    graph.add_edges_from(t.edges(),weight=0)
    for u,v,d in t.edges(data=True):
        graph[u][v]["weight"] += d["weight"]

    ###turn into adj list
    N=nx.number_of_nodes(graph)
    adj=[]
    EW =[]
    for bp in range(n,N):
        nbrs = list(nx.neighbors(graph, bp))
        adj.append(nbrs)
        EW.append([graph[bp][nbr]["weight"] for nbr in nbrs])


    return adj,EW,coords_real