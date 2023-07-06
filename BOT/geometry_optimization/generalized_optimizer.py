import numpy as np


def general_optimize(adj, EW, coords, demand_matrix, al, improv_threshold=1e-7):

    #get basic parameters
    dim = len(coords[0])
    nsites = len(coords)
    nbps = len(adj)

    #assign initial BP positions
    coords_arr = np.vstack((coords, np.random.rand(nbps,dim)))

    improv = 1
    cost_old = np.inf
    while improv > improv_threshold:
        d = np.linalg.norm(coords_arr[:,None,:]-coords_arr[None,:,:],axis=-1)+1e-12
        A = np.zeros((dim*nbps,dim*nbps))
        b = np.zeros((dim*nbps))
        for bp in range(nbps):
            adj_arr = np.array(adj[bp])
            EW_arr = np.abs(np.array(EW[bp]))**al
            b[bp*dim:(bp+1)*dim] = 1/np.sum(EW_arr/d[bp+nsites][adj_arr])*np.sum((EW_arr[adj_arr<nsites]/d[bp+nsites][adj_arr[adj_arr<nsites]])[:,None]*coords_arr[adj_arr[adj_arr<nsites]],axis=0)
            for k in range(dim):
                A[bp*dim+k,(adj_arr[adj_arr>=nsites]-nsites)*dim+k]  =  1/np.sum(EW_arr/d[bp+nsites][adj_arr]) * (EW_arr[adj_arr>=nsites]/(d[bp+nsites][adj_arr[adj_arr>=nsites]]))
        
        np.fill_diagonal(A,-1)
        b= -b
        coords_arr[nsites:] = np.linalg.solve(A,b).reshape(nbps,dim)

        cost_new=calc_cost(adj,EW,d,nsites,al)
        improv = cost_old-cost_new
        cost_old = cost_new

            
    return adj, cost_new, coords_arr, EW



def calc_cost(adj,EW,d,n,al):

    nbps = len(adj)

    cost=0
    for bp in range(nbps):
        for i,a in enumerate(adj[bp]):
            if a<bp+n:
                cost += EW[bp][i]**al * d[bp+n,a]

    return cost





    


    
