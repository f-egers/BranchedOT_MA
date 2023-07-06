import numpy as np
from numba import jit
from tqdm import tqdm
from queue import Queue

class topology():

    def __init__(self, vec_rep=None, adj=None):

        if adj is None:

            if vec_rep is None:
                raise ValueError("exactly one of (vec_rep) or (adj) must be given.")
            self.vec_rep = np.copy(vec_rep)
            self.adj = vec_to_adj(self.vec_rep)

        else:

            if adj is None:
                raise ValueError("exactly one of (vec_rep) or (adj) must be given.")
            self.adj = np.copy(adj).astype(np.intc)
            self.vec_rep = adj_to_vec(adj)


    def __str__(self):
        return "t="+str(self.vec_rep)

    def __repr__(self):
        return "t="+str(self.vec_rep)


def generate_random_topology_vecs(n,nsites):
    """
    fast way of generating array of random topology vectores. Might contain duplicates.
    Generally not very well distributed.
    """
    lim_vec = 2*np.arange(1,nsites-2)
    rand = np.random.rand(n,nsites-3)
    topvecs = lim_vec[None,:]*rand

    return topvecs.astype(int)


def generate_topology_vecs(n,nsites):
    """
    slower way of generating random topology vectores.
    But better distributed.
    """
    vec = np.zeros(nsites-3).astype(int)
    head = 0
    breaker = False
    for i in range(n):
        yield vec
        vec[head] += 1
        while vec[head]==(2*head+3):
            vec[head]=0
            head+=1
            if head==nsites-3:
                breaker=True 
                break
            vec[head]+=1
        if breaker: break
        head=0


def generate_topologies(n,nsites):
    for tvec in generate_topology_vecs(n,nsites):
        yield topology(vec_rep=tvec)

@jit
def vec_to_adj(vec):
    """
    builds adj-array for a given topology vector representation.
    """

    #initialize arrays
    n= len(vec)+3
    adj = np.zeros((n-2,3),dtype=np.intc)
    edge = np.zeros((2*n-3,2),dtype=np.intc)

    #build initial n=3-topology corresponding to nullvector
    m = n
    init_arr = np.array([0,1,2])
    adj[0]=init_arr
    edge[init_arr,0] = init_arr
    edge[init_arr,1] = m

    #build tree by processing vector elements
    for i,e in enumerate(vec):
        en=i+3
        m=i+1
        sn=m+n
        ea = edge[e][0]
        eb = edge[e][1] #eb must be a BP???
        adj[m][0]  =  ea
        adj[m][1]  =  eb
        adj[m][2]  =  en
        if ea>=n: adj[ea-n][adj[ea-n]==eb]=sn 
        if eb>=n: adj[eb-n][adj[eb-n]==ea]=sn
        edge[e][1]=sn
        edge[2*i+3][0]=en
        edge[2*i+3][1]=sn
        edge[2*i+4][0]=sn
        edge[2*i+4][1]=eb

    return adj


def adj_to_vec(adj):
    """
    builds topology vector representation from adj and edge arrays.
    """

    adj_ = np.copy(adj)

    n=len(adj_)+2

    vec = np.zeros(n-3,dtype=int)

    leafq = Queue()
    for i,a in enumerate(adj_):
        if np.sum(a<n)==2:
            leafq.put(i)

    ext_edge=2*np.arange(n)-3
    ext_edge[:3] = np.array([0,1,2])

    int_edge = 2*np.arange(n)-2
    int_edge[:3] = np.array([0,1,2])

    elim_hist = [[i] for i in range(n)]

    while not leafq.empty():
        i = leafq.get()
        a = adj_[i]
        hn = np.max(a[a<n])
        ln = np.min(a[a<n])
        if hn>2: 
            sum_elim=0
            for elim in elim_hist[ln]:
                if elim<hn: sum_elim += 1

            if sum_elim>1:
                for elim in reversed(elim_hist[ln]):
                    if elim<hn: 
                        v=int_edge[elim]
                        break
            else:
                v=ext_edge[ln]
            vec[hn-3]=v
            elim_hist[ln].append(hn)
            snp = a[a>=n][0]-n
            adj_[snp][adj_[snp]==i+n]=ln
            if np.sum(adj_[snp]<n)==2:
                leafq.put(snp)

    return vec




if __name__=="__main__":
    for tvec in tqdm(generate_topology_vecs(1000,7)):
        #print(tvec)
        a= vec_to_adj(tvec)
        b= adj_to_vec(a)
        if np.any(tvec!=b):
            print(tvec)
            print(b)







    