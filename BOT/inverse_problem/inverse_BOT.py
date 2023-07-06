import numpy as np
from tqdm import tqdm
from scipy.optimize import linprog

from ..topology import topology

def vec_angle(coords,a1,a2,b1,b2):
    vec1 = coords[a2]-coords[a1]
    vec2 = coords[b2]-coords[b1]

    angle = np.arccos(np.sum(vec1*vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))

    return angle


def Newton_Raphson(func,dfunc,init,acc):
    x = init
    dx = 1
    while dx>acc:
        x_old = x
        x = np.clip(x-func(x)/dfunc(x),0.0000001,0.9999999)
        dx = np.abs(x-x_old)
    return x


def f(al,theta,k):
    return (1-k**(2*al)-(1-k)**(2*al))/(2*k**al*(1-k)**al) - np.cos(theta)

def dfda(al,theta,k):
    return  -0.5*(1-k)**(-al)*(-k**(2*al)-(1-k)**(2*al)+1)*k**(-al)*np.log(1-k) - 0.5*(1-k)**(-al)*(-k**(2*al)-(1-k)**(2*al)+1)*k**(-al)*np.log(k) + 0.5*(1-k)**(-al)*k**(-al)*(-2*k**(2*al)*np.log(k)-2*(1-k)**(2*al)*np.log(1-k))

def dfdk(al,theta,k):
    return 0.5*al*(-(k-1)*k)**(-al-1)*(-k**(2*al)+(1-k)**(2*al)+2*k-1)

def f1(al,theta,k):
    return (1+k**(2*al)-(1-k)**(2*al))/(2*k**al) - np.cos(theta)

def df1dk(al,theta,k):
    return -(al*k**(-al - 1)*(-k**(2*al + 1) + k**(2*al) + k*(1 - k)**(2*al) + (1 - k)**(2*al) + k - 1))/(2*(k - 1))

def f2(al,theta,k):
    return (1-k**(2*al)+(1-k)**(2*al))/(2*(1-k)**al) - np.cos(theta)

def df2dk(al,theta,k):
    return -((1 - k)**(-al - 1)*(al*k*((1 - k)**(2*al) - 1) - al*(k - 2)*k**(2*al)))/(2*k)


def invert_BOT_to_al(nsites, topo: topology, coords, flows, acc = 0.001, tol=10e-9):
    lower = 0
    als = []
    for bpi in tqdm(range(nsites-2)):
        bp = bpi+nsites
        bp_pos = coords[bp]
        nb_pos = coords[topo.adj[bpi],:]
        dists = np.linalg.norm(bp_pos[None,:]-nb_pos,axis=1)
        closest_point = np.argmin(dists)
        n0_i = np.argmax(np.abs(flows[bpi]))
        
        n0 = topo.adj[bpi][n0_i]
        n12 = topo.adj[bpi][topo.adj[bpi] != n0]
        k = flows[bpi][topo.adj[bpi]==n12[0]]/(flows[bpi][topo.adj[bpi]==n12[0]]+flows[bpi][topo.adj[bpi]==n12[1]])

        if dists[closest_point]<tol:
            if closest_point==n0_i:
                psi = vec_angle(coords,n0,n12[0],n0,n12[1])
                l = 0.5
                dl = 1
                while dl>acc:
                    l_old = l
                    l = l-f(l,psi,k)/dfda(l,psi,k)
                    dl = np.abs(l-l_old)
                if l>lower:
                    lower=l

            continue


        ang = vec_angle(coords,bp,n12[0],bp,n12[1])

        al = Newton_Raphson(lambda al: f(al,ang,k),lambda al: dfda(al,ang,k),0.5,acc)

        als.append(al)

    als = np.array(als)
    als = als[als>(lower-3*acc)]
    best_guess = np.median(als[als>(lower-3*acc)])

    return als,best_guess,lower


def invert_BOT_to_flows(nsites, topo: topology, coords, EWsign, al, acc = 0.001, tol=10e-7, c=None):
    A_eq = np.zeros((1,3*nsites-6))
    A_ineq = np.zeros((1,3*nsites-6))

    sumeq1 = np.zeros(3*nsites-6)

    for bpi in tqdm(range(nsites-2)):
        bp = bpi+nsites
        bp_pos = coords[bp]
        nb_pos = coords[topo.adj[bpi],:]
        nb_EWsign = EWsign[bpi]
        dists = np.linalg.norm(bp_pos[None,:]-nb_pos,axis=1)
        closest_point = np.argmin(dists)
        n0_i = np.where(np.sum(nb_EWsign[None,:]*nb_EWsign[:,None],axis=-1)<0)[0]

        n0 = topo.adj[bpi][n0_i]
        n12_i = np.where(topo.adj[bpi] != n0)[0]
        n12 = topo.adj[bpi][n12_i]

        if dists[closest_point]<tol:
            if closest_point==n12_i[0]:
                ang1 = np.pi-vec_angle(coords,n12[0],n0,n12[0],n12[1])
                k1=Newton_Raphson(lambda k: f1(al,ang1,k),lambda k: df1dk(al,ang1,k),0.5,acc)
            elif closest_point==n12_i[1]:
                ang1 = np.pi-vec_angle(coords,n12[1],n0,n12[1],n12[0])
                k1=Newton_Raphson(lambda k: f2(al,ang1,k),lambda k: df2dk(al,ang1,k),0.5,acc)
            elif closest_point==n0_i:
                ang1 = vec_angle(coords,n0,n12[1],n0,n12[0])
                k1=Newton_Raphson(lambda k: f(al,ang1,k),lambda k: dfdk(al,ang1,k),0.5,acc)
            
        else:
            ang1 = vec_angle(coords,n0,bp,bp,n12[0])
            k1=Newton_Raphson(lambda k: f1(al,ang1,k),lambda k: df1dk(al,ang1,k),0.5,acc)
        
        eq = np.zeros(3*nsites-6)
        eq[3*bpi+n12_i[0]]=1-k1
        eq[3*bpi+n12_i[1]]=-k1
        if dists[closest_point]<tol:
            A_ineq = np.append(A_ineq,eq[None,:],axis=0)
        else:
            A_eq = np.append(A_eq,eq[None,:],axis=0)

        eq = np.zeros(3*nsites-6)
        eq[3*bpi+n0_i]=1
        eq[3*bpi+n12_i[0]]=1
        eq[3*bpi+n12_i[1]]=1
        A_eq = np.append(A_eq,eq[None,:],axis=0)

        for i in range(3):
            if topo.adj[bpi][i]<nsites:
                if EWsign[bpi][i]==1:
                    sumeq1[3*bpi+i]=1
            else:
                if topo.adj[bpi][i]-nsites>bpi:
                    A_eq = np.append(A_eq,same_flow(nsites,topo.adj,topo.adj[bpi][i]-nsites,bpi),axis=0)

    A_eq = np.append(A_eq,sumeq1[None,:],axis=0)

    A_eq=A_eq[1:]
    A_ineq=A_ineq[1:]
    b_eq = np.zeros(A_eq.shape[0])
    b_eq[-1]=1
    b_ineq = np.zeros(A_ineq.shape[0])

    sols = []
    for i in range(1):
        if c is None:
            c = np.random.rand(3*nsites-6)
        lp_res = linprog(c,A_ineq,b_ineq,A_eq,b_eq,(-1,1))["x"]
        sols.append(lp_res)

    sols = np.array(sols)
    res = np.mean(sols,axis=0)

    return res.reshape(nsites-2,3)




def same_flow(nsites,adj,n1,n2):
    eq = np.zeros(3*nsites-6)
    eq[3*n1+np.where(adj[n1]==n2+nsites)[0]]=1
    eq[3*n2+np.where(adj[n2]==n1+nsites)[0]]=1
    return eq[None,:]









