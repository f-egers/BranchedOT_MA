import os
from matplotlib.image import imread
import numpy as np
import random

def load_image(filename):
    
    my_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(my_path, filename+'.png')
    a=imread(path)
    if filename=='annulus':
        return a[:,:,3]-a[:,:,1]
    rgb_weights = [0.2989, 0.5870, 0.1140]
    
    grayscale_image = np.dot(a[...,1:], rgb_weights)
    grayscale_image = 0.5*grayscale_image/np.max(grayscale_image)

    return grayscale_image
    

def next_position_markov(current_point,aim_point,step,sigma=0.1):
    v=aim_point-current_point
    norm=np.linalg.norm(v)
    if norm !=0:
        v/=norm
    
    return current_point+v*step+np.random.normal(loc=0,scale=sigma,size=v.shape)
    
def sample_points_from_image(n,img,Random=True,noise_perc=0):
    if not Random:
        random.seed(42)
    # non_zero=np.where(img!=0)
    
    if noise_perc>0:
        non_zero=np.where((img>0.25) & (img<0.7))
        non_zero2=np.where((img<0.25) | (img>0.7))
        idx=random.sample(range(len(non_zero[0])),n-int(noise_perc*n))

        idx2=random.sample(range(len(non_zero2[0])),int(noise_perc*n))
        
        x_coord=non_zero[0][idx]
        y_coord=non_zero[1][idx]
        
        x_coord2=non_zero2[0][idx2]
        y_coord2=non_zero2[1][idx2]
        
        x_coord=np.concatenate([x_coord,x_coord2]).astype(np.int)
        y_coord=np.concatenate([y_coord,y_coord2]).astype(np.int)
    else:
        non_zero=np.where(0.25<img)
        #r
        idx=random.choices(list(range(len(non_zero[0]))),k=n)
   # max_side=max(img.shape[0],img.shape[1])
        x_coord=non_zero[0][idx]#*4/max_side
        y_coord=non_zero[1][idx]#*4/max_side
    
    
    return x_coord,y_coord




def standard_random_bot_problem(distr,n,gaus_noise=3,lapl_noise=3,noise_perc=0.01,coords_source=None,al=None):
    
    a,img = create_points(n-1,distr,gaussian_noise=gaus_noise,laplacian_noise=lapl_noise,noise_perc=noise_perc)
    scale = np.max(a)
    a[:,0]=a[:,0]/scale
    a[:,1]=a[:,1]/scale
    coords_sinks = a
    if coords_source is None:
        coords_sources = np.array([[0.95,0.29]])

    dists = np.linalg.norm(coords_sinks[:,None,:]-coords_sinks[None,:,:],axis=-1)
    quantile = np.quantile(dists,0.08)
    weights = np.sum(dists<quantile,axis=-1)
    supply_arr = np.array([1.0])
    #demand_arr = weights/np.sum(weights)
    demand_arr = 1/len(coords_sinks)*np.ones(len(coords_sinks))
    if al is None:
        al = np.random.random()

    bot_problem_dict={}
    bot_problem_dict["al"] = al
    bot_problem_dict["coords_sources"] = coords_sources
    bot_problem_dict["coords_sinks"] = coords_sinks
    bot_problem_dict["supply_arr"] = supply_arr
    bot_problem_dict["demand_arr"] = demand_arr

    return bot_problem_dict




def create_points(n=100,distribution='non_convex',Random=True,
                  gaussian_noise=0,laplacian_noise=0,noise_perc=0):
    
    """
    Parameters
    ----------
    n : int, optional
        number of nodes. The default is 100.
    distribution : str, optional
        Sampling distribution of the points.
    Random : bool, optional
        If False the points are sampled with fixed seed =42. The default is True.
    gaussian_noise : float, optional
        Standard deviation of the 0-centered gaussian noise added to the samples
    gaussian_noise : float, optional
        Standard deviation of the 0-centered laplacian noise added to the samples
    noise_perc: float, optional
        Percentage of samples that are sampled uniformly over the complementary
        area of the data sampled from a binary image. E.g if noise_perc=0.2
        80% of the data belongs to the shaded area (foreground) of the binary 
        image and 20% to the non-shaded area (background).
    """
    
    if distribution not in ['non_convex', 'non_convex2', 'tree','tree2high',"zigzag","simple_loop","figure8", "triangle","box"]:
        raise ValueError("not a valid dist.")
    img=load_image(distribution)
    
    x_coord,y_coord=sample_points_from_image(n,img,Random,noise_perc)
    P=np.vstack([x_coord,y_coord]).T
    if gaussian_noise!=0:
        P=P.astype(np.float)+np.random.normal(loc=0,scale=gaussian_noise,size=P.shape)
    if laplacian_noise!=0:
        P=P.astype(np.float)+np.random.laplace(loc=0,scale=laplacian_noise,size=P.shape)
    return P,img



def uniform_random_bot_problem(num_sources, num_sinks, normalised_to=1, dim=2, max_length=1.):
    
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




def fourier_random_bot_problem(n,dim=2,num_sources=None,smoothness=None,entropy=None,gap=None):

    if num_sources is None:
        num_sources = np.random.randint(1,n)
    if smoothness is None:
        smoothness = 5*np.random.rand()
    if entropy is None:
        entropy = np.random.rand()
    if gap is None:
        gap = 0.75*np.random.rand()

    gridsize = n
    #build frequency filter
    freq = np.fft.fftfreq(gridsize)
    freq_arr = np.linalg.norm(np.stack(np.meshgrid(*(dim*[freq]))),axis=0)
    freq_filter = 1/(freq_arr+0.01*np.min(freq_arr[freq_arr>0]))**smoothness

    #apply filter to white noise to get prob dist
    rand_freq = freq_filter*np.fft.fftn(np.random.rand(*(dim*[gridsize])))
    noise = np.real(np.fft.ifftn(rand_freq))
    noise = noise-np.median(noise)

    #dist for sources
    prob_dist = np.copy(noise)
    upper = gap*np.max(prob_dist)
    prob_dist[prob_dist<upper]=0
    prob_dist = prob_dist/np.sum(prob_dist)

    #dist for sinks
    inv_prob_dist = np.copy(noise)
    lower = gap*np.min(inv_prob_dist)
    inv_prob_dist[inv_prob_dist>lower]=0
    inv_prob_dist = inv_prob_dist/np.sum(inv_prob_dist)

    #assign terminals to prob_dist or 1-prob_dist according to entropy
    thres = np.append(np.array(num_sources*[entropy]),np.array((n-num_sources)*[1-entropy]))
    assign = np.random.rand(n)>=thres

    #sample terminals positions from distributions
    grid_ind = np.zeros(n,dtype=int)
    grid_ind[assign] = np.random.choice(gridsize**dim,size=np.sum(assign),replace=True,p=prob_dist.reshape(-1)).astype(int)
    grid_ind[~assign] = np.random.choice(gridsize**dim,size=np.sum(~assign),replace=True,p=inv_prob_dist.reshape(-1)).astype(int)
    grid_pos = np.stack(np.unravel_index(grid_ind,prob_dist.shape),axis=-1)

    #extract coordinates and demands
    coords = (grid_pos+0.5)/gridsize+0.5*(1/gridsize)*(np.random.rand(*grid_pos.shape)-0.5)
    demands = np.zeros(n)
    demands[assign] = prob_dist.reshape(-1)[grid_ind[assign]]
    demands[~assign] = inv_prob_dist.reshape(-1)[grid_ind[~assign]]

    #build problem dict
    coords_sources = coords[:num_sources]
    coords_sinks = coords[num_sources:]
    supply_arr = demands[:num_sources]/np.sum(demands[:num_sources])
    demand_arr = demands[num_sources:]/np.sum(demands[num_sources:])
    al = np.random.rand()

    bot_problem_dict = {}
    bot_problem_dict["al"] = al
    bot_problem_dict["coords_sources"] = coords_sources
    bot_problem_dict["coords_sinks"] = coords_sinks
    bot_problem_dict["supply_arr"] = supply_arr
    bot_problem_dict["demand_arr"] = demand_arr

    return bot_problem_dict#,noise,[al,num_sources,entropy,smoothness,gap]