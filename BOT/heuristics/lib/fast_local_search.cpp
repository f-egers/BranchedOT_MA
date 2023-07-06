#include <math.h>
#include <iostream>
#include <queue>
#include <cstdlib>

#include "fast_optimizer.cpp"

using namespace std;

double line_line_dist(int DIMENSION, double a1[], double a2[], double b1[], double b2[]){
    
    int m;
    
    double R1,R2,D1,D2,D3 = 0;
    double s,t;
    
    for(m=0;m<DIMENSION;m++){
        R1 += pow(a2[m]-a1[m],2.0);
        R2 += pow(b2[m]-b1[m],2.0);
        D1 += (b2[m]-b1[m])*(a2[m]-a1[m]);
        D2 += (b1[m]-a1[m])*(a2[m]-a1[m]);
        D3 += (b2[m]-b1[m])*(b1[m]-a1[m]);
    }

    s = (D1*D3+D2*R2)/(R1*R2+D1*D1);
    t = (D1*D2-D3*R1)/(R1*R2+D1*D1);

    if(s>1.0){s=1.0;}
    if(s<0.0){s=0.0;}
    if(t>1.0){t=1.0;}
    if(t<0.0){t=0.0;}

    double dist = 0;
    for(m=0;m<DIMENSION;m++){
        dist += pow(a1[m]+s*(a2[m]-a1[m])-b1[m]+t*(b2[m]-b1[m]),2.0);
    }
    dist = sqrt(dist);
    return dist;
}


double step(int *iter, double cost, int DIMENSION, int NUMSITES, int adj[][3], double EW[][3], double demands[], double XX[], double al, double improv_thres = 1e-7, int max_tries = 20, double kernel = 0.3, double beta = 1.0, bool CBOT = false){

    int i,e,c,m,bp,node,root,parent;
    int e1,e2;
    int nh1,nh2;
    int connector,disconnector;
    double new_cost;
    int num_tries;

    double a1[DIMENSION];
    double a2[DIMENSION];
    double b1[DIMENSION];
    double b2[DIMENSION];
    double dist;

    queue<int> root_queue;
    queue<int> parent_queue;

    int start_bp = rand()%(NUMSITES-2);

    num_tries = 0;
    for(i = 0;i<NUMSITES-2;i++){
        bp = (start_bp+i)%(NUMSITES-2);
        for(c=0;c<3;c++){
            //select edge to disconnect
            connector = adj[bp][c];
            disconnector = bp+NUMSITES;

            //walk down tree to find good edges to reconnect
            for(m=0;m<3;m++){
                node = adj[bp][m];
                if(node != connector){
                    if(node >= NUMSITES){root_queue.push(node);parent_queue.push(disconnector);}
                }
            }
            while(!root_queue.empty()){
                root = root_queue.front();root_queue.pop();
                parent = parent_queue.front();parent_queue.pop();

                //check edges incident on current root 
                for(e=0;e<3;e++){
                    e1 = root;
                    e2 = adj[root-NUMSITES][e];

                    if(e1<e2){continue;} //avoid double counting
                    if(e1==disconnector){continue;}
                    if(e2==disconnector){continue;}

                    //calc. dist between edges and try edge according to kernel probability
                    for(m=0;m<DIMENSION;m++){
                        a1[m] = XX[connector*DIMENSION+m];
                        a2[m] = XX[disconnector*DIMENSION+m];
                        b1[m] = XX[e1*DIMENSION+m];
                        b2[m] = XX[e2*DIMENSION+m];
                    }
                    dist = line_line_dist(DIMENSION,a1,a2,b1,b2);
                    if(((double) rand() / (RAND_MAX))>exp(-pow(dist/kernel,2.0))){continue;}

                    //all test passed, we will try this edge
                    num_tries++;

                    //make the move
                    nh1 = -1;
                    nh2 = -1;
                    for(m=0;m<3;m++){
                        node = adj[bp][m];
                        if(node != connector){
                            if(nh1 == -1){nh1 = node;}else{nh2 = node;}
                        }
                    }
                    if(nh1 >= NUMSITES){
                        for(m=0;m<3;m++){
                            if(adj[nh1-NUMSITES][m]==disconnector){adj[nh1-NUMSITES][m] = nh2;break;}
                        }
                    }
                    if(nh2 >= NUMSITES){
                        for(m=0;m<3;m++){
                            if(adj[nh2-NUMSITES][m]==disconnector){adj[nh2-NUMSITES][m] = nh1;break;}
                        }
                    }
                    if(e1 >= NUMSITES){
                        for(m=0;m<3;m++){
                            if(adj[e1-NUMSITES][m]==e2){adj[e1-NUMSITES][m] = disconnector;break;}
                        }
                    }
                    if(e2 >= NUMSITES){
                        for(m=0;m<3;m++){
                            if(adj[e2-NUMSITES][m]==e1){adj[e2-NUMSITES][m] = disconnector;break;}
                        }
                    }
                    adj[bp][0] = e1;
                    adj[bp][1] = e2;
                    adj[bp][2] = connector;

                    ////ckeck for improvement
                    //shuffle coordinates
                    for(m=DIMENSION*NUMSITES;m<DIMENSION*(2*NUMSITES-2);m++){
                        XX[m] = ((double) rand() / (RAND_MAX));
                    }
                    new_cost = iterations(iter, DIMENSION, NUMSITES, adj, EW, demands, XX, al, improv_thres = 1e-7, beta = 1.0, CBOT = CBOT);
                    if(new_cost < cost){
                        cout << "found improvement after checking " << num_tries << " topologies. New cost: " << new_cost << endl;
                        return new_cost;
                    }
                    else{
                        if(nh1 >= NUMSITES){
                            for(m=0;m<3;m++){
                                if(adj[nh1-NUMSITES][m]==nh2){adj[nh1-NUMSITES][m] = disconnector;break;}
                            }
                        }
                        if(nh2 >= NUMSITES){
                            for(m=0;m<3;m++){
                                if(adj[nh2-NUMSITES][m]==nh1){adj[nh2-NUMSITES][m] = disconnector;break;}
                            }
                        }
                        if(e1 >= NUMSITES){
                            for(m=0;m<3;m++){
                                if(adj[e1-NUMSITES][m]==disconnector){adj[e1-NUMSITES][m] = e2;break;}
                            }
                        }
                        if(e2 >= NUMSITES){
                            for(m=0;m<3;m++){
                                if(adj[e2-NUMSITES][m]==disconnector){adj[e2-NUMSITES][m] = e1;break;}
                            }
                        }
                        adj[bp][0] = nh1;
                        adj[bp][1] = nh2;
                        adj[bp][2] = connector;


                    }
                }

                //quit if max number of tries is reach
                if(num_tries>max_tries){return cost;}

                //push new roots to queue
                for(m=0;m<3;m++){
                    node = adj[root-NUMSITES][m];
                    if(node != parent){
                        if(node >= NUMSITES){root_queue.push(node);parent_queue.push(root);}
                    }
                }
            }
        }
    }
    return cost;
}



extern "C"
double downhill_climb(int *iter, int DIMENSION, int NUMSITES, int adj[][3], double EW[][3], double demands[], double XX[], double al, double improv_thres = 1e-7, int max_steps = 100, int max_tries = 20, double kernel = 0.3, double beta = 1.0, bool CBOT = false){
    
    int m;
    double cost = iterations(iter, DIMENSION, NUMSITES, adj ,EW ,demands ,XX ,al ,improv_thres = 1e-7 ,beta = 1.0, CBOT = CBOT);
    double new_cost;

    int num_steps = 0;
    while(true){
        num_steps++;
        new_cost = step(iter, cost, DIMENSION, NUMSITES, adj ,EW ,demands ,XX ,al ,improv_thres = 1e-7 ,max_tries = max_tries, kernel = kernel, beta = 1.0, CBOT = CBOT);
        if(num_steps>max_steps){break;}
        if(new_cost<cost){
            cost = new_cost;
            //cout << cost << endl;
            continue;
        }
        break;
    }
    //shuffle coordinates
    for(m=DIMENSION*NUMSITES;m<DIMENSION*(2*NUMSITES-2);m++){
        XX[m] = ((double) rand() / (RAND_MAX));
    }
    cost = iterations(iter, DIMENSION, NUMSITES, adj ,EW ,demands ,XX ,al ,improv_thres = 1e-7 ,beta = 1.0, CBOT=CBOT);
    //cout << num_steps << endl;
    return cost;
}