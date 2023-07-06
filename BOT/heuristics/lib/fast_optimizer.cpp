/*
compile with the following two commands from command line (Linux only):
g++ -c -Wall -Werror -O3 -fpic fast_optimizer.cpp
g++ -shared -Wall -O3 -o fast_optimizer.so fast_optimizer.o
The flags -Wall -O3 can technically be omitted, but should not!
*/

#include <math.h>
#include <iostream>
#include <queue>
#include <cstdlib>
using namespace std;

void optimize(int DIMENSION, int NUMSITES, int adj[][3], double EW[][3], double EL[][3], double XX[], double al, double beta = 1.0){
    double tol = 1e-12;

    double B[NUMSITES][3],C[NUMSITES][DIMENSION];
    int eqnstack[NUMSITES], leafQ[NUMSITES], val[NUMSITES], intern_edge[NUMSITES][3];
    int i,m,j,i2;
    int n0,n1,n2,lqp,eqp,k1;
    double q0,q1,q2,t;

    lqp = eqp = 0;
    k1 = NUMSITES-2;

    /* prepare equations */ 
    for(i=k1-1;i>=0;i--){
        n0 = adj[i][0];
        n1 = adj[i][1];
        n2 = adj[i][2];
        q0 = pow(fabs(EW[i][0]),al)*pow(EL[i][0]+tol,beta-2.0);
        q1 = pow(fabs(EW[i][1]),al)*pow(EL[i][1]+tol,beta-2.0);
        q2 = pow(fabs(EW[i][2]),al)*pow(EL[i][2]+tol,beta-2.0);

        t = q0+q1+q2;
        q0 /= t;
        q1 /= t;
        q2 /= t;

        val[i] = 0;
        B[i][0] = B[i][1] = B[i][2] = 0.0;
        intern_edge[i][0] = intern_edge[i][1] = intern_edge[i][2] = 0;

        for(m=0;m<DIMENSION;m++){C[i][m] = 0.0;}

        #define prep(a,b,c) if(b>=NUMSITES){val[i]++;B[i][a]=c;intern_edge[i][a]=1;}else{for(m=0;m<DIMENSION;m++){C[i][m]+=XX[b*DIMENSION+m]*c;}}
        prep(0,n0,q0);
        prep(1,n1,q1);
        prep(2,n2,q2);

        if(val[i]<=1){leafQ[lqp]=i,lqp++;}
    }
    while(lqp > 1){ 
        /* eliminate leaf i from tree*/ 
        lqp--; i = leafQ[lqp]; val[i]--; i2 = i+ NUMSITES; 
        eqnstack[eqp] = i; eqp++;/* push i in stack */ 
        for(j =0; j < 3; j++){if(intern_edge[i][j] != 0){break;}}
        q0 = B[i][j]; 
        j = adj[i][j]-NUMSITES;/* neighbor is j */ 
        val[j]-- ; 
        if(val[j] == 1){ leafQ[lqp] = j; lqp ++; }/* check if neighbor has become leaf? */ 
        for(m=0; m<3; m++){if(adj[j][m] == i2){break;}} 
        q1 = B[j][m]; B[j][m] = 0.0; intern_edge[j][m] = 0; 
        t = 1.0-q1*q0; t = 1.0/t; 
        for(m=0; m<3; m++){B[j][m] *= t;} 
        for(m=0; m<DIMENSION; m++){ C[j][m] += q1*C[i][m]; C[j][m] *= t; } 
    }
    /* Solve trivial tree */ 
    i = leafQ[0]; i2 = i + NUMSITES; 
    for(m=0; m <DIMENSION; m++){XX[i2*DIMENSION+m] = C[i][m];}
    /* Solve rest by backsolving */ 
    while(eqp > 0){ 
        eqp--; i = eqnstack[eqp]; i2 = i+ NUMSITES; 
        for(j =0; j <3; j ++){if(intern_edge[i][j] != 0){break;}}/* find neighbor j */ 
        q0 = B[i][j]; 
        j = adj[i][j];/* get neighbor indeces */ 
        for(m = 0; m < DIMENSION; m++){XX[i2*DIMENSION+m] = C[i][m] + q0*XX[j*DIMENSION+m];}
    }    
    return;
}


double length(int DIMENSION, int NUMSITES, int adj[][3],  double EW[][3], double EL[][3], double XX[], double al, double beta = 1.0) {
    /*calculates the cost of the current configuration and stores edge lengths in EL*/
    #define dist(a,b) t=0.0;for(m=0;m<DIMENSION;m++){r=XX[a*DIMENSION+m]-XX[b*DIMENSION+m];t+=r*r;}t=sqrt(t);
    int m,i2,i,j;
    int n0,n1,n2,k1;
    double leng,t,r;
    leng = 0.0;
    k1=NUMSITES-2;
    for(i=0;i<k1;i++){
        i2 = i+NUMSITES;
        n0 = adj[i][0];n1=adj[i][1];n2=adj[i][2];
        if(n0<i2){
            dist(n0,i2);leng+=pow(fabs(EW[i][0]),al)*pow(t,beta);EL[i][0]=t;n0-=NUMSITES;
            if(n0>=0)for(j=0;j<3;j++)if(adj[n0][j]==i2){EL[n0][j]=t;break;}
        }
        if(n1<i2){
            dist(n1,i2);leng+=pow(fabs(EW[i][1]),al)*pow(t,beta);EL[i][1]=t;n1-=NUMSITES;
            if(n1>=0)for(j=0;j<3;j++)if(adj[n1][j]==i2){EL[n1][j]=t;break;}
        }
        if(n2<i2){
            dist(n2,i2);leng+=pow(fabs(EW[i][2]),al)*pow(t,beta);EL[i][2]=t;n2-=NUMSITES;
            if(n2>=0)for(j=0;j<3;j++)if(adj[n2][j]==i2){EL[n2][j]=t;break;}
        }
    }
    return leng;
}

extern "C"
void calculate_EW(int NUMSITES, double EW[][3], int adj[][3], double demands[], double al){
    /*calculates the flows on all edges using the same tree elimination scheme as the equation solver in optimize*/
    int i,j,m,i2;
    int n0,n1,n2;
    int leafQ[NUMSITES],val[NUMSITES];
    int lqp = 0;
    int done_edges[NUMSITES][3];
    double d[NUMSITES];

    for(i=0;i<NUMSITES-2;i++){
        n0 = adj[i][0];
        n1 = adj[i][1];
        n2 = adj[i][2];

        d[i] = 0.0;
        done_edges[i][0] = done_edges[i][1] = done_edges[i][2] = 0;
        EW[i][0] = EW[i][1] = EW[i][2] = 0.0;

        val[i] = 0;
        if(n0>=NUMSITES){val[i]++;}else{d[i]+=demands[n0];EW[i][0]=-demands[n0];done_edges[i][0]=1;}
        if(n1>=NUMSITES){val[i]++;}else{d[i]+=demands[n1];EW[i][1]=-demands[n1];done_edges[i][1]=1;}
        if(n2>=NUMSITES){val[i]++;}else{d[i]+=demands[n2];EW[i][2]=-demands[n2];done_edges[i][2]=1;}
        if(val[i]==1){leafQ[lqp]=i;lqp++;}
    }
    while(lqp>1){
        lqp--;
        i = leafQ[lqp]; i2=i+NUMSITES; val[i]--;
        for(m=0;m<3;m++){if(done_edges[i][m]==0){break;}} /*find parent BP*/
        EW[i][m] = d[i];
        done_edges[i][m] = 1;

        j = adj[i][m]-NUMSITES;
        for(m=0;m<3;m++){if(adj[j][m]==i2){break;}}
        EW[j][m] = -d[i];
        done_edges[j][m] = 1;
        d[j]+=d[i];
        val[j]--;
        if(val[j] == 1){ leafQ[lqp] = j; lqp ++; }/* check if neighbor has become leaf? */
    }
}

void calculate_opt_angle(double opt_ang[][2], int DIMENSION, int NUMSITES, int adj[][3], double EW[][3], double demands[], double XX[], double al, double improv_thres, double cost){
    /*calculates the optimal angles at each BP for the convergence checks later. Also initializes the BP positions (in OT configuration).*/
    int k,k1,k2,m,l,breaker,d,b;
    double m1,m2,m_rat;
    int n0,n1,n2;

    queue<int> BPQ;
    queue<int> BPQ2;
    int BP_div_point[NUMSITES][3];
    bool BP_done[NUMSITES];

    n0=n1=n2=0;

    m1=m2=0.0;

    //find diverngence points for every BP; "div point"" means the edge that has a different flow direction from the other two. We map to "2sinks-1source"-case
    for(k=0;k<NUMSITES-2;k++){
        BP_done[k]=false;
        breaker=0;
        for(m=0;m<3;m++){
            for(l=0;l<3;l++){
                if(m != l){
                    if(EW[k][m]*EW[k][l] >= 0){n1=m;n2=l;breaker=1;break;}
                }
            }
            if(breaker==1){break;}
        }
        for(m=0;m<3;m++){if(m != n1 && m != n2){n0=m;break;}}
        BP_div_point[k][0] = n0;BP_div_point[k][1] = n1;BP_div_point[k][2] = n2;
        k1 = adj[k][n0];
        //The points whose div points are terminals have to be initialized first
        if(k1<NUMSITES){
            BPQ.push(k);
        }
    }
    // iterate over BPs with terminals as their div point
    while(!BPQ.empty()){
        k=BPQ.front();BPQ.pop();

        BPQ2.push(k);
        while(!BPQ2.empty()){
            b = BPQ2.front();BPQ2.pop();
            BP_done[b] = true;

            n0 = BP_div_point[b][0];n1 = BP_div_point[b][1];n2 = BP_div_point[b][2];

            //init BP positions
            k1 = adj[b][n0];
            //dist_thres = (improv_thres*cost)/(pow(fabs(EW[b][n0]),al)*(NUMSITES-2));
            for(d=0;d<DIMENSION;d++){
                XX[(b+NUMSITES)*DIMENSION+d] = XX[k1*DIMENSION+d]+0.001*(double)((rand() % (4*NUMSITES) + 4*NUMSITES)/(8*NUMSITES));
            }

            //calc optimal angles
            m1=EW[b][n1];
            m2=EW[b][n2];
            m_rat=m1/(m1+m2);
            opt_ang[b][0] = acos((1.0+pow(m_rat,2.0*al)-pow(1.0-m_rat,2.0*al))/(2.0*pow(m_rat,al))); //store optimal angle 1 in array
            opt_ang[b][1] = acos((1.0-pow(m_rat,2.0*al)+pow(1.0-m_rat,2.0*al))/(2.0*pow(1.0-m_rat,al))); //store optimal angle 2 in array

            //check for neighbors that have this BP as their div point and add it to the queue.
            for(m=0;m<3;m++){
                k2 = adj[b][m];
                if(k2>=NUMSITES){
                    if(!BP_done[k2-NUMSITES]){
                        if(adj[k2-NUMSITES][BP_div_point[k2-NUMSITES][0]]==b+NUMSITES){
                            BPQ2.push(k2-NUMSITES);
                        }
                    }
                }
            }
        }
    }
}


bool check_convergence(double opt_ang[][2], int DIMENSION, int NUMSITES, int adj[][3], double EW[][3], double XX[], double al, double cost, double improv_thres){
    /*calculates the angles between BPs and terminals, checks for V/L-Branching and compares to angles calculated in opt_ang.*/
    int k,k1,m,l,breaker,d;
    int n0,n1,n2;
    double theta1,theta2,norm1,norm2,scprod,vec1,vec2,dist,t,dist_thres,ang_thres;
    double psi,phi,rho;

    n0=n1=n2=0;

    double tol = 1e-12;
    bool not_conv = false;

    for(k=0;k<NUMSITES-2;k++){
        k1=k+NUMSITES;
        //find source; "source" means the edge that has a different flow direction from the other two, so can also be sink. We map to "2sinks-1source"-case
        breaker=0;
        for(m=0;m<3;m++){
            for(l=0;l<3;l++){
                if(m != l){
                    if(EW[k][m]*EW[k][l] >= 0){n1=m;n2=l;breaker=1;break;}
                }
            }
            if(breaker==1){break;}
        }

        for(m=0;m<3;m++){if(m != n1 && m != n2){n0=m;break;}}
        n0 = adj[k][n0];
        n1 = adj[k][n1];
        n2 = adj[k][n2];


        norm1=norm2=scprod=0;
        for(d=0;d<DIMENSION;d++){
            vec1 = XX[n1*DIMENSION+d]-XX[k1*DIMENSION+d];
            vec2 = XX[k1*DIMENSION+d]-XX[n0*DIMENSION+d];
            norm1 += vec1*vec1;
            norm2 += vec2*vec2;
            scprod += vec1*vec2;
        }
        norm1 = sqrt(norm1);
        norm2 = sqrt(norm2);
        if(norm1<tol || norm2<tol){continue;}//ignore cases where angles cannot be calculated. These are hopefully already optimal after initial optimization
        theta1 = acos(scprod/(norm1*norm2));

        norm1=norm2=scprod=0;
        for(d=0;d<DIMENSION;d++){
            vec1 = XX[n2*DIMENSION+d]-XX[k1*DIMENSION+d];
            vec2 = XX[k1*DIMENSION+d]-XX[n0*DIMENSION+d];
            norm1 += vec1*vec1;
            norm2 += vec2*vec2;
            scprod += vec1*vec2;
        }
        norm1 = sqrt(norm1);
        norm2 = sqrt(norm2);
        if(norm1<tol || norm2<tol){continue;}//ignore cases where angles cannot be calculated. These are hopefully already optimal after initial optimization
        theta2 = acos(scprod/(norm1*norm2));

        dist_thres = (improv_thres*cost)/(pow(fabs(EW[k][m]),al)*(NUMSITES-2));
        ang_thres = dist_thres*cos(theta1+theta2);

        //check for V-Branching
        norm1=norm2=scprod=0;
        for(d=0;d<DIMENSION;d++){
            vec1 = XX[n1*DIMENSION+d]-XX[n0*DIMENSION+d];
            vec2 = XX[n2*DIMENSION+d]-XX[n0*DIMENSION+d];
            norm1 += vec1*vec1;
            norm2 += vec2*vec2;
            scprod += vec1*vec2;
        }
        norm1 = sqrt(norm1);
        norm2 = sqrt(norm2);
        psi = acos(scprod/(norm1*norm2));
        if(psi >= opt_ang[k][0]+opt_ang[k][1]){
            dist=0;
            for(d=0;d<DIMENSION;d++){t=XX[k1*DIMENSION+d]-XX[n0*DIMENSION+d];t=t*t;dist+=t;}
            dist = sqrt(dist);
            if(dist>dist_thres){
                for(d=0;d<DIMENSION;d++){XX[k1*DIMENSION+d]=XX[n0*DIMENSION+d]+0.9*dist_thres*(double)((rand() % (4*NUMSITES) + 4*NUMSITES)/(8*NUMSITES));}
                //cout << "V error" << endl;
                not_conv = true;
                continue;
            }
        }

        //check for L1-Branching
        norm1=norm2=scprod=0;
        for(d=0;d<DIMENSION;d++){
            vec1 = XX[n0*DIMENSION+d]-XX[n1*DIMENSION+d];
            vec2 = XX[n2*DIMENSION+d]-XX[n1*DIMENSION+d];
            norm1 += vec1*vec1;
            norm2 += vec2*vec2;
            scprod += vec1*vec2;
        }
        norm1 = sqrt(norm1);
        norm2 = sqrt(norm2);
        phi = acos(scprod/(norm1*norm2));
        if(phi >= M_PI-opt_ang[k][1]){
            dist=0;
            for(d=0;d<DIMENSION;d++){t=XX[k1*DIMENSION+d]-XX[n1*DIMENSION+d];t=t*t;dist+=t;}
            dist = sqrt(dist);
            if(dist>dist_thres){
                for(d=0;d<DIMENSION;d++){XX[k1*DIMENSION+d]=XX[n1*DIMENSION+d]+0.9*dist_thres*(double)((rand() % (4*NUMSITES) + 4*NUMSITES)/(8*NUMSITES));}
                //cout << "L1 error" << endl;
                not_conv = true;
                continue;
            }
        }

        //check for L2-Branching
        norm1=norm2=scprod=0;
        for(d=0;d<DIMENSION;d++){
            vec1 = XX[n0*DIMENSION+d]-XX[n2*DIMENSION+d];
            vec2 = XX[n1*DIMENSION+d]-XX[n2*DIMENSION+d];
            norm1 += vec1*vec1;
            norm2 += vec2*vec2;
            scprod += vec1*vec2;
        }
        norm1 = sqrt(norm1);
        norm2 = sqrt(norm2);
        rho = acos(scprod/(norm1*norm2));
        if(rho >= M_PI-opt_ang[k][0]){
            dist=0;
            for(d=0;d<DIMENSION;d++){t=XX[k1*DIMENSION+d]-XX[n2*DIMENSION+d];t=t*t;dist+=t;}
            dist = sqrt(dist);
            if(dist>dist_thres){
                for(d=0;d<DIMENSION;d++){XX[k1*DIMENSION+d]=XX[n2*DIMENSION+d]+0.9*dist_thres*(double)((rand() % (4*NUMSITES) + 4*NUMSITES)/(8*NUMSITES));}
                //cout << "L2 error" << endl;
                not_conv = true;
                continue;
            }
        }

        //calculate angle err
        if(fabs(opt_ang[k][0]-theta1)+fabs(opt_ang[k][1]-theta2) > 2*ang_thres){
            //cout << "angle error: " << opt_ang[k][0] << " " << opt_ang[k][1] << " " << theta1 << " " << theta2 << endl;
            not_conv=true;
        }   
    }
    return not_conv;
}




extern "C"
double iterations(int *iter, int DIMENSION, int NUMSITES, int adj[][3], double EW[][3], double demands[], double XX[], double al, double improv_thres = 1e-7, double beta = 1.0, bool CBOT = false){
    /*iteratively optimizes the BP configuration until improvement threshold is reached*/
    int count,c,i,m1,m2;
    double cost,cost_old,improv;
    bool not_conv;

    double EL[NUMSITES][3];
    double opt_ang[NUMSITES][2];

    calculate_EW(NUMSITES,EW,adj,demands,al);

    if(CBOT){
        for(m1=0;m1<NUMSITES;m1++){for(m2=0;m2<3;m2++){EW[m1][m2] = fabs(EW[m1][m2])*(1-fabs(EW[m1][m2]));}}
    }

    cost_old = length(DIMENSION, NUMSITES, adj, EW, EL, XX, al, beta);
    if(beta==1.0){
        calculate_opt_angle(opt_ang, DIMENSION, NUMSITES, adj, EW, demands, XX, al, improv_thres, cost_old); //calculates optimal angles from flows AND initializes BP positions
    }

    improv = 1.0;
    *iter = 0;
    do{
        (*iter)++;
        optimize(DIMENSION, NUMSITES, adj, EW, EL, XX, al, beta);
        cost=length(DIMENSION, NUMSITES, adj, EW, EL, XX, al, beta);
        improv = (cost_old - cost)/cost_old;
        cost_old =  cost;
    }while(improv>improv_thres);
    if((beta == 1.0)&&(!CBOT)){
        i = *iter;
        for(c=0;c<4;c++){
            not_conv = check_convergence(opt_ang, DIMENSION, NUMSITES, adj, EW, XX, al, cost, improv_thres);
            if(not_conv){
                for(count=0;count<i/4;count++){
                    (*iter)++;
                    optimize(DIMENSION, NUMSITES, adj, EW, EL, XX, al, beta);
                    cost=length(DIMENSION, NUMSITES, adj, EW, EL, XX, al, beta);
                }
            }
            else{break;}
        }
    }
    return cost;
}