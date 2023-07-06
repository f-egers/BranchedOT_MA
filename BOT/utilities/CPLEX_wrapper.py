from docplex.mp.advmodel import AdvModel
from docplex.mp.solution import SolveSolution
import numpy as np

def CPLEX_MILP(c,A_eq=None,b_eq=None,A_ineq=None,b_ineq=None,up=1,lo=0,integer=None,time_limit=600,warm_start=None,verbose=False):
    '''
    simple wrapper function to solve MILP using CPLEX solver. Model needs to be build on each call.
    '''

    with AdvModel() as m:

        m.parameters.threads = 6
        m.parameters.emphasis.mip = 0

        if integer is None:
            integer = np.zeros(len(c))

        num_variables = len(integer)

        if not hasattr(up,"__len__"):
            up=[up]*len(integer)
        if not hasattr(lo,"__len__"):
            lo=[lo]*len(integer)

        if verbose: print("building CPLEX model...",end="")
        variables = [m.binary_var() if integer[i] else m.continuous_var(lb=lo[i],ub=up[i]) for i in range(num_variables)]

        if A_eq is not None:
            #for a,cst in enumerate(A_eq):
            #    m.add_constraint(m.sum([cst[i]*variables[i] for i in range(num_variables)]) == b_eq[a])
            m.add_constraints(m.matrix_constraints(A_eq,variables,b_eq,"eq"))

        if A_ineq is not None:
            #for a,cst in enumerate(A_ineq):
            #    m.add_constraint(m.sum([cst[i]*variables[i] for i in range(num_variables)]) <= b_ineq[a])
            m.add_constraints(m.matrix_constraints(A_ineq,variables,b_ineq,"le"))

        m.minimize(m.sum([c[i]*variables[i] for i in range(num_variables)]))

        m.set_time_limit(time_limit)

        if warm_start is not None:
            if hasattr(warm_start,"__len__"):
                for ws in warm_start:
                    m.add_mip_start(SolveSolution(m,{var: ws[i] for i,var in enumerate(variables)}),write_level=1)
            else:
                init_sol = SolveSolution(m,{var: warm_start[i] for i,var in enumerate(variables)})
                m.add_mip_start(init_sol,write_level=1)
            # for ws,_ in m.iter_mip_starts():
            #     is_feasible = ws.is_feasible_solution(silent=False)
            #     print(f"{'feasible' if is_feasible else 'broken'} warm start with obj.: {ws.objective_value}")
            
        if verbose: print("done")

        solution = m.solve(log_output=verbose)

    return np.array(solution.get_value_list(variables)),solution.solve_details.gap



class MILP_Model():
    def __init__(self,c,A_eq=None,b_eq=None,A_ineq=None,b_ineq=None,up=1,lo=0,integer=None,time_limit=600,verbose=False):
        self.m = AdvModel()
        self.m.parameters.threads = 6
        self.m.parameters.emphasis.mip = 0

        if integer is None:
            self.integer = np.zeros(len(c))
        else:
            self.integer = integer

        self.num_variables = len(self.integer)

        if not hasattr(up,"__len__"):
            self.up=[up]*len(self.integer)
        else:
            self.up = up
        if not hasattr(lo,"__len__"):
            self.lo=[lo]*len(self.integer)
        else:
            self.lo = lo

        if verbose: print("building CPLEX model...",end="")
        self.variables = [self.m.binary_var() if self.integer[i] else self.m.continuous_var(lb=self.lo[i],ub=self.up[i]) for i in range(self.num_variables)]

        if A_eq is not None:
            #for a,cst in enumerate(A_eq):
            #    m.add_constraint(m.sum([cst[i]*variables[i] for i in range(num_variables)]) == b_eq[a])
            self.m.add_constraints(self.m.matrix_constraints(A_eq,self.variables,b_eq,"eq"))

        if A_ineq is not None:
            #for a,cst in enumerate(A_ineq):
            #    m.add_constraint(m.sum([cst[i]*variables[i] for i in range(num_variables)]) <= b_ineq[a])
            self.m.add_constraints(self.m.matrix_constraints(A_ineq,self.variables,b_ineq,"le"))

        self.m.minimize(self.m.sum([c[i]*self.variables[i] for i in range(self.num_variables)]))

        if verbose: print("done")

        self.m.set_time_limit(time_limit)
        self.verbose = verbose



    def solve(self,clean=True):

        solution = self.m.solve(log_output=self.verbose,clean_before_solve=clean)

        return np.array(solution.get_value_list(self.variables))


    def set_objective(self,c):

        self.m.remove_objective()

        non_zero_ind = np.nonzero(c)[0]

        self.m.minimize(self.m.sum([c[i]*self.variables[i] for i in non_zero_ind]))




