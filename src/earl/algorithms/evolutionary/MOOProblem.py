from src.earl.algorithms.evolutionary.evol_problem import EvolutionaryProblem
from src.earl.models.util.counterfactual import CF


class MOOProblem(EvolutionaryProblem):

    def __init__(self, n_var, n_obj, n_ieq_constr, xl, xu, fact, obj):
        super().__init__(n_var, n_obj, n_ieq_constr, xl, xu, fact, obj)
        self.cfs = []

    def fitness_func(self, solution):
        solution = list(solution)

        if tuple(solution) in self.prev_solutions.keys():
            fitness, constraint = self.prev_solutions[tuple(solution)]

            return fitness, constraint

        output, constraint_dict, cfs = self.obj.evaluate(self.fact, solution)

        self.update_cfs(cfs, solution)

        fitness = [output[obj_name] for obj_name in self.obj.objectives]
        constraints = [int(constraint_dict[c_name]) for c_name in self.obj.constraints]

        self.prev_solutions[tuple(solution)] = (fitness, constraints)

        output.update(constraint_dict)
        self.rew_dict[tuple(solution)] = output

        return fitness, constraints

    def update_cfs(self, cfs, solution):
        for cf in cfs:
            not_exists_cf = not cf[0] in [prev_cf.cf for prev_cf in self.cfs]
            not_exists_recourse = not solution in [prev_cf.recourse for prev_cf in self.cfs]
            if not_exists_cf or not_exists_recourse:  # add new solution if it hasn't been added before
                if cf[0] != list(self.fact.forward_state):
                    new_cf = CF(self.fact, solution, cf[0], cf[1])
                    self.cfs.append(new_cf)
