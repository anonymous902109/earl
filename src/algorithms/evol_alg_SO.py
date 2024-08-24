import operator
import random
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling, FloatRandomSampling
from pymoo.optimize import minimize

from src.optimization.algs.evolutionary.SOProblem import SOProblem
from src.optimization.algs.evolutionary.evol_alg import EvolutionaryAlg


class EvolutionAlgSO(EvolutionaryAlg):

    def __init__(self, env, bb_model, obj, params):
        self.div_budget = params['div_budget']
        super(EvolutionAlgSO, self).__init__(env, bb_model, obj, params)

    def search(self, init_state, fact, target_action):
        self.fact = fact

        cf_problem = self.generate_problem(fact)

        init_population = [fact.actions] * self.pop_size
        init_population += np.random.randint(-1, 2, size=(self.pop_size, self.horizon))
        init_population = np.mod(init_population, self.xu + 1)
        init_population = np.array(init_population)

        algorithm = NSGA2(pop_size=self.pop_size,
                          crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                          mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                          sampling=init_population)

        cfs = minimize(cf_problem,
                       algorithm,
                       ('n_gen', self.n_gen),
                       seed=self.seed,
                       verbose=False)

        cfs = []
        vals = []
        for cf, v in cf_problem.prev_solutions.items():
            satisfied = sum(v[1]) == 0  # False means constraints are satisfied
            if satisfied:
                cfs.append(cf)
                vals.append(v[0])

        if self.div_budget > len(cfs):
            top_cf = cfs
            top_vals = vals
        else:
            ind = np.argpartition(vals, -self.div_budget)[:self.div_budget]
            top_cf = np.array([cfs[i] for i in ind])
            top_vals = np.array([vals[i] for i in ind])

        res = []
        for i, cf in enumerate(top_cf):
            res.append((cf, top_vals[i], cf_problem.rew_dict[tuple(cf)]))

        return res

    def generate_problem(self, fact):
        n_objectives = len(self.obj.objectives)
        n_constraints = len(self.obj.constraints)
        return SOProblem(self.horizon, n_objectives, n_constraints, self.xl, self.xu, fact, self.obj)


