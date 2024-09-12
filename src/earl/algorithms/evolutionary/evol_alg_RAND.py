import random
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize

from src.optimization.algs.evolutionary.SOProblem import SOProblem
from src.optimization.algs.evolutionary.evol_alg import EvolutionaryAlg
from src.optimization.algs.evolutionary.evol_alg_SO import EvolutionAlgSO


class EvolutionAlgRAND(EvolutionAlgSO):

    def __init__(self, env, bb_model, obj, params):
        self.div_budget = params['div_budget']
        super(EvolutionAlgRAND, self).__init__(env, bb_model, obj, params)

    def search(self, init_state, fact, target_action):

        div_res = []
        for i in range(self.div_budget):
            self.set_seed(random.randint(0, 100))
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

            if cfs.X is not None:
                div_res.append((cfs.X, cfs.F, cf_problem.rew_dict[tuple(cfs.X)]))

        return div_res

    def generate_problem(self, fact):
        n_objectives = len(self.obj.objectives)
        n_constraints = len(self.obj.constraints)
        return SOProblem(self.horizon, n_objectives, n_constraints, self.xl, self.xu, fact, self.obj)


