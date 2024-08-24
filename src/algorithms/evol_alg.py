import numpy as np
from paretoset import paretoset

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling, FloatRandomSampling
from pymoo.optimize import minimize

from src.algorithms.MOOProblem import MOOProblem


class EvolutionaryAlg:

    def __init__(self, env, bb_model, obj, params):
        self.env = env
        self.bb_model = bb_model
        self.obj = obj
        self.params = params

        self.horizon = params['horizon']
        self.xu = params['xu']
        self.xl = params['xl']

        self.n_gen = params['n_gen']
        self.pop_size = params['pop_size']

        self.rew_dict = {}
        self.seed = self.set_seed()

    def set_seed(self, seed=1):
        self.seed = seed

    def search(self, init_state, fact, target_action, allow_noop=False):
        self.fact = fact

        cf_problem = self.generate_problem(fact, allow_noop)

        # init_population = [fact.actions] * self.pop_size
        # init_population += np.random.randint(-1, 2, size=(self.pop_size, self.horizon))
        # init_population = np.mod(init_population, self.xu + 1)
        # init_population = np.array(init_population)

        algorithm = NSGA2(pop_size=self.pop_size,
                          sampling=IntegerRandomSampling(),  # TODO: works only for discrete actions maybe need to change
                          crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                          mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()))

        minimize(cf_problem,
                 algorithm,
                 ('n_gen', self.n_gen),
                 seed=self.seed,
                 verbose=0)

        cfs = cf_problem.cfs

        if len(cfs) == 0:
            return []

        best_cfs = self.get_pareto_cfs(cfs)

        for cf in best_cfs:
            cf.reward_dict.update({const_name: False for const_name in self.obj.constraints})  # update cfs with constraints which are all satisfied

        return best_cfs

    def generate_problem(self, fact, allow_noop=False):
        n_objectives = len(self.obj.objectives)
        n_constraints = len(self.obj.constraints)

        return MOOProblem(self.horizon, n_objectives, n_constraints, self.xl, self.xu, fact, self.obj)

    def get_pareto_cfs(self, cfs):
        cost_array = []
        for cf in cfs:
            cost_array.append(list(cf.reward_dict.values()))

        cost_array = np.array(cost_array)

        is_efficient = paretoset(cost_array, sense=["min"] * cost_array.shape[1])

        best_cfs = [cfs[i] for i in range(len(cfs)) if is_efficient[i]]

        return best_cfs
