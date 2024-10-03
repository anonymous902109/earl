from src.earl.algorithms.evolutionary.evol_alg import EvolutionaryAlg
from src.earl.methods.abstract_expl_alg import ExplAlgAbstract
from src.earl.methods.abstract_method import AbstractMethod
from src.earl.objectives.cf.pf_expl_obj import PfExplObj


class NSGARaccerAdvance(AbstractMethod):

    def __init__(self, env, bb_model, horizon=5,
                 n_sim=10, xu=0, xl=10, n_gen=10, pop_size=100):

        self.obj = PfExplObj(env, bb_model, n_sim=n_sim, horizon=horizon)
        self.alg = EvolutionaryAlg(env, bb_model, self.obj,
                                   horizon=horizon, xu=xu, xl=xl, n_gen=n_gen, pop_size=pop_size)

        self.n_sim = n_sim
        self.xu = xu
        self.xl = xl
        self.n_gen = n_gen
        self.pop_size = pop_size
        self.horizon = horizon

    def explain(self, fact, target):
        cfs = self.alg.search(init_state=fact, fact=fact, target_action=target)

        return cfs
