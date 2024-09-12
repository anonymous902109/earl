from src.earl.algorithms.evolutionary.evol_alg import EvolutionaryAlg
from src.earl.methods.abstract_expl_alg import ExplAlgAbstract
from src.earl.objectives.sf.spf_expl_obj import SpfExplObj


class SGRLAdvance(ExplAlgAbstract):

    def __init__(self, env, bb_model, transition_model, horizon=5,
                 n_sim=10, xu=0, xl=10, n_gen=10, pop_size=100):
        self.env = env

        self.n_sim = n_sim
        self.xu = xu
        self.xl = xl
        self.n_gen = n_gen
        self.pop_size = pop_size
        self.horizon = horizon

        self.obj = SpfExplObj(env, bb_model, transition_model, n_sim=n_sim)
        self.alg = EvolutionaryAlg(env, bb_model, self.obj, horizon=horizon,
                                   xu=xu, xl=xl, n_gen=n_gen, pop_size=pop_size)

    def get_best_cf(self, fact, target):

        cfs = self.alg.search(init_state=fact, fact=fact, target_action=target, allow_noop=True)

        return cfs