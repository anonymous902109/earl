from src.earl.algorithms.evolutionary.evol_alg import EvolutionaryAlg
from src.earl.methods.abstract_expl_alg import ExplAlgAbstract
from src.earl.methods.abstract_method import AbstractMethod
from src.earl.models.util.mc_transition_model import MonteCarloTransitionModel
from src.earl.objectives.sf.spf_expl_obj import SpfExplObj


class SGRLAdvance(AbstractMethod):

    def __init__(self, env, bb_model,
                 horizon=5, n_sim=10, xu=0, xl=10, n_gen=10, pop_size=100):
        self.env = env

        self.n_sim = n_sim
        self.xu = xu
        self.xl = xl
        self.n_gen = n_gen
        self.pop_size = pop_size
        self.horizon = horizon

        self.transition_model = MonteCarloTransitionModel(env, bb_model, n_sim=10)

        self.obj = SpfExplObj(env, bb_model, self.transition_model, n_sim=n_sim)
        self.alg = EvolutionaryAlg(env, bb_model, self.obj, horizon=horizon,
                                   xu=xu, xl=xl, n_gen=n_gen, pop_size=pop_size)

    def explain(self, fact, target):
        fact.set_target_action(target)
        sfs = self.alg.search(init_state=fact, fact=fact, target_action=target, allow_noop=True)

        return sfs