from src.earl.algorithms.evolutionary.evol_alg import EvolutionaryAlg
from src.earl.methods.abstract_expl_alg import ExplAlgAbstract
from src.earl.methods.abstract_method import AbstractMethod
from src.earl.models.util.mc_transition_model import MonteCarloTransitionModel
from src.earl.objectives.sf.scf_expl_obj import ScfExplObj


class SGRLRewind(AbstractMethod):

    def __init__(self, env, bb_model,
                 horizon=5, n_sim=10, xu=10, xl=0, n_gen=10, pop_size=100):
        self.env = env
        self.bb_model = bb_model

        self.n_sim = n_sim
        self.xu = xu
        self.xl = xl
        self.n_gen = n_gen
        self.pop_size = pop_size
        self.horizon = horizon

        self.transition_model = MonteCarloTransitionModel(env, bb_model, n_sim=10)

        self.obj = ScfExplObj(env, bb_model, self.transition_model, self.n_sim)
        self.alg = EvolutionaryAlg(env, bb_model, self.obj, horizon=horizon, xl=self.xl, xu=self.xu, n_gen=self.n_gen, pop_size=self.pop_size)

    def explain(self, fact, target):
        fact.set_target_action(target)
        cfs = self.alg.search(init_state=fact, fact=fact, target_action=target, allow_noop=True)

        return cfs