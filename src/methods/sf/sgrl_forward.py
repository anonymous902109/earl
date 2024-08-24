from src.algorithms.evol_alg import EvolutionaryAlg
from src.methods.abstract_expl_alg import ExplAlgAbstract
from src.objectives.sf.spf_expl_obj import SpfExplObj


class SGRL_Advance(ExplAlgAbstract):

    def __init__(self, env, bb_model, params, transition_model):
        self.env = env
        self.obj = SpfExplObj(env, bb_model, params, transition_model)
        self.alg = EvolutionaryAlg(env, bb_model, self.obj, params)

    def get_best_cf(self, fact, target):

        cfs = self.alg.search(init_state=fact, fact=fact, target_action=target, allow_noop=True)

        return cfs