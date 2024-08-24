
from src.algorithms.evol_alg import EvolutionaryAlg
from src.methods.abstract_expl_alg import ExplAlgAbstract
from src.objectives.cf.cf_expl_obj import CfExplObj

class NSGARaccerRewind(ExplAlgAbstract):

    def __init__(self, env, bb_model, params={}):
        self.obj = CfExplObj(env, bb_model, params)
        self.alg = EvolutionaryAlg(env, bb_model, self.obj, params)

    def get_best_cf(self, fact, target):
        cfs = self.alg.search(init_state=fact, fact=fact, target_action=target)

        return cfs
