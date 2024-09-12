from src.earl.algorithms.hts.hts import HTSAlgorithm
from src.earl.methods.abstract_expl_alg import ExplAlgAbstract
from src.earl.models.util.counterfactual import CF
from src.earl.objectives.cf.pf_expl_obj import PfExplObj


class RACCERHTS(ExplAlgAbstract):

    def __init__(self, env, bb_model, params={}):
        self.obj = PfExplObj(env, bb_model, params)
        self.alg = HTSAlgorithm(env, bb_model, self.obj, params)

        self.objectives = ['fidelity', 'reachability', 'stochastic_validity']

    def get_best_cf(self, fact, target):
        ''' Returns all cfs found in the tree '''
        res = self.alg.search(init_state=fact.states[-1], fact=fact)
        cfs = []

        best_cf = None
        for cf in res:
            cf = CF(fact, cf[0], cf[1], cf[2], cf[3])
            cfs.append(cf)

            if best_cf is None or cf.value < best_cf.value:
                best_cf = cf

        if best_cf is None:
            return []

        return [best_cf]
