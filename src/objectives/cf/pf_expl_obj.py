import copy

from src.objectives.cf.cf_expl_obj import CfExplObj


class PfExplObj(CfExplObj):
    '''
    A set of objectives and constraints used for generating backward counterfactuals in ACTER algorithm
    The action proximity is defined for continuous actions
    '''
    def __init__(self, env, bb_model, params):

        super(PfExplObj, self).__init__(env, bb_model, params)
        self.bb_model = bb_model
        self.env = env
        self.objectives = ['uncertainty', 'reachability', 'fidelity']
        self.constraints = ['validity']

        self.n_sim = params['n_sim']

    def evaluate(self, fact, actions):
        return self._evaluate(fact, actions, allow_first_noop=False)

    def get_first_state(self, fact, first_action_index):
        return copy.deepcopy(fact.forward_state), copy.deepcopy(fact.forward_env_state)



