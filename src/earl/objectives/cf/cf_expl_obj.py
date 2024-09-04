import copy

from src.earl.objectives.abstract_obj_expl import AbstractObjective



class CfExplObj(AbstractObjective):
    '''
    A set of objectives and constraints used for generating backward counterfactuals in ACTER algorithm
    The action proximity is defined for continuous actions
    '''
    def __init__(self, env, bb_model, n_sim=10, horizon=5):

        super(CfExplObj, self).__init__(env, bb_model, horizon=horizon, n_sim=n_sim)
        self.bb_model = bb_model
        self.env = env
        self.objectives = ['uncertainty', 'fidelity', 'reachability']
        self.constraints = ['validity']

        self.n_sim = n_sim

    def validity(self, outcome, obs):
        ''' Evaluates validity based on the outcome '''
        valid_outcome = outcome.cf_outcome(obs)
        # IMPORTANT: return 1 if the class has changed -- to be compatible with minimization used by NSGA
        return not valid_outcome

    def evaluate(self, fact, actions):
        return self._evaluate(fact, actions, allow_first_noop=True)

    def _evaluate(self, fact, actions, allow_first_noop=False):
        actions, first_action_index = self.process_actions(actions, allow_first_noop=allow_first_noop)
        first_state, first_env_state = self.get_first_state(fact, first_action_index)

        if len(actions) == 0:
            return {'uncertainty': 1,
                    'fidelity': 1,
                    'reachability': 1
                    }, {'validity': True}, []

        stochasticity, fidelity, _, num_cfs, cfs = self.calculate_stochastic_properties(fact,
                                                                                        actions,
                                                                                        self.bb_model,
                                                                                        first_state,
                                                                                        first_env_state)
        reachability = self.reachability(actions)
        for cf in cfs:
            cf[1].update({'reachability': reachability, 'uncertainty': stochasticity})

        objectives = {'uncertainty': stochasticity,
                      'fidelity': fidelity,
                      'reachability': reachability}

        constraints = {'validity': num_cfs == 0}

        return objectives, constraints, cfs


    def get_first_state(self, fact, first_action_index):
        return copy.copy(fact.states[first_action_index]), copy.deepcopy(fact.env_states[first_action_index])



