import copy
from src.earl.objectives.abstract_obj_expl import AbstractObjective


class SpfExplObj(AbstractObjective):
    '''
    A set of objectives and constraints used for generating backward counterfactuals in ACTER algorithm
    The action proximity is defined for continuous actions
    '''
    def __init__(self, env, bb_model, transition_model, n_sim):

        super(SpfExplObj, self).__init__(env, bb_model, transition_model, n_sim)
        self.bb_model = bb_model
        self.env = env
        self.transition_model = transition_model
        self.objectives = ['uncertainty', 'fidelity', 'sparsity', 'exceptionality']  # TODO: there is probably a better name for this
        self.constraints = ['validity']  # validity essentially

        self.n_sim = n_sim

        self.NoOp = -1

    def evaluate(self, fact, actions):
        # process actions to remove NoOp actions
        actions, first_action_index = self.process_actions(actions, allow_first_noop=False)
        first_state, first_env_state = self.get_first_state(fact, first_action_index)

        # if the action list is empty or invalid because NoOp is in the middle of the sequence
        if len(actions) == 0:
            return {'uncertainty': 1,
                    'fidelity': 1,
                    'sparsity': 1,
                    'exceptionality': 1
                    }, {'validity': True}, []

        # evaluate properties
        stochasticity, fidelity, exceptionality, num_cfs, cfs = self.calculate_stochastic_properties(fact, actions, self.bb_model, first_state, first_env_state)
        reachability = self.reachability(actions)

        for cf in cfs:
            cf[1].update({'sparsity': reachability, 'uncertainty': stochasticity})

        objectives = {'uncertainty': stochasticity,
                      'fidelity': fidelity,
                      'sparsity': reachability,
                      'exceptionality': exceptionality}

        constraints = {'validity': num_cfs == 0}

        return objectives, constraints, cfs

    def get_first_state(self, fact, first_action_index):
        return copy.deepcopy(fact.forward_state), copy.deepcopy(fact.forward_env_state)

