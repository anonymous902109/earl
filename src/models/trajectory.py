import copy


class Trajectory:

    def __init__(self, id, horizon):
        self.id = id
        self.states = []
        self.actions = []
        self.env_states = []

        self.forward_state = None
        self.backward_state = None
        self.forward_env_state = None
        self.backward_env_state = None

        self.horizon = horizon

    def append(self, state, action, env_unwrapped):
        # store only first env state to save memory
        if len(self.states) == 0:

            self.backward_env_state = copy.copy(env_unwrapped)
            self.backward_state = copy.copy(state)

        self.states.append(state)
        if action is not None:
            self.actions.append(action)
        self.env_states.append(env_unwrapped)

    def mark_outcome_state(self):
        self.forward_state = copy.copy(self.states[-1])
        self.forward_env_state = copy.copy(self.env_states[-1])  # append the last env state

        # this is actually the real factual state we're explaining
        self.fact_state = copy.copy(self.states[-1])
        self.factual_actions = copy.copy(self.actions)

        # limit to last horizon actions and states
        self.states = self.states[-(self.horizon+1):]
        self.actions = self.actions[-self.horizon:]
        self.env_states = self.env_states[-(self.horizon+1):]

        self.outcome_id = len(self.actions) - 1

    def num_actions(self):
        return len(self.actions)

    def set_outcome(self, outcome):
        self.outcome = outcome