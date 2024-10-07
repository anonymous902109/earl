import copy


class RLFact:

    def __init__(self, obs, action, prev_states, env_states, actions, horizon, target_action):
        self.state = obs
        self.action = action

        self.prev_states = prev_states

        self.env_states = env_states
        self.actions = actions
        self.horizon = horizon

        self.target_action = target_action

        self.forward_state = copy.copy(self.state)
        self.forward_env_state = None

    def set_target_action(self, target_action):
        self.target_action = target_action

    def get_state(self, index):
        return self.prev_states[index]

    def get_env_state(self, index):
        return None