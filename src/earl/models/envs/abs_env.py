import gymnasium as gym


class AbstractEnv(gym.Env):
    ''' Abstract class for defining an environment '''

    def __init__(self):

        '''
        Abstract class specifying the information about the environment.
        '''

    def step(self, action):
        return None

    def close(self):
        pass

    def render(self):
        pass

    def reset(self, seed):
        return None

    def get_actions(self, x):
        ''' Returns a list of actions available in state x'''
        return []

    def set_stochastic_state(self, state, env_state):
        ''' Changes the environment's current state to x while leaving the stochastic processes unchanged '''
        pass

    def set_nonstoch_state(self, state, env_state):
        ''' Changes the environment's current state to x and the state of the environment to env_state.
        This way the full stochastic state is copied '''
        pass

    def check_done(self, x):
        ''' Returns a boolean indicating if x is a terminal state in the environment'''
        return False

    def equal_states(self, x1, x2):
        ''' Returns a boolean indicating if x1 and x2 are the same state'''
        return False

