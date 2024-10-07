class SLFact:

    def __init__(self, obs, action, target_action):
        self.state = obs
        self.action = action
        self.target_action = target_action