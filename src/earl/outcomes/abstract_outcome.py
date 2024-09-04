class AbstractOutcome:

    def __init__(self, bb_model, target_action=None):
        self.target_action = target_action

        self.bb_model = bb_model

    def cf_outcome(self, env, state):
        return True

    def explain_outcome(self, env, state=None):
        return None