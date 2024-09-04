from src.earl.outcomes.abstract_outcome import AbstractOutcome


class FailureOutcome(AbstractOutcome):

    def __init__(self, bb_model, target_action=None, true_action=None):
        super(FailureOutcome, self).__init__(bb_model, target_action, true_action)
        self.name = 'failure'

    def cf_outcome(self, env, state):
        return env.check_success() and not env.check_failure() # counterfactual where one specific action is required

    def explain_outcome(self, env, state=None):
        return env.check_failure()   # if failure explain this outcome