import argparse

from src.baselines.sf.sgen import SGEN
from src.envs.citibikes import CitiBikes

from src.models.dqn_model import DQNModel
from src.models.ppo_model import PPOModel
from src.outcomes.one_action_outcome import OneActionOutcome

from src.tasks.sl_task import SLTask
from src.utils.util import seed_everything


def main():
    # ----------- user-defined ------------
    seed_everything(0)
    env = CitiBikes()

    bb_model = PPOModel(env, 'trained_models/citibikes')

    outcome = OneActionOutcome(bb_model, target_action=[1, 3, 2])

    params = [f'--columns={env.state_feature_names}',
              f'--categorical_features={env.categorical_features}',
              f'--continuous_features={env.continuous_features}']


    # ------ done automatically -----------
    sl_task = SLTask(env, bb_model, outcome, params)

    s_gen_1 = SGEN(bb_model, task_name='citibikes', diversity_size=1)

    cfs = sl_task.explain(s_gen_1)

    # --------- visualization/evaluation for SL methods -------------------
    # for i in sample_facts:
    #     fact_state = buffer[i].fact
    #     cf_state = buffer[i].cf
    #
    #     print('Fact')
    #     print('Cf')


if __name__ == '__main__':
    main()