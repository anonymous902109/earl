import argparse

from src.baselines.cf.ganterfactual import GANterfactual
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

    # TODO: SGEN has params like diversity size, num gens and population size -- passed through here probably
    s_gen_1 = SGEN(bb_model, task_name='citibikes', diversity_size=1)
    s_gen_3 = SGEN(bb_model, task_name='citibikes', diversity_size=3)
    s_gen_5 = SGEN(bb_model, task_name='citibikes', diversity_size=5)

    # TODO: add params to GANterfactual
    ganterfactual = GANterfactual(env,
                                  bb_model,
                                  num_features=38,
                                  nb_domains=5*5*10,
                                  domains=[(1, 2, 1), (1, 4, 1), (1, 3, 1)],
                                  dataset_size=1e3)

    sl_task.explain(s_gen_1, save_path='results/sgen1.csv')
    sl_task.explain(s_gen_3, save_path='results/sgen3.csv')
    sl_task.explain(s_gen_5, save_path='results/sgen5.csv')

    # --------- visualization/evaluation for SL methods -------------------
    # for i in sample_facts:
    #     fact_state = buffer[i].fact
    #     cf_state = buffer[i].cf
    #
    #     print('Fact')
    #     print('Cf')


if __name__ == '__main__':
    main()