import random
import time

import pandas as pd
from tqdm import tqdm

from citibikes.citibikes_env import CitiBikes
from citibikes.fact_generation import get_facts
from src.earl.methods.sf.sgen import SGEN
from src.earl.methods.sf.sgrl_backward import SGRLRewind
from src.earl.methods.sf.sgrl_forward import SGRLAdvance
from src.earl.models.bb_models.ppo_model import PPOModel
from src.earl.utils.util import seed_everything


def main():
    seed_everything(1)
    env = CitiBikes()

    bb_model = PPOModel(env,
                        'citibikes/trained_models/citibikes',
                        arch=[512, 512],
                        training_timesteps=5e5,
                        lr=0.0005,
                        batch_size=512,
                        verbose=1)

    params = [f'--columns={env.state_feature_names}',
              f'--categorical_features={env.categorical_features}',
              f'--continuous_features={env.continuous_features}']

    horizon = 5
    sl_facts, rl_facts = get_facts(env, bb_model, horizon=horizon, perc=0.1, n_states=100)

    # defining baseline models
    # s_gen_1 = SGEN(env, bb_model, params, diversity_size=1, pop_size=25, n_gen=24)
    # s_gen_3 = SGEN(env, bb_model, params, diversity_size=3, pop_size=25, n_gen=24)
    # s_gen_5 = SGEN(env, bb_model, params, diversity_size=5, pop_size=25, n_gen=24)

    SGRL_Advance = SGRLAdvance(env, bb_model, horizon=horizon, n_gen=25, pop_size=100, xl=[0, 0, 0], xu=[4, 4, 9])
    SGRL_Rewind = SGRLRewind(env, bb_model, horizon=horizon, n_gen=25, pop_size=100, xl=[0, 0, 0], xu=[4, 4, 9])

    rl_methods = [SGRL_Advance, SGRL_Rewind]
    rl_eval_paths = ['sgrl_advance', 'sgrl_rewind']

    # running RL-specific models
    for i, m in enumerate(rl_methods):
        record = []
        print('Running {}'.format(rl_eval_paths[i]))

        for f in tqdm(rl_facts):
            start = time.time()
            sfs = m.explain(f, target=f.target_action)
            end = time.time()
            if len(sfs):
                print('Generated {} sfs'.format(len(sfs)))
                for sf in sfs:
                    record.append((list(f.state), list(sf.cf), end-start))

        # recording results
        record_df = pd.DataFrame(record, columns=['fact', 'explanation', 'gen_time'])
        record_df.to_csv('citibikes/results/{}.csv'.format(rl_eval_paths[i]), index=False)

    # defining supervised learning methods
    sl_methods = [s_gen_1, s_gen_3, s_gen_5]
    sl_eval_paths = ['s_gen_1', 's_gen_3', 's_gen_5']

    # running supervised learning methods
    for i, m in enumerate(sl_methods):
        record = []
        print('Running {}'.format(sl_eval_paths[i]))

        for f in tqdm(sl_facts):
            start = time.time()
            sfs = m.explain(f)
            end = time.time()
            for sf in sfs:
                record.append((list(f.state), list(sf), end-start))

        # recording results
        record_df = pd.DataFrame(record, columns=['fact', 'explanation', 'gen_time'])
        record_df.to_csv('citibikes/results/{}.csv'.format(sl_eval_paths[i]), index=False)


if __name__ == '__main__':
    main()