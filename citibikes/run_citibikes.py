import time

import pandas as pd
from tqdm import tqdm

from citibikes.evaluation import evaluate_explanations
from citibikes.fact_generation import get_facts
from src.earl.methods.cf.ganterfactual import GANterfactual
from src.earl.methods.cf.raccer_advance import NSGARaccerAdvance
from src.earl.methods.cf.raccer_rewind import NSGARaccerRewind
from src.earl.methods.sf.sgen import SGEN
from citibikes.citibikes_env import CitiBikes
from src.earl.methods.sf.sgrl_backward import SGRLRewind
from src.earl.methods.sf.sgrl_forward import SGRLAdvance

from src.earl.models.bb_models.ppo_model import PPOModel

from src.earl.utils.util import seed_everything


def main():
    # ----------- user-defined ------------
    seed_everything(0)
    env = CitiBikes()

    bb_model = PPOModel(env, 'citibikes/trained_models/citibikes', arch=[512, 512], training_timesteps=5e5, lr=0.0005, batch_size=512, verbose=1)
    print(bb_model.evaluate())

    params = [f'--columns={env.state_feature_names}',
              f'--categorical_features={env.categorical_features}',
              f'--continuous_features={env.continuous_features}']

    horizon = 5
    sl_facts, rl_facts = get_facts(env, bb_model, horizon=horizon, perc=0.1, n_states=100)

    domains = list({tuple(bb_model.predict(f.state)) for f in sl_facts}.union({tuple(f.target_action) for f in sl_facts}))

    s_gen_1 = SGEN(env, bb_model, diversity_size=1, pop_size=24, n_gen=25, params=params)
    s_gen_3 = SGEN(env, bb_model, diversity_size=3, pop_size=24, n_gen=25, params=params)
    s_gen_5 = SGEN(env, bb_model, diversity_size=5, pop_size=24, n_gen=25, params=params)

    # ganterfactual = GANterfactual(env,
    #                               bb_model,
    #                               batch_size=64,
    #                               num_features=38,
    #                               domains=domains,
    #                               training_timesteps=4500,
    #                               dataset_size=5e5,
    #                               dataset_path='citibikes/datasets/ganterfactual_data')

    sl_methods = [s_gen_1, s_gen_3, s_gen_5]
    sl_eval_paths = ['s_gen_1', 's_gen_3', 's_gen_5', 'ganterfactual']

    for i, m in enumerate(sl_methods):
        record = []
        for f in tqdm(sl_facts):
            start = time.time()
            cfs = m.explain(f, target=tuple(f.target_action))
            end = time.time()
            for cf in cfs:
                record.append((list(f.state), list(cf), end-start))

        record_df = pd.DataFrame(record, columns=['fact', 'explanation', 'gen_time'])
        record_df.to_csv('citibikes/results/{}.csv'.format(sl_eval_paths[i]), index=False)

    # evaluate_explanations(env, 'citibikes/results/', sl_eval_paths, N_TEST=10)

    SGRL_Advance = SGRLAdvance(env, bb_model, horizon=horizon, n_gen=25, pop_size=24, xl=[0, 0, 0], xu=[4, 4, 9])
    SGRL_Rewind = SGRLRewind(env, bb_model, horizon=horizon, n_gen=25, pop_size=24, xl=[0, 0, 0], xu=[4, 4, 9])
    RACCER_Advance = NSGARaccerAdvance(env, bb_model, horizon=horizon, n_gen=25, pop_size=24, xl=[0, 0, 0], xu=[4, 4, 9])
    RACCER_Rewind = NSGARaccerRewind(env, bb_model, horizon=horizon, n_gen=25, pop_size=24, xl=[0, 0, 0], xu=[4, 4, 9])

    rl_methods = [SGRL_Advance, SGRL_Rewind, RACCER_Advance, RACCER_Rewind]
    rl_eval_paths = ['sgrl_advance', 'sgrl_rewind', 'raccer_advance', 'raccer_rewind']

    for i, m in enumerate(rl_methods):
        record = []
        for f in rl_facts:
            start = time.time()
            cfs = m.explain(f, target=f.target_action)
            end = time.time()
            if len(cfs):
                for cf in cfs:
                    record.append((list(f.state), list(cf), end-start))

        record_df = pd.DataFrame(record, columns=['fact', 'explanation', 'gen_time'])
        record_df.to_csv('citibikes/results/{}.csv'.format(rl_eval_paths[i]), index=False)


if __name__ == '__main__':
    main()