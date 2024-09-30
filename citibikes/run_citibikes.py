from tqdm import tqdm

from src.earl.methods.cf.ganterfactual import GANterfactual
from src.earl.methods.cf.raccer_advance import NSGARaccerAdvance
from src.earl.methods.cf.raccer_rewind import NSGARaccerRewind
from src.earl.methods.sf.sgen import SGEN
from citibikes.citibikes_env import CitiBikes
from src.earl.methods.sf.sgrl_backward import SGRLRewind
from src.earl.methods.sf.sgrl_forward import SGRLAdvance

from src.earl.models.bb_models.ppo_model import PPOModel
from src.earl.models.facts.rl_fact import RLFact
from src.earl.models.facts.sl_fact import SLFact

from src.earl.models.util.mc_transition_model import MonteCarloTransitionModel

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

    n_ep = 100
    horizon = 5
    n_facts = 100

    sl_facts = []
    rl_facts = []
    importances = []
    domains = []

    for i in tqdm(range(n_ep)):
        done = False
        obs, _ = env.reset()
        prev_states = []
        actions = []

        while not done:
            action = bb_model.predict(obs)

            importance = bb_model.get_importance(obs)

            if (not len(importances)) or (importance > min(importances) and len(prev_states) >= horizon):
                if len(importances) > n_facts:
                    min_index = importances.index(min(importances))
                    del importances[min_index]
                    del sl_facts[min_index]
                    del rl_facts[min_index]

                sl_fact = SLFact(obs, action)
                rl_fact = RLFact(obs, action, prev_states, env_states=[], actions=actions, horizon=horizon)

                rl_facts.append(rl_fact)
                sl_facts.append(sl_fact)
                importances.append(importance)

            actions.append(action)
            prev_states.append(env.get_state())
            obs, rew, done, trunc, info = env.step(action)

    domains = list({tuple(bb_model.predict(f.state)) for f in sl_facts})

    s_gen_1 = SGEN(env, bb_model, diversity_size=1, params=params)
    s_gen_3 = SGEN(env, bb_model, diversity_size=3, params=params)
    s_gen_5 = SGEN(env, bb_model, diversity_size=5, params=params)

    ganterfactual = GANterfactual(env,
                                  bb_model,
                                  batch_size=64,
                                  num_features=38,
                                  domains=domains,
                                  dataset_size=5e5,
                                  dataset_path='citibikes/datasets/ganterfactual_data')

    sl_methods = [s_gen_1, s_gen_3, s_gen_5, ganterfactual]

    for m, i in enumerate(sl_methods):
        for f in sl_facts:
            cf = m.explain(f, target=(1, 0, 2))
            print(cf)

    SGRL_Rewind = SGRLRewind(env, bb_model, xl=[0, 0, 0], xu=[4, 4, 9])
    SGRL_Advance = SGRLAdvance(env, bb_model)
    RACCER_Rewind = NSGARaccerRewind(env, bb_model)
    RACCER_Advance = NSGARaccerAdvance(env, bb_model)

    rl_methods = [SGRL_Advance, SGRL_Rewind, RACCER_Advance, RACCER_Rewind]
    rl_eval_paths = ['sgrl_advance.csv', 'sgrl_rewind.csv', 'raccer_advance.csv', 'raccer_rewind.csv']

    for m, i in enumerate(rl_methods):
        for f in sl_facts:
            m.explain(f, target=(1, 0, 2), eval_path=rl_eval_paths[i])


if __name__ == '__main__':
    main()