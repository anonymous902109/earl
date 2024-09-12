from tqdm import tqdm

from src.earl.methods.cf.ganterfactual import GANterfactual
from src.earl.methods.cf.raccer_advance import NSGARaccerAdvance
from src.earl.methods.cf.raccer_rewind import NSGARaccerRewind
from src.earl.methods.sf.sgen import SGEN
from citibikes.citibikes_env import CitiBikes
from src.earl.methods.sf.sgrl_backward import SGRLRewind
from src.earl.methods.sf.sgrl_forward import SGRLAdvance

from src.earl.models.bb_models.ppo_model import PPOModel
from src.earl.models.facts.sl_fact import SLFact
from src.earl.models.facts.rl_fact import RLFact
from src.earl.models.util.mc_transition_model import MonteCarloTransitionModel

from src.earl.utils.util import seed_everything


def main():
    # ----------- user-defined ------------
    seed_everything(0)
    env = CitiBikes()

    bb_model = PPOModel(env, 'citibikes/trained_models/citibikes', arch=[512, 512], verbose=1)
    print(bb_model.evaluate())

    params = [f'--columns={env.state_feature_names}',
              f'--categorical_features={env.categorical_features}',
              f'--continuous_features={env.continuous_features}']

    n_ep = 1
    horizon = 5
    n_facts = 1

    sl_facts = []
    rl_facts = []
    importances = []

    target = (1, 0, 2)
    for i in tqdm(range(n_ep)):
        done = False
        obs, _ = env.reset()
        prev_states = []
        actions = []

        while not done:
            action = bb_model.predict(obs)

            q_vals = [bb_model.get_action_prob(obs, a) for a in env.get_actions()]
            importance = max(q_vals) - min(q_vals)

            if (not len(importances)) or (importance > min(importances) and len(prev_states) >= horizon):
                if len(importances) > n_facts:
                    min_index = importances.index(min(importances))
                    del importances[min_index]
                    del sl_facts[min_index]
                    del rl_facts[min_index]

                sl_fact = SLFact(obs, action, target)
                rl_fact = RLFact(obs, action, target, prev_states, env_states=[], actions=actions, horizon=horizon)

                rl_facts.append(rl_fact)
                sl_facts.append(sl_fact)
                importances.append(importance)

            actions.append(action)
            prev_states.append(env.get_state())
            obs, rew, done, trunc, info = env.step(action)

    # s_gen_1 = SGEN(env, bb_model, diversity_size=1, params=params)
    # s_gen_3 = SGEN(env, bb_model, diversity_size=3, params=params)
    # s_gen_5 = SGEN(env, bb_model, diversity_size=5, params=params)
    #
    # ganterfactual = GANterfactual(env,
    #                               bb_model,
    #                               batch_size=64,
    #                               num_features=38,
    #                               domains=[(1, 2, 2), (1, 0, 2)],
    #                               dataset_size=5e5)

    transition_model = MonteCarloTransitionModel(env, bb_model, n_sim=10)

    SGRL_Rewind = SGRLRewind(env, bb_model, transition_model, xl=[0, 0, 0], xu=[4, 4, 9])
    SGRL_Advance = SGRLAdvance(env, bb_model, transition_model)
    RACCER_Rewind = NSGARaccerRewind(env, bb_model)
    RACCER_Advance = NSGARaccerAdvance(env, bb_model)

    SGRL_Rewind.explain(rl_facts[0], target=(1,2,0))


if __name__ == '__main__':
    main()