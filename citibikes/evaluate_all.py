from citibikes.citibikes_env import CitiBikes
from citibikes.evaluation import evaluate_explanations, transform_baseline_results
from citibikes.fact_generation import get_facts
from src.earl.methods.cf.raccer_advance import NSGARaccerAdvance
from src.earl.methods.cf.raccer_rewind import NSGARaccerRewind
from src.earl.methods.sf.sgrl_backward import SGRLRewind
from src.earl.methods.sf.sgrl_forward import SGRLAdvance
from src.earl.models.bb_models.ppo_model import PPOModel
from src.earl.utils.util import seed_everything


def main():
    seed_everything(0)
    env = CitiBikes()

    bb_model = PPOModel(env,
                        'citibikes/trained_models/citibikes',
                        arch=[512, 512],
                        training_timesteps=5e5,
                        lr=0.0005,
                        batch_size=512,
                        verbose=1)

    horizon = 5
    sl_facts, rl_facts = get_facts(env, bb_model, horizon=horizon, perc=0.1, n_states=100)
    sf_rl_eval_paths = ['sgrl_advance', 'sgrl_rewind']
    sf_sl_eval_paths = ['s_gen_1', 's_gen_3', 's_gen_5']
    eval_path = 'citibikes/results/'

    SGRL_Advance = SGRLAdvance(env, bb_model, horizon=5, n_gen=25, pop_size=24, xl=[0, 0, 0], xu=[4, 4, 9])
    SGRL_Rewind = SGRLRewind(env, bb_model, horizon=5, n_gen=25, pop_size=24, xl=[0, 0, 0], xu=[4, 4, 9])

    transform_baseline_results(sl_facts, [SGRL_Advance.obj, SGRL_Rewind.obj], sf_sl_eval_paths, eval_path)
    evaluate_explanations(env, 'citibikes/results/', sf_rl_eval_paths + sf_sl_eval_paths, N_TEST=100)

    cf_rl_eval_paths = ['raccer_advance', 'raccer_rewind']
    cf_sl_eval_paths = ['ganterfactual']

    RACCER_Advance = NSGARaccerAdvance(env, bb_model, horizon=horizon, n_gen=25, pop_size=24, xl=[0, 0, 0], xu=[4, 4, 9])
    RACCER_Rewind = NSGARaccerRewind(env, bb_model, horizon=horizon, n_gen=25, pop_size=24, xl=[0, 0, 0], xu=[4, 4, 9])

    transform_baseline_results(sl_facts, [RACCER_Advance.obj, RACCER_Rewind.obj], cf_sl_eval_paths, eval_path)
    evaluate_explanations(env, 'citibikes/results/', cf_rl_eval_paths + cf_sl_eval_paths, N_TEST=100)


if __name__ == '__main__':
    main()