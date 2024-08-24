import json
import os
import logging
from datetime import datetime

import numpy as np

from src.baselines.cf.ganterfactual import GANterfactual
from src.baselines.sf.sgen import SGEN
from src.envs.env_factory import EnvFactory
from src.evaluation.evaluating_expl import evaluate_explanations
from src.evaluation.transformation import append_feature_metrics
from src.methods.cf.raccer_hts import RACCERHTS
from src.methods.cf.raccer_rewind import NSGARaccerRewind
from src.methods.cf.raccer_advance import NSGARaccerAdvance
from src.methods.sf.sgrl_backward import SGRL_Rewind
from src.methods.sf.sgrl_forward import SGRL_Advance
from src.models.datasets.dataset_factory import BaselineDatasetFactory
from src.models.dqn_model import DQNModel
from src.models.mc_transition_model import MonteCarloTransitionModel
from src.outcomes.one_action_outcome import OneActionOutcome
from src.utils.util import seed_everything, generate_facts_with_action_outcome


def explain(task_name, scenario_name, scenario, train, eval, sf, cf):
    seed_everything(1)

    # number of evaluation instances per outcome
    N_TEST = 100
    MC_N_SIM = 10000

    model_path = 'trained_models/{}/{}'.format(task_name, scenario_name)
    param_file = 'params/{}.json'.format(task_name)
    facts_path = 'facts/explain/{}/{}/'.format(task_name, scenario_name)
    facts_baseline_path = 'facts/explain/{}/{}/'.format(task_name, scenario_name)
    eval_path = 'results/explain'
    transition_model_path = 'datasets/{}/{}/transition_model.obj'.format(task_name, scenario_name)

    # load parameters
    with open(param_file, 'r') as f:
        params = json.load(f)

    # define environment
    env = EnvFactory.get_env(task_name, params["env_{}".format(scenario['env'])])

    # load black-box model
    bb_model = DQNModel(env, model_path, params["bb_model_{}".format(scenario['bb_model'])])
    # bb_model.evaluate()

    # define outcomes
    actions = np.arange(0, env.action_space.n)
    outcomes = [OneActionOutcome(bb_model, target_action=a) for a in actions]

    # load or generate facts
    facts = []
    fact_ids = [] # ids of facts that are going to be explained
    for o in outcomes:
        fact_with_outcome = generate_facts_with_action_outcome(env, bb_model, facts_path, o, params['horizon'])
        facts.append(fact_with_outcome)

        # select N_TEST trajectories at random
        sample_facts = np.random.choice(range(len(fact_with_outcome)), N_TEST, replace=False) if (len(fact_with_outcome) >= N_TEST) \
            else np.arange(0, len(fact_with_outcome)).tolist()

        logging.info('Generated {} facts for outcome {}'.format(len(sample_facts), o.name))
        fact_ids.append(sample_facts)

    # transform facts into a baseline datasets
    baselines_datasets = []
    for o in outcomes:
        bsd = BaselineDatasetFactory.get_dataset(task_name, env, bb_model, facts, o.target_action, o.name, facts_baseline_path)
        baselines_datasets.append(bsd)

    # define baselines
    s_gen_1 = SGEN(bb_model, task_name, diversity_size=1, params=params["gen_alg"])
    s_gen_3 = SGEN(bb_model, task_name, diversity_size=3, params=params["gen_alg"])
    s_gen_5 = SGEN(bb_model, task_name, diversity_size=5, params=params["gen_alg"])
    ganterfactual = GANterfactual(env, bb_model, params["ganterfactual"], task_name, scenario_name)

    # define transition model needed for SGRL
    transition_model = MonteCarloTransitionModel(env, bb_model, transition_model_path, n_sim=MC_N_SIM)

    # define our methods
    SGRL_Rewind = SGRL_Rewind(env, bb_model, params["gen_alg"], transition_model)
    SGRL_Advance = SGRL_Advance(env, bb_model, params["gen_alg"], transition_model)
    RACCER_HTS = RACCERHTS(env, bb_model, params["hts"], transition_model)
    RACCER_Rewind = NSGARaccerRewind(env, bb_model, params["gen_alg"], transition_model)
    RACCER_Advance = NSGARaccerAdvance(env, bb_model, params["gen_alg"], transition_model)

    methods = []
    method_names = []
    baselines = []
    baseline_names = []
    if sf:
        methods += [SGRL_Rewind, SGRL_Advance]
        baselines += [s_gen_1, s_gen_3, s_gen_5]
        method_names += ['SGRL_Rewind', 'SGRL_Advance']
        baseline_names += ['S_GEN_1', 'S_GEN_3', 'S_GEN_5']
    if cf:
        methods += [RACCER_HTS, RACCER_Rewind, RACCER_Advance]
        baselines += [ganterfactual]
        method_names += ['RACCER_HTS', 'RACCER_Rewind', 'RACCER_Advance']
        baseline_names += ['GANterfactual']

    if train:
        for outcome_id, outcome in enumerate(outcomes):
            # Running baselines
            logging.info('Running outcome = {}'.format(outcome.name))
            for i, baseline in enumerate(baselines):
                logging.info('Running {}'.format(baseline_names[i]))
                path = os.path.join(eval_path, task_name, scenario_name, baseline_names[i], '{}.csv'.format(outcome.name))
                baseline.generate_explanation(baselines_datasets[outcome_id], fact_ids[outcome_id], outcome, path)

            # Running methods
            for i, method in enumerate(methods):
                logging.info('Running {}'.format(method_names[i]))
                path = os.path.join(eval_path, task_name, scenario_name, method_names[i], '{}.csv'.format(outcome.name))
                method.generate_explanation(facts[outcome_id], fact_ids[outcome_id], outcome.target_action, path)

    # evaluate
    if evaluate:
        # if sf:
        #     transform_baseline_results(env, facts, [SGRL_Rewind.obj, SGRL_Advance.obj], params, ['S_GEN'], eval_path, scenario_name)
        # if explain:
        #     transform_baseline_results(env, facts, [RACCER_Rewind.obj, RACCER_Advance.obj], params, ['GANterfactual'], eval_path, scenario_name)
        #
        append_feature_metrics(baseline_names + method_names, params, eval_path, scenario_name)

        # Evaluate semifactuals
        if sf:
            evaluate_explanations(env, params, eval_path, scenario_name, ['S_GEN', 'SGRL_Advance', 'SGRL_Rewind'], N_TEST)
        # Evaluate counterfactuals
        if cf:
            evaluate_explanations(env, params, eval_path, scenario_name,['GANterfactual', 'RACCER_HTS', 'RACCER_Advance', 'RACCER_Rewind'], N_TEST)


if __name__ == '__main__':
    logging.basicConfig(filename='logs/{}.log'.format(datetime.now().strftime('%m-%d %H-%M')),
                        level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    with open('params/scenarios.json', 'r') as f:
        scenarios = json.load(f)

    train = True
    evaluate = False
    sf = True
    cf = True

    if sf:
        sf_scenarios = scenarios['sf']
    if cf:
        cf_scenarios = scenarios['cf']

    tasks = ['gridworld', 'farm', 'frozen_lake']

    for t in tasks:
        logging.info('Running task: {}'.format(t))
        if sf:
            for scenario_name, scenario in sf_scenarios[t].items():
                logging.info('Scenario {}: {}'.format(scenario_name, scenario))
                explain(t, scenario_name, scenario, train, evaluate, sf, False)
        if cf:
            for scenario_name, scenario in cf_scenarios[t].items():
                logging.info('Scenario {}: {}'.format(scenario_name, scenario))
                explain(t, scenario_name, scenario, train, evaluate, False, cf)


    tasks = ['highway']
    # For now run highway just for cfs
    for t in tasks:
        logging.info('Running task: {}'.format(t))
        if cf:
            for scenario_name, scenario in cf_scenarios[t].items():
                logging.info('Scenario {}: {}'.format(scenario_name, scenario))
                explain(t, scenario_name, scenario, train, evaluate, False, cf)