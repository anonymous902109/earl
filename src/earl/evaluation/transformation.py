import ast
import copy
import logging
import os
import random

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem
from tqdm import tqdm

from src.earl.outcomes.exact_state_outcome import ExactStateOutcome


def transform_baseline_results(env, facts, objs, params, baseline_names, eval_path, scenario):
    ''' Finds the shortest path between the original instance and the counterfactual '''
     # for each baseline method
    for m_i, baseline_n in enumerate(baseline_names):

        baseline_path = os.path.join(eval_path, baseline_n)
        logging.info('Transforming results for baseline = {}'.format(baseline_n))

        # for each outcome file
        df = pd.read_csv(baseline_path, header=0)
        data = []

        for i, row in tqdm(df.iterrows()):
            fact_id = row['fact id']
            f = facts[fact_id]
            cf = row['explanation']

            solutions = []
            for o in objs:
                solutions += find_recourse(f, cf, o, params)

            # no cfs found in the neighborhood
            if len(solutions) == 0:
                data = append_data(data,
                                   fact_id,
                                   list(f.forward_state),
                                   cf,
                                   None,
                                   [1 for i in objs[0].objectives + objs[0].constraints],
                                   row['gen time'],
                                   0)
            else:
                 # select one at random if multiple ways to obtain the solution are present
                random_idx = random.choice(np.arange(0, len(solutions)))
                random_res = solutions[random_idx]

                # write down results
                data = append_data(data,
                                   fact_id,
                                   list(f.forward_state),
                                   cf,
                                   random_res.X,
                                   random_res.F + random_res.G,
                                   row['gen time'], 1)

        columns = ['fact id',
                   'act',
                   'Explanation',
                   'Recourse'] + objs[0].objectives + objs[0].constraints + ['gen_time', 'Found']

        df = pd.DataFrame(data, columns=columns)

        df.to_csv(baseline_path, index=False)

def find_recourse(f, cf, obj, params):
    # search for counterfactuals using these methods
    outcome = ExactStateOutcome(cf)

    # objectives are the same ones used by our approaches to find sf/explain
    objs = [
        lambda x: obj.evaluate(f, x)[0].values()
    ]

    # only constraint is to end up in the counterfactual/semifactual state
    constr = [
        lambda x: 1 - outcome.equal_states(x)  # 0 = satisfied constraint
    ]

    problem = FunctionalProblem(params['gen_alg']['horizon'],
                                objs,
                                constr_ieq=constr,
                                xl=params['gen_alg']['xl'],
                                xu=params['gen_alg']['xu']
                                )

    algorithm = NSGA2(pop_size=25,
                      sampling=IntegerRandomSampling(),
                      crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                      mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                      )

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 10),
                   seed=1,
                   verbose=False)

    if res.X is not None:
        return res

    return []


def append_data(data, fact_id, fact, expl, recourse, objs, gen_time, found):
    data.append((fact_id,
                 list(fact),
                 expl,
                 np.nan if recourse is None else recourse,
                 *objs,
                 gen_time,
                 found))

    return data
