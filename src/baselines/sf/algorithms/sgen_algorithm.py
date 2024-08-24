import time
from copy import deepcopy
from random import gauss


import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

LAMBDA1 = 30  # robustness e-neighborhood
LAMBDA2 = 10  # robustness instance
GAMMA = 1  # diversity
MAX_MC = 100
CONT_PERTURB_STD = 0.05 # perturb continuous features by 5% STD
MUTATION_RATE = 0.05
ELITIST = 4  # how many of the "best" to save


class SGENAlg:

    def __init__(self, bb_model, diversity_size, population_size):
        self.bb_model = bb_model
        self.diversity_size = diversity_size
        self.population_size = population_size

    def generate_sfs(self, dataset, df, target_action, test_ids):
        # splitting datasets into train and test -- test part contains facts for which semifactuals will be generated
        training = np.ones(df.shape[0])
        training[test_ids] = 0
        df['training'] = training

        # Split categorical and cont features because categorical ones will be encoded
        continuous_features = df[dataset.continuous_feature_names]
        categorical_features = df[dataset.categorical_feature_names]

        # Encode categorical features
        enc = OneHotEncoder().fit(categorical_features)
        categorical_features_enc = enc.transform(categorical_features).toarray()

        # NB: Continuous features are first
        data = np.concatenate((continuous_features.values, categorical_features_enc), axis=1)

        X_train = data[(df.training == 1).values]
        y_train = df[df.training == 1].Outcome.values
        X_test = data[(df.training == 0).values]

        # load data about actionability of features
        action_meta = dataset.actionability_constraints()
        cat_idxs = self.generate_cat_idxs(continuous_features, enc)
        actionable_idxs = self.get_actionable_feature_idxs(continuous_features, categorical_features, action_meta)

        # Necessary variables
        MAX_GENERATIONS = 25  # TODO: change to parameter
        reach_knn = KNeighborsClassifier(p=2).fit(X_train, y_train)

        # result holders
        sf_data = []
        found_sfs = []
        gen_times = []

        # generate sfs
        for test_idx, x in tqdm(enumerate(X_test)):
            start = time.time()
            # reorder categorical and continuous features to pass to bbmodel
            x_decoded = self.reorder_features(x, dataset, enc, len(continuous_features.columns))

            # probabilities of predicting different classes
            probs = self.bb_model.predict_proba(x_decoded)

            # sanity check -- if semifact outcome (why not Y) is already the most likely
            if probs[dataset.target_action] == max(probs):
                continue

            # this while loop exists so that the initial population has at least one semifactual
            avg_preds = 0.0
            counter_xxx = 0
            while avg_preds < 0.3:
                counter_xxx += 1
                population = self.init_population(x, X_train, continuous_features, categorical_features, cat_idxs, action_meta,
                                                  replace=True)
                decoded_pop = []
                for i, solution in enumerate(population):
                    decoded_solution = []
                    for j, x in enumerate(solution):
                        decoded_item = self.reorder_features(x, dataset, enc, len(continuous_features.columns))
                        decoded_solution.append(decoded_item)

                    decoded_pop.append(np.stack(decoded_solution).squeeze())

                decoded_pop = np.stack(decoded_pop)

                avg_preds = (self.bb_model.predict_multiple(decoded_pop.reshape(-1, *dataset.state_shape)) != dataset.target_action).mean()
                if counter_xxx == 100:
                    break

            if counter_xxx == 100:
                continue

            # Start GA
            for generation in range(MAX_GENERATIONS):
                # Evaluate fitness (meta = reachability, gain, robustness, diversity)
                fitness_scores, meta_fitness = self.fitness(x, population, cat_idxs,
                                                            actionable_idxs, self.bb_model, action_meta,
                                                            continuous_features, categorical_features,
                                                            enc, dataset, reach_knn)

                # Selection
                population, elites = self.natural_selection(population, fitness_scores)

                # Crossover
                population = self.crossover(population, continuous_features, cat_idxs, actionable_idxs)

                # Mutate
                population = self.mutation(population, continuous_features, categorical_features, cat_idxs, action_meta, actionable_idxs, x)

                # Carry over elite solutions
                population = np.concatenate((population, elites), axis=0)

                # Evaluate fitness (meta = reachability, gain, robustness, diversity)
                fitness_scores, meta_fitness = self.fitness(x, population, cat_idxs,
                                                            actionable_idxs, self.bb_model, action_meta,
                                                            continuous_features, categorical_features,
                                                            enc, dataset, reach_knn)

            result = population[np.argmax(fitness_scores)]

            if sum(fitness_scores * (meta_fitness.T[-2] == LAMBDA2)) > 0:
                for d in result:
                    end = time.time()
                    gen_times.append(end - start)
                    sf_data.append(d.tolist())
                    found_sfs.append([test_ids[test_idx], True])

            else:
                result, replaced_these_idxs = self.force_sf(result, x, enc, dataset)
                for idx, d in enumerate(result):
                    decoded_d = self.reorder_features(d, dataset, enc, len(dataset.continuous_feature_names))
                    sf_data.append(decoded_d.tolist())

                    if idx in replaced_these_idxs:
                        found_sfs.append([test_ids[test_idx], False])
                    else:
                        end = time.time()
                        gen_times.append(end - start)
                        found_sfs.append([test_ids[test_idx], True])

        sf_data = np.array(sf_data)

        success_data = []
        idx_data = []

        for i in found_sfs:
            success_data.append(int(i[1]))
            idx_data.append(i[0])

        sf_df = pd.DataFrame(sf_data)

        sf_df_readable = self.reorder_features(sf_df.values, dataset, enc, len(continuous_features.columns))
        sf_df_readable = pd.DataFrame(sf_df_readable, columns=dataset.columns)

        # return fact index data
        sf_df_readable['Fact_id'] = idx_data
        sf_df_readable['gen_time'] = gen_times

        return sf_df_readable

    def fitness(self, x, population, cat_idxs, actionable_idxs, clf, action_meta, continuous_features, categorical_features, enc, dataset, reach_knn):

        fitness_scores = list()
        meta_fitness = list()

        for solution in population:

            reachability = self.get_reachability(solution, reach_knn)
            gain = self.get_gain(x, solution)
            robustness_1 = self.get_robustness(x, solution, clf, cat_idxs,
                                            actionable_idxs, action_meta,
                                            continuous_features, categorical_features, enc, dataset) * 1

            decoded_solution = []
            for j, x in enumerate(solution):
                decoded_item = self.reorder_features(x, dataset, enc, len(continuous_features.columns))
                decoded_solution.append(decoded_item)

            decoded_solution = np.stack(decoded_solution)
            decoded_solution = decoded_solution.reshape(decoded_solution.shape[0], decoded_solution.shape[-1]) # middle dimension might be 1 removing this

            robustness_2 = (clf.predict_multiple(decoded_solution.reshape(-1, *dataset.state_shape))[0] != dataset.target_action) * 1
            diversity = self.get_diversity(solution)

            term1 = np.array(reachability.flatten() * gain)
            robustness_1 = np.array(robustness_1)
            robustness_2 = np.array(robustness_2)

            robustness_1 *= LAMBDA1
            robustness_2 *= LAMBDA2
            diversity *= GAMMA

            term1 = (term1 + robustness_1 + robustness_2).mean()

            correctness = (clf.predict_multiple(decoded_solution.reshape(-1, *dataset.state_shape)) != dataset.target_action).mean()  # hard constraint that the solution MUST contain SF
            fitness_scores.append((term1 + diversity).item() * correctness)
            meta_fitness.append([reachability.mean(), gain.mean(), robustness_1.mean(), robustness_2.mean(), diversity])

        return np.array(fitness_scores), np.array(meta_fitness)

    def get_diversity(self, solution):
        """
        Return L2 distance between all vectors (the mean)
        """

        if self.diversity_size == 1:
            return 0

        # Take average distance
        score = distance_matrix(solution, solution).sum() / (self.diversity_size ** 2 - self.diversity_size)
        return score

    def get_reachability(self, solution, reach_knn):
        """
        OOD Check using NN-dist metric
        """
        l2s, _ = reach_knn.kneighbors(X=solution, n_neighbors=1, return_distance=True)
        l2s = 1 / (l2s ** 2 + 0.1)
        return l2s

    def get_gain(self, x, solution):
        """
        Return mean distance between query and semifactuals
        """

        scores = np.sqrt(((x - solution) ** 2).sum(axis=1))
        return scores

    def get_robustness(self, x, solution, clf, cat_idxs, actionable_idxs, action_meta, continuous_features,
                       categorical_features, enc, dataset):
        """
        Monte Carlo Approximation of e-neighborhood robustness
        """

        perturbation_preds = list()
        for x_prime in solution:
            instance_perturbations = list()
            for _ in range(MAX_MC):
                x_prime_clone = deepcopy(x_prime)
                perturbed_instance = self.perturb_one_random_feature(x,
                                                                     x_prime_clone,
                                                                     continuous_features,
                                                                     categorical_features,
                                                                     action_meta,
                                                                     cat_idxs,
                                                                     actionable_idxs)

                decoded_perturbed_instance = self.reorder_features(x, dataset, enc, len(dataset.continuous_feature_names))
                instance_perturbations.append(decoded_perturbed_instance.tolist())


            predictions = clf.predict_multiple(decoded_perturbed_instance.reshape(-1, *dataset.state_shape)) != dataset.target_action
            perturbation_preds.append(predictions.tolist())
        return np.array(perturbation_preds).mean(axis=1)

    def perturb_continuous(self, x, x_prime, idx, continuous_features, categorical_features, action_meta):
        """
        slightly perturb continuous feature with actionability constraints
        """

        # Get feature max and min -- and clip it to these
        feature_names = continuous_features.columns.tolist() + categorical_features.columns.tolist()
        cat_name = feature_names[idx]

        if action_meta[cat_name]['can_increase'] and action_meta[cat_name]['can_decrease']:
            max_value = action_meta[cat_name]['max']
            min_value = action_meta[cat_name]['min']

        elif action_meta[cat_name]['can_increase'] and not action_meta[cat_name]['can_decrease']:
            max_value = action_meta[cat_name]['max']
            min_value = x[idx]

        elif not action_meta[cat_name]['can_increase'] and action_meta[cat_name]['can_decrease']:
            max_value = x[idx]
            min_value = action_meta[cat_name]['min']

        else:  # not actionable
            max_value = x[idx]
            min_value = x[idx]

        perturb = gauss(0, ((max_value - min_value) * CONT_PERTURB_STD))
        x_prime[idx] += perturb

        if x_prime[idx] > max_value:
            x_prime[idx] = max_value
        if x_prime[idx] < min_value:
            x_prime[idx] = min_value

        return x_prime

    def get_actionable_feature_idxs(self, continuous_features, categorical_features, action_meta):
        """
        sample a random actionable feature index
        """

        feature_names = continuous_features.columns.tolist() + categorical_features.columns.tolist()
        actionable_idxs = list()

        for i, f in enumerate(feature_names):
            if action_meta[f]['actionable']:
                actionable_idxs.append([i, action_meta[f]['can_increase'], action_meta[f]['can_decrease']])

        return actionable_idxs

    def get_rand_actionable_feature_idx(self, x, actionable_idxs, cat_idxs):
        """
        sample a random actionable feature index
        """

        instance_specific_actionable_indexes = deepcopy(actionable_idxs)

        # Get starting index of categories in actionable index list
        for i in range(len(actionable_idxs)):
            if actionable_idxs[i][0] == cat_idxs[0][0]:
                break
        starting_index = i

        for idx, i in enumerate(list(range(starting_index, len(actionable_idxs)))):

            sl = x[cat_idxs[idx][0]: cat_idxs[idx][1]]

            at_top = sl[-1] == 1
            can_only_go_up = actionable_idxs[i][1]

            at_bottom = sl[0] == 1
            can_only_go_down = actionable_idxs[i][2]

            # if can_only_go_up and at_top:
            #     instance_specific_actionable_indexes.remove(actionable_idxs[i])
            #
            # if can_only_go_down and at_bottom:
            #     instance_specific_actionable_indexes.remove(actionable_idxs[i])

        if len(instance_specific_actionable_indexes):
            rand = np.random.randint(len(instance_specific_actionable_indexes))

        return instance_specific_actionable_indexes[rand]

    def perturb_one_random_feature(self, x, x_prime, continuous_features, categorical_features, action_meta, cat_idxs,
                                   actionable_idxs):
        """
        perturb one actionable feature for MC robustness optimization
        """

        feature_names = continuous_features.columns.tolist() + categorical_features.columns.tolist()

        change_idx = self.get_rand_actionable_feature_idx(x, actionable_idxs, cat_idxs)[0]
        feature_num = len(feature_names)

        # if categorical feature
        if feature_names[change_idx] in categorical_features.columns:
            perturbed_feature = self.generate_category(x,
                                                       x_prime,
                                                       change_idx - len(continuous_features.columns),
                                                       # index of category for function
                                                       cat_idxs,
                                                       action_meta,
                                                       categorical_features,
                                                       replace=False)

            x_prime[cat_idxs[change_idx - len(continuous_features.columns)][0]:
                    cat_idxs[change_idx - len(continuous_features.columns)][1]] = perturbed_feature

        # if continuous feature
        else:
            x_prime = self.perturb_continuous(x,
                                              x_prime,
                                              change_idx,
                                              continuous_features,
                                              categorical_features,
                                              action_meta)

        return x_prime

    def generate_cat_idxs(self, continuous_features, enc):
        """
        Get indexes for all categorical features that are one hot encoded
        """

        cat_idxs = list()
        start_idx = len(continuous_features.columns)
        for cat in enc.categories_:
            cat_idxs.append([start_idx, start_idx + cat.shape[0]])
            start_idx = start_idx + cat.shape[0]
        return cat_idxs

    def generate_category(self, x, x_prime, idx, cat_idxs, action_meta, categorical_features, replace=True):
        """
        Randomly generate a value for a OHE categorical feature using actionability constraints
        replace: this gives the option if the generation should generate the original
        value for the feature that is present in x, or if it should only generate
        different x_primes with different values for the feature

        """

        original_rep = x[cat_idxs[idx][0]: cat_idxs[idx][1]]  # To constrain with initial datapoint
        new_rep = x_prime[cat_idxs[idx][0]: cat_idxs[idx][1]]  # to make sure we modify based on new datapoint

        cat_name = categorical_features.columns[idx]

        if replace:  # just for population initialisation

            # If you can generate new feature anywhere
            if action_meta[cat_name]['can_increase'] and action_meta[cat_name]['can_decrease']:
                new = np.eye(len(original_rep))[np.random.choice(len(original_rep))]

            # if you can only increase
            elif action_meta[cat_name]['can_increase'] and not action_meta[cat_name]['can_decrease']:
                try:
                    # To account for when it's the last value in the scale of categories
                    new = np.eye(len(original_rep) - (np.argmax(original_rep)))[
                        np.random.choice(len(original_rep) - (np.argmax(original_rep)))]
                    new = np.append(np.zeros((np.argmax(original_rep))), new)
                except:
                    new = new_rep

            # If you can only decrease
            elif not action_meta[cat_name]['can_increase'] and action_meta[cat_name]['can_decrease']:
                try:
                    # To account for when it's the first value in the scale of categories
                    new = np.eye(np.argmax(original_rep) + 1)[np.random.choice(np.argmax(original_rep) + 1)]
                    new = np.append(new, np.zeros((len(original_rep) - np.argmax(original_rep)) - 1))
                except:
                    new = new_rep

            else:
                new = new_rep

        else:  # For MC sampling, and mutation

            # If you can generate new feature anywhere
            if action_meta[cat_name]['can_increase'] and action_meta[cat_name]['can_decrease']:
                if original_rep.shape[0] == 1:
                    new = np.array([abs(1-original_rep[0])])
                else:
                    new = np.eye(len(original_rep)-1)[np.random.choice(len(original_rep)-1)]
                    new = np.insert(new, np.argmax(new_rep), 0)

            # if you can only increase
            elif action_meta[cat_name]['can_increase'] and not action_meta[cat_name]['can_decrease']:
                try:
                    # To account for when it's the last value in the scale of categories
                    new = np.eye(len(original_rep) - np.argmax(original_rep) - 1)[
                        np.random.choice(len(original_rep) - np.argmax(original_rep) - 1)]
                    new = np.insert(new, np.argmax(new_rep) - (np.argmax(original_rep)), 0)
                    new = np.concatenate(
                        (np.zeros((len(original_rep) - (len(original_rep) - np.argmax(original_rep)))), new))
                except:
                    new = new_rep

            # If you can only decrease
            elif not action_meta[cat_name]['can_increase'] and action_meta[cat_name]['can_decrease']:

                try:  # To account for when it's the first value in the scale of categories
                    new = np.eye(np.argmax(original_rep))[np.random.choice(np.argmax(original_rep))]
                    new = np.insert(new, np.argmax(new_rep), 0)
                    new = np.concatenate((new, np.zeros((len(original_rep) - np.argmax(original_rep) - 1))))

                except:
                    new = new_rep
            else:
                new = new_rep

        return new

    def init_population(self, x, X_train, continuous_features, categorical_features, cat_idxs, action_meta, replace=True):

        num_features = X_train.shape[1]
        population = np.zeros((self.population_size, self.diversity_size, num_features))

        # iterate continous features
        for i in range(len(continuous_features.columns)):

            cat_name = continuous_features.columns[i]
            value = x[i]

            # If the continuous feature can take any value
            if action_meta[cat_name]['can_increase'] and action_meta[cat_name]['can_decrease']:
                f_range = action_meta[cat_name]['max'] - action_meta[cat_name]['min']
                temp = value + np.random.normal(0, CONT_PERTURB_STD, (self.population_size,  self.diversity_size, 1)) * f_range
                # temp *= f_range
                population[:, :, i:i + 1] = temp

            # If the continous feature can only go up
            elif action_meta[cat_name]['can_increase'] and not action_meta[cat_name]['can_decrease']:
                f_range = action_meta[cat_name]['max'] - value
                temp = value + abs(np.random.normal(0, CONT_PERTURB_STD, (self.population_size,  self.diversity_size, 1))) * f_range
                # temp *= f_range
                population[:, :, i:i + 1] = temp

            # if the continuous features can only go down
            elif not action_meta[cat_name]['can_increase'] and action_meta[cat_name]['can_decrease']:
                f_range = value
                temp = value - abs(np.random.normal(0, CONT_PERTURB_STD, (self.population_size,  self.diversity_size, 1))) * f_range
                # temp *= f_range
                population[:, :, i:i + 1] = temp

            # If it's not actionable
            else:
                temp = np.zeros((self.population_size,  self.diversity_size, 1)) + value
                population[:, :, i:i + 1] = temp

        # iterate categorical features
        current_idx = len(continuous_features.columns)
        for i in range(len(categorical_features.columns)):
            cat_len = len(x[cat_idxs[i][0]: cat_idxs[i][1]])
            temp = list()

            for j in range(self.population_size):
                temp2 = list()
                for k in range(self.diversity_size):
                    x_prime = deepcopy(x)  # to keep x the same
                    temp3 = self.generate_category(x, x_prime, i, cat_idxs, action_meta, categorical_features, replace=True)
                    temp2.append(temp3.tolist())
                temp.append(temp2)

            temp = np.array(temp)
            population[:, :, current_idx:current_idx + cat_len] = temp
            current_idx += cat_len

        return population

    def mutation(self, population, continuous_features, categorical_features, cat_idxs, action_meta, actionable_idxs, x):
        """
        Iterate all features and randomly perturb them
        """

        feature_names = continuous_features.columns.tolist() + categorical_features.columns.tolist()

        for i in range(len(population)):
            for j in range(self.diversity_size):
                x_prime = population[i][j]
                for k in range(len(actionable_idxs)):
                    if np.random.rand() < MUTATION_RATE:
                        change_idx = actionable_idxs[k][0]
                        # if categorical feature
                        if feature_names[change_idx] in categorical_features.columns:
                            perturbed_feature = self.generate_category(x,
                                                                       x_prime,
                                                                       change_idx - len(continuous_features.columns),
                                                                       # index of category for function
                                                                       cat_idxs,
                                                                       action_meta,
                                                                       categorical_features,
                                                                       replace=False)
                            x_prime[cat_idxs[change_idx - len(continuous_features.columns)][0]:
                                    cat_idxs[change_idx - len(continuous_features.columns)][1]] = perturbed_feature

                        # if continuous feature
                        else:
                            x_prime = self.perturb_continuous(x,
                                                              x_prime,
                                                              change_idx,
                                                              continuous_features,
                                                              categorical_features,
                                                              action_meta)
        return population

    def natural_selection(self, population, fitness_scores):
        """
        Save the top solutions
        """
        tournamet_winner_idxs = list()
        for i in range(self.population_size - ELITIST):
            knights = np.random.randint(0, population.shape[0], 2)
            winner_idx = knights[np.argmax(fitness_scores[knights])]
            tournamet_winner_idxs.append(winner_idx)

        return population[tournamet_winner_idxs], population[(-fitness_scores).argsort()[:ELITIST]]

    def crossover(self, population, continuous_features, cat_idxs, actionable_idxs):
        """
        mix up the population
        """

        children = list()

        for i in range(0, population.shape[0], 2):
            if population.shape[0] >= i + 2:
                parent1, parent2 = population[i:i + 2]
                child1, child2 = deepcopy(parent1), deepcopy(parent2)

                crossover_idxs = np.random.randint(low=0,
                                                   high=2,
                                                   size=self.diversity_size * len(actionable_idxs)).reshape(self.diversity_size,
                                                                                                            len(actionable_idxs))

                # Crossover Children
                for j in range(self.diversity_size):
                    for k in range(len(actionable_idxs)):

                        # Child 1
                        if crossover_idxs[j][k] == 0:

                            # if continuous
                            if actionable_idxs[k][0] < len(continuous_features.columns):
                                child1[j][actionable_idxs[k][0]] = parent2[j][actionable_idxs[k][0]]

                            # if categorical
                            else:
                                cat_idx = actionable_idxs[k][0] - len(continuous_features.columns)
                                child1[j][cat_idxs[cat_idx][0]: cat_idxs[cat_idx][1]] = parent2[j][cat_idxs[cat_idx][0]:
                                                                                                   cat_idxs[cat_idx][1]]


                        # Child 2
                        else:
                            # if continuous
                            if actionable_idxs[k][0] < len(continuous_features.columns):
                                child2[j][actionable_idxs[k][0]] = parent1[j][actionable_idxs[k][0]]

                            # if categorical
                            else:
                                cat_idx = actionable_idxs[k][0] - len(continuous_features.columns)
                                child2[j][cat_idxs[cat_idx][0]: cat_idxs[cat_idx][1]] = parent1[j][cat_idxs[cat_idx][0]:
                                                                                                   cat_idxs[cat_idx][1]]

                children.append(child1.tolist())
                children.append(child2.tolist())

        return np.array(children)

    def force_sf(self, result, x, enc, dataset):
        result = self.reorder_features(x, dataset, enc, len(dataset.continuous_feature_names))
        result_preds = self.bb_model.predict(enc.inverse_transform(result))
        keep = np.where(result_preds != abs(dataset.target_action))[0]
        replace_these_idxs = np.where(result_preds == dataset.target_action)[0]
        for idx in replace_these_idxs:
            result[idx] = x  # just replace with initial sf for fairness of comparison to other methods
        return result, replace_these_idxs

    def reorder_features(self, population, dataset, enc, continuous_features_num):
        if len(population.shape) == 1:
            population = population.reshape(1, -1)

        new_population = []

        if len(population) == 0:
            return []

        for x_id, x in enumerate(population):
            # extract categorical features
            cat_x = x[continuous_features_num:]
            # decode categorical features
            if cat_x.shape[0] != 0:
                cat_x = enc.inverse_transform(cat_x.reshape(1, -1)).squeeze()

            cat_order = list(dataset.cat_order.values())

            x_decoded = []
            cont_idx = 0
            cat_idx = 0
            for i in range(len(dataset.columns)):
                if i not in cat_order:
                    x_decoded.append(x[cont_idx])
                    cont_idx += 1

                else:
                    x_decoded.append(cat_x[cat_idx])
                    cat_idx += 1

            new_population.append(x_decoded)

        return np.array(new_population)