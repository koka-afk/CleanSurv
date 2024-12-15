import warnings
import time
import numpy as np
import json
import re
import random
import os.path
from random import randint


# import classes 
from learn2clean.imputation.imputer import Imputer 
from learn2clean.duplicate_detection.duplicate_detector import Duplicate_detector
from learn2clean.feature_selection.feature_selector import Feature_selector 
from learn2clean.outlier_detection.outlier_detector import Outlier_detector 
from learn2clean.survival_analysis.cox_model import CoxRegressor
from learn2clean.survival_analysis.dh_neural_network import NeuralNetwork 
from learn2clean.survival_analysis.random_survival_forest import RSF
from learn2clean.regression.regressor import Regressor


def update_q(q, r, state, next_state, action, beta, gamma, states_dict):

    # Update Q-value using the Q-learning formula

    # Calculate the new Q-value for the current state-action pair

    # q[state, action] represents the Q-value for the current state (state) and action (action) combination.
    # new_q calculates the updated Q-value based on the Q-learning formula:
    # Q(s, a) = Q(s, a) + learning_rate * [reward + discount_factor * max(Q(s', a')) - Q(s, a)]
    # qsa represents the current Q-value for the state-action pair (state, action).

    action_name = states_dict[action]
    current_state_name = states_dict[state]
    #print(f'Action name: {action_name} \n\nCurrent State Name: {current_state_name}\n\n')
    rsa = r[current_state_name]['followed_by'][action_name] #r[state, action]
    #print(rsa)
    # rsa is the immediate reward obtained when taking the current action in the current state.
    qsa = q[state, action]

    # Update the Q-value for the current state-action pair using the Q-learning formula.
    new_q = qsa + beta * (rsa + gamma * max(q[next_state, :]) - qsa)

    # Update the Q-value matrix with the new Q-value for the current state-action pair.
    # This line effectively replaces the old Q-value with the updated one.
    q[state, action] = new_q
    # renormalize row to be between 0 and 1
    rn = q[state][q[state] > 0] / np.sum(q[state][q[state] > 0])

    q[state][q[state] > 0] = rn

    return r[current_state_name]['followed_by'][action_name] #r[state, action]


def remove_adjacent(nums):

    previous = ''

    for i in nums[:]:  # using the copy of nums

        if i == previous:

            nums.remove(i)

        else:

            previous = i

    return nums


class SurvivalQlearner:

    def __init__(self, dataset, time_col, event_col, goal, verbose=False, json_path=None, file_name=None, threshold=None):

        self.dataset = dataset

        self.time_col = time_col

        self.event_col = event_col

        self.goal = goal

        self.json_path = json_path

        if json_path is not None:
            with open(json_path) as file:
                data = json.load(file)
                self.json_file = data
        
        my_path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(my_path, "reward.json")

        with open(path) as reward:
            data = json.load(reward)
            self.rewards = data

        self.verbose = verbose

        self.file_name = file_name

        self.threshold = threshold  #sds


    def get_params(self, deep=True):

        """
            Get parameters of the QLearner instance.

            Parameters:
            - self: The QLearner instance for which parameters are to be retrieved.
            - deep (boolean): Indicates whether to retrieve parameters deeply nested within the object.

            Returns:
            - params (dictionary): A dictionary containing the parameters of the QLearner instance.
                                The keys represent parameter names, and the values are their current values.
            """

         # Create a dictionary 'params' to store the parameters of the QLearner instance.

        return {
                'goal': self.goal,           # Store the 'goal' parameter value.

                'event_col': self.event_col, # Store the 'event_col' parameter value.

                'time_col': self.time_col,   # Store the 'time_col' parameter value.

                'verbose': self.verbose,     # Store the 'verbose' parameter value.

                'file_name': self.file_name, # Store the 'file_name' parameter value.

                'threshold': self.threshold  # Store the 'threshold' parameter value.

                }

    def set_params(self, **params):

        for k, v in params.items():

            if k not in self.get_params():

                warnings.warn("Invalid parameter(s)"
                              "Check the list of available parameters with "
                              "`qlearner.get_params().keys()`")

            else:

                setattr(self, k, v)


    def get_states_actions(self):
        n = 0
        for key in self.rewards:
            if len(self.rewards[key]["followed_by"]) != 0:
                n += 1
        return n + 1
    
    
    def get_imputers(self):
        imputer_no = 0
        for key in self.rewards:
            if self.rewards[key]['type'] == "Imputer":
                imputer_no += 1
        return imputer_no
    

    def get_methods(self):
        methods = []
        for key in self.rewards:
            if self.rewards[key]['type'] not in ('Survival_Model', 'Regression'):
                methods.append(key)
        return methods
    

    def get_goals(self):
        goals = []
        for key in self.rewards:
            if self.rewards[key]['type'] in ('Survival_Model', 'Regression'):
                goals.append(key)
        return goals


    def edit_edge(self, u, v, weight):
        if weight == -1:
            self.rewards[u]['followed_by'].pop(v, None)
        else:
            self.rewards[u]['followed_by'][v] = weight

    
    def set_rewards(self, data):
        self.rewards = data


    def disable(self, op):
        ops_names = []
        for key in self.rewards:
            if self.rewards[key]['type'] == op:
                ops_names.append(key)
        for val in ops_names: # loop in case op parameter was a preprocessing step like "Imputer"
            for key in self.rewards:
                self.rewards[key]['followed_by'].pop(val, None)
            self.rewards.pop(val, None)

        for key in self.rewards: # loop in case op parameter was a single preprocessing method like "Median"
            self.rewards[key]['followed_by'].pop(op, None)
        self.rewards.pop(op, None)

    def Initialization_Reward_Matrix(self, dataset):
        """ [Data Preprocessing Reward/Connection Graph]

            This function initializes a reward matrix based on the input dataset.

            State: Initial Data

            Methods (Actions):
            1. CCA (missing values)
            2. MI (missing values)
            3. IPW (missing values)
            4. Mean (missing values)
            5. Median (missing values)
            6. UC (feature selection)
            7. LASSO (feature selection)
            8. RFE (feature selection)
            9. IG (feature selection)
            10. ED (deduplication)
            11. DBID (deduplication)
            12. DBT (deduplication)
            13. CR (outlier detection)
            14. MR (outlier detection)
            15. MUO (outlier detection)
            16. RSF (Survival model)
            17. COX (Survival model)
            18. NN (Survival model) 
        """
        # Check if there are missing values in the dataset
        if dataset.copy().isnull().sum().sum() > 0:

            r = self.rewards
            
            # Define the number of actions and states
            n_actions = self.get_states_actions()

            n_states = self.get_states_actions()

            check_missing = True

        else:  

            r = self.rewards

            # Define the number of actions and states
            n_actions = self.get_states_actions()

            n_states = self.get_states_actions()

            imputer_no = self.get_imputers()
            
            n_actions -= imputer_no
            n_states -= imputer_no

            check_missing = False

        # Initialize a Q matrix with zeros
        zeros_mat = [[0.0 for x in range(n_actions)] for y in range(n_states)]
        q = np.array(zeros_mat)

        # we prevent the transition from any survival model during preprocessing
        # r = r[~np.all(r == -1, axis=1)]

        # Print the reward matrix if verbose mode is enabled
        if self.verbose:

            print("Reward matrix")

            print(r)

        # Return the initialized Q matrix, reward matrix, number of actions, number of states, and a flag for missing values
        return q, r, n_actions, n_states, check_missing
    

    def get_config_file(self, class_name):
        config = None
        if self.json_path is not None:
            if class_name in self.json_file.keys():
                config = self.json_file[class_name]
        return config
    

    def handle_categorical(self, dataset):

        from sklearn.preprocessing import OrdinalEncoder
        from pandas.api.types import is_numeric_dtype

        data = dataset

        print(f"\n\n **HANDLE CATEGORICAL WITHOUT IMPUTATION** \n\n {data}")

        oe_dict = {}

        for col_name in data:
            if not is_numeric_dtype(data[col_name]):
                oe_dict[col_name] = OrdinalEncoder()
                col = data[col_name]
                col_not_null = col[col.notnull()]
                reshaped_values = col_not_null.values.reshape(-1, 1) # TODO is this reshaping really needed? It might cause problems
                encoded_values = oe_dict[col_name].fit_transform(reshaped_values)
                data.loc[col.notnull(), col_name] = np.squeeze(encoded_values)
        
        print(f"\n\n **HANDLE CATEGORICAL WITHOUT IMPUTATION** \n\n {data}")

        return data, oe_dict
    

    def construct_pipeline(self, dataset, actions_list, time_col, event_col, check_missing):

        """
        This function represents a data preprocessing pipeline that applies a series of actions to the input dataset
        based on the provided list of actions. It can handle missing values and perform various data preprocessing steps.

        Parameters:
        - dataset: The input dataset to be preprocessed.
        - actions_list: A list of action indices indicating which data preprocessing steps to perform.
        - time_col: The name of the column representing time information.
        - event_col: The name of the column representing event information.
        - check_missing: A flag indicating whether to check for missing values in the dataset.

        Returns:
        - n: The preprocessed dataset after applying the specified actions.
        - res: Reserved for potential future use (currently set to None).
        - t: The CPU time taken to complete the data preprocessing pipeline.
        """

        # Create a copy of the input dataset
        #dataset = dataset.copy()

        # Define names of goals (used when executing survival models)
        goals_name = self.get_goals() #["RSF", "COX", "NN", "OLS", "LASSO_REG", "MARS"]

        # Initialize the result variable as None
        res = None

        # Check if missing values should be handled
        if check_missing:

            # Define names of actions (methods) for preprocessing
            actions_name = ["Mean", "CCA", "MI", "KNN", "Median",
                            "UC", "LASSO", "RFE", "IG",
                            "DBID", "DBT", "ED",
                            "MR", "MR", "MUO"] # TODO Replace "CR" action with "MR" until fixed ... and IPW with MI
            #update on 25th of june 2024 TODO replaced Mean with my version of KNN  

            # Define a list of classes corresponding to each action (used for instantiation)
            L2C_class = [Imputer, Imputer, Imputer, Imputer, Imputer,
                         Feature_selector, Feature_selector, Feature_selector, Feature_selector,
                         Duplicate_detector, Duplicate_detector, Duplicate_detector,
                         Outlier_detector, Outlier_detector, Outlier_detector,
                         RSF, CoxRegressor, NeuralNetwork, Regressor, Regressor, Regressor
                         ]

        else:
            # If no missing values handling is needed, define names of other actions
            actions_name = [
                            "UC", "LASSO", "RFE", "IG",
                            "DBID", "DBT", "ED",
                            "MR", "MR", "MUO"] # TODO Replace "CR" action with "MR" until fixed

            # Define a list of classes corresponding to each action (used for instantiation)
            L2C_class = [Feature_selector, Feature_selector, Feature_selector, Feature_selector,
                         Duplicate_detector, Duplicate_detector, Duplicate_detector,
                         Outlier_detector, Outlier_detector, Outlier_detector,
                         RSF, CoxRegressor, NeuralNetwork, Regressor, Regressor, Regressor]

        print()

        print("Start pipeline")
        print(actions_list)
        print("-------------")

        start_time = time.time()

        n = None

        for a in actions_list:

            if not check_missing:

                if a in range(0, 6):

                    # Deduplication (0-2) and feature selection (3-6) based on the action index.
                    config = None
                    if self.json_path is not None:
                        if a <= 3:
                            config = self.get_config_file("Feature_selector")
                        else:
                            config = self.get_config_file("Duplicate_detector")
                    
                    dataset = L2C_class[a](dataset = dataset, time_col = time_col, event_col = event_col, strategy = actions_name[a], config=config, verbose = self.verbose).transform()

                if a in (7, 8, 9):
                    # Execute outlier detectors (7-9) based on the action index.

                    config = self.get_config_file("Outlier_detector")

                    print(f"IN OUTLIER CALL: \n\n\n\n\n\n\n")
                    print(dataset)
                    print("\n\n\n\n\n\n\n")
                    dataset = L2C_class[a](dataset = dataset, time_col = time_col, event_col = event_col, strategy = actions_name[a], verbose = self.verbose).transform()
                    print(f"IN OUTLIER CALL: \n\n\n\n\n\n\n")
                    print(dataset)
                    print("\n\n\n\n\n\n\n")
                if a == 10:
                    # Execute Random Survival Forest
                    print(f'\nIN RSF --------------------------------> {dataset}\n\n')
                    config = self.get_config_file("RSF")
                    rsf = L2C_class[a](dataset=dataset, time_column=time_col, target_goal=event_col, config=config, verbose=self.verbose)
                    survival_probabilities, c_index = rsf.fit_rsf_model()
                    n = {"quality_metric": c_index}
                    print(f"IN RSF --------------------------------> {c_index} \n\n\n\n {n}")
                
                if a == 11:
                    # Execute Cox Model
                    print("\n\n\n\n\n\n\n")
                    print(dataset)
                    print("\n\n\n\n\n\n\n")
                    # TODO continue developing this file starting by adding "mode" parameter and then adjusting this part (cox) and then continue
                    config = self.get_config_file("CoxRegressor")
                    res = dataset # taking the final dataset after cleaning as a result # TODO check if needed in the unnecessary final call in show_traverse
                    cox_model = L2C_class[a](dataset = dataset, time_column = time_col, target_goal = event_col, config=config, verbose = self.verbose)
                    c_index = cox_model.updated_fit()
                    if isinstance(c_index, np.generic):
                            c_index = c_index.item()
                    time_dif = time.time() - start_time
                    n = {"quality_metric": c_index, 'time': time_dif}
                    print(f"IN SURVIVAL_QLEARNER --------------------------------> {c_index} \n\n\n\n {n}")

                if a == 12:
                    # Execute Neural Network
                    config = self.get_config_file("NeuralNetwork")
                    res = dataset # taking the final dataset after cleaning as a result # TODO check if needed in the unnecessary final call in show_traverse
                    nn = L2C_class[a](dataset = dataset, time_column = time_col, target_goal = event_col, config=config, verbose = self.verbose)
                    c_index = nn.fit_dh()
                    n = {"quality_metric": c_index}
                
                if a in (13, 14, 15):
                    config = self.get_config_file(self.goal)
                    res = dataset
                    reg = L2C_class[a](dataset=dataset, target=self.event_col, strategy=self.goal, verbose=self.verbose)
                    quality_metric = reg.transform()
                    n = quality_metric
                    

            else:

                if (dataset is not None and len(dataset.dropna()) == 0):

                    pass

                else:

                    config = None

                    if a in (0, 1, 2, 3, 4):
                        # Execute missing values handling methods (0-4) based on the action index.

                        config = self.get_config_file("Imputer")
                        dataset = L2C_class[a](dataset = dataset, time_col = time_col, event_col = event_col, config=config, strategy = actions_name[a], verbose = self.verbose).transform()
                        #print("HANDLING MISSING VALUES: " + str(len(n)))
                    if a in (5, 6, 7, 8):
                        # Execute Feature selection methods (5-8) based on the action index.

                        config = self.get_config_file("Feature_selector")
                        dataset = L2C_class[a](dataset = dataset, time_col = time_col, event_col = event_col, config=config, strategy = actions_name[a], verbose = self.verbose).transform()

                    if a in (9, 10, 11):
                        # Execute deduplication methods (9-11) based on the action index.

                        config = self.get_config_file("Duplicate_detector")
                        dataset = L2C_class[a](dataset = dataset, time_col = time_col, event_col = event_col, config=config, strategy = actions_name[a], verbose = self.verbose).transform()

                    if a in (12, 13, 14):
                        # Execute outlier detection methods (12-14) based on the action index.

                        config = self.get_config_file("Outlier_detector")
                        dataset = L2C_class[a](dataset = dataset, time_col = time_col, event_col = event_col, config=config, strategy = actions_name[a], verbose = self.verbose).transform()

                    if a == 15:
                        # Execute Random Survival Forest
                        print(f'\nIN RSF --------------------------------> {dataset}\n\n')
                        config = self.get_config_file("RSF")
                        rsf = L2C_class[a](dataset=dataset, time_column=time_col, target_goal=event_col, config=config, verbose=self.verbose)
                        survival_probabilities, c_index = rsf.fit_rsf_model()
                        n = {"quality_metric": c_index}
                        print(f"IN RSF --------------------------------> {c_index} \n\n\n\n {n}")

                    if a == 16:
                        # Execute Cox Model

                        config = self.get_config_file("CoxRegressor")
                        res = dataset # taking the final dataset after cleaning as a result # TODO check if needed in the unnecessary final call in show_traverse
                        cox_model = L2C_class[a](dataset=dataset, time_column=time_col, target_goal=event_col, config=config, verbose=self.verbose)
                        c_index = cox_model.updated_fit()
                        if isinstance(c_index, np.generic):
                            c_index = c_index.item()
                        time_dif = time.time() - start_time
                        n = {"quality_metric": c_index, 'time': time_dif}
                        print(f"IN SURVIVAL_QLEARNER --------------------------------> {c_index} \n\n\n\n {n}")

                    if a == 17:
                        # Execute Neural Network

                        config = self.get_config_file("NeuralNetwork")
                        res = dataset # taking the final dataset after cleaning as a result # TODO check if needed in the unnecessary final call in show_traverse
                        nn = L2C_class[a](dataset = dataset, time_column = time_col, target_goal = event_col, config=config, verbose = self.verbose)
                        c_index = nn.fit_dh()
                        n = {"quality_metric": c_index}
                    
                    if a in (18, 19, 20):
                        config = self.get_config_file(self.goal)
                        res = dataset
                        reg = L2C_class[a](dataset=dataset, target=self.event_col, strategy=self.goal, verbose=self.verbose)
                        quality_metric = reg.transform()
                        n = quality_metric
                        if n == 0:
                            n = {'quality_metric': 0}

        
        # Calculate the elapsed CPU time
        t = time.time() - start_time

        print("End Pipeline CPU time: %s seconds" % (time.time() - start_time))

        # Return the preprocessed dataset, result, and CPU time
        return n, res, t

    def show_traverse(self, dataset, q, g, check_missing):
        # show all the greedy traversals
        """
        This function displays all the greedy traversals of the reinforcement learning agent based on the learned Q-values.
        It explores different strategies for preprocessing data based on the Q-matrix.

        Parameters:
        - dataset: The input dataset to be preprocessed.
        - q: The Q-matrix representing the learned state-action values.
        - g: The index of the survival model goal.
        - check_missing: A flag indicating whether to check for missing values in the dataset.

        Returns:
        - actions_strategy: A list of strings describing the actions taken in each strategy.
        - strategy: A list of quality metrics corresponding to each strategy.
        """

         # Define lists of methods and goals based on whether missing values should be handled
        if check_missing:

            methods, goals = ["Mean", "CCA", "MI", "KNN", "Median",
                            "UC", "LASSO", "RFE", "IG",
                            "DBID", "DBT", "ED",
                            "MR", "MR", "MUO"], ["RSF", "COX", "NN", "OLS", "LASSO_REG", "MARS"]

        else:

            methods, goals = ["UC", "LASSO", "RFE", "IG",
                            "DBID", "DBT", "ED",
                            "MR", "MR", "MUO"], ["RSF", "COX", "NN", "OLS", "LASSO_REG", "MARS"]

        n_states = len(methods) + 1

        # Append the current goal to the list of methods (for traversal visualization)
        methods.append(str(goals[g]))

        strategy = []

        actions_strategy = []

        final_dataset = None

        for i in range(len(q)-1):
            
            # This 'for' loop iterates through the states (methods) represented by the Q-matrix, excluding the last state.
            actions_list = []

            current_state = i

            current_state_name = methods[i]
            # traverse = "%i -> " % current_state

            traverse_name = "%s -> " % current_state_name

            n_steps = 0

            while current_state != n_states-1 and n_steps < 17:
                # This 'while' loop continues until either the current state is the final goal state or a maximum of 17 steps is reached.
                actions_list.append(current_state)

                next_state = np.argmax(q[current_state])

                current_state = next_state

                current_state_name = methods[next_state]
                # traverse += "%i -> " % current_state

                traverse_name += "%s -> " % current_state_name

                actions_list.append(next_state)

                n_steps = n_steps + 1

                actions_list = remove_adjacent(actions_list)

            if not check_missing:

                traverse_name = traverse_name[:-4]

                del actions_list[-1]
                actions_list.append(g+len(methods)-1)

            else:

                del actions_list[-1]

                actions_list.append(g+len(methods)-1)

                traverse_name = traverse_name[:-4]

            print(f'BEFORE CHECK MISSING IF CONDITION ---> {check_missing}')

            if check_missing: # this if statement ensures that if there are NaNs in the dataset, there must be imputation 'before' model
                print("\n\n IN IMPUTATION CHECK >>>>>>>>>>>>>>>>\n\n")
                temp = traverse_name.split(" -> ")
                print(f'HERE IS TEMP ---> {temp} \n\n {actions_list} \n\n')
                has_imputer = False
                name = "" 
                imputer_list = ["Mean", "CCA", "MI", "KNN", "Median"]
                for im in imputer_list:
                    if im in temp:
                        has_imputer = True
                        name = im
                        break
                
                if not has_imputer:
                    random_index = 0
                    random_imputer = randint(0, 4)
                    temp.insert(random_index, imputer_list[random_imputer])
                    actions_list.insert(random_index, random_imputer)
                    traverse_name = ""
                    for item in range(len(temp) - 1):
                        traverse_name = traverse_name + temp[item] + " -> "
                    traverse_name += str(self.goal)
                    traverse_name = traverse_name.strip()
                else:
                    index = temp.index(name)
                    temp.pop(index)
                    temp.insert(0, name) # adjusting string sequence of strategy
                    pos = imputer_list.index(name)
                    index_action_list = actions_list.index(pos)
                    actions_list.pop(index_action_list)
                    actions_list.insert(0, pos) # adjusting actual list of actions
                    traverse_name = ""
                    for item in range(len(temp) - 1):
                        traverse_name = traverse_name + temp[item] + " -> "
                    traverse_name += str(self.goal)
                    traverse_name = traverse_name.strip()
            else:
                dataset = self.handle_categorical(dataset)[0] # encoding categorical values outside Imputer class
                

            for idx in range(len(actions_list)): # convert numbers that for some reason randomly change to numpy type Ex. 2 -> np.int64(2)
                num = actions_list[idx]
                if isinstance(num, np.generic):
                    actions_list[idx] = num.item()
            
            print("\n\nStrategy#", i, ": Greedy traversal for "
                  "starting state %s" % methods[i])

            print(traverse_name)

            print(actions_list)

            
                    

            actions_strategy.append(traverse_name)

            # Execute the preprocessing pipeline for the current strategy and store the quality metric
            dataset_copy = dataset.copy()
            temp_val = self.construct_pipeline(dataset_copy, actions_list, self.time_col, self.event_col, check_missing)
            print(f'temp_val pipeline \n {temp_val}')
            strategy.append(temp_val[0])
            final_dataset = temp_val[1]

        # Execute the preprocessing pipeline for the final strategy (goal) and store the quality metric TODO: do not know if needed
        #strategy.append(self.construct_pipeline(final_dataset, [g+len(methods)-1], self.time_col, self.event_col, check_missing)[1])

        print()

        print("==== Recap ====\n")

        print("List of strategies tried by Learn2Clean:")

        print(actions_strategy)

        print('\nList of corresponding quality metrics ****\n',
              strategy)

        print()

        return actions_strategy, strategy

    def Learn2Clean(self):

        """
        This function represents the main Learn2Clean algorithm. It learns an optimal policy for data preprocessing using Q-learning
        and executes the best strategy based on quality metrics for the specified survival analysis goal.

        Returns:
        - rr: A tuple containing information about the best strategy and its performance.
        """

        goals = ["RSF", "COX", "NN", "OLS", "LASSO_REG", "MARS"]

         # Check if the specified goal is valid
        if self.goal not in goals:

            raise ValueError("Goal invalid. Please choose between RSF, COX, NN, OLS, LASSO or MARS")

        else:

            g = goals.index(self.goal)

            pass

        start_l2c = time.time()

        print("Start Learn2Clean")

        gamma = 0.91 #0.8

        beta = 0.1 #1.

        n_episodes = 1E3

        epsilon = 0.2 #0.05

        random_state = np.random.RandomState(1999)

        # Initialize Q-matrix, reward matrix, number of actions, number of states, and missing value flag
        q, r, n_actions, n_states, check_missing = self.Initialization_Reward_Matrix(self.dataset)

        state_names = []
        for key in self.rewards:
            if (self.rewards[key]["type"] == "Survival_Model" or self.rewards[key]["type"] == "Regression") and key != self.goal:
                continue
            if check_missing:
                state_names.append(key)
            else:
                if self.rewards[key]['type'] != 'Imputer':
                    state_names.append(key)

        print(state_names)
        states_dict = {}
        states_dict_reversed = {}
        i = 0
        for x in state_names:
            states_dict[i] = x
            states_dict_reversed[x] = i
            i += 1

        for e in range(int(n_episodes)):

            states = list(range(n_states))

            random_state.shuffle(states)

            current_state = states[0]

            goal = False

            r_mat = r.copy()

            if e % int(n_episodes / 10.) == 0 and e > 0:

                pass

            while (not goal) and (current_state != n_states-1):

                #print("HEREEEEEE")
                 # Implement epsilon-greedy exploration strategy to select actions
                valid = r_mat[states_dict[current_state]]['followed_by']

                temp = [] #r[current_state] >= 0
                for valid_state in valid:
                    if valid_state in states_dict_reversed.keys():
                        temp.append(states_dict_reversed[valid_state])
                valid_moves = [False for x in  range(n_states)]
                for x in temp:
                    valid_moves[x] = True
                valid_moves = np.array(valid_moves)
                

                if random_state.rand() < epsilon:

                    actions = np.array(list(range(n_actions)))

                    actions = actions[valid_moves]

                    if type(actions) is int:

                        actions = [actions]

                    random_state.shuffle(actions)

                    action = actions[0]

                    next_state = action

                else:


                    if np.sum(q[current_state]) > 0:

                        action = np.argmax(q[current_state])

                    else:

                        actions = np.array(list(range(n_actions)))

                        actions = actions[valid_moves]

                        random_state.shuffle(actions)

                        action = actions[0]

                    next_state = action
               
                reward = update_q(q, r, current_state, next_state, action, beta, gamma, states_dict)

                if reward > 1:

                    goal = True

                np.delete(states, current_state)

                current_state = next_state

        if self.verbose:

            print("Q-value matrix\n", q)

        print("Learn2Clean - Pipeline construction -- CPU time: %s seconds"
              % (time.time() - start_l2c))

        metrics_name = ["C-Index", "C-Index", "C-Index", "MSE", "MSE", "MSE"]

        print("=== Start Pipeline Execution ===")

        print(q)

        start_pipexec = time.time()

        # Execute strategies and store results
        result_list = self.show_traverse(self.dataset, q, g, check_missing)

        quality_metric_list = []
        best_overall = [] # for testing purposes -> maintain best obtained value so far
        timestamps = [] # for testing purposes -> when the pipelines finished

        print(f'result_list: \n {result_list}')

        if result_list[1]:
            

            for dic in range(len(result_list[1])):
                print(f'In QLearning: \n {result_list[1][dic]}')

                if result_list[1][dic] != None:
                    for key, val in result_list[1][dic].items():

                        if key == 'quality_metric':

                            quality_metric_list.append(val)
                        elif key == 'time':
                            timestamps.append(val)

            if g in range(0, 2):

                result = max(x for x in quality_metric_list if x is not None)  # changed from min to max

                result_l = quality_metric_list.index(result)

                result_list[0].append(goals[g])

                print("Strategy", result_list[0][result_l], 'for maximal ', # print changed from 'minimal' to 'maximal'
                      result, 'for', goals[g])

                print()

            else:

                result = min(x for x in quality_metric_list if x is not None)

                result_l = quality_metric_list.index(result)

                result_list[0].append(goals[g])

                print("Strategy", result_list[0][result_l], 'for Minimal',
                      metrics_name[g], ':', result, 'for', goals[g])

                print()

        else:

            result = None

            result_l = None
        
        best_so_far = 0
        for num in quality_metric_list: # populate the best_overall
            best_so_far = max(num, best_so_far)
            best_overall.append(best_so_far)
        for i in range(1, len(timestamps)):
            timestamps[i] += timestamps[i - 1]

        t = time.time() - start_pipexec

        print("=== End of Learn2Clean - Pipeline execution "
              "-- CPU time: %s seconds" % t)

        print()

        if result_l is not None:

            rr = (self.file_name, "Learn2Clean", goals[g], result_list[0][result_l], metrics_name[g], result, t)

        else:

            rr = (self.file_name, "Learn2Clean", goals[g], None, metrics_name[g], result, t)

        print("**** Best strategy ****")

        # Return information about the best strategy and its performance
        print(rr)
        
        with open('./save/'+str(self.file_name)+'_results.txt',
                  mode='a+') as rr_file:

            print("{}".format(rr), file=rr_file)

        best_overall.insert(0, "Best So Far")
        best_overall.insert(0, "CleanSurv")
        timestamps.insert(0, "Timestamps")
        timestamps.insert(0, "CleanSurv")
        with open('./save/'+str(self.file_name)+'_timestamps.txt', mode='a') as rr_file:
            print("{}".format(best_overall), file=rr_file)
            print("{}".format(timestamps), file=rr_file)

        

    def random_cleaning(self, dataset_name="None", loop=1):

        """
         This function generates a random cleaning strategy and executes it on the dataset.
        
         Args:
         - dataset_name: The name of the dataset being cleaned.
        
         Returns:
         - p[1]: The result of the cleaning strategy, including quality metrics.
        """

        check_missing = self.dataset.isnull().sum().sum() > 0
        rr = ""
        average = 0
        obtained_scores = []
        for repeat in range(loop):
            random.seed(time.perf_counter())

            # Check if the dataset contains missing values
            if check_missing:

                # Define methods and action list for datasets with missing values
                methods = ["-", "CCA", "MI", "Mean", "KNN", "Median", "-", "UC", "LASSO", "RFE", "IG",
                        "-", "DBID", "DBT", "ED",
                        "-", "MR", "MR", "MUO",
                        "-",  "-", "-"]
                

                rand_actions_list = [randint(1, 5), randint(6, 10), randint(11, 14),
                                    randint(15, 18), randint(19, 21)]

            else:
                # Define methods and action list for datasets without missing values
                methods = ["-", "UC", "LASSO", "RFE", "IG",
                        "-",  "DBID", "DBT", "ED",
                        "-",  "MR", "MR", "MUO",
                        "-", "-", "-"]
                

                rand_actions_list = [randint(0, 4), randint(5, 8), randint(9, 12),
                                    randint(13, 15)]

            # Define survival analysis goals and metric names
            goals = ["RSF", "COX", "NN"]

            metrics_name = ["C-Index", "C-Index", "C-Index"]

            if self.goal not in goals:
                raise ValueError("Goal invalid. Please choose between RSF, COX, NN")

            else:

                g = goals.index(self.goal)

            # Create a string representation of the random cleaning strategy
            traverse_name = methods[rand_actions_list[0]] + " -> "

            for i in range(1, len(rand_actions_list)):

                traverse_name += "%s -> " % methods[rand_actions_list[i]]

            traverse_name = re.sub('- -> ', '', traverse_name) + goals[g]

            name_list = re.sub(' -> ', ',', traverse_name).split(",")

            print()

            print()

            print("--------------------------")

            print("Random cleaning strategy:\n", traverse_name)

            print("--------------------------")

            if check_missing:

                rand_actions_list[len(rand_actions_list)-1] = g+len(methods)-6

                methods = ["CCA", "MI", "Mean", "KNN", "Median",
                                "UC", "LASSO", "RFE", "IG",
                                "DBID", "DBT", "ED",
                                "MR", "MR", "MUO"]

                new_list = []

                for i in range(len(name_list)-1):

                    m = methods.index(name_list[i])

                    new_list.append(m)

                new_list.append(g+len(methods))

            else:
                self.dataset = self.handle_categorical(self.dataset)
                rand_actions_list[len(rand_actions_list)-1] = g+len(methods)-5

                methods = ["UC", "LASSO", "RFE", "IG",
                        "DBID", "DBT", "ED",
                        "MR", "MR", "MUO"]
                new_list = []

                for i in range(len(name_list)-1):

                    m = methods.index(name_list[i])

                    new_list.append(m)

                new_list.append(g+len(methods))
            dataset_copy = self.dataset.copy()
            p = self.construct_pipeline(dataset=dataset_copy, actions_list=new_list, time_col=self.time_col, event_col=self.event_col, check_missing=check_missing)
            rr += str((dataset_name, "random", goals[g], traverse_name, metrics_name[g], "Quality Metric: ", p[0]['quality_metric'])) + "\n"
            average += p[0]['quality_metric']
            obtained_scores.append(p[0]['quality_metric'])
        mean = average / loop
        for i in  range(len(obtained_scores)):
            obtained_scores[i] -= mean
            obtained_scores[i] = obtained_scores[i] ** 2
        standard_deviation = (sum(obtained_scores) / len(obtained_scores)) ** (1.0 / 2.0)
        print(rr)
        print(f"**Average score over {loop} experiments is: {average/loop}**")
        print(f"**Standard deviation:{standard_deviation}**")
        average_score_str = f"**Average score over {loop} experiments is: {average/loop}**\n**Standard deviation:{standard_deviation}**\n\n"
        rr += average_score_str

        if p[1] is not None:

            with open('./save/'+dataset_name+'_results.txt',
                    mode='a+') as rr_file:

                print("{}".format(rr), file=rr_file)

        return p[1]
    

    def custom_pipeline(self, pipelines_file, model_name, dataset_name="None"):
        
        pipeline_counter = 0
        rr = ""

        for line in pipelines_file:
            steps = list(line.split(" "))
            goals = ["RSF", "COX", "NN"]
            metrics_name = ["C-Index", "C-Index", "C-Index"]
            methods = ["UC", "LASSO", "RFE", "IG",
                        "DBID", "DBT", "ED",
                        "MR", "MR", "MUO"]
            
            g = goals.index(model_name)
            missing = False
            for step in steps:
                if step not in methods:
                    methods = ["CCA", "MI", "Mean", "KNN", "Median",
                                "UC", "LASSO", "RFE", "IG",
                                "DBID", "DBT", "ED",
                                "MR", "MR", "MUO"]
                    missing = True
                    break 

            steps.append(model_name)
            action_list = []
            traverse_name = ""

            for i in range(len(steps) - 1):
                name = "".join(steps[i].splitlines())
                print(name)
                steps[i] = name
                traverse_name += steps[i] + " -> "
                m = methods.index(steps[i])
                action_list.append(m)

            traverse_name += model_name
            action_list.append(g+len(methods)) 
            #check_missing = self.dataset.isnull().sum().sum() > 0

            check_missing = missing

            print()

            print()

            print("--------------------------")

            print("Custom Pipeline strategy:\n", traverse_name)

            print("--------------------------")

            print(traverse_name)
            print(action_list)
            dataset_copy = self.dataset.copy()
            p = self.construct_pipeline(dataset=dataset_copy, actions_list=action_list, time_col=self.time_col, event_col=self.event_col, check_missing=check_missing)
            print(f'P IS HERE {p}')
            print()
            rr += str((dataset_name, "Custom", goals[g], traverse_name, metrics_name[g], "Quality Metric: ", p[0]['quality_metric'])) + "\n"
            pipeline_counter += 1
        print(rr)

        with open('./save/'+str(self.file_name)+'_results.txt',
                  mode='a+') as rr_file:
            print("{}".format(rr), file=rr_file)

        print(f'**{pipeline_counter} Strategies Have Been Tried**')
        return p
    

    def no_prep(self, dataset_name='None'):

        goals = ["RSF", "COX", "NN"]

        metrics_name = ["C-Index", "C-Index", "C-Index"]

        if self.goal not in goals:

            raise ValueError("Goal invalid. Please choose between RSF, COX, NN")

        else:

            g = goals.index(self.goal)

        check_missing = self.dataset.isnull().sum().sum() > 0

        if check_missing:
            self.dataset.dropna(inplace=True)
            len_m = 15

        else:
            len_m = 10
        
        self.dataset = self.handle_categorical(self.dataset)[0]

        p = self.construct_pipeline(dataset=self.dataset, actions_list=[g+len_m], time_col=self.time_col, event_col=self.event_col, check_missing=check_missing)
        rr = (dataset_name, "no-prep", goals[g], goals[g], metrics_name[g],"Quality Metric: ", p[0]['quality_metric'])

        print(f'\n\n{rr}\n\n')

        if p[1] is not None:

            with open('./save/'+dataset_name+'_results.txt',
                      mode='a') as rr_file:

                print("{}".format(rr), file=rr_file)
    

    
    def get_imputers(self):
        imputers = []
        for method in self.rewards:
            if self.rewards[method]['type'] == 'Imputer':
                imputers.append(method)
        return imputers
    

    def generate_pipeline(self, current_step, pipeline, imputers):
        pipeline.append(current_step)
        if current_step == self.goal or len(self.rewards[current_step]['followed_by']) == 0:
            pipeline.pop()
            random_imputer = imputers[random.randint(0, len(imputers) - 1)]
            pipeline.insert(0, random_imputer)
            formatted_pipeline = ""
            for method in pipeline:
                formatted_pipeline += method + " "
            formatted_pipeline = formatted_pipeline[:-1]
            res = self.custom_pipeline([formatted_pipeline], self.goal)
            return res
        else:
            next_steps = self.rewards[current_step]['followed_by']
            for next_step, reward in next_steps.items():
                if next_step not in pipeline:
                    # if len(pipeline) != 0 and self.rewards[pipeline[-1]]['type'] == self.rewards[next_step]['type']:
                    #     continue
                    self.generate_pipeline(next_step, pipeline.copy(), imputers)

    
    # def grid_search(self, dataset_name='None'):
    #     start_time = time.time()
    #     timestamps = []
    #     best_so_far = -1.0
    #     imputers = self.get_imputers()
    #     for start in self.rewards:
    #         ans = self.generate_pipeline(start, [], imputers)
    #         best_so_far = max(best_so_far, ans[0]['quality_metric'])
    #         timestamps.append((best_so_far, time.time() - start_time))
    #         time_dif = time.time() - start_time
    #         if time_dif >= 300:
    #             time_in_mins = time_dif / 60
    #             print()
    #             print(f"Time Limit of {time_in_mins} mins have been reached!")
    #             print()
    #             break
    #     else:
    #         print()
    #         print(f"Grid Search completed in {(time.time() - start_time) / 60} mins")
    #         print()
    #     with open('./save/'+dataset_name+'_results.txt', mode='a') as rr_file:
    #          print("{}".format(timestamps), file=rr_file)

    

    # def grid_search(self, dataset_name='None'):
    #     imputers, feature_selectors, duplicate_detectors, outlier_detectors = [], [], [], []

    #     for method in self.rewards:
    #         if method == "CR":
    #             continue
    #         method_type = self.rewards[method]['type']
    #         if method_type == 'Imputer':
    #             imputers.append(method)
    #         elif method_type == 'Feature_selector':
    #             feature_selectors.append(method)
    #         elif method_type == 'Duplicate_detector':
    #             duplicate_detectors.append(method)
    #         elif method_type == 'Outlier_detector':
    #             outlier_detectors.append(method)

    #     random.shuffle(imputers)
    #     start_time = time.time()
    #     results = []
    #     timestamps = []
    #     timeout = False
    #     best_so_far = 0
    #     for i in imputers:
    #         for j in feature_selectors:
    #             for k in duplicate_detectors:
    #                 for z in outlier_detectors:
    #                     string = i + " " + j + " " + k + " " + z
    #                     pipeline = [string]
    #                     res = self.custom_pipeline(pipeline, self.goal)[0]['quality_metric']
    #                     best_so_far = max(best_so_far, res)
    #                     time_dif = time.time() - start_time
    #                     results.append(best_so_far)
    #                     timestamps.append(time_dif)
    #                     if time_dif >= 300:
    #                         timeout = True
    #                         print()
    #                         print(f"Time limit for Grid Search reached in {time_dif / 60} mins")
    #                         break
    #                 if timeout:
    #                     break
    #             if timeout:
    #                 break
    #         if timeout:
    #             break
    #     else:
    #         print()
    #         print(f'Grid Search Completed in {(time.time() - start_time) / 60} mins')

    #     with open('./save/'+dataset_name+'_results.txt', mode='a') as rr_file:
    #         print("{}".format(results), file=rr_file)
    #         print("{}".format(timestamps), file=rr_file)



    def grid_search(self, dataset_name='None', trials=1):
        imputers, feature_selectors, duplicate_detectors, outlier_detectors = [], [], [], []

        for method in self.rewards:
            if method == "CR":
                continue
            method_type = self.rewards[method]['type']
            if method_type == 'Imputer':
                imputers.append(method)
            elif method_type == 'Feature_selector':
                feature_selectors.append(method)
            elif method_type == 'Duplicate_detector':
                duplicate_detectors.append(method)
            elif method_type == 'Outlier_detector':
                outlier_detectors.append(method)
        for trial in range(trials):
            random.shuffle(imputers)
            random.shuffle(feature_selectors)
            random.shuffle(duplicate_detectors)
            random.shuffle(outlier_detectors)
            all_methods = [feature_selectors, duplicate_detectors, outlier_detectors]
            random.shuffle(all_methods)
            all_methods.insert(0, imputers)
            start_time = time.time()
            results = []
            timestamps = []
            timeout = False
            best_so_far = 0
            for i in all_methods[0]:
                for j in all_methods[1]:
                    for k in all_methods[2]:
                        for z in all_methods[3]:
                            string = i + " " + j + " " + k + " " + z
                            pipeline = [string]
                            res = self.custom_pipeline(pipeline, self.goal)[0]['quality_metric']
                            best_so_far = max(best_so_far, res)
                            time_dif = time.time() - start_time
                            results.append(best_so_far)
                            timestamps.append(time_dif)
                            if time_dif >= 600:
                                timeout = True
                                print()
                                print(f"Time limit for Grid Search reached in {time_dif / 60} mins")
                                break
                        if timeout:
                            break
                    if timeout:
                        break
                if timeout:
                    break
            else:
                print()
                print(f'Grid Search Completed in {(time.time() - start_time) / 60} mins')
            results.insert(0, "Best So Far")
            results.insert(0, "Grid_Search")
            timestamps.insert(0, "Timestamps")
            timestamps.insert(0, "Grid_Search")

            with open('./save/'+dataset_name+'_timestamps.txt', mode='a') as rr_file:
                print("{}".format(results), file=rr_file)
                print("{}".format(timestamps), file=rr_file)


            

        