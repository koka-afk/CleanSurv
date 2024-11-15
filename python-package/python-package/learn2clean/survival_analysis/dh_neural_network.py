import numpy as np
import tensorflow as tf
import random
import os
import sys
import pandas as pd

from .Deephit.class_DeepHit import Model_DeepHit
from .Deephit.utils_network import *
from .Deephit.get_main import get_valid_performance

class NeuralNetwork:
    def __init__(self, dataset, time_column, target_goal, config=None, verbose=False):
        self.dataset = dataset
        self.time_column = time_column
        self.event_column = target_goal
        self.config = config
        self.verbose = verbose
    

    # Example function to prepare your dataset
    def prepare_dataset_single_event(self):
        dataset = self.dataset

        time = dataset[self.time_column].values  # Event times
        label = dataset[self.event_column].values  # 0 for censored, 1 for event
        # temp = []
        # for i in range(len(label)):
        #     temp.append(1)
        
        # label = np.concatenate((label, temp))
        dataset.drop([self.event_column, self.time_column], axis=1, inplace=True) # get rid of the time & event columns
        data = dataset[dataset.columns].values  # Features
        
        
        num_time_intervals = 100  # Example number of time intervals
        
        num_samples = len(label)
        
        # Create masks (single event type)
        mask1 = np.random.rand(num_samples, 1, num_time_intervals)  # Mask for single event type
        mask2 = np.random.rand(num_samples, num_time_intervals)  # Overall survival mask
        
        return (data, time, label), (mask1, mask2)
    

    def get_config_dict(self, function_name):
        config_dict = {}
        if self.config is not None:
            if function_name in self.config.keys():
                config_dict = self.config[function_name]
        return config_dict
    

    def fit_dh(self):
        # Prepare dataset
        DATA, MASK = self.prepare_dataset_single_event()

        current_method = sys._getframe().f_code.co_name # method name
        hyperparameters = self.get_config_dict(current_method)

        # Define default hyperparameters
        default_settings = {
            'mb_size': 64,
            'iteration': 10000,
            'keep_prob': 0.8,
            'lr_train': 0.001,
            'alpha': 0.2,
            'beta': 0.5,
            'gamma': 0.1,
            'h_dim_shared': 100,
            'num_layers_shared': 2,
            'h_dim_CS': 50,
            'num_layers_CS': 2,
            'active_fn': 'relu',
            'out_path': './model_output'
        }

        in_parser = {}
        for key in default_settings.keys():
            in_parser[key] = hyperparameters.get(key, default_settings[key])

        # Other parameters
        out_itr = 1
        eval_time = [12, 24, 36]
        MAX_VALUE = -99
        OUT_ITERATION = 5
        seed = 1234

        out_itr = hyperparameters.get('out_itr', out_itr)
        eval_time = hyperparameters.get('eval_time', eval_time)
        MAX_VALUE = hyperparameters.get('MAX_VALUE', MAX_VALUE)
        OUT_ITERATION = hyperparameters.get('OUT_ITERATION', OUT_ITERATION)
        seed = hyperparameters.get('seed', seed)

        # Train and validate the model
        max_valid = get_valid_performance(DATA, MASK, in_parser, out_itr, eval_time, MAX_VALUE, OUT_ITERATION, seed)

        print(f"\n\n Maximum validation performance: {max_valid} \n\n")

        return max_valid

