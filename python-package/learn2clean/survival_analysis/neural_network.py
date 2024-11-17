import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from .Deepsurv.deep_surv import DeepSurv

class NeuralNetwork:

    def __init__(self, dataset, time_column, target_goal, config=None, verbose=False):
        self.dataset = dataset
        self.time_column = time_column
        self.event_column = target_goal
        self.config = config
        self.verbose = verbose


    def get_config_dict(self, function_name):
        config_dict = {}
        if self.config is not None:
            if function_name in self.config.keys():
                config_dict = self.config[function_name]
        return config_dict
    

    def fit(self):
        current_method = sys._getframe().f_code.co_name
        hyperparameters = self.get_config_dict(current_method)

        n_in = len(self.dataset.columns)
    
        learning_rate = hyperparameters.get("learning_rate", 0.001)
        hidden_layers_sizes = hyperparameters.get("hidden_layers_sizes", None)
        lr_decay = hyperparameters.get("lr_decay", 0.0)
        momentum = hyperparameters.get("momentum", 0.9)
        L2_reg = hyperparameters.get("L2_reg", 0.0)
        L1_reg = hyperparameters.get("L1_reg", 0.0)
        activation = hyperparameters.get("activation", "rectify")
        dropout = hyperparameters.get("dropout", None)
        batch_norm = hyperparameters.get("batch_norm", False)
        standardize = hyperparameters.get("standardize", False)

        nn = DeepSurv(n_in=n_in, learning_rate=learning_rate, hidden_layers_sizes=hidden_layers_sizes,
                      lr_decay=lr_decay, momentum=momentum, L2_reg=L2_reg, L1_reg=L1_reg,
                      activation=activation, dropout=dropout, batch_norm=batch_norm, standardize=standardize)
        
        n_epochs = hyperparameters.get("n_epochs", 500)
        validation_frequency = hyperparameters.get("validation_frequency", 250)
        patience = hyperparameters.get("patience", 2000)
        improvement_threshold = hyperparameters.get("improvement_threshold", 0.99999)
        patience_increase = hyperparameters.get("patience_increase", 2)
        logger = hyperparameters.get("logger", None)
        verbose = hyperparameters.get("verbose", True)

        x = self.dataset
        x[self.event_column] = x[self.event_column].astype(bool)
        y = x[[self.time_column, self.event_column]]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        observations = x_train.drop([self.event_column, self.time_column], axis=1).values
        times = x_train[self.time_column].values
        events = x_train[self.event_column].values

        df = {'x': observations, 't': times, 'e': events}

        log = nn.train(df, n_epochs=n_epochs, validation_frequency=validation_frequency, patience=patience,
                       improvement_threshold=improvement_threshold, patience_increase=patience_increase,
                       logger=logger, verbose=verbose)
        

        observations = x_test.drop([self.event_column, self.time_column], axis=1).values
        times = x_test[self.time_column].values
        events = x_test[self.event_column].values

        c_index = nn.get_concordance_index(observations, times, events)

        print(f"INSIDE DEEP SURV -------> {c_index}")
        
        return c_index
        

        
