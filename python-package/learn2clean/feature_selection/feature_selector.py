import warnings
import time
import numpy as np
import pandas as pd


class Feature_selector():
    """
    Select the features for the train dataset using a
    particular strategy and keep the same features in the test dataset

    Parameters
    ----------
    * dataset: input dataset dict
        including dataset['train'] pandas DataFrame, dataset['test']
        pandas DataFrame and dataset['target'] pandas DataSeries
        obtained from train_test_split function of Reader class

    * strategy: str, default = 'LC'
        The choice for the feature selection strategy:
            - 'MR', 'VAR and 'LC' are agnostic to the task
            - 'Tree', 'WR', 'SVC' are used for classification task
            -  'L1', 'IMP' are used  for regression task
            Available strategies=
            'MR': using a default threshold on the missing ratio per variable,
            i.e., variables with 20% (by default) and more missing values
            are removed
            'LC': detects pairs of linearly correlated variables and remove one
            'VAR': uses threshold on the variance
            'Tree': uses decision tree classification as model for feature
                selection given the target set for classification task
                'SVC': uses linear SVC as model for feature selection given
                 the target set for classification task
            'WR': uses the selectKbest (k=10) and Chi2 for feature selection
                given the target set for classification task
            'L1': uses Lasso L1 for feature selection given the target set for
                regression task
            'IMP': uses Random Forest regression for feature selection given
                the target set for regression task

    * exclude: str, default = 'None' name of variable to be excluded from
        feature selection

    * threshold: float, default = '0.3' only for MR, VAR, LC, L1, and IMP

    * verbose: Boolean,  default = 'False' otherwise display information
    about the applied feature selection
    """

    def __init__(self, dataset, time_col=None, event_col=None,  strategy='LC', exclude=None,
                 threshold=0.3, config=None, verbose=False):

        self.dataset = dataset

        self.time_column = time_col

        self.event_column = event_col


        self.config = config

        self.strategy = strategy

        self.exclude = exclude

        self.threshold = threshold

        self.verbose = verbose

    def get_params(self, deep=True):

        return {'strategy': self.strategy,

                'exclude': self.exclude,

                'threshold':  self.threshold,

                'verbose': self.verbose

                }

    def set_params(self, **params):

        for k, v in params.items():

            if k not in self.get_params():

                warnings.warn("Invalid parameter(s) for normalizer. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`feature_selector.get_params().keys()`")

            else:

                setattr(self, k, v)
    

    def univariate_coxph_selection(self):
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        import numpy as np
        import pandas as pd

        print(">>Feature Selection started with UC method..... ")

        # Drop rows with NaN values
        x = self.dataset.dropna()

        # Ensure the time and event columns are of correct data types
        x[self.time_column] = x[self.time_column].astype(float)
        x[self.event_column] = x[self.event_column].astype(bool)

        print(x)

        # Encode event column
        event_labels = LabelEncoder().fit_transform(x[self.event_column])
        y = np.array(list(zip(event_labels, x[self.time_column])), dtype=[('event', '?'), ('time', '<f8')])

        # Standardize feature columns
        feature_columns = x.columns.difference([self.event_column, self.time_column])
        x[feature_columns] = StandardScaler().fit_transform(x[feature_columns])

        # Perform feature selection using Univariate Cox Proportional Hazards Model
        cph = CoxPHSurvivalAnalysis(alpha=0.1, verbose=10)
        print(y)

        try:
            cph.fit(x[feature_columns], y)
        except ValueError as e:
            print("Error during fitting:", e)
            return None

        # Get the magnitudes of estimated coefficients as feature importance scores
        coefs = cph.coef_
        feature_importances = np.abs(coefs)

        # Select the top features based on importance scores
        k = int(len(feature_importances) * 0.8)  # Select top 80% features
        top_features_indices = np.argsort(feature_importances)[-k:]

        selected_features = np.zeros(len(self.dataset.columns), dtype=bool)

        # Ensure that event and time columns are not removed
        event_column_index = x.columns.get_loc(self.event_column)
        time_column_index = x.columns.get_loc(self.time_column)

        if event_column_index not in top_features_indices:
            top_features_indices = np.append(top_features_indices, event_column_index)

        if time_column_index not in top_features_indices:
            top_features_indices = np.append(top_features_indices, time_column_index)

        # Ensure all indices are within bounds
        top_features_indices = [idx for idx in top_features_indices if idx < len(selected_features)]

        selected_features[top_features_indices] = True

        selected_columns = []

        # Print the selected features
        print("UC Selected Features:")
        for i, feature in enumerate(x.columns):
            if selected_features[i]:
                selected_columns.append(feature)
                if feature != self.event_column and feature != self.time_column:
                    print(feature)

        new_dataset = x[selected_columns]
        return new_dataset

    

    def lasso_selection(self):

        from sklearn.preprocessing import LabelEncoder
        from sklearn.linear_model import LassoCV

        print(">>Feature Selection started with lasso method..... ")

        x = self.dataset.dropna() #TODO I have to include dropna() as NaNs are not allowed and lasso can be a starting state

        # Encode the event column in the target array
        event_labels1 = LabelEncoder().fit_transform(x[self.event_column].astype(bool))
        y = np.array(list(zip(event_labels1, x[self.time_column])), dtype=[('event', '?'), ('time', '<f8')])
        event_labels = LabelEncoder().fit_transform(y["event"])

        # Fit LassoCV with encoded event column and time column
        # warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lasso = LassoCV(cv=5)
        lasso.fit(x, event_labels)
        # Return the selected features based on non-zero coefficient values
        selected_features = lasso.coef_ != 0

        event_column_index = x.columns.get_loc(self.event_column)
        time_column_index = x.columns.get_loc(self.time_column)
        selected_features[event_column_index] = True # Ensuring that event column excluded
        selected_features[time_column_index] = True # Ensuring that time column excluded

        print("LASSO Selected Features:")
        
        selected_columns = []

        for feature, selected in zip(x, selected_features):
            if selected:
                selected_columns.append(feature)
                print(feature)
        
        return x[selected_columns]
    

    def rfe_selection(self):

        from sklearn.preprocessing import LabelEncoder
        from sklearn.feature_selection import RFECV
        from sklearn.linear_model import LassoCV
        from sklearn.exceptions import ConvergenceWarning

        print(">>Feature Selection started with RFE method..... ")
        
        x = self.dataset.dropna() #TODO I have to include dropna() as NaNs are not allowed and rfe can be a starting state

        event_labels = LabelEncoder().fit_transform(x[self.event_column].astype(bool))
        
        y = np.array(list(zip(event_labels, x[self.time_column])), dtype=[('event', '?'), ('time', '<f8')])

        # Encode the event column in the target array
        Endcoded_event_labels = LabelEncoder().fit_transform(y["event"])

        # Perform feature selection using Recursive Feature Elimination (RFE)

        # if not self.verbose:
        #     warnings.filterwarnings("ignore", category=ConvergenceWarning)

        rfecv = RFECV(estimator=LassoCV(cv=5, max_iter=10000), step=1, scoring='neg_mean_squared_error')
        # rfecv = RFECV(estimator=LassoCV(cv=5, max_iter=10000), step=1, scoring='accuracy')
        rfecv.fit(x, Endcoded_event_labels)

        selected_features = rfecv.support_

        event_column_index = x.columns.get_loc(self.event_column)
        time_column_index = x.columns.get_loc(self.time_column)
        selected_features[event_column_index] = True # Ensuring that event column excluded
        selected_features[time_column_index] = True # Ensuring that time column excluded

        print("RFE Selected Features:")

        selected_columns = []

        for feature, selected in zip(x, selected_features):
            if selected:
                selected_columns.append(feature)
                print(feature)
    
        return x[selected_columns]
    

    def information_gain_selection(self):

        from sklearn.preprocessing import LabelEncoder
        from sklearn.feature_selection import SelectKBest, f_regression

        print(">>Feature Selection started with information gain method..... ")

        x = self.dataset.dropna()#TODO I have to include dropna() as NaNs are not allowed and IG can be a starting state

        event_labels1 = LabelEncoder().fit_transform(x[self.event_column].astype(bool))
       
        time_values = x[self.time_column]

        y = np.array(list(zip(event_labels1, time_values)), dtype=[('event', '?'), ('time', '<f8')])

        # Encode the event column in the target array
        event_labels = LabelEncoder().fit_transform(y["event"])

        # Perform feature selection using Information Gain
        #warnings.filterwarnings("ignore", category=ConvergenceWarning)

        k_best = SelectKBest(score_func=f_regression, k='all')  # changing from k = 'all' to k = 3
        k_best.fit(x, event_labels)

        selected_features = k_best.get_support()

        event_column_index = x.columns.get_loc(self.event_column)
        time_column_index = x.columns.get_loc(self.time_column)
        selected_features[event_column_index] = True # Ensuring that event column excluded
        selected_features[time_column_index] = True # Ensuring that time column excluded


        print("Information Gain Selected Features:")

        selected_columns = []

        for feature, selected in zip(x, selected_features):
            if selected:
                selected_columns.append(feature)
                print(feature)

        return x[selected_columns]
        

    def transform(self):

        start_time = time.time()
        
        if self.dataset is None:
            return self.dataset

        print()

        print(">>Feature selection ")

        if self.strategy == 'UC':
            selected = self.univariate_coxph_selection()
        
        elif self.strategy == 'LASSO':
            selected = self.lasso_selection()
        
        elif self.strategy == 'RFE':
            selected = self.rfe_selection()
        
        elif self.strategy == 'IG':
            selected = self.information_gain_selection()
        
        else:
            raise ValueError("Invalid feature selection strategy. Please choose from 'UC', 'LASSO', 'RFE', or 'IG'.")
        
        print("Feature selection done -- CPU time: %s seconds" %
            (time.time() - start_time))
        
        return selected
            
        
