import warnings
import time
import numpy as np


class Outlier_detector():
    """
    Identify and remove outliers using a particular strategy

    Parameters
    ----------
    * dataset: input dataset dict
        including dataset['train'] pandas DataFrame, dataset['test']
        pandas DataFrame and dataset['target'] pandas DataSeries
        obtained from train_test_split function of Reader class

    * threshold: float, default = '0.3' for any outlying value in a row"
        or a value in [0,1] for multivariate "
        outlying row. For example, with threshold=0.5
        if a row has outlying values in half of the attribute set and more,
        it is considered as an outlier and removed"

    * strategy: str, default = 'ZSB'
        The choice for outlier detection and removal strategy:
            - 'ZSB', 'IQR and 'LOF' for numerical values
            Available strategies =
            'ZS': detects outliers using the robust Zscore as a function
            of median and median absolute deviation (MAD)
            'IQR': detects outliers using Q1 and Q3 +/- 1.5*InterQuartile Range
            'LOF': detects outliers using Local Outlier Factor

    * verbose: Boolean,  default = 'False' otherwise display
        about outlier detected and removed

    * exclude: str, default = 'None' name of variable to be
        excluded from outlier detection
    """

    def __init__(self, dataset, strategy='ZSB', threshold=0.3, time_col=None,
                 event_col=None, config=None, verbose=False, exclude=None): # mode determines original function or survival analysis

        self.dataset = dataset

        self.strategy = strategy

        self.threshold = threshold

        self.time_column = time_col

        self.event_column = event_col

        self.config = config

        self.verbose = verbose

        self.exclude = exclude  # to implement

    def get_params(self, deep=True):

        return {'strategy': self.strategy,

                'threshold': self.threshold,

                'verbose': self.verbose,

                'exclude': self.exclude

                }

    def set_params(self, **params):

        for k, v in params.items():

            if k not in self.get_params():

                warnings.warn("Invalid parameter(s) for normalizer. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`outlier_detector.get_params().keys()`")

            else:

                setattr(self, k, v)
    

    def survival_analysis_with_fdr_control(self): # TODO Fix this method from NaNs and return correct data

        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
        from statsmodels.stats import multitest

        #self.dataset.dropna(inplace=True)

        # Split the data into groups for comparison (e.g., treatment vs. control)
        groups = self.dataset[self.event_column].unique()

        # Initialize Kaplan-Meier estimator
        kmf = KaplanMeierFitter()

        # Create a dictionary to store results for each group
        results_dict = {}

        for group in groups:
            # Select data for the current group
            group_data = self.dataset[self.dataset[self.event_column] == group]

            # Fit the Kaplan-Meier survival curve for the current group
            kmf.fit(group_data[self.time_column], event_observed=group_data[self.event_column])

            # Perform log-rank test for the current group
            results = logrank_test(group_data[self.time_column], group_data[self.time_column])

            # Store the results for the current group
            results_dict[group] = results

        # Correct p-values for multiple testing using the Benjamini-Hochberg procedure
        p_values = [result.p_value for result in results_dict.values()]
        _, adjusted_p_values, _, _ = multitest.multipletests(p_values)

        # Threshold for controlling FDR
        fdr_threshold = 0.05  # Adjust as needed

        # Identify outliers based on adjusted p-values
        outliers = [group for group, adj_p in zip(groups, adjusted_p_values) if adj_p < fdr_threshold]

        print(outliers)

        # Number of detected outliers
        num_outliers = len(outliers)

        # Number of remaining rows
        num_remaining_rows = len(self.dataset) - num_outliers

        # Print results
        print("Number of Detected Outliers:", num_outliers)
        print("Number of Remaining Rows:", num_remaining_rows)
        return self.dataset

    def martingale_residuals(self): # TODO Fix this method by actually discovering outliers correctly

        from lifelines import KaplanMeierFitter

        x = self.dataset #.dropna()
        kmf = KaplanMeierFitter()
        kmf.fit(x[self.time_column], event_observed=x[self.event_column])
        
        observed_events = x[self.event_column]
        expected_events = kmf.event_table['observed'][:-1]
        
        x['martingale_residuals'] = observed_events - expected_events
        
        # Calculate outliers based on a threshold (e.g., absolute value > 1)
        outlier_threshold = 1
        x['is_outlier'] = abs(x['martingale_residuals']) > outlier_threshold
        
        num_dropped_outliers = x['is_outlier'].sum()
        num_remaining_rows = len(x) - num_dropped_outliers
        
        print("Number of Dropped Outliers (Martingale Residuals):", num_dropped_outliers)
        print("Number of Remaining Rows (Martingale Residuals):", num_remaining_rows)
        
        # Drop outliers from the dataset
        data = x[~x['is_outlier']]

        data.drop(["is_outlier", "martingale_residuals"], axis=1, inplace=True)
        
        return data
    

    def multivariate_outliers(self):

        from sklearn.covariance import EllipticEnvelope

        dataset = self.dataset #.dropna()
        print(dataset)
        if dataset is None: # fail-safe procedure
            return dataset
        
        envelope = EllipticEnvelope()

        needed_values = dataset[[self.time_column, self.event_column]]
        temp = envelope.fit_predict(needed_values)
        dataset['multivariate_outliers'] = temp
        
        num_dropped_outliers = dataset[dataset['multivariate_outliers'] != 1].shape[0]
        num_remaining_rows = len(dataset) - num_dropped_outliers
        
        print("Number of Dropped Outliers (Multivariate Outliers):", num_dropped_outliers)
        print("Number of Remaining Rows (Multivariate Outliers):", num_remaining_rows)
        
        # Drop outliers from the dataset
        dataset = dataset[dataset['multivariate_outliers'] == 1]

        dataset.drop(['multivariate_outliers'], axis=1, inplace=True)

        return dataset
    

    def transform(self):

        start_time = time.time()

        if self.dataset is None:
            return self.dataset

        print()

        print(">>Outlier detection and removal:")

        if (self.strategy == "CR"):
            dn = self.survival_analysis_with_fdr_control()

        elif(self.strategy == "MR"):
            dn = self.martingale_residuals()

        elif(self.strategy == "MUO"):
            dn = self.multivariate_outliers()

        else:
            raise ValueError("Strategy invalid."
                            "Please choose between "
                            "'CR', 'MR' or 'MUO'")
            
        print("Outlier detection and removal done --  CPU time: %s seconds" % (time.time() - start_time))

        print()

        return dn
