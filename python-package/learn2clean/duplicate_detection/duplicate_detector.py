import time
import warnings
import numpy as np
import jellyfish as jf
import pandas as pd
pd.options.mode.chained_assignment = None


def add_key_reindex(dataset, rand=False):

    if rand:

        dataset = dataset.reindex(np.random.permutation(dataset.index))

    dataset['New_ID'] = range(1, 1+len(dataset))

    dataset['New_ID'].apply(str)

    return(dataset)


class Duplicate_detector():
    """
    Remove the duplicate records from the dataset

    Parameters
    ----------
    * dataset: input dataset dict
        including dataset['train'] pandas DataFrame, dataset['test']
        pandas DataFrame and dataset['target'] pandas DataSeries
        obtained from train_test_split function of Reader class

    * threshold: float, default = '0.6' only for 'AD' strategy

    * strategy: str, default = 'ED'
        The choice for the deduplication strategy : 'ED', 'AD' or 'METRIC'
        Available strategies =
        'ED':  exact duplicate detection/removal or
        'AD':  for aproximate duplicate records detection and removal
        based on Jaccard similarity or
        'METRIC': using a particular distance specificied in 'metric':
                'DL' (by default) for  Damerau Levenshtein Distance
                'LM for Levenshtein Distance or
                'JW' for Jaro-Winkler Distance

    * metric: str, default = 'DL'  only used for 'AD' strategy

    * verbose: Boolean,  default = 'False' otherwise display the list of
        duplicate rows that have been removed

    * exclude: str, default = 'None' name of variable to
        be excluded from deduplication
    """

    def __init__(self, dataset, strategy='ED', threshold=0.6, time_col=None,
                 event_col=None, metric='DL', config=None,  verbose=False, exclude=None):
         
        self.dataset = dataset

        self.strategy = strategy

        self.threshold = threshold

        self.time_column = time_col

        self.event_column = event_col

        self.config = config

        self.metric = metric

        self.verbose = verbose

        self.exclude = exclude  # to implement

    def get_params(self, deep=True):

        return {'strategy': self.strategy,

                'metric': self.metric,

                'threshold':  self.threshold,

                'verbose': self.verbose,

                'exclude': self.exclude

                }

    def set_params(self, **params):

        for k, v in params.items():

            if k not in self.get_params():

                warnings.warn("Invalid parameter(s) for normalizer. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`duplicate_detector.get_params().keys()`")

            else:

                setattr(self, k, v)
    
    

    # Function for generating unique event IDs
    def generate_event_ids(self):
        # Combine selected columns to create a unique event identifier
        data = self.dataset #.copy()
        data['event_id'] = data[self.time_column].astype(str) + data[self.event_column].astype(str)
        return data
    

    # Function for Unique Event Identifier-Based Deduplication
    def deduplicate_by_event_id(self, df_with_event_ids):
        dataset = df_with_event_ids

        initial_rows = dataset.shape[0]  # Get the initial number of rows

        df_deduplicated = dataset.drop_duplicates(subset=["event_id"])

        final_rows = df_deduplicated.shape[0]  # Get the final number of rows after deduplication

        num_duplicates = initial_rows - final_rows  # Calculate the number of duplicate rows removed

        print(f"Number of Duplicate Rows identified by event id: {num_duplicates}")

        print(f"Number of Rows After Deduplication by event id: {final_rows}")

        print(df_deduplicated)

        # Remove the 'event_id' column from df_deduplicated
        df_deduplicated.drop(["event_id"], axis=1, inplace=True)

        return df_deduplicated
    

    # Function for Timestamp-Based Deduplication
    def deduplicate_by_timestamp(self):

        dataset = self.dataset

        initial_rows = dataset.shape[0]  # Get the initial number of rows

        new_col = "timestamp"

        dataset[new_col] = pd.to_datetime(dataset[self.time_column])

        df_deduplicated = dataset.sort_values(by=self.time_column).drop_duplicates(subset=new_col)

        final_rows = df_deduplicated.shape[0]  # Get the final number of rows after deduplication

        num_duplicates = initial_rows - final_rows  # Calculate the number of duplicate rows removed

        print(f"Number of Duplicate Rows identified by time: {num_duplicates}")
        print(f"Number of Rows After Deduplication identified by time of event: {final_rows}")

        dataset.drop([new_col], axis=1, inplace=True)
        df_deduplicated.drop([new_col], axis=1, inplace=True) # no need for the column any more because it causes problems

        return df_deduplicated
    

    def Exact_duplicate_removal(self):
        
        dataset = self.dataset

        if len(dataset) > 0:

            df = dataset.drop_duplicates()

            print('Initial number of rows:', len(dataset))

            print('After deduplication: Number of rows:', len(df))

        else:
            df = dataset
            print("No duplicate detection, empty dataframe")

        return df
    

    def transform(self):

        start_time = time.time()

        if self.dataset is None:
            return self.dataset

        print()

        print(">>Duplicate detection and removal:")

        if (self.strategy == "DBID"):
            print (" started using event ID based deduplication .....")
            event_id = self.generate_event_ids()
            print(event_id)
            dn = self.deduplicate_by_event_id(event_id)

        elif (self.strategy == 'DBT'):
            print(" started using timestamp-based deduplication .....")
            dn = self.deduplicate_by_timestamp()

        elif (self.strategy == "ED"):
            print(" started using exact duplicate removal mehtod .....")
            dn = self.Exact_duplicate_removal()

        else:
            raise ValueError("Strategy invalid."
                            "Please choose between "
                            "'DBID', 'DBT' or 'ED'")
        
        print("Deduplication done -- CPU time: %s seconds" %
            (time.time() - start_time))
        print()

        return dn
