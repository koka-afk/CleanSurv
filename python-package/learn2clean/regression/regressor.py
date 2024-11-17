
import time
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from scipy.stats import skew
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import numpy as np
np.seterr(divide='ignore', invalid='ignore')


def LT_log_transform_skew_features(dataset):

    numeric_feats = dataset.dtypes[dataset.dtypes != "object"].index

    Y = dataset.select_dtypes(['object'])

    skewed_feats = dataset[numeric_feats].apply(
        lambda x: skew(x.dropna()))  # compute skewness

    skewed_feats = skewed_feats[skewed_feats >= 0.75]

    skewed_feats = skewed_feats.index

    dataset[skewed_feats] = np.log1p(dataset[skewed_feats])

    return dataset[skewed_feats].join(Y)


class Regressor():
    """
    Regression task
    ----------
    * dataset: input dataset dict
        including dataset['train'] pandas DataFrame, dataset['test']
        pandas DataFrame and dataset['target'] pandas DataSeries
        obtained from train_test_split function of Reader class

    * strategy: str, default = 'MARS'
        The choice for the regression method:
            - 'MARS, 'LASSO or 'OLS'

   * target: str, name of the target numerical variable from  dataset['target']
       pandas DataSeries

   * k_folds: int, default = 10, number of folds for cross-validation

   * verbose: Boolean,  default = 'False' otherwise display the list of
       duplicate rows that have been removed
   """

    def __init__(self, dataset, target, strategy='LASSO',
                 k_folds=10, verbose=False):

        self.dataset = dataset

        self.target = target

        self.strategy = strategy

        self.k_folds = k_folds

        self.verbose = verbose

    def get_params(self, deep=True):

        return {'strategy': self.strategy,

                'target': self.target,

                'k_folds': self.k_folds,

                'verbose': self.verbose}

    def set_params(self, **params):

        for k, v in params.items():

            if k not in self.get_params():

                warnings.warn("Invalid parameter(s) for clusterer. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`regressor.get_params().keys()`")

            else:

                setattr(self, k, v)

    def OLS_regression(self, dataset, target):  # quality metrics : accuracy
        # Split the dataset into training and test sets
        X = dataset.select_dtypes(['number']).dropna()
        y = dataset[target].loc[X.index]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if (len(X_train.columns) <= 1) or (len(X_train) < self.k_folds):
            print('Error: Need at least one continuous variable and', self.k_folds, 'observations for regression')
            mse = None
        else:
            X1Train = sm.add_constant(X_train)
            reg = sm.OLS(y_train, X1Train)
            resReg = reg.fit()

            X1Test = sm.add_constant(X_test)
            ypReg = reg.predict(resReg.params, X1Test)

            if self.verbose:
                print(resReg.summary())

            mse = mean_squared_error(y_test, ypReg)
            print("MSE of OLS with", self.k_folds, "folds for cross-validation:", mse)

        return mse

    def LASSO_regression(self, dataset, target):  # quality metrics : accuracy
        # Split the dataset into training and test sets
        X = dataset.select_dtypes(['number']).dropna()
        y = dataset[target].loc[X.index]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if (len(X_train.columns) <= 1) or (len(X_train) < self.k_folds):
            print('Error: Need at least one continuous variable and', self.k_folds, 'observations for regression')
            mse = None
        else:
            my_alphas = np.array([0.001, 0.01, 0.02, 0.025, 0.05, 0.1, 0.25, 0.5, 0.8, 1.0, 1.2])
            lcv = LassoCV(alphas=my_alphas, normalize=False, fit_intercept=False, random_state=0, cv=self.k_folds, tol=0.0001)

            lcv.fit(X_train, y_train)

            if self.verbose:
                print("MSE values of cross validation")
                print(lcv.mse_path_)

            avg_mse = np.mean(lcv.mse_path_, axis=1)
            if self.verbose:
                print("alphas vs. MSE in cross-validation")
                print(pd.DataFrame({'alpha': lcv.alphas_, 'MSE': avg_mse}))

            print("Best alpha =", lcv.alpha_)

            ypLasso = lcv.predict(X_test)
            mse = mean_squared_error(y_test, ypLasso)
            print("MSE of LASSO with", self.k_folds, "folds for cross-validation:", mse)

        return mse

    def MARS_regression(self, dataset, target, k_folds=5, verbose=False):
        # Split the dataset into training and test sets
        X = dataset.select_dtypes(['number']).dropna()
        y = dataset[target].loc[X.index]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if len(X_train.columns) <= 1 or len(X_train) < k_folds:
            print('Error: Need at least one continuous variable and', k_folds, 'observations for regression')
            return None

        model = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=32, learning_rate=0.1, min_samples_leaf=20)

        model.fit(X_train, y_train)

        def rmse_cv(model):
            rmse = np.sqrt(-cross_val_score(model, X_train, np.log1p(y_train), scoring="neg_mean_squared_error", cv=k_folds))
            return rmse

        cv_mars = rmse_cv(model).mean()

        if verbose:
            print("Model coefficients and performance:")
            print(model)

        print("MSE of MARS with", k_folds, "folds for cross-validation:", cv_mars)

        return cv_mars

    def transform(self):

        start_time = time.time()

        d = self.dataset

        if d is None:
            return 10000

        if self.target in d.columns:

            print()

            print(">>Regression task")

            if (self.strategy == "OLS"):

                dn = self.OLS_regression(d, self.target)

            elif (self.strategy == "LASSO"):

                dn = self.LASSO_regression(d, self.target)

            elif (self.strategy == "MARS"):
                
                dn = self.MARS_regression(d, self.target)

            else:

                raise ValueError(
                    "The regression function should be OLS, LASSO, or MARS")

            print("Regression done -- CPU time: %s seconds" %

                  (time.time() - start_time))

        else:

            raise ValueError("Target variable invalid.")

        return {'quality_metric': dn}
