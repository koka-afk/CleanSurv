# CleanSurvival

**CleanSurvival** is a Python-based tool that automatically finds the optimal data cleaning pipeline for survival analysis models using Q-Learning.

## Q-Learning

**Q-Learning** is a reinforcement learning algorithm used to find the optimal sequence of data cleaning methods. The tool uses a reward matrix to guide the Q-Learning agent and find the pipeline that maximizes the performance of the chosen survival analysis model.


 The tool supports three different survival analysis models:

- **Cox PH Model**
- **Random Survival Forest**
- **DeepHit Neural Network**

## Imputation Methods

1. **Complete Case Analysis (CCA)**: Removes rows with missing values.
2. **Multiple Imputation using Chained Equations (MICE)**: Uses an iterative imputer to fill in missing values.
3. **Simple Mean Imputation**: Replaces missing values with the mean of the corresponding column.
4. **Simple Median Imputation**: Replaces missing values with the median of the corresponding column.
5. **KNN Imputation**: Uses K-Nearest Neighbors to impute missing values.

## Feature Selection Methods

1. **Univariate CoxPH Selection (UC)**: Selects features based on their importance scores from a univariate Cox Proportional Hazards model.
2. **Lasso Selection**: Uses Lasso regularization to select features.
3. **Recursive Feature Elimination (RFE)**: Recursively removes features based on their importance.
4. **Information Gain Selection (IG)**: Selects features based on their information gain.

## Deduplication Methods

1. **Exact Duplicate Removal (ED)**: Removes exact duplicate rows.
2. **Deduplication by Event ID (DBID)**: Removes duplicate rows based on a unique event identifier.
3. **Deduplication by Timestamp (DBT)**: Removes duplicate rows based on timestamps.

## Outlier Detection Methods

1. **Martingale Residuals (MR)**: Uses martingale residuals to detect outliers.
2. **Multivariate Outliers (MUO)**: Detects outliers using the EllipticEnvelope method.

## Different Modes
1. **CleanSurvival**: This mode utilizes the Q-Learning algorithm to determine the optimal data cleaning pipeline for your chosen survival analysis model.
2. **Random Cleaning**: This mode generates random cleaning pipelines and evaluates their performance. You can specify the number of random experiments to run.
3. **Custom Pipeline Design**: This mode allows you to define your own custom cleaning pipelines and evaluate their performance. You need to provide a text file containing the pipelines.
4. **No Preparation**: This mode applies no data cleaning and directly executes the chosen survival analysis model.

## Flexibility and Configuration
CleanSurvival offers flexibility in configuring its behavior and parameters. Two main configuration files are used:

1. **config.json**: This file allows you to tune the hyperparameters of the various data cleaning and machine learning methods used in the tool.

   - You can modify settings for methods like multiple imputation, KNN imputer, and neural networks.
   - This enables you to fine-tune the tool's performance for different datasets and scenarios.
2. **reward.json**: This file defines the reward matrix used in the Q-Learning algorithm.
   
   - You can adjust the rewards associated with different data cleaning actions and their combinations.
   - This allows you to influence the Q-Learning agent's exploration and learning process.


In addition to these configuration files, CleanSurvival supports importing different reward definitions. You can provide your own reward matrix or modify the existing one to suit your specific needs.

This flexibility allows you to customize CleanSurvival's behavior and adapt it to various data cleaning challenges and survival analysis tasks.


## How to Use

CleanSurvival can be used in two ways:

1. **Using app.py in an editor:**

    - Open app.py in your preferred Python editor.
    - Modify the `path` variable to point to your dataset.
    - Modify the `json_path` variable to point to your reward configuration file (optional).
    - Run the script.
    - Follow the prompts to select a survival analysis model, configure the Q-Learning graph, and choose a cleaning algorithm.

2. **Using run.py in a terminal:**

   - Open a terminal and navigate to the CleanSurvival directory.
   - Run the script with the following arguments:
   ```bash
   python run.py -d <dataset_path> -r <rewards_path> -md <model> -lm <load_mode> -lf <load_file> -a <algo> -ao <algo_op>
   ```
   - `-d`: Path to the dataset.
   - `-r`: Path to the JSON file containing rewards (optional).
   - `-md`: Model for survival mode (RSF, COX, or NN).
   - `-lm`: Load mode for editing the Q-Learning graph (T, J, D or else for no change).
   - `-lf`: Path to the file for editing the Q-Learning graph.
   - `-a`: Cleaning algorithm (CleanSurvival (L), random (R), custom (C), or no_preparation (N)).
   - `-ao`: Options for the cleaning algorithm (number of experiments for random, or path to a file containing pipelines for custom).