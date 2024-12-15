from learn2clean.qlearning import survival_qlearner as survival_ql
import pandas as pd
import json


path = "learn2clean/datasets/rotterdam_missing_MNAR/rotterdam_missing_50_MNAR.csv"  
file_name = path.split("/")[-1]
json_path = 'C:/Users/yosef/Desktop/Learn2Clean_extended/python-package/config.json'
dataset = pd.read_csv(path)
dataset.drop('rownames', axis=1, inplace=True)
dataset.drop('pid', axis=1, inplace=True)
time_column = "dtime"
event_column = "death"
model = ""
available_models = ['RSF', 'COX', 'NN', 'OLS', 'LASSO_REG', 'MARS']
while True:
    model = input("Please choose 'RSF', 'COX', 'NN', 'OLS', 'LASSO_REG' or 'MARS': ")
    model = model.upper()
    if model in available_models:
        break   

l2c = survival_ql.SurvivalQlearner(file_name=file_name, dataset=dataset, time_col=time_column, event_col=event_column, goal=model, json_path=json_path, threshold=0.6)

edit = str(input("Choose 'T' to Add/Edit Edges using txt file, 'J' to import graph from JSON file or 'D' for disable mode: ")).upper()
if edit == 'T':
    txt_path = str(input("Provide path to txt file: "))
    with open(txt_path, 'r+') as edges:
        for line in edges:
            edge = list(line.split(" "))
            u = edge[0]
            v = edge[1]
            weight = int(edge[2])
            l2c.edit_edge(u, v, weight)
elif edit == 'J':
    graph_path = str(input("Provide path to txt file: "))
    with open(graph_path, 'r+') as graph:
        data = json.load(graph)
        l2c.set_rewards(data)
elif edit == 'D':
    disable_path = str(input("Provide path to txt file: "))
    with open(disable_path, 'r+') as disable:
        for op in disable:
            l2c.disable(op)

# print(l2c.rewards)

job = ""
while True:
    job = str(input("Please choose 'L' for Learn2Clean, 'R' for Random, 'C' for Custom Pipeline Design, 'G' for Grid Search or 'N' for a No Preparation job: ")).upper()
    if job == 'L' or job == 'R' or job == 'C' or job == 'N' or job == 'G':
        break

if job == "L":
    restarts = 15
    for times in range(restarts):
        l2c.Learn2Clean()
elif job == "R":
    repeat = eval(input("Please enter the number of random experiments: ")) # TODO Allow user to choose number of random trials
    l2c.random_cleaning(dataset_name=file_name, loop=repeat)
elif job == 'C':
    pipelines_file_path = str(input("Please enter pipelines file name: "))
    pipelines = open(pipelines_file_path, 'r')
    l2c.custom_pipeline(pipelines, model, dataset_name=file_name)
elif job == 'G':
    trials = eval(input("Please input the number of trials/restarts of the grid search: "))
    l2c.grid_search(dataset_name=file_name, trials=trials)
else:
    l2c.no_prep()

