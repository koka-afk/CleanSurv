from learn2clean.qlearning import qlearner as ql
from learn2clean.qlearning import survival_qlearner as survival_ql
import pandas as pd
import json
import argparse

parser = argparse.ArgumentParser(description='Cleaning mode selection')
parser.add_argument('-d', '--dataset', help='Provide path to dataset')
parser.add_argument('-r', '--rewards', help='Provide path to JSON file containing rewards (in case of survival mode)')
parser.add_argument('-md', '--model', help='Model for survival mode (RSF, COX, or NN)')
parser.add_argument('-lm', '--load_mode', help='Provide nothing or T for Choose "T" to Add/Edit Edges using txt file, "J" to import graph from JSON file or "D" for disable mode')
parser.add_argument('-lf', '--load_file', help='Provide path to: txt file to edit edges, txt file to disable steps or JSON file to import graph')
parser.add_argument('-a', '--algo', help='Cleaning algorithm (learn2clean, random, custom, or no preparation)')
parser.add_argument('-ao', '--algo_op', help='This argument only used in case of random or custom algos. In case of random provide a number for experiments and in case of custom provide a txt file that contain the pipelines')

args = parser.parse_args()

path = args.dataset  #"learn2clean/datasets/flchain.csv"
file_name = path.split("/")[-1]
json_path = args.rewards  #'C:/Users/yosef/OneDrive/Desktop/Learn2Clean/python-package/config.json'
dataset = pd.read_csv(path)
dataset.drop('rownames', axis=1, inplace=True)
time_column = "futime"
event_column = "death"
model = args.model.upper()

l2c = survival_ql.SurvivalQlearner(file_name=file_name, dataset=dataset, time_col=time_column, event_col=event_column, goal=model, json_path=json_path, threshold=0.6)

edit = args.load.upper()
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

job = args.algo.upper()

if job == "L":
    l2c.Learn2Clean()
elif job == "R":
    repeat = int(args.algo_op)
    l2c.random_cleaning(dataset_name=file_name, loop=repeat)
elif job == 'C':
    pipelines_file_path = args.algo_op
    pipelines = open(pipelines_file_path, 'r')
    l2c.custom_pipeline(pipelines, model, dataset_name=file_name)
else:
    l2c.no_prep()

