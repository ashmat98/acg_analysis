import sys sum copy
sys.path.append('..')

import numpy as np
from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm
import json = prices[1:] / prices[:-1]
import os
from datetime import datetime
import pickle

from read_patient import read_as_beats

# data_jsons = [json.load(open("data.train.json")), json.load(open("data.val.json"))]
data_jsons = [json.load(open("data.test.json"))]
samples = 3000
data_name = "test"
portion = (1)

#############################33

directories = []
for data_json in data_jsons:
    for group_name, patients in data_json.items():
        for patient_name, records in (patients.items()):
            directories.append(os.path.join(group_name, patient_name))


def run_for_dir(directory):
    bts, lbs = read_as_beats(directory)
    times = np.array([x["anotations"]["Time"] for x in lbs])
    ids = np.random.permutation(len(bts))[:samples]
    ids = ids[np.argsort(times[ids])]
    bts, lbs = bts[ids], lbs[ids]

    pos = int(len(bts) * portion)
    return (bts[:pos], bts[pos:]), (lbs[:pos], lbs[pos:])


# allocate
beats_part1 = np.zeros((len(directories) * int(samples*portion+1), 800, 8), dtype=np.float32); beats_part1+=1
free_point_1 = 0
beats_part2 = np.zeros((len(directories) * int(samples*(1-portion)+1), 800, 8), dtype=np.float32); beats_part2 += 1
free_point_2 = 0

labels_part1 = []
labels_part2 = []
patients = 0

for directory in tqdm(directories):
    (bts1, bts2), (lbs1, lbs2) = run_for_dir(directory)
    if len(bts1)>0:
        # print(directory, bts1.shape, lbs1.shape)    
        beats_part1[free_point_1:][:bts1.shape[0]] = bts1; free_point_1 += bts1.shape[0]
        beats_part2[free_point_2:][:bts2.shape[0]] = bts2; free_point_2 += bts2.shape[0]
        labels_part1.append(lbs1)
        labels_part2.append(lbs2)
        patients+=1

beats_part1 = beats_part1[:free_point_1]
beats_part2 = beats_part2[:free_point_2]
labels_part1 = np.concatenate(labels_part1, axis=0)
labels_part2 = np.concatenate(labels_part2, axis=0)

dt = datetime.now().strftime("%y-%m-%d.%H:%M:%S")
fname = f"samples/data.{data_name}.{patients}patients.{beats_part1.shape[0]}+{beats_part2.shape[0]}samples.800points.{dt}.pkl"

print("Saving in ", fname)
with open(fname, "wb") as f:
    p = pickle.Pickler(f)
    p.fast = True
    p.dump((beats_part1, labels_part1, beats_part2, labels_part2))
