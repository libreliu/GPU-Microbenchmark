#!/usr/bin/env python3

import os, json
from timeit import repeat

import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
RESULT_DIR = os.path.join(ROOT_DIR, "result/")

def analyze_pchase_cpu(jsonData):

    numRepeats = jsonData['numRepeats']
    numSize = len(jsonData['data'][0]['size'])

    avgChaseTime = np.ndarray((numRepeats, numSize), dtype=np.double)
    size = np.asarray(jsonData['data'][0]['size'], dtype=np.int32)

    for repeatIdx in range(0, numRepeats):
        avgChaseTime[repeatIdx] = jsonData['data'][repeatIdx]['avgChaseTime']
    
    avgChaseTimeMean = np.mean(avgChaseTime, axis=0)
    avgChaseTimeStd = np.std(avgChaseTime, axis=0)

    plt.errorbar(size, avgChaseTimeMean, yerr=avgChaseTimeStd, fmt = "-", ecolor = "red", elinewidth = 0.5, capsize = 2, capthick = 1)

    plt.show()

def dispatch_analyze(jsonData):
    if jsonData['type'] == 'pchase-cpu':
        analyze_pchase_cpu(jsonData)

def main():
    jsonFiles = []
    for fileName in os.listdir(RESULT_DIR):
        if fileName.endswith(".json"):
            jsonFiles.append(fileName)
    
    print(f"Got {len(jsonFiles)} json files to be processed")

    for fileName in jsonFiles:
        print(f"- Processing {fileName}...", end='')
        with open(os.path.join(RESULT_DIR, fileName)) as f:
            jsonData = json.load(f)
            dispatch_analyze(jsonData)
        print(" [ Done ]")

if __name__ == "__main__":
    main()