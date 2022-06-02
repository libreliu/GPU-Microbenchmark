#!/usr/bin/env python3

import os, json

import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
RESULT_DIR = os.path.join(ROOT_DIR, "result/")

def pretty_size(s):
    if s < 1024:
        return f"{s} B"
    elif s < 1024 * 1024:
        return f"{s // 1024} K"
    else:
        return f"{s // (1024 * 1024)} M"


def analyze_pchase_cpu(jsonData):

    repeats = jsonData['repeats']
    strides = jsonData['strides']
    workingSets = jsonData['workingSets']
    
    numRepeats = len(repeats)
    numStrides = len(strides)
    numWorkingSets = len(workingSets)

    avgChaseTimes = np.asarray(jsonData['avgChaseTime'], dtype=np.double).reshape(
        (numRepeats, numStrides, numWorkingSets)
    )

    fig, ax = plt.subplots()

    legendHandles = []

    for strideIdx, stride in enumerate(strides):
        avgChaseTimeMean = np.mean(avgChaseTimes[:, strideIdx, :], axis=0)
        avgChaseTimeStd = np.std(avgChaseTimes[:, strideIdx, :], axis=0)

        # print("avgChaseTimeMean")

        handle = ax.errorbar(
            workingSets, avgChaseTimeMean, yerr=avgChaseTimeStd,
            fmt = "-", ecolor = "red", elinewidth = 0.5, capsize = 2, capthick = 1
        )
        legendHandles.append(handle)

        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=10)
        ax.set_ylabel('Time Consumed')
    
    plt.legend(handles=legendHandles, labels=[f"Stride {i}" for i in strides], loc='best')

    plt.xticks(
        workingSets,
        [pretty_size(i) for i in workingSets],
        rotation=45,
        size='small'
    )

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