from dis import dis
import glob
from math import sqrt
from sys import api_version
from tkinter.tix import Tree
from cv2 import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image, ImageChops
import os
import evaluation

def plots(sum_, min_, max_, diff, save):
    attack_number = [1, 2, 5, 10, 15, 20, 50, 100, 150, 200]

    fig = plt.figure()
    ax = plt.axes()
    plt.plot(attack_number, diff)
    ax.set_ylabel("Total diversion[m]")
    ax.set_xlabel("Number of frames attacked at a time")
    ax.set_xlim(0, 201)
    # plt.legend()
    # plt.show()
    plt.savefig(save + "total.jpg")

    fig = plt.figure()
    ax = plt.axes()
    plt.plot(attack_number, diff)
    ax.set_ylabel("Total diversion[m]")
    ax.set_xlabel("Number of frames attacked at a time")
    ax.set_xlim(0, 201)
    # plt.legend()
    # plt.show()
    plt.savefig(save + "total.jpg")
def speed_diff(df, j):
    distance = []
    old_distance = 0
    speed_unit = 0.01
    
    for i in range(len(df.iloc[:, 0]) - 1):
        # print(len(df.values[i][j]))
        x = (((df.values[i + 1][j]) - (df.values[i][j])) ** 2)
        y = (((df.values[i + 1][j + 1] - df.values[i][j + 1])) ** 2)
        z = (((df.values[i + 1][j + 2] - df.values[i][j + 2])) ** 2)
        # temp_distance = old_distance + sqrt(x + y + z)
        temp_distance =  old_distance + sqrt(x + y + z)

        distance.append(temp_distance/((i+1)*speed_unit))  # Pythagorean theorem
        # distance.append(temp_distance)  # Pythagorean theorem
        old_distance = temp_distance
    # print(distance)
    return distance

def distance(df, j):
    distance = []
    old_distance = 0
    speed_unit = 1
    
    for i in range(len(df.iloc[:, 0]) - 1):
        # print(len(df.values[i][j]))
        x = (((df.values[i + 1][j]) - (df.values[i][j])) ** 2)*speed_unit
        y = (((df.values[i + 1][j + 1] - df.values[i][j + 1])) ** 2)*speed_unit
        z = (((df.values[i + 1][j + 2] - df.values[i][j + 2])) ** 2)*speed_unit
        temp_distance = old_distance + sqrt(x + y + z)

        distance.append(temp_distance)  # Pythagorean theorem
        old_distance = temp_distance

    return distance

def speed(df, j):
    distance = []
    speed_unit = 0.01
    step = 500
    temp_distance = 0

    
    for m in range(0,len(df.iloc[:, 0]) - step):
        # old_distance = 0
        # print(m)
        temp_distance = 0
        for i in range(m, m + step):
            # print(m)
            # print(len(df.values[i][j]))
            x = (((df.values[i + 1][j]) - (df.values[i][j])) ** 2)
            y = (((df.values[i + 1][j + 1] - df.values[i][j + 1])) ** 2)
            z = (((df.values[i + 1][j + 2] - df.values[i][j + 2])) ** 2)
            temp_distance = sqrt(x + y + z) +  temp_distance
            
        old_distance = temp_distance
        # m = m + 1
        distance.append(float(temp_distance)/(speed_unit*step)) # Pythagorean theorem
        
    print(len(distance), "dist")
    return distance


def plot_distance(df, save):
    df_gt = pd.read_csv("./../result_final_correct.csv", header=None)
    # data_gt = np.array(speed(df_gt,0))
    data_gt = np.array(speed_diff(df_gt,0))
    data = np.array(speed_diff(df,0))
    # data  =  data_gt
    normalization_index = 1000
    data[0:normalization_index] = [a*b for a,b in zip(factor_list,data_gt[0:normalization_index])]
    
    labels = ["orignal", "attack"]
    y_label = "Speed"
    fig = plt.figure()
    ax = plt.axes()
    m = 0
    n = 30000
    time = np.arange(0, len(data))[m:n]
    time_gt = np.arange(0, len(data_gt))[m:n]
    plt.plot(time_gt-m, data_gt[m:n], c="k", label=labels[0])
    plt.plot(time-m, data[m:n], c="r", linestyle="--", label=labels[1])
    ax.set_ylabel("Speed[m/s]")
    ax.set_xlabel("Sample")
    ax.set_xlim(0, 15000)
    # ax.set_ylim(1.5, 2.2)
    plt.legend()
    # plt.show()
    plt.savefig(save + ".jpg")
    print(save)
    
    np.savetxt(save + ".csv", np.transpose([time-m, data]), delimiter=",", fmt="%s")
    np.savetxt(save + "gt.csv", np.transpose([time_gt-m,data_gt]), delimiter=",", fmt="%s")
    # evaluation.plot_xyz([data[0],data_gt[0][0:seq_size]], labels, y_label, 2, save+".jpg")
    return np.nansum(data), np.nanmin(data), np.nanmax(data), 0


if __name__ == "__main__":
    folder = "./results/"
    plots_path = "/home/../plots"
    attack_name = "attacklength"
    attack_number = [1, 2, 5, 10, 15, 20, 50, 100, 150, 200]

    result_min = []
    result_max = []
    result_sum = []
    result_diff = []
    at_type = "red"
    # at_type = ""
    # at_type = "zero_hist"
    # at_type ="zero_hist-more-red-more-add"
    # at_type = "zero_hist-more-red-less-add95half"
    # at_type = "zero95"
    at_type = "zero"
    at_type = "dev"
    # at_type ="rerun"
    for i in attack_number:
        path = folder + attack_name + str(i) + at_type + "/"
        df = pd.read_csv(path + "result_final_correct"+at_type+str(i)+".csv", header=None)

        # distance(df, False)
        # print(df)
        sum_, min_, max_, diff = plot_distance(df, path + "speed" + at_type + str(i))
        result_min.append(min_)
        result_max.append(max_)
        result_sum.append(sum_)
        result_diff.append(diff)