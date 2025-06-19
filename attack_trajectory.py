from cProfile import label
import glob
from math import sqrt
from sys import api_version
from tkinter.tix import Tree
from matplotlib import projections, scale
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image, ImageChops
import os
import quaternion
import evaluation
from sklearn.metrics import mean_squared_error
from math import sqrt

# df_result = pd.concat([df_pos, df_ori, t_df_pos,t_df_ori, tmp_df], axis= 1)
seq_size = 13900
df_gt = pd.read_csv(
    "./results/.../result_final_correct.csv",
    header=None,
)
# df_gt = pd.read_csv(
#     "./results/attacklength10sanitycheck/result_final_correct.csv",
#     header=None,
# )

def distance(df, j):
    distance = []
    old_distance = 15
    for i in range(seq_size):
        # print(len(df.values[i][j]))
        x = ((df.values[i + 1][j]) - (df.values[i][j])) ** 2
        y = ((df.values[i + 1][j + 1] - df.values[i][j + 1])) ** 2
        z = ((df.values[i + 1][j + 2] - df.values[i][j + 2])) ** 2
        temp_distance = old_distance + sqrt(x + y + z)

        distance.append(temp_distance)  # Pythagorean theorem
        old_distance = temp_distance
    return [distance]


def force_quaternion_uniqueness(q):

    q_data = quaternion.as_float_array(q)

    if np.absolute(q_data[0]) > 1e-05:
        if q_data[0] < 0:
            return -q
        else:
            return q
    elif np.absolute(q_data[1]) > 1e-05:
        if q_data[1] < 0:
            return -q
        else:
            return q
    elif np.absolute(q_data[2]) > 1e-05:
        if q_data[2] < 0:
            return -q
        else:
            return q
    else:
        if q_data[3] < 0:
            return -q
        else:
            return q


def traj(df):
    a = 1
    window_size = 10
    stride = 10
    pos_data = df[df.columns[a : a + 3]].values
    ori_data = df[df.columns[a + 3 : a + 7]].values
    pos_data_g = df_gt[df_gt.columns[a : a + 3]].values
    ori_data_g = df_gt[df_gt.columns[a + 3 : a + 7]].values
    y_delta_p = []
    y_delta_q = []

    # init_p = pos_data[window_size//2 - stride//2, :]
    # # print
    # init_q = ori_data[window_size//2 - stride//2, :]
    for idx in range(0, len(df.iloc[:, a]) - window_size - 1, stride):
        p_a = pos_data[idx + window_size // 2 - stride // 2, :]
        p_b = pos_data[idx + window_size // 2 + stride // 2, :]

        q_a = quaternion.from_float_array(
            ori_data[idx + window_size // 2 - stride // 2, :]
        )
        q_b = quaternion.from_float_array(
            ori_data[idx + window_size // 2 + stride // 2, :]
        )

        delta_p = np.matmul(quaternion.as_rotation_matrix(q_a).T, (p_b.T - p_a.T)).T

        delta_q = force_quaternion_uniqueness(q_a.conjugate() * q_b)

        y_delta_p.append(delta_p)
        y_delta_q.append(quaternion.as_float_array(delta_q))

    # x = np.reshape(x, (len(x), x[0].shape[0], x[0].shape[1]))
    y_delta_p = np.reshape(y_delta_p, (len(y_delta_p), y_delta_p[0].shape[0]))
    y_delta_q = np.reshape(y_delta_q, (len(y_delta_q), y_delta_q[0].shape[0]))
    print(len(y_delta_p))
    y_delta_p = y_delta_p[0:100, :]
    fig = plt.figure()
    ax = plt.axes()
    plt.plot(y_delta_p[:, 0], y_delta_p[:, 1])
    # plt.plot(gt_x,gt_y)
    # plt.plot(time, data_x,data_y, label =  "Attack-X")
    # plt.plot(time,data_x_gt,data_y_gt, label = "Original-X")
    # plt.plot(time, data_y, label =  "Attack-Y")
    # plt.plot(time,data_y_gt, label = "Original-Y")
    ax.set_ylabel("X")
    ax.set_xlabel("Sample")
    # ax.set_xlim(0,8000)
    plt.legend()
    plt.show()

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_distance(df, save, attacklist):
    a = 12 # 12 gives right trajectory
    seq_size = 13900
    r2 = np.sqrt(2)
    # df_result = pd.concat([df_pos, df_ori, t_df_pos,t_df_ori, tmp_df, tmp_trans], axis= 1)

    time = np.arange(0, seq_size)
    data_x = np.array(df.iloc[:, a].dropna())[0:seq_size]
    data_y = np.array(df.iloc[:, a + 1].dropna())[0:seq_size]
    data_z = np.array(df.iloc[:, a + 2].dropna())[0:seq_size]
    data_x_gt = np.array(df_gt.iloc[:, a].dropna())[0:seq_size]
    data_y_gt = np.array(df_gt.iloc[:, a + 1].dropna())[0:seq_size]
    data_z_gt = np.array(df_gt.iloc[:, a + 2].dropna())[0:seq_size]
    x = []
    y = []
    z = []
    gt_x = []
    gt_y = []
    gt_z = []
    xdirection = []
    ydirection = []

    m = 0
    n = 13000
    data_x = data_x[m:n]
    data_y =data_y[m:n]
    data_x_gt = data_x_gt[m:n]
    data_y_gt =data_y_gt[m:n]

    data_x =  moving_average(data_x,2000)
    data_y =  moving_average(data_y,2000)

    data_x_gt =  moving_average(data_x_gt,2000)
    data_y_gt =  moving_average(data_y_gt,2000)

    normalization_index = 1 # if trajectory needs to be closer add more normalization
    data_x[0:normalization_index] = [a*b for a,b in zip(factor_list,data_x_gt[0:normalization_index])]
    factor =  data_y[normalization_index] /data_y_gt[normalization_index]
    data_y[0:normalization_index] = [a*b for a,b in zip(factor_list,data_y_gt[0:normalization_index])]

    fig = plt.figure()
    ax = plt.axes()
    # ax = plt.axes(projection = '3d')
    # ax.set_xlabel('Z')
    # plt.plot(x,y,z, label= 'attack')
    # plt.plot(gt_x,gt_y,gt_z, label = "original")
    # plt.plot(x, y, label="attack", markevery=[0,-1])
    # plt.plot(gt_x, gt_y, label="original", marker = "o",markevery=[0,-1])
    # plt.plot(x, y, label="attack", markevery=[0,-1])
    scale_x = 1
    # np.mean(data_x_gt) 
    scale_y = 1
    #  np.mean(data_y_gt)

    # data_x = ((data_x- np.min(data_x_gt)) / (np.max(data_x_gt) - np.min(data_x_gt)))*scale_x
    # data_y = ((data_y- np.min(data_y_gt)) / (np.max(data_y_gt) - np.min(data_y_gt)))*scale_y
    # data_x_gt = ((data_x_gt- np.min(data_x_gt)) / (np.max(data_x_gt) - np.min(data_x_gt)))
    # data_y_gt = ((data_y_gt- np.min(data_y_gt)) / (np.max(data_y_gt) - np.min(data_y_gt)))
    from scipy.signal import savgol_filter
    


    plt.plot(data_x, data_y, label="attacked", marker = "o",markevery=[0,-1])
    plt.plot(data_x_gt, data_y_gt, label="original", marker = "o",markevery=[0,-1])
    ax.set_ylabel("Y")
    ax.set_xlabel("X")

    # ax.set_xlim(0,8000)
    plt.legend()
    # plt.show()
    plt.savefig(save + "2d.jpg")
    return 0, 0, 0, 0

if __name__ == "__main__":
    folder = "./results/"
    plots_path = "/home/.../plots"
    attack_name = "attacklength"
    attack_number = [1, 2, 5, 10, 15, 20, 50, 100, 150, 200]
    attack_number = [1, 2, 5, 10, 15, 20, 50]
    attack_number = [1, 2, 5, 10, 15, 20, 50, 100, 150, 200]
    attack_number = [10]

    result_min = []
    result_max = []
    result_sum = []
    result_diff = []
    sequence = "..."
    sequence_type = "orignal"  # attack or original
    config_string = "attack_length_steady"
    num_percentile = 90
    at_type = "red"
    # at_type = ""
    # at_type ="rerun"
    at_type="dev"
    # at_type ="zero"
    # at_type ="sanitycheck"
    for i in attack_number:
        file_name = ( "./results/attack_frames/attacklist" + str(sequence)+ "-"+ sequence_type+ config_string+ str(num_percentile)
                + "wposes"+ str(i)+ at_type+ ".csv")
        # attacklist = pd.read_csv(file_name, header=None)
        # print(attacklist.iloc[:, 4])

        path = folder + attack_name + str(i) + at_type + "/"
        df = pd.read_csv(path + "result_traj"+at_type+str(i)+".csv", header=None)
        # print(df)
        # distance(df, False)
        # print(df)
        # traj(df)
        # sum_, min_, max_, diff = plot_distance(df, path + "trajectory" + str(i), attacklist.iloc[:, 4])
        sum_, min_, max_, diff = plot_distance(df, path + "trajectory" + str(i), [])
        # result_min.append(min_)
        # result_max.append(max_)
        # result_sum.append(sum_)
        # result_diff.append(diff[-1])

    # plots(result_sum, result_min, result_max, result_diff, folder + attack_name+"distance_skiprow")
