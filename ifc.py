#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from importlib.resources import path
from PIL import Image
import six

# import image module from pillow
from itertools import count
from re import A
import time
from PIL import Image
import glob
from click import Path
import numpy as np
import os
import pandas as pd
import evaluation
import matplotlib.pyplot as plt
import hashlib
import cdf_index

seq_size = 1300
# seq_size = 50
def pose_correlation(imu, image):
    cor_x = []
    cor_y = []
    for i in range(seq_size - 1):
        cor_x.append(np.corrcoef(imu[i], imu[i + 1]))
        cor_y.append(np.corrcoef(image[i], image[i + 1]))
    return cor_x, cor_y

    # interframe corelataion coefficient


def calcualate_ifc(path, filename_img, filename_imu):
    imagelist = [f for f in glob.glob(path + "*.png")]
    imagelist = sorted(imagelist)
    imulist = [f for f in glob.glob(path + "imu/*.txt")]
    imulist = sorted(imulist)
    # print(imulist)
    # open only for new sequence create sepearate function
    ifc_img = []
    ifc_imu = []
    img = []
    imu = []
    for i in range(seq_size):
        img.append(np.array(Image.open(imagelist[i]).convert("L")) / 255)
    print("all ifc are done")
    for i in range(1, seq_size):
        ifc_img.append(np.var(np.corrcoef(img[i - 1], img[i])))
    

    # tmp_img = pd.DataFrame({"acc1":ifc_img, "acc2":ifc_img})
    tmp_img = pd.DataFrame({"acc1": ifc_img, "acc2": ifc_img})
    # filename = path  +'/'+ foldername +'/ifc_copy-paste.csv'
    np.savetxt(filename_img, tmp_img.values, delimiter=",")

    # tmp_imu = pd.DataFrame({"acc1":imu, "acc2":ifc_imu})
    # np.savetxt(filename_imu,tmp_imu.values, delimiter=',')
    # print(path, filename_img, filename_imu,"saving path")
    return ifc_img, ifc_imu, img, imu


def plot_xyz(data, labels, y_label, axis_num, save):
    fig = plt.figure()
    ax = plt.axes()
    start = 0
    end = 2700 - 1

    time = (np.arange(2700))[start:end]  # number of samples

    X = data[0][start:end]
    y = data[1][start:end]
    ax.set_ylim(min(min(X), min(y)), max(max(y), max(X)))
    plt.plot(time, X, c="k", label=labels[0])
    plt.plot(time, y, c="r", linestyle="--", label=labels[1])

    ax.set_ylabel(y_label)

    ax.set_xlabel("Time(samples)")
    ax.set_xlim(0, end)

    plt.legend()
    plt.show()
    if not os.path.exists(save):
        os.makedirs(save)

    plt.savefig(save)


if __name__ == "__main__":
    name = ".."
    input_path = "/home/../PV/"
    num_percentile = [60, 70, 75, 80, 85, 90, 95, 100]
    configure_parameter = [1, 5, 10, 15, 20, 25, 50, 70, 80, 90, 100, 150, 200]

    eval_type = "ifc"
    gt_path = "/home/../results/gt"

    filetype = "image"
    gt_image_path = "{}/{}/{}_{}_{}.csv".format(
        gt_path, eval_type, eval_type, filetype, name
    )
    filetype = "imu"
    gt_imu_path = "{}/{}/{}_{}_{}.csv".format(
        gt_path, eval_type, eval_type, filetype, name
    )

    ifc_img, ifc_imu, img, imu = calcualate_ifc(input_path, gt_image_path, gt_imu_path)

    if not os.path.exists("{}/{}/".format(gt_path, eval_type)):
        os.makedirs("{}/{}/".format(gt_path, eval_type))
    num_percentile = [90]
    configure_parameter = [200]
    for j in range(len(num_percentile)):
        for i in range(len(configure_parameter)):

            input_path = (
                "/home/../PV/attack/attacklength_steady"
                + str(num_percentile[j])
                + str(configure_parameter[i])
                + "/"
            )
            output_path = (
                "/home/../results/attacklength"
                + str(configure_parameter[i])
                + "/steady"
                + str(num_percentile[j])
            )
            attack_path = input_path
            imu_path = input_path + "imu/"
            imulist = [f for f in glob.glob(imu_path + "*.txt")]
            imulist = sorted(imulist)
            imagelist = [f for f in glob.glob(input_path + "*.png")]
            imagelist = sorted(imagelist)

            attack_list = []
            attack = 0  # 0 only original, 1 attack and original

            if not os.path.exists(output_path + "ifc/"):
                os.makedirs(output_path + "ifc/")
            filetype = "imu"
            name = "de_" + str(num_percentile[j]) + str(configure_parameter[i])
            filename_imu = (
                output_path + "ifc/" + "ifc_" + filetype + "_" + name + ".csv"
            )
            filetype = "image"
            filename_img = (
                output_path + "ifc/" + "ifc_" + filetype + "_" + name + ".csv"
            )
            ifc_img, ifc_imu, img, imu = calcualate_ifc(
                input_path, filename_img, filename_imu
            )
