# import image module from pillow
from itertools import count
# from re import A
import time
from PIL import Image
import glob
from click import Path
import numpy as np
import os
import pandas as pd
import shutil
import matplotlib.pyplot as plt

# attack specific
# all files and directories ending with .png
input_path = "/home/../PV/"
output_path = input_path + "attack/"

if not os.path.exists(output_path):
    os.makedirs(output_path)

copy = 0  # index of the frame that needs to be copied
paste_length = 20  # num frames where the copied image needs to be pasted
attack_type = "de"  # distance enlargment

def create_list(sequence, sequence_type, configure_parameter, config_string, num_percentile, at_type):  # option = 1 for complex method, option = 2 when using simple method
    path = ("./results/attack_frames/"
        + str(sequence)
        + "-"
        + sequence_type
        + config_string
        + str(num_percentile)
        + str(configure_parameter)
        + ".csv"
    )  # independent file
    imagelist = [f for f in glob.glob(input_path + "*.png")]
    imagelist = sorted(imagelist)

    imu_path = input_path + "imu/"
    imulist = [f for f in glob.glob(imu_path + "*.txt")]
    imulist = sorted(imulist)

    df_pose = pd.read_csv(input_path + "poses.txt", header=None)
    count = 0
    counter = 0
    size = len(imagelist) - 1
    dumb_counter = 0

    df = pd.read_csv(path, header=None)

    fake_counter = 0
    real_counter = 0
    attack_flag = 0
    attack_count = 0
    attack_frames = 0
    # print(len(df.values))
    total_frames = len(df.values) - 1
    after_attack_array = []
    attack_count = 0
    alternate = 0
    added = []
    drop = []
    i = 0
    alter_count = []

    img_output_path = (
        output_path
        + "attacklength_steady"
        + str(num_percentile)
        + str(configure_parameter)
        + at_type
        + "/")

    imu_output_path = str(img_output_path) + "imu/"
    print(imu_output_path)
    attack_image_path = []
    attack_imu_path = []
    attack_list = []
    attack_imu = []
    attack_pose = []

    # attack_list = []
    attack_imu = []
    j = 0

    while fake_counter < total_frames:
        # base case
        i += 1
        if attack_flag == 0:
            after_attack_array.append(real_counter)

            image_name = str(j).zfill(10) + ".png"
            attack_image_path.append(str(img_output_path + image_name))
            attack_list.append(str(imagelist[real_counter]))
            imu_name = str(j).zfill(10) + ".txt"
            attack_imu_path.append(str(imu_output_path + imu_name))
            attack_imu.append(str(imulist[real_counter]))
            j +=1
            ##### 
            fake_counter += 1

        # transition to attack, 0 to 1 transition
        if (int(df.values[real_counter][1]) == 0 and int(df.values[real_counter + 1][1]) == 1):
            chosen_frame_index = real_counter
            attack_flag = 1

        # transition to attack, 1 to 0 transition
        if (int(df.values[real_counter][1]) == 1 and int(df.values[real_counter + 1][1]) == 0):
            if attack_flag == 1:
                real_counter = chosen_frame_index
            attack_flag = 0

        if attack_flag == 1:  # add
            after_attack_array.append(chosen_frame_index)
            # changing picture from the new one from the hist images
            image_name = str(j).zfill(10) + ".png"
            attack_image_path.append(str(img_output_path + image_name))
            address_string =  str(str(imagelist[chosen_frame_index])).split("PV/")
            attack_list.append(str(address_string[0]+"PV/histimages/"+address_string[1]))
            address_string =  str(str(imulist[chosen_frame_index])).split("imu/")
            imu_name = str(j).zfill(10) + ".txt"
            attack_imu_path.append(str(imu_output_path + imu_name))
            attack_imu.append(str(address_string[0]+"imunegate/"+address_string[1]))
            # print("this",str(address_string[0]+"imunegate/"+address_string[1]))
            dumb_counter +=1
            j +=1
            fake_counter += 1
            ##### 
        real_counter += 1

    attack_pose = after_attack_array
    print(len(after_attack_array) - len(df.values),len(attack_list), len(attack_imu_path), len(attack_image_path))

    file_name = (
        "./results/attack_frames/attacklist"
        + str(sequence)
        + "-"
        + sequence_type
        + config_string
        + str(num_percentile)
        + "wposes"
        + str(configure_parameter)
        + at_type
        + ".csv"
    )
    tmp_df = pd.DataFrame(
        {
            "acc1": attack_list,
            "acc2": attack_image_path,
            "acc3": attack_imu,
            "acc4": attack_imu_path,
            "pose": attack_pose,
        }
    )
    np.savetxt(file_name, tmp_df.values, delimiter=",", fmt="%s")
    return attack_count, len(after_attack_array) - len(df.values)


def copy_images(file_name):
    df = pd.read_csv(file_name, header=None)
    for i in range(len(df[1])):
        original = Image.open(df.values[i][0])
        attack = original.copy()
        paste = attack.save(df.values[i][1])


def copy_imu(file_name):
    df = pd.read_csv(file_name, header=None)
    for i in range(len(df[3])):
        # df.values[rows][columns]
        shutil.copyfile(df.values[i][2], df.values[i][3])


def copy_poses(file_name, configure_parameter, num_percentile):
    df = pd.read_csv(file_name, header=None)
    df_pose = pd.read_csv(input_path + "poses.txt", header=None)
    pose = []
    for i in range(len(df[3])):
        # for i in range(0,1000):
        temp = df_pose.values[df.values[i][4]][0]
        pose.append(df_pose.values[df.values[i][4]][0])
    # pose.append(temp)
    tmp_df = pd.DataFrame({"acc1": pose})
    file = (
        output_path
        + "attacklength_steady"
        + str(num_percentile)
        + str(configure_parameter)
        + at_type
        + "/"
    )
    np.savetxt(file + "poses.txt", tmp_df.values, delimiter=",", fmt="%s")


def plots(attack_count, attack_frames, save):
    attack_number = [1, 5, 10, 15, 20, 25, 50, 70, 80, 90, 100, 150, 200]

    fig = plt.figure()
    ax = plt.axes()
    plt.plot(attack_number, attack_count)
    ax.set_ylabel("Number of attack launched")
    ax.set_xlabel("Number of frames attacked at a time")
    ax.set_xlim(0, 201)
    plt.savefig(save + "number.jpg")

    fig = plt.figure()
    ax = plt.axes()
    plt.plot(attack_number, attack_frames)
    ax.set_ylabel("Number of frame attacked")
    ax.set_xlabel("Number of frames attacked at a time")
    ax.set_xlim(0, 201)
    plt.savefig(save + "frames.jpg")


if __name__ == "__main__":
    sequence = ".."
    sequence_type = "orignal"  # attack or original
    config_string = "attack_length_steady"
    configure_parameter = [1, 2, 5, 10, 15, 20, 50, 100, 150, 200]
    num_percentile = [60, 70, 75, 80, 85, 90, 95, 100]
    num_percentile = [90]
    configure_parameter = [2, 5, 10, 15, 20, 50, 100, 150]
    configure_parameter = [10]
    num_frames = []
    num_attacks = []
    at_type = "zero_hist"
    at_type ="dev"
    # at_type =""
    # at_type = 'zero'
    for j in range(len(num_percentile)):
        num_frames = []
        num_attacks = []
        # print(configure_parameter)
        for i in range(len(configure_parameter)):
            # print(sequence, sequence_type, configure_parameter[i],config_string,num_percentile[j])
            attack_count, attack_frames = create_list(
                sequence,
                sequence_type,
                configure_parameter[i],
                config_string,
                num_percentile[j],
                at_type,
            )

            config = configure_parameter[i]
            # print(num_percentile[j], configure_parameter[i], attack_count,attack_frames)
            file_name = (
                "./results/attack_frames/attacklist"
                + str(sequence)
                + "-"
                + sequence_type
                + config_string
                + str(num_percentile[j])
                + "wposes"
                + str(config)
                + at_type
                + ".csv"
            )
            img_output_path = (
                output_path
                + "attacklength_steady"
                + str(num_percentile[j])
                + str(configure_parameter[i])
                + at_type
                + "/"
            )
            imu_output_path = img_output_path + "imu/"
            if not os.path.exists(img_output_path):
                os.makedirs(img_output_path)
            if not os.path.exists(imu_output_path):
                os.makedirs(imu_output_path)
            copy_images(file_name)
            copy_imu(file_name)
            copy_poses(file_name, config, num_percentile[j])