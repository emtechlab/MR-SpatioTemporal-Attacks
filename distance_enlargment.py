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
import shutil
import matplotlib.pyplot as plt

# all files and directories ending with .png
input_path = "/home/../PV/"
output_path = input_path + "attack/"

if not os.path.exists(output_path):
    os.makedirs(output_path)

copy = 0  # index of the frame that needs to be copied
paste_length = 20  # num frames where the copied image needs to be pasted
attack_type = "de"  # distance enlargment
# output_path = input_path + 'attack'+attack_type+str(copy)+str(paste_length)
# warm  = [200]
# sample = [200] # history
# steady_state = [5,10,20]
# steady_state = [10]
# u_limit = [20,10, 25]
# u_limit = [90]


def create_list(sequence, sequence_type, configure_parameter, config_string, num_percentile):  
    path = (
        "./results/attack_frames/"
        + str(sequence)
        + "-"
        + sequence_type
        + config_string
        + str(num_percentile)
        + str(configure_parameter)
        + ".csv"
    )
    imagelist = [f for f in glob.glob(input_path + "*.png")]
    imagelist = sorted(imagelist)

    imu_path = input_path + "imu/"
    imulist = [f for f in glob.glob(imu_path + "*.txt")]
    imulist = sorted(imulist)
    print(input_path, len(imagelist), len(imulist))

    df_pose = pd.read_csv(input_path + "poses.txt", header=None)
    # print((df_pose.values[0]))

    attack_list = []
    attack_imu = []
    attack_pose = []
    count = 0
    counter = 0
    size = len(imagelist)

    df = pd.read_csv(path, header=None)

    fake_counter = 0
    real_counter = 0
    attack_flag = 0
    attack_count = 0
    attack_frames = 0
    total_frames = len(df.values) - 1
    after_attack_array = []
    attack_count = 0
    while fake_counter < total_frames:
        # base case
        if attack_flag == 0:
            after_attack_array.append(real_counter)
            fake_counter += 1

        # transition to attack, 0 to 1 transition
        if (
            int(df.values[real_counter][1]) == 0
            and int(df.values[real_counter + 1][1]) == 1
        ):
            chosen_frame_index = real_counter
            attack_flag = 1
            attack_count += (
                1  # for stats only does not effect any thing in the algorithm
            )

        if (
            int(df.values[real_counter][1]) == 1
            and int(df.values[real_counter + 1][1]) == 0
        ):
            if attack_flag == 1:
                real_counter = chosen_frame_index
            attack_flag = 0

        if attack_flag == 1:
            after_attack_array.append(chosen_frame_index)

        real_counter += 1

    # print(after_attack_array)
    print(len(after_attack_array) - len(df.values))

    # print(configure_parameter, attack_count, attack_frames)
    attack_image_path = []
    attack_imu_path = []
    img_output_path = (
        output_path
        + "attacklength_steady"
        + str(num_percentile)
        + str(configure_parameter)
        + "/"
    )
    imu_output_path = img_output_path + "imu/"
    # if not os.path.exists(img_output_path):
    #     os.makedirs(img_output_path)


    attack_list = []
    attack_imu = []
    for i in range(len(after_attack_array)):
        image_name = str(i).zfill(10) + ".png"
        imu_name = str(i).zfill(10) + ".txt"
        attack_imu_path.append(str(imu_output_path + imu_name))
        attack_image_path.append(str(img_output_path + image_name))
        attack_list.append(str(imagelist[after_attack_array[i]]))
        attack_imu.append(str(imulist[after_attack_array[i]]))

    attack_pose = after_attack_array
    print(len(attack_list), len(attack_imu_path), len(attack_image_path))

    file_name = (
        "./results/attack_frames/attacklist"
        + str(sequence)
        + "-"
        + sequence_type
        + config_string
        + str(num_percentile)
        + "wposes"
        + str(configure_parameter)
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
    # print(tmp_df)
    # # print(tmp_df)
    np.savetxt(file_name, tmp_df.values, delimiter=",", fmt="%s")
    # # print("file saved with path", file_name)
    # return 0,0
    return attack_count, len(after_attack_array) - len(df.values)


def copy_images(file_name):

    df = pd.read_csv(file_name, header=None)
    for i in range(len(df[1])):
        # df.values[rows][columns
        original = Image.open(df.values[i][0])
        attack = original.copy()
        paste = attack.save(df.values[i][1])
        # print("Image number --- " , i)


def copy_imu(file_name):
    df = pd.read_csv(file_name, header=None)
    for i in range(len(df[3])):
        # df.values[rows][columns]
        shutil.copyfile(df.values[i][2], df.values[i][3])


def copy_poses(file_name, configure_parameter, num_percentile):
    df = pd.read_csv(file_name, header=None)
    # f = open(input_path+ "poses.txt", "r")
    # print(f.readlines())
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
        + "/"
    )
    # print(tmp_df)
    # print(file, configure_parameter)
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
    # num_percentile = [60,70,80,85,90,95]
    # num_percentile = [70,80,85,95]
    # num_percentile = [80]
    # do 80 200 combination
    # "attack_length_steady"+str(num_percentile)
    config_string = "attack_length_steady"
    configure_parameter = [1, 2, 5, 10, 15, 20, 50, 100, 150, 200]
    # configure_parameter = [1,10,50,100,200]
    # configure_parameter = [200]
    num_percentile = [60, 70, 75, 80, 85, 90, 95, 100]
    num_percentile = [90]
    configure_parameter = [1, 10, 20, 50, 100, 150, 200]
    # configure_parameter = [50]
    num_frames = []
    num_attacks = []
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
            )
            # num_attacks.append(attack_count)
            # num_frames.append(attack_frames)
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
                + ".csv"
            )

            img_output_path = (
                output_path
                + "attacklength_steady"
                + str(num_percentile[j])
                + str(configure_parameter[i])
                + "/"
            )
            imu_output_path = img_output_path + "imu/"
         