# print(__doc__)
from cProfile import label
from fileinput import filename
from hashlib import algorithms_available
from re import L, T
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from torch import positive
import pandas as pd
import quaternion
from csv import reader, writer

epoch = 0
fusion_mode = 3
copy = 100000  # index of the frame that needs to be copied [100,50]
paste_length = (
    50000  # num frames where the copied image needs to be pasted [5,10,15,20]
)
attack_type = "one_frame"  # one_frame : where is one frama is pasted over, multi_frame: multiple frames are replaced

seq_size = 13937 


def generate_trajectory_6d_quat(init_p, init_q, y_delta_p, y_delta_q):
    cur_p = np.array(init_p)
    cur_q = quaternion.from_float_array(init_q)
    pred_p = []
    pred_p.append(np.array(cur_p))

    for [delta_p, delta_q] in zip(y_delta_p, y_delta_q):
        cur_p = cur_p + np.matmul(quaternion.as_rotation_matrix(cur_q), delta_p.T).T
        cur_q = cur_q * quaternion.from_float_array(delta_q).normalized()
        pred_p.append(np.array(cur_p))
        # print(cur_p, cur_q)
    # print(np.reshape(pred_p, (len(pred_p), 3)),)
    return np.reshape(pred_p, (len(pred_p), 3))


def generate_trajectory_3d(
    init_l, init_theta, init_psi, y_delta_l, y_delta_theta, y_delta_psi
):
    cur_l = np.array(init_l)
    cur_theta = np.array(init_theta)
    cur_psi = np.array(init_psi)
    pred_l = []
    pred_l.append(np.array(cur_l))

    for [delta_l, delta_theta, delta_psi] in zip(y_delta_l, y_delta_theta, y_delta_psi):
        cur_theta = cur_theta + delta_theta
        cur_psi = cur_psi + delta_psi
        cur_l[0] = cur_l[0] + delta_l * np.sin(cur_theta) * np.cos(cur_psi)
        cur_l[1] = cur_l[1] + delta_l * np.sin(cur_theta) * np.sin(cur_psi)
        cur_l[2] = cur_l[2] + delta_l * np.cos(cur_theta)
        pred_l.append(np.array(cur_l))
        print(cur_l)

    return np.reshape(pred_l, (len(pred_l), 3))

def save_files(attack_dir, path, folder):
    pose_x = []
    pose_y = []
    pose_z = []
    gt_x = []
    gt_y = []
    gt_z = []
    attack_pose_x = []
    attack_pose_y = []
    attack_pose_z = []
    pose = []
    attack = []
    gt_p = []
    gt_q = []

    count = 0
    acc = [0 for i in range(seq_size)]
    print("Reading data files in to pose, groundtruth and attack")
    for i in range(seq_size):
        pose_filename = (
            path
            + folder
            + "result_seq"
            + str(i)
            + "_"
            + str(epoch)
            + "_fusion_"
            + str(fusion_mode)
            + ".csv"
        )
        gt_filename = (
            path
            + folder
            + "truth_pose_seq"
            + str(i)
            + "_"
            + str(epoch)
            + "_fusion_"
            + str(fusion_mode)
            + ".csv"
        )
        attack_pose_filename = (
            path
            + attack_dir
            + "result_seq"
            + str(i)
            + "_"
            + str(epoch)
            + "_fusion_"
            + str(fusion_mode)
            + ".csv"
        )
        attack_gt_filename = (
            path
            + attack_dir
            + "truth_pose_seq"
            + str(i)
            + "_"
            + str(epoch)
            + "_fusion_"
            + str(fusion_mode)
            + ".csv"
        )
        # print(pose_filename, gt_filename)

        df_pose = pd.read_csv(pose_filename, header=None)
        df_gt = pd.read_csv(gt_filename, header=None)
        df_pose_attack = pd.read_csv(attack_pose_filename, header=None)

        df_ap = pd.DataFrame({"acc1": acc})
        df_pose = df_pose.join(df_ap)
        df_pose_attack = df_pose_attack.join(df_ap)
        df_gt = df_gt.join(df_ap)

        start = 0  # combination 1 = 0 combination 2 = 3

        pose_x.append(df_pose.values[0][start + 0])
        pose_y.append(df_pose.values[0][start + 1])
        pose_z.append(df_pose.values[0][start + 2])
        gt_x.append(df_gt.values[0][start + 0])
        gt_y.append(df_gt.values[0][start + 1])
        gt_z.append(df_gt.values[0][start + 2])
        attack_pose_x.append(df_pose_attack.values[0][start + 0])
        attack_pose_y.append(df_pose_attack.values[0][start + 1])
        attack_pose_z.append(df_pose_attack.values[0][start + 2])

        if df_pose.values[0][0] == df_gt.values[0][0]:
            count = count + 1

        pose.append(df_pose.values[0])
        attack.append(df_pose_attack.values[0])
        gt_p.append(df_gt.values[0])
        gt_q.append(df_gt.values[1])

        # attack.append([df_pose_attack.values[0][start+0],df_pose_attack.values[0][start+0],df_pose_attack.values[0][start+0]])
    print(count)
    pose = np.array(pose)
    attack = np.array(attack)
    gt_p = np.array(gt_p)
    gt_q = np.array(gt_q)

    gt_p = gt_p[:, 0:3]
    pose_p = pose[:, 0:3]
    pose_q = pose[:, 3:7]
    # print(attack[1][6])
    # attack[0], len(attack))
    attack_p = attack[:, 0:3]
    attack_q = attack[:, 3:7]

    window_size = 200
    stride = 10
    init_p = pose_p[window_size // 2 - stride // 2, :]
    init_q = pose_q[window_size // 2 - stride // 2, :]
    # print(init_p, init_q, len(init_p), len(init_q))
    pose_trajectory = generate_trajectory_6d_quat(
        init_p, init_q, pose_p, pose_q
    )  # original pose
    init_p = attack_p[window_size // 2 - stride // 2, :]
    init_q = attack_q[window_size // 2 - stride // 2, :]
    at_trajectory = generate_trajectory_6d_quat(
        init_p, init_q, attack_p, attack_q
    )  # attack pose
    init_p = gt_p[window_size // 2 - stride // 2, :]
    init_q = gt_q[window_size // 2 - stride // 2, :]
    gt_trajectory = generate_trajectory_6d_quat(
        init_p, init_q, gt_p, gt_q
    )  # ground truth

    fig = plt.figure()
    ax = plt.axes()
    plt.plot(pose_trajectory[:, 0], pose_trajectory[:, 1])
    plt.plot(gt_trajectory[:, 0], gt_trajectory[:, 1])

    ax.set_xlabel("X(t)")
    ax.set_ylabel("Y(t)")
    # ax.set_zlabel('Z(t)')

    plt.legend()
    plt.show()

    print(len(gt_trajectory))

    file_name = path + attack_dir + "all_pose_" + ".csv"
    # tmp_df = pd.DataFrame({"acc1":pose_p , "acc2":pose_q})
    tmp_df = pd.DataFrame(
        {
            "pose_x": pose_p[:, 0],
            "pose_y": pose_p[:, 1],
            "pose_z": pose_p[:, 2],
            "or_x": pose_q[:, 0],
            "or_y": pose_q[:, 1],
            "or_z": pose_q[:, 2],
            "at_x": attack_p[:, 0],
            "at_y": attack_p[:, 1],
            "at_z": attack_p[:, 2],
            "at_or_x": attack_q[:, 0],
            "at_or_y": attack_q[:, 1],
            "at_or_z": attack_q[:, 2],
            "gt_x": gt_p[:, 0],
            "gt_y": gt_p[:, 1],
            "gt_z": gt_p[:, 2],
            "gt_or_x": gt_q[:, 0],
            "gt_or_y": gt_q[:, 1],
            "gt_or_z": gt_q[:, 2],
        }
    )
    np.savetxt(file_name, tmp_df.values, delimiter=",", fmt="%s")

    file_name = path + attack_dir + "trajectory_" + ".csv"
    tmp_df = pd.DataFrame(
        {
            "col1": np.array(gt_trajectory[:, 0]),
            "col2": np.array(gt_trajectory[:, 1]),
            "col3": np.array(gt_trajectory[:, 2]),
            "col4": np.array(pose_trajectory[:, 0]),
            "col5": np.array(pose_trajectory[:, 1]),
            "col6": np.array(pose_trajectory[:, 2]),
            "col7": np.array(at_trajectory[:, 0]),
            "col8": np.array(at_trajectory[:, 1]),
            "col9": np.array(at_trajectory[:, 2]),
        }
    )
    # np.savetxt(file_name], (gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2], pose_trajectory[:, 0], pose_trajectory[:, 1], pose_trajectory[:, 2],at_trajectory[:, 0], at_trajectory[:, 1], at_trajectory[:, 2]))
    np.savetxt(file_name, tmp_df.values, delimiter=",", fmt="%s")


if __name__ == "__main__":
    copy = 0  # index of the frame that needs to be copied
    paste_length = 20  # num frames where the copied image needs to be pasted
    attack_type = "one_frame"
    warm = [100]
    sample = [100]
    steady_state = [5, 10, 20]
    u_limit = [20, 10, 25]

    path = "/home/../results"
    folder = "/../"

    attack_dir = folder
    save_files(attack_dir, path, folder)
    fig = plt.figure()
    # ax = plt.axes(projection='3d')
    ax = plt.axes()

    time = np.arange(0, seq_size + 1)

    print("plotting trajectory")