from operator import gt
from unittest import result
from cv2 import reduce
import numpy as np
from pandas import concat, read_csv
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
import utils
import math
import functools

seq_size = 17567

def euler2mat(angle):

    B = angle.shape[0]
    x, y, z = angle[:,0], angle[:,1], angle[:,2]
    # print(x)
    Ms = []

    cosz = np.cos(z)
    sinz = np.sin(z)

    Ms.append(np.array([[cosz, -sinz, 0],[sinz, cosz, 0],[0, 0, 1]]))
    cosy = np.cos(y)
    siny = np.sin(y)
    Ms.append(np.array([[cosy, 0, siny],[0, 1, 0],[-siny, 0, cosy]]))
    
    cosx = np.cos(x)
    sinx = np.sin(x)
    Ms.append(np.array([[1, 0, 0],[0, cosx, -sinx],[0, sinx, cosx]]))

    return functools.reduce(np.dot, Ms[::-1])

def generate_trajectory_3d(init_l, init_theta, init_psi, y_delta_l, y_delta_theta, y_delta_psi):
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

def get_quaternion_from_euler(pos):
  """
  Convert an Euler angle to a quaternion.
   
  Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
 
  Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
  """
  
  roll = pos.iloc[:,0] 
  pitch=  pos.iloc[:,1]
  yaw =  pos.iloc[:,2]
#   print(roll,pitch, yaw)
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  df= pd.DataFrame([qx,qy,qz,qw])
  df = df.T

  return df

def save_rot_mat(rotmat):
    df = pd.DataFrame(rotmat)
    np.savetxt("dummy.txt",df.values, delimiter=',', fmt= '%s')
    from csv import reader, writer 
    with open('dummy.txt') as f, open('destination.csv', 'w') as fw: 
        writer(fw, delimiter=',').writerows(zip(*reader(f, delimiter=',')))



def plot_trajectory(dim, pose_trajectory, gt_trajectory):
    start = 0
    end = len(pose_trajectory[:, 1])
    gt_end = 15000
    if dim == 2:
        ax = plt.axes()
        plt.plot(pose_trajectory[:, 0][start:end], pose_trajectory[:, 1][start:end], c = 'k', label = 'orignal pose')
        plt.plot(gt_trajectory[:, 0][start:gt_end], gt_trajectory[:, 1][start:gt_end], c = 'r', label = 'ground truth')
        
    else:
        ax = plt.axes(projection='3d')
        plt.plot(pose_trajectory[:, 0], pose_trajectory[:, 1], pose_trajectory[:, 2], c = 'k', label = 'orignal pose')
        plt.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2], c = 'r', label = 'ground truth')
        ax.set_zlabel('Z(t)')
    ax.set_xlabel('X(t)')
    ax.set_ylabel('Y(t)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    path = '/home/../results/5-27-7PM/'
    folder = '/home/../results/'
    attack_name = "attacklength"
    attack_number = [1,2,5,10,15,20,50,100,150,200]
    attack_number = [1,10]
    
    for i in attack_number:
        path = folder + attack_name + str(i) + "/"
        # path = '/home/../results/5-27-7PM-train/'
        window_size = 1
        stride = 1
        
        df_ori = read_csv(path+"result_ori1_0_fusion_3.csv",header= None)
        df_pos = read_csv(path+"result_pose1_0_fusion_3.csv",header= None)
        t_df_ori = read_csv(path+"truth_euler_seq1_0_fusion_3.csv",header= None)
        t_df_pos = read_csv(path+"truth_pose_seq1_0_fusion_3.csv",header= None)

        df_ori = df_ori[df_ori.reset_index().index % 2 == 0]  # Selects every 2nd raw starting from 0
        t_df_ori = t_df_ori[t_df_ori.reset_index().index % 2 == 0]
        df_pos = df_pos[df_pos.reset_index().index % 2 == 0]
        t_df_pos = t_df_pos[t_df_pos.reset_index().index % 2 == 0]
        print(len(t_df_ori), len(t_df_pos))
        
        acc = [1 for i in range(len(df_ori))]
        df_ad = pd.DataFrame({'acc1': acc})

        ori = df_ori.join(df_ad)
        t_ori = t_df_ori.join(df_ad)
        # ori = get_quaternion_from_euler(df_ori)
        # t_ori = get_quaternion_from_euler(t_df_ori)
        print(len(ori), len(t_ori))
        df_ap = pd.DataFrame()
        df_ap = pd.concat([df_pos, df_ori], axis= 1)

        translation = df_ap.to_numpy()[:, :3]
        rot = df_ap.to_numpy()[:,3:]
        # rot_mat = euler2mat(rot)  # [B, 3, 3]
        # save_rot_mat(rot_mat)

        init_p = df_pos.to_numpy()[window_size//2 - stride//2, :]
        init_q = ori.to_numpy()[window_size//2 - stride//2, :]
        pose_trajectory = generate_trajectory_6d_quat(init_p,init_q, df_pos.to_numpy(),ori.to_numpy()) # original pose 

        init_p = t_df_pos.to_numpy()[window_size//2 - stride//2, :]
        init_q = t_ori.to_numpy()[window_size//2 - stride//2, :]
        gt_trajectory = generate_trajectory_6d_quat(init_p,init_q,t_df_pos.to_numpy(),t_ori.to_numpy()) # ground truth

        
        tmp_df = pd.DataFrame({ 'col1': np.array(gt_trajectory[:, 0]),
        'col2' :  np.array(gt_trajectory[:, 1]),
        'col3':np.array(gt_trajectory[:, 2]),
        'col4': np.array(pose_trajectory[:, 0]),
        'col5' :np.array(pose_trajectory[:, 1]),
        'col6' : np.array(pose_trajectory[:, 2])})
        df_result = pd.DataFrame()
        df_result = pd.concat([df_pos, df_ori, t_df_pos,t_df_ori, tmp_df], axis= 1)
        print(len(df_result), len(gt_trajectory))
        print(df_result)
        np.savetxt(path + "result_final_skiprow.csv", df_result.values[0:13927], delimiter=',', fmt= '%s')
        dim = 2