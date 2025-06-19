# import the necessary packages
from matplotlib.cbook import is_math_text
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import imutils
from PIL import Image
import pandas as pd
import shutil

def copy_imu(imulist):
    # len(imulist)
    for i in range(len(imulist)):
        # print(i)
        df = pd.read_csv(imulist[i], header=None)
        # df.values[rows][columns]
        y = np.genfromtxt(df.values[0]).astype(np.float32).reshape(-1, 6)[0]
        for j in range(3):
            y[j+3] = -y[j+3]
        filename = input_path + "imunegate/"
        np.savetxt(filename + imulist[i].split("imu/")[1], [y], delimiter=" ", fmt="%s")

if __name__ == "__main__":
    input_path = "/home/../PV/"
    sequence = ".."
    sequence_type = "orignal"  # attack or original
    imu_path = input_path + "imu/"
    imulist = [f for f in glob.glob(imu_path + "*.txt")]
    imulist = sorted(imulist)

    copy_imu(imulist)


