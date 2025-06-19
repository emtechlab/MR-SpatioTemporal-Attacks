import pandas as pd
from collections import deque
import numpy as np

# This is independent of the attacks
seq_size = 13947
# seq_size =  1000
class _State:
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.data = deque(maxlen=self.num_samples)


class CopyHoldAttack:
    def __init__(
        self,
        num_warmup=50,
        num_samples=100,
        num_steadystate=10,
        num_percentile=90,
        attack_length=10,
    ):
        self.num_warmup = num_warmup
        self.num_steadystate = num_steadystate
        self.steadystate_counter = 0
        self.attack_counter = 0
        self.attack_length = attack_length
        self.warm_flag = False
        self.num_samples = num_samples
        self.upper_bound = None
        self.lower_bound = None
        self.upperbound_percentile = num_percentile
        self.history = _State(num_samples=self.num_samples)
        # print(self.num_warmup)

    def UpdateHistory(self, new_sample=None):
        self.history.data.appendleft(new_sample)
        if len(self.history.data) >= self.num_warmup:
            self.warm_flag = True

    def ComputeBounds(
        self,
    ):
        # print(self.history.data)
        self.upper_bound = np.percentile(
            np.array(self.history.data), self.upperbound_percentile
        )  # 90
        self.lower_bound = np.percentile(
            np.array(self.history.data), 100 - self.upperbound_percentile
        )

    def PickAttackFrames(self, new_sample=None):

        # add new sample to the history
        self.UpdateHistory(new_sample=new_sample)

        # if we have not had enough samples in history
        # the attack cannot be launched
        if self.warm_flag == False:
            return -1

        # if we have enough samples, compute bounds
        self.ComputeBounds()

        # if new samples remain in bound, increase the counter
        # as soon as it goes out, reset it
        if self.lower_bound < new_sample < self.upper_bound:
            self.steadystate_counter += 1
        else:
            self.steadystate_counter = 0

        # if steady state has achieved, launch the attack.
        if (self.steadystate_counter > self.num_steadystate) and (
            self.attack_counter < self.attack_length
        ):
            self.attack_counter += 1
            return 1
        else:
            self.attack_counter = 0
            return -1

    # this function should not be a part of this class
    # keep it here for now.
    def PlotData(
        self,
    ):
        pass


def vector_steadiness(
    sequence, sequence_type, configure_parameter, config_string, num_percentile
):

    img_file_name = (
        "./results/attack_frames/image_"
        + sequence
        + "-"
        + sequence_type
        + config_string
        + str(num_percentile)
        + str(configure_parameter)
        + ".csv"
    )
    imu_file_name = (
        "./results/attack_frames/imu_"
        + sequence
        + "-"
        + sequence_type
        + config_string
        + str(num_percentile)
        + str(configure_parameter)
        + ".csv"
    )

    # print(sequence, sequence_type, configure_parameter,config_string)
    df_imu = pd.read_csv(imu_file_name, header=None)
    df_img = pd.read_csv(img_file_name, header=None)

    steady = []
    imu_value = df_imu.iloc[:, 0]
    img_value = df_img.iloc[:, 0]
    # print(len(img_value), len(imu_value))
    for i in range(seq_size):
        if imu_value[i] == img_value[i]:
            steady.append(1)
        else:
            steady.append(0)
    # print(sum(steady))
    both_steady = []
    for i in range(seq_size):
        if steady[i] == 1 and img_value[i] == 1 and imu_value[i] == 1:
            both_steady.append(1)
        else:
            both_steady.append(0)

    print(
        num_percentile,
        configure_parameter,
        sum(both_steady),
        sum(both_steady) / len(df_imu.iloc[:, 0]),
    )
    output_path = (
        "./results/attack_frames/"
        + str(sequence)
        + "-"
        + sequence_type
        + config_string
        + str(num_percentile)
        + str(configure_parameter)
        + ".csv"
    )
    tmp_df = pd.DataFrame({"steady": steady, "both_steady": both_steady})
    np.savetxt(output_path, tmp_df.values, delimiter=",", fmt="%s")
    # print(output_path)
    # print(steady)


def frame_count(
    vector, sequence, sequence_type, configure_parameter, config_string, num_percentile
):
    # vector = 'image' # imu  OR image
    # print(vector, sequence, sequence_type,configure_parameter, config_string)
    file_name = (
        "./results/gt/hash/hash_"
        + vector
        + "_"
        + sequence
        + "-"
        + sequence_type
        + ".csv"
    )

    attack_df = pd.read_csv(file_name, header=None)
    attack_df.columns = ["original", "attack"]
    # attack_counter = 0
    flag = []
    for index, row in attack_df.iterrows():
        # print(index)
        attack_flag = attack_class.PickAttackFrames(new_sample=row["original"])
        # if attack_flag > 0:
        #     attack_counter += 1
        flag.append(attack_flag)
        # print("{index},{attack_flag}".format(index=index, attack_flag=attack_flag))
    tmp_df = pd.DataFrame({"flag": flag})

    output_path = (
        "./results/attack_frames/"
        + vector
        + "_"
        + sequence
        + "-"
        + sequence_type
        + config_string
        + str(num_percentile)
        + str(configure_parameter)
        + ".csv"
    )
    print(output_path)
    np.savetxt(output_path, tmp_df.values, delimiter=",", fmt="%s")


if __name__ == "__main__":

    #### temporary data reader ####
    sequence = "5-27-7PM"
    sequence_type = "orignal"  # attack or original

    ###############################

    num_warmup = 200  # samples
    num_samples = 200
    num_steadystate = 10
    num_percentile = [65, 70, 75, 80, 85, 90, 95, 100]
    configure_parameter = [1, 2, 5, 10, 15, 20, 50, 100, 150, 200]
    configure_parameter = [1, 2, 3, 4, 5, 10, 15, 20, 50, 100, 150, 200]
    configure_parameter = np.arange(0, 210, 10)
    # configure_parameter = [200]
    for j in range(len(num_percentile)):
        for i in range(0, len(configure_parameter)):
            if i == 0:
                attack_length = configure_parameter[i] + 1
            else:
                attack_length = configure_parameter[i]
            # print(attack_length)
            attack_class = CopyHoldAttack(
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_steadystate=num_steadystate,
                num_percentile=num_percentile[j],
                attack_length=attack_length,
            )
            config_string = "attack_length_steady"
            frame_count(
                "image",
                sequence,
                sequence_type,
                attack_length,
                config_string,
                num_percentile[j],
            )
            frame_count(
                "imu",
                sequence,
                sequence_type,
                attack_length,
                config_string,
                num_percentile[j],
            )
            vector_steadiness(
                sequence, sequence_type, attack_length, config_string, num_percentile[j]
            )
