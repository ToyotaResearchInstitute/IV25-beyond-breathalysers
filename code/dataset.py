import glob
import os
import random
import warnings
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
import torch

warnings.simplefilter(action="ignore", category=FutureWarning)

TOBII_RENAME = {
    "tobii_right_eye_gaze_pt_in_display_x": "r_gpd_x",
    "tobii_right_eye_gaze_pt_in_display_y": "r_gpd_y",
    "tobii_right_eye_gaze_pt_validity": "r_gpd_v",
    "tobii_right_eye_pupil_diameter": "r_p_d",
    "tobii_right_eye_pupil_validity": "r_p_v",
    "tobii_right_eye_gaze_origin_in_user_x": "r_gou_x",
    "tobii_right_eye_gaze_origin_in_user_y": "r_gou_y",
    "tobii_right_eye_gaze_origin_in_user_z": "r_gou_z",
    "tobii_right_eye_gaze_origin_validity": "r_gou_v",
    "tobii_left_eye_gaze_pt_in_display_x": "l_gpd_x",
    "tobii_left_eye_gaze_pt_in_display_y": "l_gpd_y",
    "tobii_left_eye_gaze_pt_validity": "l_gpd_v",
    "tobii_left_eye_pupil_diameter": "l_p_d",
    "tobii_left_eye_pupil_validity": "l_p_v",
    "tobii_left_eye_gaze_origin_in_user_x": "l_gou_x",
    "tobii_left_eye_gaze_origin_in_user_y": "l_gou_y",
    "tobii_left_eye_gaze_origin_in_user_z": "l_gou_z",
    "tobii_left_eye_gaze_origin_validity": "l_gou_v",
    "tobii log time": "ts",
}


class VisuomotorDataset(torch.utils.data.Dataset):
    def __init__(self, participant_dict, split, cfg, norm_data=None):
        self.split = split
        self.participant_list = participant_dict[split]
        self.window_obs = cfg.data.window_obs
        self.test_sample_gap = cfg.test.sample_gap
        self.resample_rate = cfg.data.resample_rate
        self.normalize = cfg.data.normalize
        self.n_samples = cfg.data.n_samples
        self.cfg = cfg
        self.use_single_eye = cfg.data.single_eye
        self.tdfs, self.edfs, self.norm_mean, self.norm_std = load_visuomotor_test_data(
            cfg.data.path, participant_list=self.participant_list
        )
        self.exp_keys = ["fixed_gaze", "gaze_tracking", "silent_reading", "choice_reaction"]
        self.exp_to_use = cfg[split].exp_to_use
        self.data_list = self.generate_exppid_list()
        self.data_keys = cfg.data.data_keys

        if norm_data is not None:
            self.norm_mean = norm_data[0]
            self.norm_std = norm_data[1]
        else:
            self.norm_mean = dict(
                zip(self.data_keys, [self.norm_mean[k] if k in self.norm_mean.keys() else 0 for k in self.data_keys])
            )
            self.norm_std = dict(
                zip(self.data_keys, [self.norm_std[k] if k in self.norm_std.keys() else 1 for k in self.data_keys])
            )

    def get_raw_data(self):
        return self.tdfs, self.edfs

    def get_norm_data(self):
        return self.norm_mean, self.norm_std

    def generate_exppid_list(self):
        return list(product(self.exp_to_use, self.participant_list))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        exp, pid = self.data_list[index]
        tdfs = self.tdfs[exp][pid]
        edfs = self.edfs[exp][pid]
        if pid.startswith("P"):
            intoxication_change = np.array(1, dtype=np.int64)
        else:
            intoxication_change = np.array(0, dtype=np.int64)

        assert self.window_obs < len(tdfs[0]), print("window_obs must be smaller than length of data sample")

        if self.split == "train":
            # random start index
            start_idx_list = np.random.choice(
                np.arange(
                    self.cfg.train.timing_jitter, len(tdfs[0]) - self.window_obs - self.cfg.train.timing_jitter + 1
                ),
                size=self.n_samples,
                replace=False,
            )
            # randomly sample eye
            if self.use_single_eye:
                if random.random() < 0.5:
                    remove_eye = "l"
                else:
                    remove_eye = "r"
        elif self.split == "val":
            # tile data with no overlap and randomly sample eye
            start_idx_list = np.arange(60 * 1, len(tdfs[0]) - self.window_obs + 1, self.window_obs)
            if self.use_single_eye:
                if random.random() < 0.5:
                    remove_eye = "l"
                else:
                    remove_eye = "r"
        elif self.split == "test":
            # tile the data, sample both eyes
            if self.use_single_eye:
                remove_eye = None
            # use first 1-31s of each test (same amount of time for each test)
            start_idx_list = np.arange(60, 31 * 60 - self.window_obs, self.test_sample_gap)

        if self.use_single_eye:
            # single eye sampling logic
            if remove_eye is not None:
                data_keys = sorted([d for d in self.data_keys if not d.startswith(remove_eye)])

                x = torch.zeros(len(start_idx_list), 2, len(data_keys), 512, dtype=torch.float32)

                for ii, start_idx in enumerate(start_idx_list):
                    if self.split == "train":
                        jitter0 = random.randint(-self.cfg.train.timing_jitter, self.cfg.train.timing_jitter)
                        jitter1 = random.randint(-self.cfg.train.timing_jitter, self.cfg.train.timing_jitter)
                    else:
                        jitter0, jitter1 = 0, 0

                    sample0 = tdfs[0].iloc[start_idx + jitter0 : start_idx + self.window_obs + jitter0]
                    sample1 = tdfs[1].iloc[start_idx + jitter1 : start_idx + self.window_obs + jitter1]
                    subsample0 = sample0[data_keys].values
                    subsample1 = sample1[data_keys].values

                    if self.normalize:
                        if self.split == "train":
                            for kk, dk in enumerate(data_keys):
                                subsample0[:, kk] = (
                                    subsample0[:, kk] - (1 + (random.random() - 0.5) * 0.1) * self.norm_mean[dk]
                                ) / ((1 + (random.random() - 0.5) * 0.1) * self.norm_std[dk])
                                subsample1[:, kk] = (
                                    subsample1[:, kk] - (1 + (random.random() - 0.5) * 0.1) * self.norm_mean[dk]
                                ) / ((1 + (random.random() - 0.5) * 0.1) * self.norm_std[dk])
                        else:
                            for kk, dk in enumerate(data_keys):
                                subsample0[:, kk] = (subsample0[:, kk] - self.norm_mean[dk]) / self.norm_std[dk]
                                subsample1[:, kk] = (subsample1[:, kk] - self.norm_mean[dk]) / self.norm_std[dk]

                    if self.split == "train":
                        flip = random.random() < 0.5
                    else:
                        flip = False

                    # flipping augmentations, or not
                    if flip and intoxication_change == 1 and self.cfg[self.split].drunkenflip:
                        x[ii, 0, :, :] = torch.tensor(subsample1, dtype=torch.float32).t()
                        x[ii, 1, :, :] = torch.tensor(subsample0, dtype=torch.float32).t()
                        intoxication_change = np.array(0, dtype=np.int64)  # could be negative but try 0 for now
                    elif flip and intoxication_change == 0 and self.cfg[self.split].soberflip:
                        x[ii, 0, :, :] = torch.tensor(subsample1, dtype=torch.float32).t()
                        x[ii, 1, :, :] = torch.tensor(subsample0, dtype=torch.float32).t()
                    else:
                        x[ii, 0, :, :] = torch.tensor(subsample0, dtype=torch.float32).t()
                        x[ii, 1, :, :] = torch.tensor(subsample1, dtype=torch.float32).t()

                intoxication_change = intoxication_change.repeat(len(start_idx_list))
                
            else:
                # sample both eyes separately then combine
                data_keys = sorted([d for d in self.data_keys if not d.startswith("l")])
                x = torch.zeros(len(start_idx_list) * 2, 2, len(data_keys), 512, dtype=torch.float32)
                for jj, remove_eye in enumerate(["l", "r"]):
                    data_keys = sorted([d for d in self.data_keys if not d.startswith(remove_eye)])

                    for ii, start_idx in enumerate(start_idx_list):
                        sample0 = tdfs[0].iloc[start_idx : start_idx + self.window_obs]
                        sample1 = tdfs[1].iloc[start_idx : start_idx + self.window_obs]
                        subsample0 = sample0[data_keys].values
                        subsample1 = sample1[data_keys].values

                        for kk, dk in enumerate(data_keys):
                            subsample0[:, kk] = (subsample0[:, kk] - self.norm_mean[dk]) / self.norm_std[dk]
                            subsample1[:, kk] = (subsample1[:, kk] - self.norm_mean[dk]) / self.norm_std[dk]

                        try:
                            x[ii * 2 + jj, 0, :, :] = torch.tensor(subsample0, dtype=torch.float32).t()
                            x[ii * 2 + jj, 1, :, :] = torch.tensor(subsample1, dtype=torch.float32).t()
                        except:
                            import IPython
                            IPython.embed()

                intoxication_change = intoxication_change.repeat(len(start_idx_list) * 2)
        else:
            # sample both eyes
            data_keys = sorted([d for d in self.data_keys])

            x = torch.zeros(len(start_idx_list), 2, len(data_keys), 512, dtype=torch.float32)

            for ii, start_idx in enumerate(start_idx_list):
                if self.split == "train":
                    jitter0 = random.randint(-self.cfg.train.timing_jitter, self.cfg.train.timing_jitter)
                    jitter1 = random.randint(-self.cfg.train.timing_jitter, self.cfg.train.timing_jitter)
                else:
                    jitter0, jitter1 = 0, 0

                sample0 = tdfs[0].iloc[start_idx + jitter0 : start_idx + self.window_obs + jitter0]
                sample1 = tdfs[1].iloc[start_idx + jitter1 : start_idx + self.window_obs + jitter1]
                subsample0 = sample0[data_keys].values
                subsample1 = sample1[data_keys].values

                if self.normalize:
                    if self.split == "train":
                        for kk, dk in enumerate(data_keys):
                            subsample0[:, kk] = (
                                subsample0[:, kk] - (1 + (random.random() - 0.5) * 0.1) * self.norm_mean[dk]
                            ) / ((1 + (random.random() - 0.5) * 0.1) * self.norm_std[dk])
                            subsample1[:, kk] = (
                                subsample1[:, kk] - (1 + (random.random() - 0.5) * 0.1) * self.norm_mean[dk]
                            ) / ((1 + (random.random() - 0.5) * 0.1) * self.norm_std[dk])
                    else:
                        for kk, dk in enumerate(data_keys):
                            subsample0[:, kk] = (subsample0[:, kk] - self.norm_mean[dk]) / self.norm_std[dk]
                            subsample1[:, kk] = (subsample1[:, kk] - self.norm_mean[dk]) / self.norm_std[dk]

                if self.split == "train":
                    flip = random.random() < 0.5
                else:
                    flip = False

                # flipping augmentations, or not
                if flip and intoxication_change == 1 and self.cfg[self.split].drunkenflip:
                    x[ii, 0, :, :] = torch.tensor(subsample1, dtype=torch.float32).t()
                    x[ii, 1, :, :] = torch.tensor(subsample0, dtype=torch.float32).t()
                    intoxication_change = np.array(0, dtype=np.int64)  # could be negative but try 0 for now
                elif flip and intoxication_change == 0 and self.cfg[self.split].soberflip:
                    x[ii, 0, :, :] = torch.tensor(subsample1, dtype=torch.float32).t()
                    x[ii, 1, :, :] = torch.tensor(subsample0, dtype=torch.float32).t()
                else:
                    x[ii, 0, :, :] = torch.tensor(subsample0, dtype=torch.float32).t()
                    x[ii, 1, :, :] = torch.tensor(subsample1, dtype=torch.float32).t()

            intoxication_change = intoxication_change.repeat(len(start_idx_list))

        # replace nans with 0
        x = torch.nan_to_num(x, nan=0.0)

        if self.resample_rate < 60:
            assert 60 % self.resample_rate == 0
            # Note: this only resamples the gaze tracking, not the "event" which is dim 0
            if self.resample_rate == 30:
                x[:, :, 1:, 1::2] = x[:, :, 1:, ::2]
            if self.resample_rate == 20:
                # Define the start and step values for indexing
                step = 3
                last_dim_length = x.shape[-1]
                valid_length = last_dim_length - (last_dim_length % step)  # Make it divisible by 3
                x[:, :, 1:, 1:valid_length:3] = x[:, :, 1:, :valid_length:3]
                x[:, :, 1:, 2:valid_length:3] = x[:, :, 1:, :valid_length:3]
            if self.resample_rate == 10:
                step = 6
                last_dim_length = x.shape[-1]
                valid_length = last_dim_length - (last_dim_length % step)  # Make it divisible by 3
                x[:, :, 1:, 1:valid_length:6] = x[:, :, 1:, :valid_length:6]
                x[:, :, 1:, 2:valid_length:6] = x[:, :, 1:, :valid_length:6]
                x[:, :, 1:, 3:valid_length:6] = x[:, :, 1:, :valid_length:6]
                x[:, :, 1:, 4:valid_length:6] = x[:, :, 1:, :valid_length:6]
                x[:, :, 1:, 5:valid_length:6] = x[:, :, 1:, :valid_length:6]

        return x, intoxication_change, (exp, pid)


def fetch_participants(data_path, version="all"):
    """Return a list of participants in the Participants/ path"""
    if version == "all":
        search_string = "*"
    elif version == "v1":
        search_string = "P7*"  # v1 participants
    elif version == "v2":
        search_string = "72*"  # v2 participants
    participant_list = glob.glob(os.path.expanduser(os.path.join(data_path, search_string)))
    participant_list = sorted([p.split("/")[-1] for p in participant_list])
    
    # exclude due to missing data / pilot
    EXCLUDE_SET = {
        "P701",  # extended pilot, ignore >>
        "P711",  # dropped (overly intoxicated) >>
        "7218",  # missing round 2 data due to dropout >>
        "7219",
        "7225",
        "7228",
        "7229",
        "7237",
        "7238",  # unused (round 30 control)
        "7203",  # didn't respond to choice reaction test - likely button issue
    }
    return [p for p in participant_list if p not in EXCLUDE_SET]


def load_visuomotor_test_data(data_path, participant_list):
    """Loads all visuomotor test data and computes mean + std, from a list of participants"""

    assert len(participant_list) > 0

    tdfs_out = {}
    edfs_out = {}

    exp_list = ["fixed_gaze", "gaze_tracking", "silent_reading", "choice_reaction"]
    event_replace_dict = {
        r"^START.*": np.float32(0.0),  # Matches any string starting with 'START'
        r"^END.*": np.float32(0.0),  # Matches any string starting with 'END'
        r"^TARGET.*": np.float32(1.0),  # Matches any string starting with 'Target'
        r"^Correct": np.float32(2.0),  # Matches any string starting with 'Target'
        r"^Incorrect.*": np.float32(0.0),  # Matches any string starting with 'Incorrect'
        r".* .* .*": np.float32(-1.0),  # must contain at least 2 spaces (reading text)
    }

    norm_values = defaultdict(list)
    norm_subsample_rate = 10

    for exp in exp_list:
        tdfs_out[exp] = {}
        edfs_out[exp] = {}

        durations = []  # to make all data for a given exp the same duration

        for pid in participant_list:
            edf = []
            tdf = []

            for R in ["R1", "R2"]:
                if R == "R1":
                    # first round 
                    test_name = f"stationary_tasks/{exp}"
                else:
                    # second round
                    test_name = f"stationary_tasks_recap/{exp}"

                csv_name = f"sim_bag/experiment_{exp.replace('_','') + 'test'}_event_log.csv"
                csv_path = os.path.expanduser(os.path.join(data_path, pid, R, test_name, csv_name))
                assert os.path.exists(csv_path), print(f"{csv_path} not found.")
                df = pd.read_csv(csv_path)
                df = df.rename(columns={f"{exp} log time": "ts"})
                df = df.rename(columns={f"{exp}_data": "data"})
                df["ts"] -= df["ts"].iloc[0]
                df = df.set_index("ts")
                assert df.iloc[-1].values == "END", print(f"last val of edf is not END: {csv_path}")
                durations.append(df.index[-1])
                edf.append(df)

                csv_name = "sim_bag/experiment_tobii_frame.csv"
                csv_path = os.path.expanduser(os.path.join(data_path, pid, R, test_name, csv_name))
                assert os.path.exists(csv_path), print(f"{csv_path} not found.")
                df = pd.read_csv(csv_path, dtype=np.float64)
                df = df[list(TOBII_RENAME.keys())]
                df = df.rename(columns=TOBII_RENAME)
                df["ts"] -= df["ts"].iloc[0]
                df = df.set_index("ts")
                tdf.append(df)

            edfs_out[exp][pid] = edf
            tdfs_out[exp][pid] = tdf

        max_duration = max(durations)
        t_common = np.arange(0, max_duration, 1 / 60)
        for pid in participant_list:
            for ii in [0, 1]:
                # forward fill through nans
                tdf_reindexed = tdfs_out[exp][pid][ii].reindex(t_common, method="nearest").ffill().copy()

                # compute norm and mean
                for k in tdfs_out[exp][pid][ii]:
                    if k[-1] != "v":  # ignore validity messages
                        norm_values[k].extend(tdfs_out[exp][pid][ii][k].values[::norm_subsample_rate])

                # add edf data to tdf as a column "event"
                tdf_reindexed["event"] = 0

                edf = edfs_out[exp][pid][ii].copy().infer_objects(copy=False).replace(event_replace_dict, regex=True)
                # edf = edf_replaced.astype(np.float64)

                for idx1 in edf.index:
                    closest_idx = tdf_reindexed.index.get_indexer([idx1], method="nearest")[0]
                    closest_ts = tdf_reindexed.index[closest_idx]
                    tdf_reindexed.loc[closest_ts, "event"] = edf.loc[idx1, "data"]

                tdfs_out[exp][pid][ii] = tdf_reindexed

    # compute norm constants - per data stream over all experiments
    norm_mean = {}
    norm_std = {}
    for k in norm_values:
        norm_mean[k] = np.nanmean(norm_values[k])
        norm_std[k] = np.nanstd(norm_values[k])

    return tdfs_out, edfs_out, norm_mean, norm_std
