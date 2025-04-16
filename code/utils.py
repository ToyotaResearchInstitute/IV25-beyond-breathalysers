import glob
import os
import pickle
import random
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from box import Box
from sklearn.metrics import confusion_matrix


def load_config_as_box(cfg_path):
    try:
        with open(cfg_path, "r") as f:
            return Box(yaml.safe_load(f))
    except:
        raise ValueError


def set_random_seed(seed):
    print(f"Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RunningLoss:
    def __init__(self):
        self.reset()

    def reset(self):
        """reset loss counters"""
        self.train_loss = 0.0
        self.train_correct = 0.0
        self.train_samples = 0
        self.train_gt = []
        self.train_pred = []
        self.val_loss = 0.0
        self.val_correct = 0.0
        self.val_samples = 0
        self.val_gt = []
        self.val_pred = []

    def update_train(self, loss, predicted, labels):
        """Updates the running training stats."""
        correct = (predicted == labels).sum().item()
        batch_size = labels.size(0)
        self.train_loss += loss * batch_size
        self.train_correct += correct
        self.train_samples += batch_size
        self.train_gt.extend(labels.cpu().numpy())
        self.train_pred.extend(predicted.cpu().numpy())

    def update_val(self, loss, predicted, labels):
        """Updates the running validation stats."""
        correct = (predicted == labels).sum().item()
        batch_size = labels.size(0)
        self.val_loss += loss * batch_size
        self.val_correct += correct
        self.val_samples += batch_size
        self.val_gt.extend(labels.cpu().numpy())
        self.val_pred.extend(predicted.cpu().numpy())

    @property
    def avg_train_loss(self):
        """Returns the average training loss for the epoch."""
        return self.train_loss / self.train_samples if self.train_samples != 0 else 0.0

    @property
    def avg_train_acc(self):
        """Returns the average training loss for the epoch."""
        return 100 * self.train_correct / self.train_samples if self.train_samples != 0 else 0.0

    @property
    def avg_val_loss(self):
        """Returns the average validation loss for the epoch."""
        return self.val_loss / self.val_samples if self.val_samples != 0 else 0.0

    @property
    def avg_val_acc(self):
        """Returns the average validation loss for the epoch."""
        return 100 * self.val_correct / self.val_samples if self.val_samples != 0 else 0.0

    def get_tpr_fpr_F1(self, mode):
        """Returns TPR, FPR"""

        if mode == "train":
            # compute confusion matrix
            tn, fp, fn, tp = confusion_matrix(self.train_gt, self.train_pred).ravel()
        elif mode == "val":
            tn, fp, fn, tp = confusion_matrix(self.val_gt, self.val_pred).ravel()

        # Calculate TPR and FPR
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        F1 = (2 * tp) / (2 * tp + fp + fn)

        return tpr, fpr, F1


def get_results_paths(unique_name, wandb_path):
    results_paths = sorted(glob.glob(os.path.expanduser(f"{wandb_path}base-{unique_name}-*/wandb/run-*/files/results.pkl")))
    print(f"Found {len(results_paths)} files.")
    return results_paths


def aggregate_results(unique_name, wandb_path="~/wandb/"):
    results_paths = get_results_paths(unique_name, wandb_path)
    results = {}
    for f in results_paths:
        with open(f, "rb") as file:
            data = pickle.load(file)
            for exp in data:
                if exp not in results:
                    results[exp] = {}
                for pid in data[exp]:
                    if pid not in results[exp]:
                        results[exp][pid] = data[exp][pid]
    return results


def logit_to_probability(logit):
    return 1 / (1 + np.exp(-logit))


def get_avg_scores(results):
    scores = {}
    scores['F1'] = {}
    scores['Acc'] = {}
    scores['BAcc'] = {}
    for exp in results.keys():
        tn, fp, fn, tp = confusion_matrix(list(chain(*[_['labels'] for _ in results[exp].values()])), 
                                          list(chain(*[_['predicted'] for _ in results[exp].values()]))).ravel()
        scores['F1'][exp] = (2 * tp) / (2 * tp + fp + fn)
        scores['Acc'][exp] = (tp + tn) / (tp + tn + fp + fn)
        scores['BAcc'][exp] = 1/2 * tp / (tp + fn) + 1/2 * tn / (tn + fp)            
    return scores    


def get_LR_max_avg_scores(results):
    scores = {}
    scores['F1'] = {}
    scores['Acc'] = {}
    scores['BAcc'] = {}
    for exp in results.keys():
        labels = np.array(list(chain(*[_['labels'] for _ in results[exp].values()])))
        predicted = np.array(list(chain(*[_['predicted'] for _ in results[exp].values()])))
        assert(len(labels) % 2 == 0)
        
        # Reshape to (*, 2) and take max (across L & R eyes)
        labels_max = np.max(labels.reshape(-1, 2), axis=1)
        labels = labels_max.reshape(-1, 1)

        # Reshape to (*, 2) and take max (across L & R eyes)
        predicted_max = np.max(predicted.reshape(-1, 2), axis=1)
        predicted = predicted_max.reshape(-1, 1)
                
        tn, fp, fn, tp = confusion_matrix(labels, predicted).ravel()
        scores['F1'][exp] = (2 * tp) / (2 * tp + fp + fn)
        scores['Acc'][exp] = (tp + tn) / (tp + tn + fp + fn)
        scores['BAcc'][exp] = 1/2 * tp / (tp + fn) + 1/2 * tn / (tn + fp)            
    return scores    


def set_plot_style(font_size=24, grid_line_width="1.0"):
    ggplot_styles = {
        "font.family": "Times New Roman",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": font_size,
        "axes.edgecolor": "white",
        "axes.facecolor": "EBEBEB",
        "axes.grid": True,
        "axes.grid.which": "both",
        "axes.spines.left": False,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.spines.bottom": False,
        "grid.color": "white",
        "grid.linewidth": grid_line_width,
        "xtick.color": "555555",
        "xtick.major.bottom": True,
        "xtick.minor.bottom": False,
        "ytick.color": "555555",
        "ytick.major.left": True,
        "ytick.minor.left": False,
    }
    plt.rcParams.update(ggplot_styles)


TITLE_TRANSLATE = {
    "gaze_tracking": "Gaze Tracking Test",
    "fixed_gaze": "Fixed Gaze Test",
    "silent_reading": "Silent Reading Test",
    "choice_reaction": "Choice Reaction Test",
}


METRIC_TRANSLATE = {
    "BAcc": "Balanced Accuracy",
    "F1": "F1 Score",
}


LABEL_TRANSLATE = {
    "gaze_tracking": "GT",
    "fixed_gaze": "FG",
    "silent_reading": "SR",
    "choice_reaction": "CR",
    "all": "all",
}