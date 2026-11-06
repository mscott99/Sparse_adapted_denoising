import pandas as pd
import numpy as np
from .utils_plot import log_relplot
import matplotlib.pyplot as plt


def explode_with_pos(df: pd.DataFrame, opt_col: str = "opt_data"):
    if "experiment_found" in df.columns:
        df = df[df["experiment_found"] == True]
    df["tracked_opt"] = df[opt_col].apply(lambda x: list(enumerate(x)))
    df = df.explode("tracked_opt")
    df[["pos", "val"]] = pd.DataFrame(
        df["tracked_opt"].tolist(), index=df.index)
    return df[df["pos"].notna()]


def compare_recoveries(df:pd.DataFrame, num_imgs = 5, true_col = "true_signal", recov_col = "recovered_signal"):
    rows = df.sample(num_imgs)
    fig, axes = plt.subplots(2, num_imgs, figsize=(num_imgs*3, 6))
    for i, (_, row) in enumerate(rows.iterrows()):
        true_img = np.transpose(row[true_col], (1, 2, 0))
        recov_img = np.transpose(row[recov_col], (1, 2, 0))
        axes[0, i].imshow(true_img)
        axes[0, i].axis('off')
        axes[1, i].imshow(recov_img)
        axes[1, i].axis('off')
    # Add row labels on the left of the full plot
    plt.tight_layout()
    plt.show()

def compare_recoveries_and_reconstructed(df:pd.DataFrame, num_imgs = 5, true_col = "true_signal", recov_col = "recovered_signal", recon_col = "reconstructed_img"):
    rows = df.sample(num_imgs)
    fig, axes = plt.subplots(3, num_imgs, figsize=(num_imgs*3, 9))
    for i, (_, row) in enumerate(rows.iterrows()):
        true_img = np.transpose(row[true_col], (1, 2, 0))
        recov_img = np.transpose(row[recov_col], (1, 2, 0))
        recon_img = np.transpose(row[recon_col], (1, 2, 0))
        axes[0, i].imshow(true_img)
        axes[0, i].axis('off')
        axes[1, i].imshow(recov_img)
        axes[1, i].axis('off')
        axes[2, i].imshow(recon_img)
        axes[2, i].axis('off')
    # Add row labels on the left of the full plot
    plt.tight_layout()
    plt.show()
