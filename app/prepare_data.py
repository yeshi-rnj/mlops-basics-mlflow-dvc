"""Create the dataset file (tracked by DVC) and a small overview plot."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

import config


def main():
    df = load_breast_cancer(as_frame=True).frame
    df.to_csv(config.DATA_PATH, index=False)
    print(f"Saved {config.DATA_PATH}  shape={df.shape}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts = df["target"].value_counts().sort_index()
    axes[0].bar(config.CLASS_NAMES, counts.values, color=["#d95f5f", "#4c9a6a"])
    axes[0].set_title("Class distribution")
    axes[0].set_ylabel("Samples")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 4, str(v), ha="center")
    for label, name, color in [(0, "malignant", "#d95f5f"), (1, "benign", "#4c9a6a")]:
        axes[1].hist(df.loc[df.target == label, "mean radius"], bins=25,
                     alpha=0.65, label=name, color=color)
    axes[1].set_title("Mean radius by class")
    axes[1].set_xlabel("mean radius")
    axes[1].legend()
    fig.suptitle(f"Breast Cancer Wisconsin dataset ({df.shape[0]} samples, "
                 f"{df.shape[1] - 1} features)")
    fig.tight_layout()
    fig.savefig(config.IMG_DIR / "01_dataset_overview.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
