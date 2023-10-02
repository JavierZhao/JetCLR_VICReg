import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from sklearn.metrics import auc
import argparse

plt.style.use(hep.style.ROOT)

def main(args):
    label = args.label
    model_label = "vicreg_" + label
    cov_loss_train_epochs = np.load(f"/ssl-jet-vol-v2/JetCLR_VICReg/models/model_performances/{label}/{model_label}_cov_loss_train_epochs.npy")
    cov_loss_val_epochs = np.load(f"/ssl-jet-vol-v2/JetCLR_VICReg/models/model_performances/{label}/{model_label}_cov_loss_val_epochs.npy")
    loss_train_batches = np.load(f"/ssl-jet-vol-v2/JetCLR_VICReg/models/model_performances/{label}/{model_label}_loss_train_batches.npy")
    loss_train_epochs = np.load(f"/ssl-jet-vol-v2/JetCLR_VICReg/models/model_performances/{label}/{model_label}_loss_train_epochs.npy")
    loss_val_batches = np.load(f"/ssl-jet-vol-v2/JetCLR_VICReg/models/model_performances/{label}/{model_label}_loss_val_batches.npy")
    loss_val_epochs = np.load(f"/ssl-jet-vol-v2/JetCLR_VICReg/models/model_performances/{label}/{model_label}_loss_val_epochs.npy")
    repr_loss_train_epochs = np.load(f"/ssl-jet-vol-v2/JetCLR_VICReg/models/model_performances/{label}/{model_label}_repr_loss_train_epochs.npy")
    repr_loss_val_epochs = np.load(f"/ssl-jet-vol-v2/JetCLR_VICReg/models/model_performances/{label}/{model_label}_repr_loss_val_epochs.npy")
    std_loss_train_epochs = np.load(f"/ssl-jet-vol-v2/JetCLR_VICReg/models/model_performances/{label}/{model_label}_std_loss_train_epochs.npy")
    std_loss_val_epochs = np.load(f"/ssl-jet-vol-v2/JetCLR_VICReg/models/model_performances/{label}/{model_label}_std_loss_val_epochs.npy")
    lct_auc_epochs = np.load(f"/ssl-jet-vol-v2/JetCLR_VICReg/models/model_performances/{label}/{model_label}_lct_auc_epochs.npy")

    # Plot loss curves in training
    fontsize = 20
    fig, ax = plt.subplots(2, 2)

    ax[0, 0].plot(loss_train_epochs, 'r') #row=0, col=0
    ax[0, 0].set_xlabel("Epochs", fontsize=fontsize)
    ax[0, 0].set_ylabel("Total loss", fontsize=fontsize)
    ax[0, 0].legend(
        title=label,
        loc="upper right",
        fontsize=18,
        title_fontsize=18,
    )

    ax[1,0].plot(repr_loss_train_epochs, 'b') #row=1, col=0
    ax[1,0].set_xlabel("Epochs", fontsize=fontsize)
    ax[1,0].set_ylabel("Invariance loss", fontsize=fontsize)
    ax[1,0].legend(
        title=label,
        loc="upper right",
        fontsize=18,
        title_fontsize=18,
    )

    ax[0,1].plot(std_loss_train_epochs, 'g') #row=0, col=1
    ax[0,1].set_xlabel("Epochs", fontsize=fontsize)
    ax[0,1].set_ylabel("Variance loss", fontsize=fontsize)
    ax[0,1].legend(
        title=label,
        loc="upper right",
        fontsize=18,
        title_fontsize=18,
    )

    ax[1,1].plot(cov_loss_train_epochs, 'y') #row=1, col=1
    ax[1,1].set_xlabel("Epochs", fontsize=fontsize)
    ax[1,1].set_ylabel("Covariance loss", fontsize=fontsize)
    ax[1,1].legend(
        title=label,
        loc="upper right",
        fontsize=18,
        title_fontsize=18,
    )
    plt.subplots_adjust(hspace=0.5, wspace=0.5) # adjust spacing between plots
    plt.figtext(0.5, 0.01, "Different loss terms in training", ha="center", fontsize=20)
    plt.savefig(f"/ssl-jet-vol-v2/JetCLR_VICReg/models/model_performances/{label}/{model_label}_loss_train_epochs.png")
    # plt.show()

    # Plot loss curves in validation
    fontsize = 20
    fig, ax = plt.subplots(2, 2)

    ax[0, 0].plot(loss_val_epochs, 'r') #row=0, col=0
    ax[0, 0].set_xlabel("Epochs", fontsize=fontsize)
    ax[0, 0].set_ylabel("Total loss", fontsize=fontsize)
    ax[0, 0].legend(
        title=label,
        loc="upper right",
        fontsize=18,
        title_fontsize=18,
    )

    ax[1,0].plot(repr_loss_val_epochs, 'b') #row=1, col=0
    ax[1,0].set_xlabel("Epochs", fontsize=fontsize)
    ax[1,0].set_ylabel("Invariance loss", fontsize=fontsize)
    ax[1,0].legend(
        title=label,
        loc="upper right",
        fontsize=18,
        title_fontsize=18,
    )

    ax[0,1].plot(std_loss_val_epochs, 'g') #row=0, col=1
    ax[0,1].set_xlabel("Epochs", fontsize=fontsize)
    ax[0,1].set_ylabel("Variance loss", fontsize=fontsize)
    ax[0,1].legend(
        title=label,
        loc="upper right",
        fontsize=18,
        title_fontsize=18,
    )

    ax[1,1].plot(cov_loss_val_epochs, 'y') #row=1, col=1
    ax[1,1].set_xlabel("Epochs", fontsize=fontsize)
    ax[1,1].set_ylabel("Covariance loss", fontsize=fontsize)
    ax[1,1].legend(
        title=label,
        loc="upper right",
        fontsize=18,
        title_fontsize=18,
    )
    plt.subplots_adjust(hspace=0.5, wspace=0.5) # adjust spacing between plots
    plt.figtext(0.5, 0.01, "Different loss terms in validation", ha="center", fontsize=20)
    plt.savefig(f"/ssl-jet-vol-v2/JetCLR_VICReg/models/model_performances/{label}/{model_label}_loss_val_epochs.png")
    # plt.show()


if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label",
        type=str,
        action="store",
        dest="label",
        default="new",
        help="a label for the model used for inference",
    )
    args = parser.parse_args()
    main(args)