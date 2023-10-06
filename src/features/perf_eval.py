# import standard python modules
import os
import sys
import numpy as np
from sklearn import metrics

# import torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# import simple FCN network
from src.models.fcn_linear import fully_connected_linear_network
from src.models.fcn import fully_connected_network

# import preprocessing functions
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, RobustScaler


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_perf_stats(labels, measures):
    measures = np.nan_to_num(measures)
    auc = metrics.roc_auc_score(labels, measures)
    fpr, tpr, thresholds = metrics.roc_curve(labels, measures)
    fpr2 = [fpr[i] for i in range(len(fpr)) if tpr[i] >= 0.5]
    tpr2 = [tpr[i] for i in range(len(tpr)) if tpr[i] >= 0.5]
    try:
        imtafe = np.nan_to_num(
            1 / fpr2[list(tpr2).index(find_nearest(list(tpr2), 0.5))]
        )
        # imtafe: inverse of background rejection at 50% signal efficiency
    except:
        imtafe = 1
    return auc, imtafe

# ( self, input_size, output_size, hidden_size, n_hidden, dropout_rate, opt, learning_rate ):
def linear_classifier_test(
    linear_input_size,
    linear_batch_size,
    linear_n_epochs,
    linear_learning_rate,
    reps_tr_in,
    trlab_in,
    reps_te_in,
    telab_in,
    val_fraction=0.1,      # Fraction of training data to use as validation
    linear_opt="adam",
    n_hidden=0,
    hidden_size=0
):
    # define the global base device
    world_size = torch.cuda.device_count()
    if world_size:
        device = torch.device("cuda:0")
        for i in range(world_size):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        device = "cpu"
        print("Device: CPU")
    xdevice = device
#     xdevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Splitting training data into training and validation sets
    num_val_samples = int(val_fraction * reps_tr_in.shape[0])
    shuffled_indices = torch.randperm(reps_tr_in.shape[0])
    reps_val_in = reps_tr_in[shuffled_indices[:num_val_samples]]
    vallab_in = trlab_in[shuffled_indices[:num_val_samples]]
    reps_tr_in = reps_tr_in[shuffled_indices[num_val_samples:]]
    trlab_in = trlab_in[shuffled_indices[num_val_samples:]]

    if n_hidden == 0:
        fcn_linear = fully_connected_linear_network(
            linear_input_size, 1, linear_opt, linear_learning_rate
        )
    else: 
        fcn_linear = fully_connected_network(linear_input_size, 1, hidden_size, n_hidden, 0.1, linear_opt, linear_learning_rate)
    fcn_linear.to(xdevice)
    bce_loss = nn.BCELoss()
    sigmoid = nn.Sigmoid()
    losses = []
    val_losses = []    # List to store validation losses
    
    if linear_opt == "sgd":
        scheduler = torch.optim.lr_scheduler.StepLR(
            fcn_linear.optimizer, 100, gamma=0.6, last_epoch=-1, verbose=False
        )
#     elif linear_opt == "adam":
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(fcn_linear.optimizer, 'min', patience=10)
    best_val_loss = float('inf')  # Initialize to a very high value
    best_model_state = None  # Placeholder for the best model state
    redo_lct = False
    epoch = 0
    while epoch < linear_n_epochs:
        if redo_lct:
            # re-initialize the linear classifier
            print("re-initialized LCT")
            losses, val_losses = [], []
            best_val_loss = float('inf')
            best_model_state = None
            if n_hidden == 0:
                fcn_linear = fully_connected_linear_network(
                    linear_input_size, 1, linear_opt, linear_learning_rate
                )
            else: 
                fcn_linear = fully_connected_network(linear_input_size, 1, hidden_size, n_hidden, 0.1, linear_opt, linear_learning_rate)
            fcn_linear.to(xdevice)
            redo_lct = False
            epoch = 0
            
        indices_list = torch.split(
            torch.randperm(reps_tr_in.shape[0]), linear_batch_size
        )
        losses_e = []
        
        for i, indices in enumerate(indices_list):
            fcn_linear.optimizer.zero_grad()
            x = reps_tr_in[indices, :]
            l = trlab_in[indices]
            x = torch.Tensor(x).view(-1, linear_input_size).to(xdevice)
            l = torch.Tensor(l).float().view(-1, 1).to(xdevice)
            z = sigmoid(fcn_linear(x)).to(xdevice)
            loss = bce_loss(z, l).to(xdevice)
            loss.backward()
            fcn_linear.optimizer.step()
            losses_e.append(loss.detach().cpu().numpy())
        
        # Calculate validation loss at the end of the epoch
        with torch.no_grad():
            x_val = torch.Tensor(reps_val_in).view(-1, linear_input_size).to(xdevice)
            l_val = torch.Tensor(vallab_in).float().view(-1, 1).to(xdevice)
            z_val = sigmoid(fcn_linear(x_val)).to(xdevice)
            val_loss = bce_loss(z_val, l_val).to(xdevice)
            val_losses.append(val_loss.detach().cpu().numpy())

            # Save model state if current validation loss is the best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = fcn_linear.state_dict()

        losses.append(np.mean(np.array(losses_e)))
        if linear_opt == "sgd":
            scheduler.step(val_losses[-1])
        
        if epoch == 50:
            print(losses[epoch])
            if losses[epoch] >= 40:
                # redo LCT
                redo_lct = True
        epoch += 1
            
    # Load the best model state
    fcn_linear.load_state_dict(best_model_state)
    out_dat = (
        fcn_linear(torch.Tensor(reps_te_in).view(-1, linear_input_size).to(xdevice))
        .detach()
        .cpu()
        .numpy()
    )
    out_lbs = telab_in
    return out_dat, out_lbs, losses, val_losses    # Return the validation losses as well


