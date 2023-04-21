import torch
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F

def train_model(model, train_dataloader, val_dataloader, num_epochs, loss_fn, lr = 1e-3):
    """
    Trains the given model for `num_epochs` epochs. Use SGD as an optimizer.
    You may need to use `train_step` and `val_step`.
    :param model: Pytorch nn.Module.
    :param train_dataloader: Pytorch DataLoader with the training data.
    :param val_dataloader: Pytorch DataLoader with the validation data.
    :param num_epochs: <int> number of epochs to train the model.
    :param lr: <float> learning rate for the weight update.
    :return:
    """
    optimizer = None
    # Initialize the optimizer
    # --- Your code here

    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas = (0.9, 0.990), eps = 1e-08, weight_decay=0, amsgrad=False)

    # ---
    pbar = tqdm(range(num_epochs))
    train_losses = []
    val_losses = []
    for epoch_i in pbar:
        train_loss_i = None
        val_loss_i = None
        # --- Your code here

        train_loss_i = train_step(model, train_dataloader, optimizer, loss_fn)
        val_loss_i = val_step(model, val_dataloader, loss_fn)

        # ---
        pbar.set_description(
            f'Train Loss: {train_loss_i:.4f} | Validation Loss: {val_loss_i:.4f}')
        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)
    return train_losses, val_losses

def train_step(model, train_loader, optimizer, loss_fn) -> float:
    """
    Performs an epoch train step.
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param optimizer: Pytorch optimizer
    :return: train_loss <float> representing the average loss among the different mini-batches.
        Loss needs to be MSE loss.
    """
    train_loss = 0.  # TODO: Modify the value
    # Initialize the train loop
    # --- Your code here

    model.train()
    

    # ---
    for batch_idx, (data, target) in enumerate(train_loader):
        # --- Your code here

        optimizer.zero_grad()

        res = model(data)
        loss = loss_fn(res, target)
        loss.backward()
        optimizer.step()

        # ---
        train_loss += loss.item()
    return train_loss/len(train_loader)


def val_step(model, val_loader, loss_fn) -> float:
    """
    Perfoms an epoch of model performance validation
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param optimizer: Pytorch optimizer
    :return: val_loss <float> representing the average loss among the different mini-batches
    """
    val_loss = 0.  # TODO: Modify the value
    # Initialize the validation loop
    # --- Your code here

    model.eval()

    # ---
    for batch_idx, (data, target) in enumerate(val_loader):
        loss = None
        # --- Your code here

        res = model(data)
        loss = loss_fn(res, target)

        # ---
        val_loss += loss.item()
    return val_loss/len(val_loader)