import torch
from tqdm import tqdm


def Train(train_loader, val_loader, model, loss_fn, optimizer, n_epochs):
    val_losses = []
    train_losses =[]
    for epoch in tqdm(range(n_epochs)):
        total_train_loss = 0.0
        total_val_loss = 0.0
        for data,target in train_loader:
            model.train()                   # set model in training mode
            yhat = model(data)    # forward propogation (prediction)
            loss = loss_fn(yhat, target)  # compute the loss
            total_train_loss += loss.item()
            loss.backward()                       # back propogation
            optimizer.step()                      # update b, w using grads and lr
            optimizer.zero_grad()                 # we don't want to accumulate grad
        with torch.no_grad():
            model.eval()
            for data_val, target_val in val_loader:
                y_hat_val = model(data_val)
                val_loss = loss_fn(y_hat_val, target_val)
                total_val_loss += val_loss.item()

        train_losses.append(total_train_loss/len(train_loader))
        val_losses.append(total_val_loss / len(val_loader))
        """
        print(f"Epoch {epoch:4}:    ECE655 Model State:   ", end='')
        print(f"Train error:{train_losses[epoch]}")
        print(f"Val error:{val_losses[epoch]}")
        """
    print(f"Number of epochs {n_epochs}")
    print(f"Final training error {train_losses[-1]}")
    print(f"Final validation error {val_losses[-1]}")
    return train_losses, val_losses
    print("\n========================DONE====================:")
