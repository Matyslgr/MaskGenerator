##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## earlystopping
##

import torch

class EarlyStopping:
    def __init__(self, model_path: str, patience: int, delta: float, verbose: bool):
        self.model_path = model_path
        self.patience = patience
        self.delta = delta
        self.verbose = verbose

        self.counter = 0
        self.best_model_wts = None
        self.early_stop = False
        self.best_loss = float("inf")

    def __call__(self, val_loss, model: torch.nn.Module):
        if self.best_loss > val_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)

        elif self.best_loss - val_loss > self.delta:
            self.counter = 0
            if self.verbose:
                print(f"Validation loss improved by {self.best_loss - val_loss:.6f} (below delta)")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered")

    def save_checkpoint(self, model: torch.nn.Module):
        """Saves model when validation loss decrease."""
        self.best_model_wts = model.state_dict()
        torch.save(self.best_model_wts, self.model_path)
        if self.verbose:
            print(f"Model saved at {self.model_path}")

    def load_best_model(self, model: torch.nn.Module):
        if self.best_model_wts is not None:
            model.load_state_dict(self.best_model_wts)
        else:
            raise ValueError("No model weights to load. The model hasn't been saved yet.")
        return model
