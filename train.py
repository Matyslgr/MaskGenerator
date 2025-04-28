##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## train
##

import os
import argparse
from utils import set_deterministic_behavior, get_all_pairs_path
from model import MyUNet
from config import DATASETS_DIR

def parse_args():
    parser = argparse.ArgumentParser(description='Script to train the model.')

    # Group for model-related parameters
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument('--n_convs', type=int, default=2, help='The number of convolutional layers for each block in the UNet model')
    model_group.add_argument('--num_filters', type=int, nargs='+', default=[32, 64, 128, 256], help='The number of filters for each encoder block in the UNet model')
    model_group.add_argument('--dropout', type=float, default=0.0, help='The dropout rate for the model')

    # Group for training-related parameters
    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument('--dataset_version', type=str, required=True, help='The version of the dataset to use for training')
    training_group.add_argument('--seed', type=int, default=42, help='The seed for random number generation')
    training_group.add_argument('--batch_size', type=int, default=16, help='The batch size for training')
    training_group.add_argument('--num_epochs', type=int, default=100, help='The number of epochs for training')
    training_group.add_argument('--lr', type=float, default=0.001, help='The learning rate for the optimizer')
    training_group.add_argument('--step_size', type=int, default=10, help='The step size for the learning rate scheduler')
    training_group.add_argument('--gamma', type=float, default=0.1, help='The gamma value for the learning rate scheduler')
    training_group.add_argument('--patience', type=int, default=30, help='The patience value for early stopping')
    training_group.add_argument('--delta', type=float, default=0.0, help='The delta value for early stopping')

    # Group for other parameters (non-model-related)
    other_group = parser.add_argument_group('Other Parameters')
    other_group.add_argument('--mode', type=str, choices=['crossval', 'fulltrain'], default='crossval', help='The mode of training (cross-validation or full training)')
    other_group.add_argument('--hash', type=str, required=True, help='The hash of the model to train')
    other_group.add_argument('--model_name_template', type=str, required=True, help='The template for the model name')
    other_group.add_argument('--verbose', action='store_true', help='Whether to print verbose output')

    return parser.parse_args()

def create_model(args):
    return MyUNet(
        in_channels=3,
        out_channels=1,
        num_filters=args.num_filters,
        n_convs=args.n_convs,
        dropout=args.dropout
    )

def main():
    args = parse_args()

    # Set the random seed for reproducibility
    set_deterministic_behavior(args.seed)

    dataset_dir = os.path.join(DATASETS_DIR, args.dataset_version).replace("\\", "/")

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist.")

    pairs_path = get_all_pairs_path(dataset_dir)

    # pairs_path = pairs_path[:100]  # Limit to 100 pairs for testing

    print(f"[INFO] Successfully loaded {len(pairs_path)} pairs of images and masks.")

    if args.mode == "crossval":
        from crossval_trainer import CrossvalTrainer
        trainer = CrossvalTrainer(args, create_model)
        trainer.train(pairs_path)
    elif args.mode == "fulltrain":
        from full_trainer import FullTrainer
        trainer = FullTrainer(args, create_model)
        trainer.train(pairs_path)
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Choose 'crossval' or 'fulltrain'.")

if __name__ == "__main__":
    main()
