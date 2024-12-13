import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets

from model_factory import ModelFactory

from train import train

self_supervised_models = ["SimCLR"]

# cd "C:\Users\Mathis\Desktop\MVA\S1\Object Recognition\TP vrai\recvis24_a3"
# Usage: python main.py --model_name SimCLR --lr 0.001 --batch-size 1028 --optimizer Adam --epochs 20 --temperature 0.05
# python main.py --model_name SimCLR --lr 0.001 --batch-size 1028 --optimizer Adam --epochs 20 --temperature 0.05

def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="RecVis A3 training script")
    parser.add_argument(
        "--data",
        type=str,
        default="sketch_recvis2024",
        metavar="D",
        help="folder where data is located. train_images/ and val_images/ need to be found in the folder",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="basic_cnn",
        metavar="MOD",
        help="Name of the model for model and transform instantiation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        metavar="B",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment",
        metavar="E",
        help="folder where experiment outputs are located.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        metavar="NW",
        help="number of workers for data loading",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        metavar="OPT",
        help="optimizer to use (default: SGD)",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0001,
        metavar="WD",
        help="weight decay for AdamW optimizer (default: 0.0)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.05,
        metavar="T",
        help="temperature parameter for the loss function (default: 0.05)",
    )

    # Optional model path for trained SimCLR
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        metavar="M",
        help="the model file to be evaluated. Usually it is of the form model_X.pth",
    )

    parser.add_argument(
        "--early_stopping",
        type=bool,
        default=False,
        metavar="ES",
        help="Whether to use early stopping or not",
    )

    parser.add_argument(
        "--stopping_patience",
        type=int,
        default=5,
        metavar="P",
        help="Number of epochs to wait before early stopping",
    )

    parser.add_argument(
        "--scheduler",
        type=str,
        default="none",
        metavar="SCH",
        help="What scheduler to use (default: none)",
    )

    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=8,
        metavar="SP",
        help="Number of epochs to wait before scheduler",
    )


    args = parser.parse_args()
    assert args.model_name in ["basic_cnn", "SimCLR", "trained_SimCLR"], "Model not implemented"
    assert args.scheduler in ["plateau", "cosine", "none"], f"Scheduler {args.scheduler} not implemented"
    assert args.optimizer in ["SGD", "Adam", "AdamW"], "Optimizer not implemented"
    assert args.model_name == "trained_SimCLR" or args.model_path is None, "Model path only for trained SimCLR"
    args.supervised = not args.model_name in self_supervised_models
    print(args)
    return args


def opts2() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RecVis A3 training script")
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.yaml",
        metavar="C",
        help="path to the config file",
    )

    args = parser.parse_args()
    return args


def main2():
    args = opts2()
    modelFactory = ModelFactory.from_yaml(args.config_file)

    train(modelFactory)

from model_files import to_do_list
def main3():
    for file in to_do_list:
        modelFactory = ModelFactory.from_yaml(file)
        train(modelFactory)
        del modelFactory
        torch.cuda.empty_cache()

def main():
    """Default Main Function."""
    # options
    args = opts()

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    # Set the seed (for reproducibility)
    torch.manual_seed(args.seed)

    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

    # load model and transform and loss
    kwargs = {
        "temperature": args.temperature,
        "model_path": args.model_path if args.model_name == "trained_SimCLR" else None,
    }
    model, data_transforms, criterion = ModelFactory(args.model_name, **kwargs).get_all()
    if use_cuda:
        print("Using GPU")
        model.cuda()
    else:
        print("Using CPU")

    # Data initialization and loading
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + "/train_images", transform=data_transforms),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + "/val_images", transform=data_transforms),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Setup optimizer
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError("Optimizer not implemented")
    
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.scheduler_patience, verbose=True)
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, verbose=True)
    else:
        scheduler = None        

    train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        use_cuda=use_cuda,
        n_epochs=args.epochs,
        args=args,
        supervised=args.supervised,
        scheduler=scheduler,
        early_stopping_patience=args.stopping_patience if args.early_stopping else None
    )


if __name__ == "__main__":
    main2()
