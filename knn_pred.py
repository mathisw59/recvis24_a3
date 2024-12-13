import torch
import torch.nn as nn
from timm import create_model

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from model_factory import ModelFactory
from model import getBackboneModel
from yaml_parser import TrainingConfig
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from data import data_transforms

import clip

from torchvision import models
import torchvision.transforms as transforms

import os

# List of configuration files
config_files = [
    # 'configuration_files/supervised_mobilenetv3.yaml',
    # 'configuration_files/supervised_mobilenetv3_scratch.yaml',
    # 'configuration_files/supervised_resnet.yaml',
    # 'configuration_files/supervised_resnet_scratch.yaml',
    # 'configuration_files/supervised_squeezenet.yaml',
    # 'configuration_files/supervised_squeezenet_scratch.yaml',
    # 'configuration_files/supervised_vit.yaml',
    'configuration_files/supervised_vit_scratch.yaml',
    # 'configuration_files/supervised_basic_cnn.yaml',
    # 'configuration_files/supervised_basic_cnn_scratch.yaml',
]

from PIL import Image
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")



class KNN(nn.Module):
    def __init__(self, n_neighbors=3):
        super(KNN, self).__init__()
        self.n_neighbors = n_neighbors

    def forward(self, x_train, y_train, x_test):
        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric='cosine')
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        return y_pred


if __name__ == '__main__':
    # Initialize rich console
    console = Console()

    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold cyan]Using device:[/bold cyan] {device}")

    torch.manual_seed(0)
    np.random.seed(0)

    # Create a table for results
    results_table = Table(title="KNN Classification Results", show_lines=True)
    results_table.add_column("Configuration File", justify="left", style="cyan")
    results_table.add_column("Best n_neighbors", justify="center", style="green")
    results_table.add_column("Accuracy", justify="center", style="magenta")

    for config_file in config_files:
        console.print(f"\n[bold cyan]Processing configuration file:[/bold cyan] {config_file}")
        config = TrainingConfig.load(config_file)
        modelFactory = ModelFactory(config)
        # modelFactory.model, in_features = getBackboneModel(config.model)
        # modelFactory.model = models.vit_h_14(weights=models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1)
        # modelFactory.model = create_model("vit_small_patch16_224_dino", pretrained=True)
        # model, preprocess = clip.load("ViT-L/14", device=device)
        from transformers import AutoImageProcessor, AutoModel
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")
        model = AutoModel.from_pretrained("facebook/dinov2-giant")

        feature_extractor = model.

        # modelFactory.model = model


        dir = str(config.data.directory)
        # test_dir = os.path.join(dir, "/test_images/mistery_category")
        test_dir = dir + "/test_images/mistery_category"
        print(dir, test_dir)

        # Move model to GPU
        modelFactory.model = modelFactory.model.to(device)
        modelFactory.model.eval()

        if config.model.weights_path:
            modelFactory.model.load_state_dict(torch.load(config.model.weights_path))

        console.print("[bold green]Model loaded and moved to GPU successfully![/bold green]")

        train_loader, val_loader = modelFactory.train_loader, modelFactory.val_loader
        # train_loader.dataset.transform = preprocess
        # val_loader.dataset.transform = preprocess

        features_train = []
        labels_train = []

        # Progress bar for training feature extraction
        console.print("[bold yellow]Extracting features from training data...[/bold yellow]")
        with Progress(console=console) as progress:
            train_task = progress.add_task("[green]Processing training data...", total=len(train_loader))
            for x, y in train_loader:
                x = x.to(device)
                with torch.no_grad():
                    # features = modelFactory.model.encode_image(x)
                    features = modelFactory.model(x)
                    features_train.extend(features.cpu().numpy())  # Move to CPU for KNN
                    labels_train.extend(y.cpu().numpy())
                progress.update(train_task, advance=1)

        features_train = np.array(features_train)
        labels_train = np.array(labels_train)

        
        np.save("features_train.npy", features_train)
        np.save("labels_train.npy", labels_train)

        features_val = []
        labels_val = []

        # Progress bar for validation feature extraction
        console.print("[bold yellow]Computing features for validation set...[/bold yellow]")
        with Progress(console=console) as progress:
            val_task = progress.add_task("[blue]Processing validation data...", total=len(val_loader))
            for x, y in val_loader:
                x = x.to(device)
                with torch.no_grad():
                    # features = modelFactory.model.encode_image(x)
                    features = modelFactory.model(x)
                    features_val.extend(features.cpu().numpy())  # Move to CPU for KNN
                    labels_val.extend(y.cpu().numpy())
                progress.update(val_task, advance=1)

        features_val = np.array(features_val)
        labels_val = np.array(labels_val)

        np.save("features_val.npy", features_val)
        np.save("labels_val.npy", labels_val)


        console.print("[bold green]Feature extraction completed successfully![/bold green]")
        console.print("[bold yellow] Computing features for test set...[/bold yellow]")

        features_test = []
        files = os.listdir(test_dir)

        with Progress(console=console) as progress:
            test_task = progress.add_task("[blue]Processing test data...", total=len(files))
            for f in files:
                if "jpeg" in f:
                    data = pil_loader(test_dir + "/" + f)
                    data = data_transforms(data)
                    data = data.view(1, data.size(0), data.size(1), data.size(2))
                    data = data.to(device)
                    with torch.no_grad():
                        # features = modelFactory.model.encode_image(data)
                        features = modelFactory.model(data)
                        features_test.extend(features.cpu().numpy())
                progress.update(test_task, advance=1)


        # Hyperparameter search for n_neighbors
        console.print("[bold magenta]Performing hyperparameter search for KNN...[/bold magenta]")
        best_accuracy = 0
        best_n_neighbors = 0

        knn = KNN(n_neighbors=1)

        # we cheat here and use the validation set as well to train the KNN
        y_train = np.concatenate([labels_train, labels_val])
        X_train = np.concatenate([features_train, features_val])

        y_pred = knn(X_train, y_train, features_test)

        # Write the predictions to a file
        output_file = open("kaggle_knn.csv", "w")
        output_file.write("Id,Category\n")
        for i, f in enumerate(files):
            if "jpeg" in f:
                pred = y_pred[i]
                output_file.write("%s,%d\n" % (f[:-5], pred))



        for n in range(1, 21):  # Searching for n_neighbors in the range [1, 20]
            knn = KNN(n_neighbors=n)
            y_pred = knn(features_train, labels_train, features_val)
            accuracy = accuracy_score(labels_val, y_pred)

            console.print(f"[cyan]n_neighbors={n} -> Accuracy: {accuracy:.4f}[/cyan]")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_n_neighbors = n

        # Add results to the table
        results_table.add_row(config_file, str(best_n_neighbors), f"{best_accuracy:.4f}")

    # Display the table
    console.print("\n[bold green]Summary of Results:[/bold green]")
    console.print(results_table)


# Summary of Results:
#                                KNN Classification Results
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
# ┃ Configuration File                                      ┃ Best n_neighbors ┃ Accuracy ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
# │ configuration_files/supervised_mobilenetv3.yaml         │        1         │  0.1464  │
# ├─────────────────────────────────────────────────────────┼──────────────────┼──────────┤
# │ configuration_files/supervised_mobilenetv3_scratch.yaml │        1         │  0.6840  │
# ├─────────────────────────────────────────────────────────┼──────────────────┼──────────┤
# │ configuration_files/supervised_resnet.yaml              │        1         │  0.4872  │
# ├─────────────────────────────────────────────────────────┼──────────────────┼──────────┤
# │ configuration_files/supervised_resnet_scratch.yaml      │        1         │  0.6944  │
# ├─────────────────────────────────────────────────────────┼──────────────────┼──────────┤
# │ configuration_files/supervised_squeezenet.yaml          │        1         │  0.2432  │
# ├─────────────────────────────────────────────────────────┼──────────────────┼──────────┤
# │ configuration_files/supervised_squeezenet_scratch.yaml  │        1         │  0.0636  │
# ├─────────────────────────────────────────────────────────┼──────────────────┼──────────┤
# │ configuration_files/supervised_vit_scratch.yaml         │        1         │  0.7616  │
# └─────────────────────────────────────────────────────────┴──────────────────┴──────────┘
# | ViT_H_14.yaml                                           |        1         |  0.8868  |
# ├─────────────────────────────────────────────────────────┼──────────────────┼──────────┤
# | CLIP_H_14.yaml                                          |        1         |  0.80    |
# ├─────────────────────────────────────────────────────────┼──────────────────┼──────────┤
