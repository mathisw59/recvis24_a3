## This is the GitHub repository for the SketchNet classification challenge.

SkecthNet is a dataset of 500 classes, with 5 samples per class, augmented into 20 different views. The goal of the challenge is to classify the sketches into the 500 classes.

I explored many approaches to this problem, including:
- Using a pre-trained models like ResNet50, ViT, Swin Transformer, etc. and fine-tuning them on the dataset.
- Using a foundation model like CLIP and use it for image classification by retrieval using the KNN algorithm or linear probing.
- Pre-train a model using contrastive learning and fine-tune it on the dataset.

In order to train all of these models, I created a simple pipeline that can be used to train any of these models using yaml files, specifying the model, the dataset, the optimizer, the loss function, etc., similar to Hydra.