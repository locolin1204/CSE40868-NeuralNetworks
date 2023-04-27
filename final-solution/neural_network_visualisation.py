from turtle import shape
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.io import read_image
import cv2
import os

from argparse import ArgumentParser
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms.functional import resize
from torchvision.transforms import CenterCrop
from tqdm import tqdm
from matplotlib import pyplot as plt

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
from torchvision.models import resnet18

from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from PIL import Image

torch.cuda.empty_cache()


# create CNN model
class CNN(nn.Module):

    def __init__(self, numChannels, numClasses):
        super(CNN, self).__init__()
        self.classes = numClasses

        # initailize ReLU and MaxPooling layer
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize first set of CONV, BatchNormal layers
        self.conv1 = nn.Conv2d(in_channels=numChannels, out_channels=96,
                               kernel_size=(11, 11), stride=(4, 4))
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256,
                               kernel_size=(5, 5), stride=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384,
                               kernel_size=(3, 3), stride=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384,
                               kernel_size=(3, 3), stride=(1, 1))
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256,
                               kernel_size=(3, 3), stride=(1, 1))

        # initialize first set of CONV, BatchNormal layers
        self.batchnorm1 = nn.BatchNorm2d(96)
        self.batchnorm2 = nn.BatchNorm2d(256)

        # initialize first set of FC layers
        self.fc1 = nn.Linear(in_features=2304, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=self.classes)

        # As we are using CrossEntropy loss during training, a Softmax layer is not needed here
        # See details: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

    # Evaluation function
    def evaluate(self, model, dataloader, classes, device):

        # We need to switch the model into the evaluation mode
        model.eval()

        # Prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        # For all test data samples:
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs, enc = model(images)
            _, predictions = torch.max(outputs, 1)

            images = images.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            predictions = predictions.detach().cpu().numpy()

            # Count the correct predictions for each class
            for label, prediction in zip(labels, predictions):

                # If you want to see real and predicted labels for all samples:
                # print("Real class: " + classes[label] + ", predicted = " + classes[prediction])

                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

        # Calculate the overall accuracy on the test set
        acc = sum(correct_pred.values()) / sum(total_pred.values())

        # Accuracy per class
        acc_per_class = dict()
        for key in correct_pred.keys():
            acc_per_class[key] = correct_pred[key] / total_pred[key]

        return acc, acc_per_class

    # This function will request to freeze the gradients for layer
    # specified in "tuning_layer_name"
    def freeze_layers(self, model, tuning_layer_name):
        print("Fine-tuning the model by freezing all layers except the one selected")
        for name, param in model.named_parameters():
            if name.find(tuning_layer_name) == -1:
                param.requires_grad = False
        return model

    # This function checks with layers are "trainable"
    def check(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print(name)

    def forward(self, x):

        x = resize(x, size=[256])

        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.batchnorm1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.batchnorm2(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # flatten all dimensions except batch
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.relu(x)

        # Let's return the encodings right before the last fully-connected
        # layer for the t-SNE visualization
        enc = x
        x = self.fc2(x)

        return x, enc


class FaceDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform):
        self.img_labels = pd.read_csv(label_dir, header=0)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        image = image.to(torch.float) / 256.
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)

        return (image, label)


# Transform images:
# a) to tensor: convert the PIL image or numpy.ndarray to tensor
# b) Z-normalize a tensor image (using its mean and standard deviation)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


device = torch.device(
        "mps:0" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu"
    )
print("currently using: ", device)
model = CNN(3, 2)
model.to(device)

# Loading the weigths, either the ones provided, or -- better -- the ones you obtained in your practical #2
# my_weights = "../first-solution/CSE40868_final_project_best_model.pth"
my_weights = "CSE40868_final_project_best_model_improved.pth"

print(f"Loading the weights from {my_weights} ...")
model.load_state_dict(torch.load(my_weights))
print("Successfully loaded the model checkpoint!")

# *** TASK ***: You may want to skip the above code if you want to visualize a network
# initialized with random weights

# *** TASK *** Here you can define which filters (from which layer) you want to visualize
for name, param in model.named_parameters():
    if name == "conv5.weight":      # conv2.weight, conv3.weight, etc.
        filters = param

# Let's normalize the intensity of the filters (to [0,1] range) just for plotting as images
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
filters = filters.detach().cpu().numpy()


################################################################################
# *** TASK ***: Write a simple code to display all 96 filters in the first layer
# (display it as an array of 11x11-pixel color images)
print(f"Filters shape: {filters.shape} (num of filters, num of input channels, width and height of filters)")
plt.figure(figsize=(16,16))
for i in range(filters.shape[0]):
  plt.subplot(16, 16, i+1)
  plt.imshow(filters[i, 0])

plt.savefig("figures/kernels.png")
plt.show()

