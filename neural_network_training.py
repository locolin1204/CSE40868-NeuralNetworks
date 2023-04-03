import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms.functional import resize
from torchvision.transforms import CenterCrop
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from torchsummary import summary
from tqdm import tqdm
import numpy as np
import pandas as pd


class CNN(nn.Module):

    def __init__(self, numChannels, numClasses):
        super(CNN, self).__init__()
        self.classes = numClasses

        # you can check https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

        # https://poloclub.github.io/cnn-explainer/

        # What is the relationship between output feature channel and number of kernels?
        # Can you draw a picture to describe the relationship among input size,
        # kernel size and output size in a convolution layer?

        # Convolutional layers:
        self.conv1 = nn.Conv2d(numChannels, 96, 11, stride=4)
        self.conv2 = nn.Conv2d(96, 256, 5, stride=1)
        self.conv3 = nn.Conv2d(256, 384, 3, stride=1)
        self.conv4 = nn.Conv2d(384, 384, 3, stride=1)
        self.conv5 = nn.Conv2d(384, 256, 3, stride=1)

        # Activation function:
        # check https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.relu = nn.ReLU()

        # Pooling layer:
        # check https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # Batch normalization layers:
        self.batchnorm1 = nn.BatchNorm2d(96)
        self.batchnorm2 = nn.BatchNorm2d(256)

        # Fully-connected layers:
        self.fc1 = nn.Linear(2304, 1024)
        self.fc2 = nn.Linear(1024, 2)

        # We have very specific numbers of neurons (2304 and 1024)
        #               in fully-connected layers (fc1 and fc2). Make sure you understand
        #               where these numbers are comming from (and when they can be different).

        # As we are using CrossEntropy loss during training, so a softmax activation
        #               function (= implemented as a softmax layer to apply it to the whole layer)
        #               is not needed here since it's already included in Pytorch's implementation
        #               of cross-entropy loss. See this for details:
        #               https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

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
            outputs = model(images)
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

            print("------")
            for i in range(2):
                print(classes[i], ":", correct_pred[classes[i]] / total_pred[classes[i]])
                # print("correct", correct_pred)
                # print("total", total_pred)
        # Calculate the overall accuracy on the test set
        acc = sum(correct_pred.values()) / sum(total_pred.values())

        return acc

    def forward(self, x):

        x = resize(x, size=[256])


        # Convolutional, ReLU, MacPooling and Batchnorm layers go first
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

        #  What if we remove one of the layers? Will the network still work?

        # After the last pooling operation, and before the first
        # fully-connected layer, we need to "flatten" our tensors
        x = torch.flatten(x, 1)

        # Can fully-connected layers accept data if they are not flattened?

        # Finally, we need our two-layer perceptron (two fully-connected layers) at the end of the network:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



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


# Define number of epochs and batch size:

image_dir = 'resources/image-source/test-all-source/'

label_dir = 'resources/cse40868-final-project-labels.csv'

epochs = 5
batch_size = 50

# *** THINK *** Should we set the # of epochs as large as possible? Why?
# *** THINK *** If we find that the evaluation accuracy is low, should we increase or decrease the # of epochs?

data = FaceDataset(img_dir=image_dir, label_dir=label_dir, transform=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

train_len = int(len(data) * 0.8)
val_len = int(len(data) * 0.1)
test_len = int(len(data) - train_len - val_len)
train_data, val_data, test_data = random_split(data, [train_len, val_len, test_len])
classes = ["me-source", "everyone-source"]

# Prepare data loaders for train, validation and test data splits
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2)

if __name__ == '__main__':

    # Specify the operation mode:
    # 'train' = training with your train and validation data splits
    # 'eval'  = evaluation of the trained model with your test data split

    ############################
    ######## Start Here ########
    ############################
    mode = 'eval'

    # Path where you plan to save the best model during training
    my_best_model = "CSE40868_final_project_best_model.pth"


    device = torch.device("cpu")
    # Initialize the model and print out its configuration
    model = CNN(numChannels=3, numClasses=2)
    model.to(device)
    summary(model, input_size=(3, 32, 32))
    # Set the device (GPU or CPU, depending on availability)
    device = torch.device(
        "mps:0" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)

    print("Currently using device: ", device)
    print("\n\nModel summary:\n\n")

    if mode == "train":

        print("\n\nTraining starts!\n\n")

        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        # Try different optimizers (can be Adam or SGD).
        # Try different parameters (learning rate, momentum).

        running_loss = .0
        best_acc = .0
        for epoch in range(epochs):
            print(f"Starting epoch {epoch + 1}")
            for idx, data in tqdm(enumerate(train_loader), total=len(train_loader)):
                # Get the inputs (data is a list of [inputs, labels])
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                loss = loss.detach().cpu().numpy()
                inputs = inputs.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                running_loss += loss

            # Evaluate the accuracy after each epoch
            acc = model.evaluate(model, val_loader, classes, device)
            if acc > best_acc:
                print(f"Better validation accuracy achieved: {acc * 100:.2f}%")
                best_acc = acc
                print(f"Saving this model as: {my_best_model}")
                torch.save(model.state_dict(), my_best_model)

    # And here we evaluate the trained model with the test data
    elif mode == "eval":

        print("\n\nValidating the trained model:")
        print(f"Loading checkpoint from {my_best_model}")
        model.load_state_dict(torch.load(my_best_model))
        acc = model.evaluate(model, test_loader, classes, device)
        print(f"Accuracy on the test (unknown) data: {acc * 100:.2f}%")

    else:
        print("'mode' argument should either be 'train' or 'eval'")
