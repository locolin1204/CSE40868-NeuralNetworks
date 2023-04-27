import torch
import torch.nn as nn
from torchvision.transforms.functional import resize

class CNN(nn.Module):

    def __init__(self, numChannels, numClasses):
        super(CNN, self).__init__()
        self.classes = numClasses

        # Convolutional layers:
        self.conv1 = nn.Conv2d(numChannels, 96, 11, stride=4)
        self.conv2 = nn.Conv2d(96, 256, 5, stride=1)
        self.conv3 = nn.Conv2d(256, 384, 3, stride=1)
        self.conv4 = nn.Conv2d(384, 384, 3, stride=1)
        self.conv5 = nn.Conv2d(384, 256, 3, stride=1)


        # Activation function:
        self.relu = nn.ReLU()

        # Pooling layer:
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # Batch normalization layers:
        self.batchnorm1 = nn.BatchNorm2d(96)
        self.batchnorm2 = nn.BatchNorm2d(256)

        # Fully-connected layers:
        self.fc1 = nn.Linear(2304, 1024)
        self.fc2 = nn.Linear(1024, 2)

        self.dropout = nn.Dropout(p=0.5)

    # Evaluation function
    def evaluate(self, model, dataloader, classes, device, criterion):

        # We need to switch the model into the evaluation mode
        model.eval()
        running_loss = .0

        # Prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        # For all test data samples:
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs, _ = model(images)
            _, predictions = torch.max(outputs, 1)

            loss = criterion(outputs, labels)
            loss = loss.detach().cpu().numpy()
            running_loss += loss

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
            for i in range(len(classes)):
                print(classes[i], ":", correct_pred[classes[i]] / total_pred[classes[i]])
                # print("correct", correct_pred)
                # print("total", total_pred)
        # Calculate the overall accuracy on the test set
        acc = sum(correct_pred.values()) / sum(total_pred.values())
        total_running_loss = running_loss/len(dataloader)

        return acc, total_running_loss

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

        # After the last pooling operation, and before the first
        # fully-connected layer, we need to "flatten" our tensors
        x = torch.flatten(x, 1)

        # Can fully-connected layers accept data if they are not flattened?

        # Finally, we need our two-layer perceptron (two fully-connected layers) at the end of the network:
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)

        # return the encodings right before the last fully-connected
        # layer for the t-SNE visualization
        enc = x
        x = self.fc2(x)
        return x, enc
