import os
from torch.optim import lr_scheduler
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
from tqdm import tqdm
import matplotlib.pyplot as plt
from CNN import CNN
from FaceDataset import FaceDataset


image_dir = "../resources/image-source/test-all-source"
label_dir = "../resources/cse40868-final-project-labels.csv"

# Define number of epochs and batch size:

epochs = 15
batch_size = 32

# Transform images:
# a) to tensor: convert the PIL image or numpy.ndarray to tensor
# b) Z-normalize a tensor image (using its mean and standard deviation)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
data = FaceDataset(img_dir=image_dir, label_dir=label_dir, transform=transform)

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
    mode = "eval"

    # Path where you plan to save the best model during training
    my_best_model = "CSE40868_final_project_best_model_improved.pth"

    device = torch.device("cpu")
    # Initialize the model and print out its configuration
    model = CNN(numChannels=3, numClasses=2)
    model.to(device)
    print("\n\nModel summary:\n\n")
    summary(model, input_size=(3, 250, 250))
    # Set the device (GPU or CPU, depending on availability)
    device = torch.device(
        "mps:0" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)
    print("Currently using device: ", device)
    criterion = nn.CrossEntropyLoss()

    if mode == "train":

        print("\n\nTraining starts!\n\n")

        model.train()
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        # Try different optimizers (can be Adam or SGD).

        running_loss = .0
        best_acc = .0

        train_losses = []
        val_losses = []
        train_acc = []
        val_acc = []


        for epoch in range(epochs):
            running_loss = .0
            train_correct_predictions = 0
            train_total_samples = 0
            print(f"Starting epoch {epoch + 1}")
            for idx, data in tqdm(enumerate(train_loader), total=len(train_loader)):
                # Get the inputs (data is a list of [inputs, labels])
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                train_total_samples += labels.size(0)
                train_correct_predictions += (predicted == labels).sum().item()

                loss = loss.detach().cpu().numpy()
                inputs = inputs.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                running_loss += loss

            train_losses.append(running_loss/len(train_loader))
            train_acc.append((train_correct_predictions / train_total_samples))

            # Evaluate the accuracy after each epoch
            scheduler.step()
            acc, val_running_loss = model.evaluate(model, val_loader, classes, device, criterion)
            if acc > best_acc:
                print(f"Better validation accuracy achieved: {acc * 100:.2f}%")
                best_acc = acc
                print(f"Saving this model as: {my_best_model}")
                torch.save(model.state_dict(), my_best_model)

            val_losses.append(val_running_loss)
            val_acc.append(best_acc)

            fig, axs = plt.subplots(2)
            axs[0].plot(list(range(1, epoch+2)), train_acc, label="Training Accuracy")
            axs[0].plot(list(range(1, epoch+2)), val_acc, label="Validation Accuracy")
            axs[0].legend()
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlim(1, epochs)
            axs[0].set_ylim(0, 1)

            axs[1].plot(list(range(1, epoch + 2)), train_losses, label="Training Loss")
            axs[1].plot(list(range(1, epoch + 2)), val_losses, label="Validation Loss")
            axs[1].legend()
            axs[1].set_xlabel("Epochs")
            axs[1].set_ylabel("Loss")
            axs[1].set_xlim(1, epochs)
            axs[1].set_ylim(0, 1)
            plt.savefig("figures/accuracy_loss.jpeg")
            plt.show()

    # And here we evaluate the trained model with the test data
    elif mode == "eval":

        print("\n\nValidating the trained model:")
        print(f"Loading checkpoint from {my_best_model}")
        model.load_state_dict(torch.load(my_best_model))
        acc, _ = model.evaluate(model, test_loader, classes, device, criterion)
        print(f"Accuracy on the test (unknown) data: {acc * 100:.2f}%")

    else:
        print("'mode' argument should either be 'train' or 'eval'")


