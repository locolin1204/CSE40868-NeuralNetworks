import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from neural_network_visualisation import CNN
from neural_network_visualisation import FaceDataset

device = torch.device(
    "mps:0" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu"
)
print("currently using: ", device)
model = CNN(3, 2)
model.to(device)

################################################################################
# *** TASK *** select appropriate weights and test data to run 4 experiments:
# Exp. 1: Model trained on CIFAR-10 and tested on CIFAR-10
# Exp. 2: Model trained on CIFAR-10 and tested on CIFAR-100
# Exp. 3: Model fine-tuned (all layers) for CIFAR-100 and tested on CIFAR-10
# Exp. 4: Model fine-tuned (all layers) for CIFAR-100 and tested on CIFAR-100
#
# Question: what are the differences among the obtained t-SNE projections in these experiments?
# Provide your observations at the end of this notebook.

# Provide correct path to the weights:
# my_weights = '/content/drive/MyDrive/CSE40868/CIFAR-100_best_model_all_layers.pth'

# Provide correct path to the CIFAR-100 subset (the same as in Practical 2):
image_dir = "../resources/image-source/test-all-source"
# Select the test data:
################################################################################

my_weights = "../first-solution/CSE40868_final_project_best_model.pth"
my_weights = "CSE40868_final_project_best_model_improved.pth"
model.load_state_dict(torch.load(my_weights))
print("Successfully loaded the model checkpoint!")

label_dir = "../resources/cse40868-final-project-labels.csv"
face_test = FaceDataset(img_dir=image_dir, label_dir=label_dir, transform=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
sample_data_len = int(len(face_test) * 0.2)
sample_data,_ = random_split(face_test, [sample_data_len, len(face_test)-sample_data_len])
classes = ["me-source", "everyone-source"]

# Create a data loader (for sample data from either CIFAR-10 or selected CIFAR-100, as you defined above)
sample_data_loader = DataLoader(sample_data, batch_size=50, shuffle=False, drop_last=False, num_workers=0)

# Get the encodings for your sample data
for idx, data in tqdm(enumerate(sample_data_loader), total=len(sample_data_loader)):

    # Get the inputs (data is a list of [inputs, labels])
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Get the features (encodings) from the one-before-last fully-connected layer
    # in our Convolutional Neural Network:
    with torch.no_grad():
      _,enc = model(inputs)
    enc = enc.detach().cpu().numpy()
    inputs = inputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    if idx == 0:
        labels_all = labels
        enc_all = enc
    else:
        enc_all = np.vstack((enc_all, enc))
        labels_all = np.hstack((labels_all, labels))

# t-SNE (selected) hyperparameters
perplexity = 50
n_iter = 1000

# Run the t-SNE for the encodings you extracted
tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=n_iter)
tsne_results = tsne.fit_transform(enc_all)
class_names = [classes[i] for i in labels_all]
feat_cols = ['pixel'+str(i) for i in range(enc.shape[1])]

# Visualize the t-SNE projection to 2 dimensions
df = pd.DataFrame(enc_all, columns=feat_cols)
df['labels'] = labels_all
df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(12, 8))
ax = sns.scatterplot(
    data=df,
    x="tsne-2d-one",
    y="tsne-2d-two",
    s=100,                  # marker size
    hue="labels",
    legend="full",
    alpha=0.8,
)
plt.legend(title='Classes', labels=classes)
plt.savefig('figures/tSNE.png')
plt.show()