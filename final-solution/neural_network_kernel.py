import torch
from matplotlib import pyplot as plt
from CNN import CNN

torch.cuda.empty_cache()

device = torch.device(
        "mps:0" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu"
    )
print("currently using: ", device)
model = CNN(3, 2)
model.to(device)

# Loading the weights
# first solution
# my_weights = "../first-solution/CSE40868_final_project_best_model.pth"
# improved solution
my_weights = "CSE40868_final_project_best_model_improved.pth"

print(f"Loading the weights from {my_weights} ...")
model.load_state_dict(torch.load(my_weights))
print("Successfully loaded the model checkpoint!")


# define which filters (from which layer) you want to visualize
for name, param in model.named_parameters():
    if name == "conv5.weight":
        filters = param

# normalize the intensity of the filters (to [0,1] range) just for plotting as images
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
filters = filters.detach().cpu().numpy()

#  display all 96 filters in the first layer

print(f"Filters shape: {filters.shape} (num of filters, num of input channels, width and height of filters)")
plt.figure(figsize=(16, 16))
for i in range(filters.shape[0]):
  plt.subplot(16, 16, i+1)
  plt.imshow(filters[i, 0])

plt.savefig("figures/kernels.png")
plt.show()

