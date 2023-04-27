import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
from PIL import Image
from neural_network_visualisation import CNN

device = torch.device(
        "mps:0" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu"
    )
print("currently using: ", device)

# *** TASK *** Find a picture (in the internet) of an object representing
# one of the ImageNet classes (for instance, hippo), upload it to your Google Drive folder
# and run Task 3 for that picture. We have the ImageNet class list in "image_classes.txt".
# You can also look here: https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/

# filename = '../resources/image-source/test-all-source/IMG-277.JPG'

filename = "../resources/image-source/sample-images-for-testing/sample-image-5.jpeg"

input_image = Image.open(filename)
rgb_img = np.float32(input_image) / 255

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor_cls = preprocess(input_image)
input_batch = input_tensor_cls.unsqueeze(0).to(device)

input_tensor = preprocess_image(rgb_img,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]).to(device)

# In this task we will use one of the popular models -- ResNet-18 -- pre-trained on the ImageNet dataset
# (ImageNet has 1000 classes, all are listed in "image_classes.txt")
device = torch.device(
        "mps:0" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu"
    )
model = CNN(3, 2)


# model.load_state_dict(torch.load("../first-solution/CSE40868_final_project_best_model.pth"))

# model.to(device)
# model.eval()
# model_paths = ["CSE40868_final_project_best_model_improved.pth", "CSE40868_final_project_best_model_improved.pth"]
model_paths = ["../first-solution/CSE40868_final_project_best_model.pth", "CSE40868_final_project_best_model_improved.pth"]

for loop in range(0, 2):
    print(model_paths[loop])
    model.load_state_dict(torch.load(model_paths[loop]))
    model.to(device)
    model.eval()
    # Get the model's prediction
    with torch.no_grad():
        output = model(input_batch)
    softmax_scores = torch.nn.functional.softmax(output[0][0], dim=0)
    # Read the categories
    with open("../resources/image_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    # Show top "topk" categories per image
    topk = 2
    topk_prob, topk_catid = torch.topk(softmax_scores, topk)
    print("\nTop {} categories:".format(topk))
    for i in range(topk_prob.size(0)):
        print("{}: {:.5f}".format(categories[topk_catid[i]], topk_prob[i].item()))
    # We will be checking which are salient regions for the last conv layer in ResNet
    # (right before the classification fully-connected layer)

    # select all target layers in CNN including full connect layers
    list_of_layers = [module for module in model.modules() if not isinstance(module, nn.Sequential)]

    target_layers = [model.conv1, model.conv2, model.conv3, model.conv4, model.conv5, model.batchnorm1, model.batchnorm2]

    # Get the Grad Class Activation Map
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    # *** TASK ***
    # We have to specify the target class we want to generate
    # the Class Activation Maps for. By default (if target_category = None),
    # the class with the highest softmax score will be used as target class:

    chosen_category = None
    chosen_category = 0     # me-source
    # chosen_category = 1     # everyone-source
    # or:

    # my_target_category = 0  # me-source
    # my_target_category = 1  # everyone-source
    # target_category = [ClassifierOutputTarget(my_target_category)]

    if chosen_category is None:
        target_category = [ClassifierOutputTarget(topk_catid[0])]
    else:
        target_category = [ClassifierOutputTarget(chosen_category)]

    # But, for what it's worth, we can generate CAM for a different target class
    # than the one represented by the input image. In that way we could check which
    # features of class X were found (and important) in a sample image representing class Y.
    # For instance, 284 (counting from 0) is the ID of "Siamese cat" category. For any
    # image provided as the input (including or not the Siamese cat), we can check
    # which features in that image are similar to those of the Siamese cat.

    # # *** TASK *** find an image that contains objects from TWO ImageNet classes,
    # generate GradCAMs for both classes (by providing appropriate target in "target_category"),
    # and check if salient features obtained for ResNet-18 are correct. Write a few sentences
    # describing your observations.
    # Visualization code
    grayscale_cam = cam(input_tensor=input_tensor, targets=target_category)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    vis = Image.fromarray(visualization)

    if chosen_category is None:
      print("\nClass Activation Map for: {}".format(categories[topk_catid[0]]))
    else:
      print("\nClass Activation Map for: {}".format(categories[chosen_category]))

    gradcam_image_path = f"figures/GradCAM_visualization{loop}.png"
    vis.save(gradcam_image_path)

    image = cv2.imread(gradcam_image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.show()

