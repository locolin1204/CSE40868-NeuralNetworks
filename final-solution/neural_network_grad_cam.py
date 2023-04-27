import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from PIL import Image
from CNN import CNN

device = torch.device(
        "mps:0" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu"
    )
print("currently using: ", device)

# image from dataset
# filename = '../resources/image-source/test-all-source/IMG-277.JPG'

# extreme data for testing
# there are 12 images to choose, sample-image-0.jpeg to sample-image-12.jpeg
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


device = torch.device(
        "mps:0" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu"
    )
model = CNN(3, 2)


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

    # select all target layers in CNN including full connect layers
    list_of_layers = [module for module in model.modules() if not isinstance(module, nn.Sequential)]

    target_layers = [model.conv1, model.conv2, model.conv3, model.conv4, model.conv5, model.batchnorm1, model.batchnorm2]

    # Get the Grad Class Activation Map
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    # We have to specify the target class we want to generate
    # the Class Activation Maps for. By default, (if chosen_category = None),
    # the class with the highest softmax score will be used as target class:

    chosen_category = None
    # chosen_category = 0     # me-source
    # chosen_category = 1     # everyone-source


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

