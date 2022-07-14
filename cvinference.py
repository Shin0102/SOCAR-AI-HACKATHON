#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torchvision
from torchvision import transforms


emotions = {0: "Normal", 1: "Drowsy", 2: "Call", 3: "Tobacco", 4: "Out of Camera"}
emotion_to_color = {
    "Normal": [0, 0, 255],  # blue
    "Drowsy": [255, 0, 0],  # red
    "Call": [255, 255, 0],  # yellow
    "Tobacco" : [0, 255, 255],  # aqua
    "Out of Camera": [0, 0, 0], # black
}


def convert_img2tensor(img):
    #img = np.array(img)
    img_tensor = inference_transform()(img)
    img_tensor = img_tensor.view(-1, 1, 224, 224)
    return img_tensor

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

def inference_transform():
    transform_list = [
        transforms.ToTensor(),
        transforms.Resize((224)),
        transforms.Normalize((0.5,), (0.5,)),
    ]
    return Compose(transform_list)


def cv_inference(model, img):
    img_tensor = convert_img2tensor(img)
    result = model(img_tensor).argmax(dim=1).item()
    return result, emotions[result]







