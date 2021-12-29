import torch
import torchvision
from torchsummary import summary
from celeba_loader import data_loader, data_loader_val
import numpy as np
import datetime
import os

NUM_CLASSES = 2
NUM_EPOCHS = 10
MODEL_SAVE_PATH = './history/'
RESULTS_PATH = './results/'


#use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load VGG19 model
model = torchvision.models.vgg19(pretrained=True, progress=True)
#Fine tuning the model
model.classifier[6] = torch.nn.Linear(4096, NUM_CLASSES)
model = model.to(device)
#Model Summary
# print(summary(model, (3,224,224)))