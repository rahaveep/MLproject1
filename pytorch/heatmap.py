###Work with this code only after setting the batch size in celeba_loader to 1

import torch
from torchsummary import summary
#Importing the model
print(">> Loading model...")
from model import model, device
from celeba_loader import data_loader
import numpy as np
from matplotlib import pyplot as plt

MODEL_PATH = './history/2021-12-01 22:34/epoch_04.pth'
#Loading the weights
model.load_state_dict(torch.load(MODEL_PATH)['model_state_dict'])
print(">> Model loaded successfully")
#Setting the model for inference
model.eval()
# print(model)

#Helper function for feature extraction
features = dict()

def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

layer_keys = ['ReLU_{:02d}'.format(i) for i in range(16)]  ###There are 18 ReLU layers (16 in feature extractor, 2 in classifier)
total_layer_num = 37
count = 0
for i in range(total_layer_num):
    layer = model.features[i]
    if isinstance(layer, torch.nn.ReLU):
        model.features[i].register_forward_hook(get_features(layer_keys[count]))
        count += 1

#Feature extraction loop
#place holder
PREDS = list()
FEATS = list()
#place holder for batch features
features = {}
#loop through the batches
for images, labels in iter(data_loader):
    images = images.to(device)
    labels = labels.to(device)

    preds = model(images)

    PREDS.append(preds.detach().cpu().numpy())
    for a_key in layer_keys:
        FEATS.append(features[a_key].cpu().numpy())

    break

HEATMAP_SAVE_PATH = './heatmaps/male_no_male/'

plt.imshow(np.transpose(images[0].cpu().detach().numpy(), (1, 2, 0)))
plt.savefig(HEATMAP_SAVE_PATH+"input_image.png")
# plt.show()

for i in range(len(layer_keys)):
    heatmap = np.transpose(FEATS[i][0], (1,2,0))
    heatmap =  np.mean(heatmap,axis=2)
    plt.imshow(heatmap, cmap='hot')
    plt.savefig(HEATMAP_SAVE_PATH+'relu_{}.png'.format(i))



    