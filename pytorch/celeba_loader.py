import torch
import torchvision
from torchvision import transforms
import torch.utils.data as data_utils
from matplotlib import pyplot as plt
import numpy as np

#Loading the dataset from dataset location
DATASET_PATH = './dataset/'
BATCH_SIZE = 1

celeba_data = torchvision.datasets.CelebA(DATASET_PATH, split='train', transform= transforms.Compose([transforms.ToTensor(),
                                                                                                    transforms.Resize([224,224])
                                                                                                    ]))
celeba_data_validation = torchvision.datasets.CelebA(DATASET_PATH, split='valid', transform= transforms.Compose([transforms.ToTensor(),
                                                                                                    transforms.Resize([224,224])
                                                                                                    ]))


#The image is already normalized --> How?

#Just keeping in the smile class and removing all other classes
index =20 #Male class index = 20  ###Smile class index = 31
attributes_list = celeba_data.attr.tolist()
attributes_list = [[i[index]] for i in attributes_list]
celeba_data.attr = torch.tensor(attributes_list, dtype=float) #######int64

val_attributes_list = celeba_data_validation.attr.tolist()
val_attributes_list = [[i[index]] for i in val_attributes_list]
celeba_data_validation.attr = torch.tensor(val_attributes_list, dtype=float)

#Reducing the dataset size
indices = torch.arange(10000)
celeba_1k = data_utils.Subset(celeba_data, indices)

celeba_1k_val = data_utils.Subset(celeba_data_validation, indices)

#Defining the data loader for training
data_loader = torch.utils.data.DataLoader(celeba_1k,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False,
                                        )
data_loader_val = torch.utils.data.DataLoader(celeba_data_validation,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        )


#TODO: Create a subset of images with just male attribute --> Completed

#Helper code to visualize the dataloader
# for images, labels in iter(data_loader):
#     plt.imshow(np.transpose(images[0].cpu().detach().numpy(), (1, 2, 0)))
#     plt.show()

#     break