import torch
import torchvision
from torchsummary import summary
from celeba_loader import data_loader, data_loader_val
import numpy as np
import datetime
import os

NUM_CLASSES = 2
NUM_EPOCHS = 5
MODEL_SAVE_PATH = './history/'
RESULTS_PATH = './results/'

#Checkpoints write-path
CHECKPOINT_PATH = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")+'/'  #################
if not os.path.isdir(MODEL_SAVE_PATH+CHECKPOINT_PATH):
	os.mkdir(MODEL_SAVE_PATH+CHECKPOINT_PATH)

#Loss writing text file name
loss_text = 'loss_'+datetime.datetime.now().strftime("%Y-%m-%d %H:%M")+'.txt'
with open(RESULTS_PATH+loss_text, 'w+'):
	pass

#use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load VGG19 model
model = torchvision.models.vgg19(pretrained=True, progress=True)
#Fine tuning the model
model.classifier[6] = torch.nn.Linear(4096, NUM_CLASSES)
model = model.to(device)
#Model Summary
# print(summary(model, (3,224,224)))

#Loss function
loss_function = torch.nn.CrossEntropyLoss()
# Using an Adam Optimizer with lr = 0.3
optimizer = torch.optim.Adam(model.parameters(),
							lr = 1e-4,
							weight_decay = 1e-8)

#Training
losses = list()
for epoch in range(NUM_EPOCHS):
    print(">> Epoch Number = {}".format(epoch))
    for phase in ['train', 'val']:
        if phase == "train":
            model.train()
            for images, labels in iter(data_loader):
                images = images.to(device)
                labels = torch.squeeze(labels)

                labels = labels.long().to(device)
                optimizer.zero_grad()
                outputs = model(images)

                loss = loss_function(outputs, labels)
                
                loss.backward()
                optimizer.step()
            print(">>Training Loss = {}".format(loss))
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
    # }, MODEL_SAVE_PATH+CHECKPOINT_PATH+'{:02d}_epoch.pth'.format(epoch))
    }, MODEL_SAVE_PATH+CHECKPOINT_PATH+'epoch_{:02d}.pth'.format(epoch))

    
    #Writing the loss function to a text file
    with open(RESULTS_PATH+loss_text, 'a') as f:
        f.writelines(str(loss.detach().cpu().numpy())+'\n')

model.eval()
for images, labels in iter(data_loader_val):
    images = images.to(device)
    labels = torch.squeeze(labels)

    labels = labels.long().to(device)
    optimizer.zero_grad()
    outputs = model(images)

    loss = loss_function(outputs, labels)
print(">>Validation Loss = {}".format(loss))



 




