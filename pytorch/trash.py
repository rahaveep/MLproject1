import torch
import torchvision
from torchsummary import summary



# use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from model import model
model = model.to(device)
summary(model, (3,224,224))
# #Load VGG19 model
# model = torchvision.models.vgg19(pretrained=True, progress=True)
# #Fine tuning the model
# model.classifier[6] = torch.nn.Linear(4096, 2)
# model = model.to(device)
# summary(model, (3,224,224))

#Testing out the loss function
# loss = torch.nn.CrossEntropyLoss()
# input = torch.randn(3,5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)
# output = loss(input, target)
# output.backward()

# print(input.shape)
# print(target.shape)

#The input shape is --> (3,5)
#The target shape is --> (3)

#labels.shape --> (64,1) --> change this to (64)
#outputs.shape --> (64,2)

        # label_array = labels.cpu().detach().numpy()
        # label_array = label_array.reshape((1,len(label_array)))
        # label_array = np.squeeze(label_array, axis=0)
        # print(label_array.shape)
        # print(label_array)
        # new_labels = torch.tensor(label_array)
        # new_labels = torch.squeeze(new_labels)
        # new_labels = labels.long().to(device)
        # print(new_labels.shape)