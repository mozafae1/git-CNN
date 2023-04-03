import numpy as np
import torch
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

Transforms=transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(0.5,0.5)])
traindata=datasets.FashionMNIST(root='data',train=True,download=True,transform=Transforms)
traindata_loader=DataLoader(traindata, batch_size=5,shuffle=True)

testdata=datasets.FashionMNIST(root='data', train=False, download=True, transform=Transforms)
testdata_loader=DataLoader(testdata, batch_size=5, shuffle=True)

dataiter = iter(traindata_loader)
images, labels = next(dataiter)

def imgshow(img):
   img=img/2+0.5
   np_img=img.numpy()
   np_img=np_img.transpose(1,2,0)
   plt.imshow(np_img)

imgshow(torchvision.utils.make_grid(images))



# Create the network
class CNN_Glory(nn.Module):
    def __init__(self,C_in,C_1,K_1,C_2,K_2,KM_1,S_1,KM_2,S_2):
      super().__init__()
      self.conv1=nn.Conv2d(C_in,C_1,K_1)
      self.conv2=nn.Conv2d(C_1,C_2,K_2)
      self.bn1=nn.BatchNorm2d(C_1)
      self.bn2=nn.BatchNorm2d(C_2)
      self.max1=nn.MaxPool2d(KM_1,S_1)
      self.max2=nn.MaxPool2d(KM_2,S_2)
      self.conv3=nn.Conv2d(C_2,C_2,kernel_size=3, stride=1, padding=1)
      self.conv4=nn.Conv2d(C_2,C_2,kernel_size=3, stride=1, padding=1)
      self.bn3=nn.BatchNorm2d(C_2)
      self.bn4=nn.BatchNorm2d(C_2)


      self.fc1=nn.Linear(C_2*4*4,160)
      self.bnfc1 = nn.BatchNorm1d(160)
      self.fc2=nn.Linear(160,80)
      self.bnfc2 = nn.BatchNorm1d(80)
      self.fc3=nn.Linear(80,10)
      

    def forward(self,X):
       Z=self.conv1(X)
       Z=self.bn1(Z)
       Z=F.relu(Z)
       Z=self.max1(Z)

       Z=self.conv2(Z)
       Z=self.bn2(Z)
       Z=F.relu(Z)
       Z=self.max2(Z)

       Z=self.conv3(Z)
       Z=self.bn3(Z)
       Z=F.relu(Z)
      
       Z=self.conv4(Z)
       Z=self.bn4(Z)
       Z+=F.relu(Z)

       Z=Z.view(-1,16*4*4)
     
       Z=self.fc1(Z)
       Z=self.bnfc1(Z)
       Z=F.relu(Z)

       Z=self.fc2(Z)
       Z=self.bnfc2(Z)
       Z=F.relu(Z)

       Z=self.fc3(Z)

       return Z
    

# Neural Network Parameters

c_in=images.size()[1]
c_1=6
c_2=16
k_1=5
k_2=5
km_1=2
km_2=2
s_1=2
s_2=2 
oCNN=CNN_Glory(C_in=c_in,C_1=c_1,K_1=k_1,C_2=c_2,K_2=k_2,KM_1=km_1,S_1=s_1,KM_2=km_2,S_2=s_2)

# Hyperparameters

LEARNING_RATE=1e-4
NUMBER_OF_EPOCHS=5
MOMENTUM=0.9

# Loss Function
loss_function=nn.CrossEntropyLoss()
optimizer=opt.SGD(oCNN.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)



# Optimizer



# Training and Optimization

for i in range(NUMBER_OF_EPOCHS):
     
     ls_epoch=0.0

     for (idx_batch,batch) in enumerate(traindata_loader):


        X,y=batch
        y_pred=oCNN(X)
        ls=loss_function(y_pred,y)

        optimizer.zero_grad()
        ls.backward()
        optimizer.step()

        ls_epoch+=ls.item()


        if idx_batch % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{i + 1}, {idx_batch + 1:5d}] loss: {ls_epoch / 2000:.3f}')
            ls_epoch = 0.0

   


classes=('T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot')

print('Finished Training')

dataiter = iter(testdata_loader)
images, labels = next(dataiter)

# print images
imgshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(5)))

outputs = oCNN(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(5)))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testdata_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = oCNN(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


       
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testdata_loader:
        images, labels = data
        outputs = oCNN(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')      
   



params=oCNN.parameters() 
for p in params:
    print(p.shape) 
conv1_weight=oCNN.state_dict()['conv1.weight']