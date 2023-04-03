import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform
)


train_dataloader = DataLoader(training_data, batch_size=4,shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)

# get some random training images
train_dataiter=iter(train_dataloader)
images, labels = next(train_dataiter)

batch_size=4

# functions to show an image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # making it a valid image data
    plt.show()




classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# show images
torchvision.utils.make_grid(images).size() # make_grid converts the batch of images into a grid of images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


IN_CHANNELS=images.size()[1]
OUT_CHANNELS_1=16
OUT_CHANNELS_2=32
KERNEL_SIZE_1=5
KERNEL_SIZE_2=5
KERNEL_MAX_1=2
KERNEL_MAX_2=2
STRIDE=2



import torch.nn as nn
import torch.nn.functional as F


class CnnEmad(nn.Module):
    def __init__(self,in_channels,out_channels_1,out_channels_2,kernel_size_1,kernel_size_2,kernel_max_1,kernel_max_2,stride):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels_1,kernel_size_1)
        self.conv2=nn.Conv2d(out_channels_1,out_channels_2,kernel_size_2)
        #self.conv3=nn.Conv2d(out_channels_2,32,2)
        self.max1=nn.MaxPool2d(kernel_max_1,stride)
        self.max2=nn.MaxPool2d(kernel_max_2,stride)
        #self.max3=nn.MaxPool2d(kernel_max_2,stride)
        self.fc1=nn.Linear(32*5*5,240)
        self.fc2=nn.Linear(240,84)
        self.fc3=nn.Linear(84,10)


    def forward(self,input):
        z1=F.relu(self.conv1(input))
        z1=self.max1(z1)
        z1=F.relu(self.conv2(z1))
        z1=self.max2(z1)
        #z1=F.relu(self.conv3(z1))
        #z1=self.max3(z1)

        z1=torch.flatten(z1,1)
        z1=F.relu(self.fc1(z1))
        z1=F.relu(self.fc2(z1))
        z1=self.fc3(z1)
        return z1

NUM_EPOCHS=5
LEARNING_RATE = 1e-3
MOMENTUM=0.9

# Initialize model
model = CnnEmad(in_channels=IN_CHANNELS,out_channels_1=OUT_CHANNELS_1,out_channels_2=OUT_CHANNELS_2,kernel_size_1=KERNEL_SIZE_1,kernel_size_2=KERNEL_SIZE_2,kernel_max_1=KERNEL_MAX_1,kernel_max_2=KERNEL_MAX_2,stride=STRIDE)
print (model.named_parameters)

#from torch.optim import Adam
import torch.optim as optim
loss_fn = nn.CrossEntropyLoss()
#optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
optimizer=optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)






for epoch in range(2):

    epoch_loss = 0.0 

     
    for (batch_idx,batch) in enumerate(train_dataloader):
        
        
        X,y=batch

        optimizer.zero_grad() # RESET GRADIENTS

        # Forward pass
        y_pred = model(X) # INPUTS
        loss = loss_fn(y_pred, y) # COMPUTE THE LOSS
       
        

        loss.backward() #  BACKWARD PASS
        optimizer.step() # UPDATE WEIGHTS

        epoch_loss += loss.item()  


        if batch_idx % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {epoch_loss / 2000:.3f}')
            epoch_loss = 0.0



    
print('Finished Training')

dataiter = iter(test_dataloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

outputs = model(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = model(images)
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
    for data in test_dataloader:
        images, labels = data
        outputs = model(images)
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


