## Collect data
from torchvision import datasets
from torchvision.transforms import ToTensor
train_data = datasets.MNIST(
    root = 'data',
    train = True,
    transform = ToTensor(),
    download = True,)

test_data = datasets.MNIST(
    root = 'data',
    train = False,
    transform = ToTensor())


## DataLoaders
import torch
# get the labeled images from the MNIST dataset
from torch.utils.data import DataLoader
loaders = {
    'train' : torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1),
    'test'  : torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1),}


## Dense
import torch.nn as nn
import torch.nn.functional as F

hidden_size = 25

class DENSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, hidden_size)
        #self.fc2 = nn.Linear(64, 64)
        #self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x

##
model = DENSE()
loss_func = F.nll_loss

from torch import optim
optimizer = optim.Adam(model.parameters(), lr = 0.01)


from torch.autograd import Variable
num_epochs = 3

## Train
def train(num_epochs, model, loaders) :
    model.train()
    total_step = len(loaders['train'])
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            model.zero_grad()
            output = model(Variable(images))
            loss = loss_func(output, Variable(labels))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print (f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.2f}')


train(num_epochs, model, loaders)

# torch.save(model.state_dict(), 'mnist_model_dense.pth')

## Test

def test():
    # Test the model
    model.eval()
    with torch.no_grad():
        accuracy = 0
        for images, labels in loaders['test']:
            test_output = model(images)
            pred_y = torch.max(test_output, 1)[1]
            accuracy += (pred_y == labels).sum().item() / float(labels.size(0))
        accuracy /= len(loaders['test'])
        print(f'The model performed at {accuracy:.2f} on the dataset.')

test()


## Get a good colormap and printing function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap



def print_arouser_for_neurones(model) :

    figure = plt.figure(figsize=(12, 10))
    # define color
    top = cm.get_cmap('Oranges_r', 128)
    bottom = cm.get_cmap('Blues', 128)
    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                        bottom(np.linspace(0, 1, 128))))
    cmap = ListedColormap(newcolors, name='OrangeBlue')
    cols, rows = int(hidden_size**(1/2)), int(hidden_size**(1/2))
    for i in range(1, cols * rows + 1):
        img = model.fc1.weight[i-1].reshape((28,28))
        figure.add_subplot(rows, cols, i)
        plt.title(f"{i}'th neuron")
        plt.axis("off")
        plt.imshow(img.squeeze().squeeze().detach(), cmap = cmap)

    plt.show()


print_arouser_for_neurones(model)
