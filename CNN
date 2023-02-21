## Collect data
import torch
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
# form the labeled images from the MNIST dataset
from torch.utils.data import DataLoader
loaders = {
    'train' : torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1),
    'test'  : torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1),}

## CNN

import torch.nn as nn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),)

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),)

        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x


## Define
cnn = CNN()
loss_func = nn.CrossEntropyLoss()

from torch import optim
optimizer = optim.Adam(cnn.parameters(), lr = 0.01)

from torch.autograd import Variable
num_epochs = 10

## Train
def train(num_epochs, cnn, loaders):
    # train the model
    cnn.train()
    total_step = len(loaders['train'])

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            # define variables
            b_x = Variable(images)
            b_y = Variable(labels)
            # get the output
            output = cnn(b_x)[0]
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print (f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.2f}')
train(num_epochs, cnn, loaders)

# torch.save(model.state_dict(), 'mnist_model.pth')

## Test
def test():
    # Test the model
    cnn.eval()
    with torch.no_grad():
        accuracy = 0
        for images, labels in loaders['test']:
            test_output, last_layer = cnn(images)
            pred_y = torch.max(test_output, 1)[1]
            # add the current accuracy on this batch
            accuracy += (pred_y == labels).sum().item() / float(labels.size(0))
        accuracy /= len(loaders['test'])
        print(f'The model performed at {accuracy:.2f} on the dataset.')

test()

## Getting result

sample = next(iter(loaders['test']))
imgs, lbls = sample

actual_number = lbls[:10].numpy()
actual_number

test_output, last_layer = cnn(imgs[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(f'Prediction number: {pred_y}')
print(f'Actual number: {actual_number}')

## Printing digits
import matplotlib.pyplot as plt

plt.imshow(train_data.data[0], cmap='gray')
plt.title('%i' % train_data.targets[0])
plt.show()

figure = plt.figure(figsize=(10, 8))
cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()



## Print images

def imshow(img, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    img = img.squeeze(0).squeeze(0)
    plt.imshow(img.detach(), cmap ="gray")
    plt.xticks([])
    plt.yticks([])
    plt.show()


## Img gen

class decorrImage(nn.Module):
    def __init__(self, scale=1.0, sigmoid=False):
        super().__init__()
        self.image = nn.Parameter(torch.rand(1, 1, 28, 28))
        projection = torch.Tensor(
            [[0.26, 0.09, 0.02],
             [0.27, 0.00, -0.05],
             [0.27, -0.09, 0.03]])
        self.register_buffer('projection', projection)
        self.scale = scale
        self.sigmoid = sigmoid

    def forward(self):
        if self.sigmoid:
            return torch.sigmoid(self.image * self.scale + 0.5)
        else:
            return torch.clip(self.image * self.scale + 0.5, min=0.0, max=1.0)

class sigmoidImage(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = nn.Parameter(torch.randn(1, 1, 28, 28))

    def forward(self):
        return self.image.sigmoid()



from tqdm import trange
from torchvision import transforms
TFORM = transforms.Compose([
    transforms.Pad(12),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.RandomAffine(
        degrees=10, translate=(0.05, 0.05), scale=(1.2, 1.2), shear=10
    ),
])

def optimize_img(img, model, lr=0.05, n_epochs=200, class_id=0, apply_transforms=False):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = img.to(DEVICE)
    model = model.to(DEVICE)

    opt = torch.optim.Adam(img.parameters(), lr=lr)

    for i in trange(n_epochs):
        opt.zero_grad()
        image = img()
        if apply_transforms:
            image = TFORM(image)
        loss = -model(image)[0][0][class_id]
        loss.backward()
        opt.step()

    imshow(img().cpu())


def optimize_img_no_print(img, model, lr=0.05, n_epochs=200, class_id=0, apply_transforms=False):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = img.to(DEVICE)
    model = model.to(DEVICE)

    opt = torch.optim.Adam(img.parameters(), lr=lr)

    for i in range(n_epochs):
        opt.zero_grad()
        image = img()
        if apply_transforms:
            image = TFORM(image)
        loss = -model(image)[0][0][class_id]
        loss.backward()
        opt.step()

    return img().cpu()


## Application

img = decorrImage(scale=1.0, sigmoid=False)
imshow(img())

img = sigmoidImage()

optimize_img(img, cnn, apply_transforms=False, n_epochs=200)

## Find what awoke the network the most according to the expected output

def print_arouser_depending_on_digit(model) :

    figure = plt.figure(figsize=(10, 8))
    cols, rows = 5, 2
    for i in range(1, cols * rows + 1):
        img = decorrImage(scale=1.0, sigmoid=False)
        img = optimize_img_no_print(img, model, class_id=i-1, apply_transforms = False,  n_epochs=200)

        figure.add_subplot(rows, cols, i)
        plt.title(f"{i-1}'th number'")
        plt.axis("off")
        plt.imshow(img.squeeze().squeeze().detach(), cmap="gray")
    plt.show()


print_arouser_depending_on_digit(cnn)


## Inner viz

class SaveOutput:
    def __init__(self):
        self.output = None

    def __call__(self, module, module_in, module_out):
        self.output = module_out

    def clear(self):
        self.output = None

def optimize_img_inner(img, model, relu_id, channel_id, lr=0.05, n_epochs=200, apply_transforms=False):
    # optimizes an image to activate some intermediate-layer channel of post-ReLU activation
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = img.to(DEVICE)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(img.parameters(), lr=lr)

    post_relu = []
    for i, m in enumerate(model.modules()):
        if isinstance(m,nn.Conv2d) :
            hook = SaveOutput()
            post_relu.append(hook)
            m.register_forward_hook(hook)

    for i in range(n_epochs):
        opt.zero_grad()
        image = img()
        if apply_transforms:
            image = TFORM(image)

        for hook in post_relu:
            hook.clear()
        model(image)

        loss = -(post_relu[relu_id].output[0, channel_id] **2).sum()
        loss.backward()
        opt.step()

    return img().cpu()

## Find what awoke the neuron the most

def print_intern_arouser(model) :

    figure = plt.figure(figsize=(10, 8))
    cols, rows = 5, 2
    for i in range(1, cols * rows + 1):
        img = decorrImage(scale=25.0, sigmoid=False)
        img = optimize_img_inner(img, model, relu_id=1, channel_id=i, n_epochs=400)

        figure.add_subplot(rows, cols, i)
        plt.title(f"{i}'th neuron")
        plt.axis("off")
        plt.imshow(img.squeeze().squeeze().detach(), cmap="gray")

    plt.show()


print_intern_arouser(cnn)

