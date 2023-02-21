# MNIST-Visualization

This Python script is an implementation of a dense neural network using PyTorch for the MNIST dataset. The script includes the following steps:

1. Collecting the MNIST dataset using `torchvision`.
2. Creating `DataLoaders` to load the data and split it into batches.
3. Defining a `DENSE` or a `CNN` neural network model, respectively with one hidden layer and 36 neurons and 2 convolutional 2d, and using the `nn.Module` class.
4. Training the model with stochastic gradient descent using `Adam` optimizer and negative log-likelihood loss function.
5. Testing the model and calculating its accuracy on the test data.
6. Displaying the résults in function of the important parts. For the `DENSE` the weights of the neurons in the hidden layer and for the `CNN` the maximizing input for each number or for the intern convolutional layers.

## Dependencies
The script requires the following Python packages:

- `torch`
- `torchvision`
- `numpy`
- `matplotlib`
- `tqdm`

## Usage
1. Install the dependencies by running `pip install torch torchvision numpy matplotlib tqdm`.
2. Download or clone the script to your local machine.
3. Navigate to the directory where the script is saved.
4. Run the script by typing `python dense.py` or `python CNN.py`.

## Outputs
The script outputs the following:

- The training progress will be printed to the console, including the loss for each epoch and batch.
- The final accuracy of the model on the test set will be printed to the console.
- A visualization of the corresponding expected output will be displayed.

## Greeting
I'd like to thanks Nutan whom I derived my code from [the website](https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118) and [this course](https://colab.research.google.com/drive/1wzfFReLo9JCVUabb0qfb-ypcuSLAJkAJ?usp=sharing) that gave me the main (yet not successful) ideas for the CNN visualization.
