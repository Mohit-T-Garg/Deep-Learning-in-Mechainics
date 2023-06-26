import numpy as np
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax
import scipy.io
from matplotlib import pyplot as plt
import pandas as pd
np.random.seed(42)
# %% Load data

train_images =  scipy.io.loadmat('train_images.mat')['train_images']
train_labels = scipy.io.loadmat('train_labels.mat')['train_labels'].reshape((-1,))
test_images = scipy.io.loadmat('test_images.mat')['test_images']
test_labels = scipy.io.loadmat('test_labels.mat')['test_labels'].reshape((-1,))
#print(train_images.shape,train_labels.shape)


# %% Model definition
conv = Conv3x3(9)                   # 28x28x1 -> 26x26x9
pool = MaxPool2()                  # 26x26x9 -> 13x13x9
softmax = Softmax(13 * 13 * 9, 10)                # 13x13x9 -> 10

def forward(image, label):
  '''
  Completes a forward pass of the CNN and calculates the accuracy and
  cross-entropy loss.
  - image is a 2d numpy array
  - label is a digit
  '''
  # Transform the grayscale image from [0, 255] to [-0.5, 0.5] to make it easier
  # to work with.
  transform_image = image/255 - 0.5
  out = conv.forward(transform_image)
  out = pool.forward(out)
  out = softmax.forward(out)

  # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
  loss = -np.log(out[label])
  acc=0
  #print(image.shape,label.shape,out)
  if np.argmax(out) == label:
    acc = 1 
  
  return out, loss, acc
  
def train(im, label, lr=.005):
  '''
  A training step on the given image and label.
  Shall return the cross-entropy loss and accuracy.
  - image is a 2d numpy array
  - label is a digit
  - lr is the learning rate
  '''
  # Forward
  out, loss, acc = forward(im, label)

  # Calculate initial gradient
  gradient = np.zeros(10)
  gradient[label] = -1 / out[label]

  # Backprop
  gradient = softmax.backprop(gradient, lr)
  gradient = pool.backprop(gradient)
  gradient = conv.backprop(gradient, lr)

  return loss, acc

print('MNIST CNN initialized! Number of epochs = 3')

# Train the CNN for 2 epochs
for epoch in range(3):
  print('--- Epoch %d ---' % (epoch + 1))

  # Shuffle the training data
  permutation = np.random.permutation(len(train_images))
  train_images = train_images[permutation]
  train_labels = train_labels[permutation]

  # Train!
  loss = 0
  num_correct = 0
  for i in range(1000):
    im,label = train_images[i],train_labels[i]
    if i % 100 == 99:
      print(
        '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
        (i + 1, loss / (i+1), 100*num_correct/(i+1))
      )
    l, acc = train(im, label)
    loss += l
    num_correct += acc

# Test the CNN for Training data
print('\n--- Testing the CNN on Training Data---')
loss = 0
num_correct = 0
for im, label in zip(train_images, train_labels):
  _, l, acc = forward(im, label)
  loss += l
  num_correct += acc

num_train = len(train_images)
print('Train Loss:', loss / num_train)
print('Train Accuracy:', num_correct / num_train)

# Test the CNN
print('\n--- Testing the CNN for Test data ---')
loss = 0
num_correct = 0
pred1=[]
for im, label in zip(test_images, test_labels):
  p, l, acc = forward(im, label)
  pred1.append(np.argmax(p))
  loss += l
  num_correct += acc

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)

def displayData(X,ax):
    """
    Displays the data from X
    """ 
    pixel = np.reshape(X,(28,28))
    ax.imshow(pixel)
    ax.axis("off")


# Randomly select data points to display
plt.figure(figsize = (8,15))
for i in range(40):
    ax = plt.subplot(4,10,i+1)
    k = np.random.randint(test_images.shape[0])

    displayData(test_images[k],ax)
    if pred1[k] != test_labels[k]:
        ax.set_title(f"{test_labels[k]} ({pred1[k]})", color="red")
    else:
        ax.set_title(f"{test_labels[k]} ({pred1[k]})", color="blue")

plt.show()
