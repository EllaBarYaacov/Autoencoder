# -*- coding: utf-8 -*-
"""Copy of ex2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lTGIqGEjYZimxU0tQ_oXvmvVb_hlw02c
"""

import torchvision
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import tensorflow as tf
from google.colab import drive

# read celeb data from drive
drive.mount('/content/drive')
celeb_data_path = "drive/My Drive/limudim/year_D_semester_B/NN/ex2/CelebA64.npy"
celeb_data = np.load(celeb_data_path)

device = torch.device("cuda")

batch_size = 5


def imshow(image):
  plt.imshow(image)
  plt.show()


def get_train_loader_celebs():
  # celeb_data_path = "CelebA64.npy"
  # drive.mount('/content/drive')
  # celeb_data_path = "drive/My Drive/limudim/year_D_semester_B/NN/ex2/CelebA64.npy"
  # celeb_data = np.load(celeb_data_path)
  dataloader = torch.utils.data.DataLoader(celeb_data, batch_size=batch_size, shuffle=True)
  return dataloader


def get_train_loader_mnist():
  transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
  train_dataset = torchvision.datasets.MNIST(root="~/torch_datasets", train=True, transform=transform, download=True)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
  return train_loader


def get_test_loader():
  transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
  test_dataset = torchvision.datasets.MNIST(root="~/torch_datasets", train=False, transform=transform, download=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
  return test_loader


def from_pixels_to_channels_view(image):
  return tf.stack([image[:,:,0], image[:,:,1], image[:,:,2]])

def from_channels_to_pixels_view(image):
  pixels_view = np.zeros(image.shape).ravel()
  pixels_view[0:: 3] = image[0].ravel()
  pixels_view[1:: 3] = image[1].ravel()
  pixels_view[2:: 3] = image[2].ravel()
  return pixels_view.reshape(image.shape[1], image.shape[2], image.shape[0])


class AE(nn.Module):
  def __init__(self, height, width, num_channels, filters_first_conv, bottle_neck_dim):
      super(AE, self).__init__()
      self.height = height
      self.width = width

      self.encoder_conv1 = nn.Conv2d(in_channels=num_channels, out_channels=filters_first_conv, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      self.encoder_conv2 = nn.Conv2d(in_channels=filters_first_conv, out_channels=2 * filters_first_conv, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
      self.encoder_conv3 = nn.Conv2d(in_channels=2 * filters_first_conv, out_channels=4 * filters_first_conv, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      self.encoder_fc1 = nn.Linear(4 * filters_first_conv * (int(width / 8) * int(height / 8)) * batch_size, 2 * bottle_neck_dim)
      self.encoder_fc2 = nn.Linear(2 * bottle_neck_dim, bottle_neck_dim)

      self.decoder_fc2 = nn.Linear(bottle_neck_dim, 2 * bottle_neck_dim)
      self.decoder_fc1 = nn.Linear(2 * bottle_neck_dim, 4 * filters_first_conv * (int(width / 8) * int(height / 8)) * batch_size)
      self.decoder_tconv3 = nn.ConvTranspose2d(in_channels=4 * filters_first_conv, out_channels=2 * filters_first_conv, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
      self.decoder_tconv2 = nn.ConvTranspose2d(in_channels=2 * filters_first_conv, out_channels=filters_first_conv, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
      self.decoder_tconv1 = nn.ConvTranspose2d(in_channels=filters_first_conv, out_channels=num_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))

  def forward(self, x):
      # ENCODER
      x = F.relu(self.encoder_conv1(x))
      x = F.relu(self.encoder_conv2(x))
      x = F.relu(self.encoder_conv3(x))
      self.shape_to_return = x.shape
      x = self.flat_features(x)
      x = F.relu(self.encoder_fc1(x))
      x = F.relu(self.encoder_fc2(x))

      # DECODER
      x = F.relu(self.decoder_fc2(x))
      x = F.relu(self.decoder_fc1(x))
      x = x.reshape(self.shape_to_return)
      x = F.relu(self.decoder_tconv3(x))
      x = F.relu(self.decoder_tconv2(x))
      x = F.relu(self.decoder_tconv1(x))
      return x

  def flat_features(self, x):
      size = x.size()
      num_features = 1
      for s in size:
          num_features *= s
      return x.reshape(num_features)


class AE_copy_leaky_relu(nn.Module):
  # def __init__(self, height, width, num_channels, filters_first_conv, bottle_neck_dim):
  def __init__(self, other_AE):
      super(AE_copy_leaky_relu, self).__init__()
      self.height = other_AE.height
      self.width = other_AE.width

      self.encoder_conv1 = other_AE.encoder_conv1
      self.encoder_conv2 = other_AE.encoder_conv2
      self.encoder_conv3 = other_AE.encoder_conv3
      self.encoder_fc1 = other_AE.encoder_fc1
      self.encoder_fc2 = other_AE.encoder_fc2

      self.decoder_fc2 = other_AE.decoder_fc2
      self.decoder_fc1 = other_AE.decoder_fc1
      self.decoder_tconv3 = other_AE.decoder_tconv3
      self.decoder_tconv2 = other_AE.decoder_tconv2
      self.decoder_tconv1 = other_AE.decoder_tconv1
      self.leaky_relu = nn.LeakyReLU(0.1)
      # self.drop_outs = nn.Dropout(p=0.5)

  def forward(self, x):
      # ENCODER
      x = self.leaky_relu(self.encoder_conv1(x))
      x = self.leaky_relu(self.encoder_conv2(x))
      x = self.leaky_relu(self.encoder_conv3(x))
      self.shape_to_return = x.shape
      x = self.flat_features(x)
      x = self.leaky_relu(self.encoder_fc1(x))
      x = self.leaky_relu(self.encoder_fc2(x))

      # DECODER
      x = self.leaky_relu(self.decoder_fc2(x))
      x = self.leaky_relu(self.decoder_fc1(x))
      x = x.reshape(self.shape_to_return)
      x = self.leaky_relu(self.decoder_tconv3(x))
      x = self.leaky_relu(self.decoder_tconv2(x))
      x = self.leaky_relu(self.decoder_tconv1(x))
      return x

  def flat_features(self, x):
      size = x.size()
      num_features = 1
      for s in size:
          num_features *= s
      return x.reshape(num_features)



class AE_copy_dropouts(nn.Module):
  # def __init__(self, height, width, num_channels, filters_first_conv, bottle_neck_dim):
  def __init__(self, other_AE):
      super(AE_copy_dropouts, self).__init__()
      self.height = other_AE.height
      self.width = other_AE.width

      self.encoder_conv1 = other_AE.encoder_conv1
      self.encoder_conv2 = other_AE.encoder_conv2
      self.encoder_conv3 = other_AE.encoder_conv3
      self.encoder_fc1 = other_AE.encoder_fc1
      self.encoder_fc2 = other_AE.encoder_fc2

      self.decoder_fc2 = other_AE.decoder_fc2
      self.decoder_fc1 = other_AE.decoder_fc1
      self.decoder_tconv3 = other_AE.decoder_tconv3
      self.decoder_tconv2 = other_AE.decoder_tconv2
      self.decoder_tconv1 = other_AE.decoder_tconv1
      self.leaky_relu = nn.LeakyReLU(0.1)
      self.drop_outs = nn.Dropout(p=0.5)

  def forward(self, x):
      # ENCODER
      x = F.relu(self.encoder_conv1(x))
      x = F.relu(self.encoder_conv2(x))
      x = F.relu(self.encoder_conv3(x))
      shape_to_return = x.shape
      x = self.flat_features(x)
      x = F.relu(self.drop_outs(self.encoder_fc1(x)))
      x = F.relu(self.drop_outs(self.encoder_fc2(x)))

      # DECODER
      x = F.relu(self.drop_outs(self.decoder_fc2(x)))
      x = F.relu(self.drop_outs(self.decoder_fc1(x)))
      x = x.reshape(shape_to_return)
      x = F.relu((self.decoder_tconv3(x)))
      x = F.relu((self.decoder_tconv2(x)))
      x = F.relu(self.decoder_tconv1(x))
      return x

  def flat_features(self, x):
      size = x.size()
      num_features = 1
      for s in size:
          num_features *= s
      return x.reshape(num_features)


class AE_tanh(nn.Module):
  def __init__(self, height, width, num_channels, filters_first_conv, bottle_neck_dim):
      super(AE_tanh, self).__init__()
      self.height = height
      self.width = width

      self.encoder_conv1 = nn.Conv2d(in_channels=num_channels, out_channels=filters_first_conv, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      self.encoder_conv2 = nn.Conv2d(in_channels=filters_first_conv, out_channels=2 * filters_first_conv, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
      self.encoder_conv3 = nn.Conv2d(in_channels=2 * filters_first_conv, out_channels=4 * filters_first_conv, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      self.encoder_fc1 = nn.Linear(4 * filters_first_conv * (int(width / 8) * int(height / 8)) * batch_size, 2 * bottle_neck_dim)
      self.encoder_fc2 = nn.Linear(2 * bottle_neck_dim, bottle_neck_dim)

      self.decoder_fc2 = nn.Linear(bottle_neck_dim, 2 * bottle_neck_dim)
      self.decoder_fc1 = nn.Linear(2 * bottle_neck_dim, 4 * filters_first_conv * (int(width / 8) * int(height / 8)) * batch_size)
      self.decoder_tconv3 = nn.ConvTranspose2d(in_channels=4 * filters_first_conv, out_channels=2 * filters_first_conv, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
      self.decoder_tconv2 = nn.ConvTranspose2d(in_channels=2 * filters_first_conv, out_channels=filters_first_conv, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
      self.decoder_tconv1 = nn.ConvTranspose2d(in_channels=filters_first_conv, out_channels=num_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))

  def forward(self, x):
      # ENCODER
      x = F.relu(self.encoder_conv1(x))
      x = F.relu(self.encoder_conv2(x))
      x = F.relu(self.encoder_conv3(x))
      self.shape_to_return = x.shape
      x = self.flat_features(x)
      x = F.relu(self.encoder_fc1(x))
      x = F.relu(self.encoder_fc2(x))

      # DECODER
      x = F.relu(self.decoder_fc2(x))
      x = F.relu(self.decoder_fc1(x))
      x = x.reshape(self.shape_to_return)
      x = F.relu(self.decoder_tconv3(x))
      x = F.relu(self.decoder_tconv2(x))
      x = F.relu(self.decoder_tconv1(x))
      x = F.tanh(x)
      return x

  def flat_features(self, x):
      size = x.size()
      num_features = 1
      for s in size:
          num_features *= s
      return x.reshape(num_features)


class AE_sigmoid(nn.Module):
  def __init__(self, height, width, num_channels, filters_first_conv, bottle_neck_dim):
      super(AE_sigmoid, self).__init__()
      self.height = height
      self.width = width

      self.encoder_conv1 = nn.Conv2d(in_channels=num_channels, out_channels=filters_first_conv, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      self.encoder_conv2 = nn.Conv2d(in_channels=filters_first_conv, out_channels=2 * filters_first_conv, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
      self.encoder_conv3 = nn.Conv2d(in_channels=2 * filters_first_conv, out_channels=4 * filters_first_conv, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      self.encoder_fc1 = nn.Linear(4 * filters_first_conv * (int(width / 8) * int(height / 8)) * batch_size, 2 * bottle_neck_dim)
      self.encoder_fc2 = nn.Linear(2 * bottle_neck_dim, bottle_neck_dim)

      self.decoder_fc2 = nn.Linear(bottle_neck_dim, 2 * bottle_neck_dim)
      self.decoder_fc1 = nn.Linear(2 * bottle_neck_dim, 4 * filters_first_conv * (int(width / 8) * int(height / 8)) * batch_size)
      self.decoder_tconv3 = nn.ConvTranspose2d(in_channels=4 * filters_first_conv, out_channels=2 * filters_first_conv, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
      self.decoder_tconv2 = nn.ConvTranspose2d(in_channels=2 * filters_first_conv, out_channels=filters_first_conv, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
      self.decoder_tconv1 = nn.ConvTranspose2d(in_channels=filters_first_conv, out_channels=num_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))

  def forward(self, x):
      # ENCODER
      x = F.relu(self.encoder_conv1(x))
      x = F.relu(self.encoder_conv2(x))
      x = F.relu(self.encoder_conv3(x))
      self.shape_to_return = x.shape
      x = self.flat_features(x)
      x = F.relu(self.encoder_fc1(x))
      x = F.relu(self.encoder_fc2(x))

      # DECODER
      x = F.relu(self.decoder_fc2(x))
      x = F.relu(self.decoder_fc1(x))
      x = x.reshape(self.shape_to_return)
      x = F.relu(self.decoder_tconv3(x))
      x = F.relu(self.decoder_tconv2(x))
      x = F.relu(self.decoder_tconv1(x))
      x = torch.sigmoid(x)*255
      return x
      
  def flat_features(self, x):
      size = x.size()
      num_features = 1
      for s in size:
        num_features *= s
      return x.reshape(num_features)

def train_model(model, train_loader, optimizer, criterion, epochs, m_name):
  for epoch in range(epochs):
      loss = 0
      i = 0
      model_name = m_name +"_" +str(epoch+1)+"_epochs"
      for batch_images in train_loader:
          if i % 100 == 0:
              print("epoch=", str(epoch+1),"  batch=",str(i),"batch shape=", str(batch_images.shape))
          if batch_images.shape[0] != 5:
              continue
          batch_images = tf.stack([from_pixels_to_channels_view(image) for image in batch_images])
          batch_images = torch.Tensor(batch_images.numpy().astype(float)).to(device)
          optimizer.zero_grad()  # reset the gradients back to zero
          outputs = model(batch_images)  # compute reconstructions
          train_loss = criterion(outputs, batch_images)  # compute training reconstruction loss
          train_loss.backward()  # compute accumulated gradients
          optimizer.step()  # perform parameter update based on current gradients

          # add the mini-batch training loss to epoch loss
          loss += train_loss.item()
          i += 1
      
      # compute the epoch training loss
      loss = loss / len(train_loader)
      torch.save(model, "drive/My Drive/limudim/year_D_semester_B/NN/ex3/"+model_name+"_Q4_loss_"+str(loss))
      # display the epoch training loss
      print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))


def train_model_tanh(model, train_loader, optimizer, criterion, epochs, m_name):
    for epoch in range(epochs):
        loss = 0
        i = 0
        model_name = m_name + "_" + str(epoch + 1) + "_epochs"
        for batch_images in train_loader:
            if i % 100 == 0:
                print("this_is_epoch_number ", str(epoch), "  batch=", str(i))
            if batch_images.shape[0] != 5:
                continue

            batch_images = tf.stack([from_pixels_to_channels_view(image) for image in batch_images])
            tanh_batch_images = (batch_images.numpy()/255)*2-1  
            batch_images = torch.Tensor(batch_images.numpy().astype(float)).to(device)
            tanh_batch_images = torch.Tensor(tanh_batch_images).to(device)

            optimizer.zero_grad()  # reset the gradients back to zero
            outputs = model(batch_images)  # compute reconstructions 
            train_loss = criterion(outputs, tanh_batch_images)  # compute training reconstruction loss
            train_loss.backward()  # compute accumulated gradients
            optimizer.step()  # perform parameter update based on current gradients

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
            i += 1

        # compute the epoch training loss
        loss = loss / len(train_loader)
        torch.save(model, "drive/My Drive/limudim/year_D_semester_B/NN/ex2/" + model_name + "_loss_" + str(loss))
        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

if __name__ == '__main__':
    model = AE_sigmoid(64, 64, 3, 16, 300).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    train_loader = get_train_loader_celebs()
    train_model(model, train_loader, optimizer, criterion, 10, "AE_16f_sigmoid")