import torchvision
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import tensorflow as tf

batch_size = 5


def imshow(image1, image2=None, title=None):
    # print(image1.shape)
    # print(image2.shape)
    if image2:
        plt.imshow(np.concatenate([image1, image2], axis=1))
    else:
        plt.imshow(image1)
    plt.title(title)
    plt.show()


def get_train_loader_celebs():
    celeb_data_path = "CelebA64.npy"
    celeb_data = np.load(celeb_data_path)
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

class AE_copy(nn.Module):
    # def __init__(self, height, width, num_channels, filters_first_conv, bottle_neck_dim):
    def __init__(self, other_AE):
        super(AE_copy, self).__init__()
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


class AE_batch_norm(nn.Module):
    def __init__(self, height, width, num_channels, filters_first_conv, bottle_neck_dim):
        super(AE_batch_norm, self).__init__()
        self.height = height
        self.width = width

        self.encoder_conv1 = nn.Conv2d(in_channels=num_channels, out_channels=filters_first_conv, kernel_size=(3, 3),
                                       stride=(2, 2), padding=(1, 1))
        self.encoder_conv2 = nn.Conv2d(in_channels=filters_first_conv, out_channels=2 * filters_first_conv,
                                       kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.encoder_conv3 = nn.Conv2d(in_channels=2 * filters_first_conv, out_channels=4 * filters_first_conv,
                                       kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.encoder_fc1 = nn.Linear(4 * filters_first_conv * (int(width / 8) * int(height / 8)) * batch_size,
                                     2 * bottle_neck_dim)
        self.encoder_fc2 = nn.Linear(2 * bottle_neck_dim, bottle_neck_dim)

        self.decoder_fc2 = nn.Linear(bottle_neck_dim, 2 * bottle_neck_dim)
        self.decoder_fc1 = nn.Linear(2 * bottle_neck_dim,
                                     4 * filters_first_conv * (int(width / 8) * int(height / 8)) * batch_size)
        self.decoder_tconv3 = nn.ConvTranspose2d(in_channels=4 * filters_first_conv,
                                                 out_channels=2 * filters_first_conv, kernel_size=(3, 3), stride=(2, 2),
                                                 padding=(1, 1), output_padding=(1, 1))
        self.decoder_tconv2 = nn.ConvTranspose2d(in_channels=2 * filters_first_conv, out_channels=filters_first_conv,
                                                 kernel_size=(5, 5), stride=(2, 2), padding=(2, 2),
                                                 output_padding=(1, 1))
        self.decoder_tconv1 = nn.ConvTranspose2d(in_channels=filters_first_conv, out_channels=num_channels,
                                                 kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                                 output_padding=(1, 1))
        self.batch_norm3 = nn.BatchNorm2d(num_channels)
        self.batch_norm16 = nn.BatchNorm2d(filters_first_conv)
        self.batch_norm32 = nn.BatchNorm2d(2 * filters_first_conv)
        self.batch_norm64 = nn.BatchNorm2d(4 * filters_first_conv)

    def forward(self, x):
        # ENCODER
        x = F.relu(self.batch_norm16(self.encoder_conv1(x)))
        x = F.relu(self.batch_norm32(self.encoder_conv2(x)))
        x = F.relu(self.batch_norm64(self.encoder_conv3(x)))
        self.shape_to_return = x.shape
        x = self.flat_features(x)
        x = F.relu(self.encoder_fc1(x))
        x = F.relu(self.encoder_fc2(x))

        # DECODER
        x = F.relu(self.decoder_fc2(x))
        x = F.relu(self.decoder_fc1(x))
        x = x.reshape(self.shape_to_return)
        x = F.relu(self.batch_norm32(self.decoder_tconv3(x)))
        x = F.relu(self.batch_norm16(self.decoder_tconv2(x)))
        x = F.relu(self.batch_norm3(self.decoder_tconv1(x)))
        return x

    def flat_features(self, x):
        size = x.size()
        num_features = 1
        for s in size:
            num_features *= s
        return x.reshape(num_features)


class AE_specific_latent_space_batch_norm(nn.Module):
    def __init__(self, height, width, num_channels, filters_first_conv, bottle_neck_dim):
        super(AE_specific_latent_space_batch_norm, self).__init__()
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
        self.batch_norm3 = nn.BatchNorm2d(num_channels)
        self.batch_norm16 = nn.BatchNorm2d(filters_first_conv)
        self.batch_norm32 = nn.BatchNorm2d(2 * filters_first_conv)
        self.batch_norm64 = nn.BatchNorm2d(4 * filters_first_conv)

    def forward(self, x):
        # ENCODER
        x = F.relu(self.batch_norm16(self.encoder_conv1(x)))
        x = F.relu(self.batch_norm32(self.encoder_conv2(x)))
        x = F.relu(self.batch_norm64(self.encoder_conv3(x)))
        self.shape_to_return = x.shape
        x = self.flat_features(x)
        x = F.relu(self.encoder_fc1(x))
        x = F.relu(self.encoder_fc2(x))
        x = F.sigmoid(x)
        z_vec = x

        # DECODER
        x = F.relu(self.decoder_fc2(x))
        x = F.relu(self.decoder_fc1(x))
        x = x.reshape(self.shape_to_return)
        x = F.relu(self.batch_norm32(self.decoder_tconv3(x)))
        x = F.relu(self.batch_norm16(self.decoder_tconv2(x)))
        x = F.relu(self.batch_norm3(self.decoder_tconv1(x)))
        return x, z_vec

    def flat_features(self, x):
        size = x.size()
        num_features = 1
        for s in size:
            num_features *= s
        return x.reshape(num_features)



class AE_copy_returns_z_too(nn.Module):
    def __init__(self, autoencoder):
        super(AE_copy_returns_z_too, self).__init__()

        self.encoder_conv1 = autoencoder.encoder_conv1
        self.encoder_conv2 = autoencoder.encoder_conv2
        self.encoder_conv3 = autoencoder.encoder_conv3
        self.encoder_fc1 = autoencoder.encoder_fc1
        self.encoder_fc2 = autoencoder.encoder_fc2

        self.decoder_fc2 = autoencoder.decoder_fc2
        self.decoder_fc1 = autoencoder.decoder_fc1
        self.decoder_tconv3 = autoencoder.decoder_tconv3
        self.decoder_tconv2 = autoencoder.decoder_tconv2
        self.decoder_tconv1 = autoencoder.decoder_tconv1
        self.batch_norm3 = autoencoder.batch_norm3
        self.batch_norm16 = autoencoder.batch_norm16
        self.batch_norm32 = autoencoder.batch_norm32
        self.batch_norm64 = autoencoder.batch_norm64

    def forward(self, x):
        # ENCODER
        x = F.relu(self.batch_norm16(self.encoder_conv1(x)))
        x = F.relu(self.batch_norm32(self.encoder_conv2(x)))
        x = F.relu(self.batch_norm64(self.encoder_conv3(x)))
        self.shape_to_return = x.shape
        x = self.flat_features(x)
        x = F.relu(self.encoder_fc1(x))
        x = F.relu(self.encoder_fc2(x))
        z_vec = x

        # DECODER
        x = F.relu(self.decoder_fc2(x))
        x = F.relu(self.decoder_fc1(x))
        x = x.reshape(self.shape_to_return)
        x = F.relu(self.batch_norm32(self.decoder_tconv3(x)))
        x = F.relu(self.batch_norm16(self.decoder_tconv2(x)))
        x = F.relu(self.batch_norm3(self.decoder_tconv1(x)))
        return x, z_vec

    def flat_features(self, x):
        size = x.size()
        num_features = 1
        for s in size:
            num_features *= s
        return x.reshape(num_features)


class AE_copy_rdecoder_only(nn.Module):
    def __init__(self, autoencoder):
        super(AE_copy_rdecoder_only, self).__init__()

        self.shape_to_return = autoencoder.shape_to_return
        self.decoder_fc2 = autoencoder.decoder_fc2
        self.decoder_fc1 = autoencoder.decoder_fc1
        self.decoder_tconv3 = autoencoder.decoder_tconv3
        self.decoder_tconv2 = autoencoder.decoder_tconv2
        self.decoder_tconv1 = autoencoder.decoder_tconv1
        self.batch_norm3 = autoencoder.batch_norm3
        self.batch_norm16 = autoencoder.batch_norm16
        self.batch_norm32 = autoencoder.batch_norm32
        self.batch_norm64 = autoencoder.batch_norm64

    def forward(self, x):
        # DECODER
        x = F.relu(self.decoder_fc2(x))
        x = F.relu(self.decoder_fc1(x))
        x = x.reshape(self.shape_to_return)
        x = F.relu(self.batch_norm32(self.decoder_tconv3(x)))
        x = F.relu(self.batch_norm16(self.decoder_tconv2(x)))
        x = F.relu(self.batch_norm3(self.decoder_tconv1(x)))
        return x

    def flat_features(self, x):
        size = x.size()
        num_features = 1
        for s in size:
            num_features *= s
        return x.reshape(num_features)


def train_model(model, train_loader, optimizer, criterion, epochs):
    for epoch in range(epochs):
        loss = 0
        i = 0
        model_name = "autoencoder_model_" +str(epoch)+"_epochs"
        for batch_images in train_loader:

            if i % 100 == 0:
                print("epoch=", str(epoch),"  batch=",str(i),"batch shape=", str(batch_images.shape))
            if batch_images.shape[0] != batch_size:
                continue
            batch_images = tf.stack([from_pixels_to_channels_view(image) for image in batch_images])
            batch_images = torch.Tensor(batch_images.numpy().astype(float))
            optimizer.zero_grad()  # reset the gradients back to zero
            outputs = model(batch_images)  # compute reconstructions
            train_loss = criterion(outputs, batch_images)  # compute training reconstruction loss
            train_loss.backward()  # compute accumulated gradients
            optimizer.step()  # perform parameter update based on current gradients

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
            i += 1
        torch.save(model, model_name)  # save model

        # compute the epoch training loss
        loss = loss / len(train_loader)

        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))


def train_to_specific_latent_space(model, train_loader, optimizer, criterion1, criterion2, epochs, m_name):
    for epoch in range(epochs):
        loss = 0
        i = 0
        model_name = m_name + "_" + str(epoch + 1) + "_epochs"
        for batch_images in train_loader:
            if i % 100 == 0:
                print("this_is_epoch_number ", str(epoch), "  batch=", str(i))
            if batch_images.shape[0] != batch_size:
                continue

            batch_images = tf.stack([from_pixels_to_channels_view(image) for image in batch_images])
            batch_images = torch.Tensor(batch_images.numpy().astype(float))
            optimizer.zero_grad()  # reset the gradients back to zero
            outputs, z_vec = model(batch_images)  # compute reconstructions
            reconstruction_loss = criterion1(outputs, batch_images)
            vec_from_normal_dist = torch.Tensor(np.random.normal(loc = 0, scale=1, size = z_vec.shape))
            latent_space_loss = criterion2(vec_from_normal_dist, z_vec)
            train_loss = reconstruction_loss + latent_space_loss # compute training reconstruction loss + loss in latent space
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


def show_encoder_results(autoencoder):
    if isinstance(autoencoder, str):
        autoencoder = torch.load(autoencoder, map_location=torch.device('cpu'))
    data_loader = get_train_loader_celebs()
    i = 0
    for batch in data_loader:
        if i == 1:
            break
        # imshow(batch.reshape(64 * batch_size, 64, 3))
        images_channels_view = tf.stack([from_pixels_to_channels_view(image) for image in batch])
        images_channels_view = torch.Tensor(images_channels_view.numpy().astype(float))
        restored = autoencoder(images_channels_view)
        show_batch(restored, batch)


def show_batch(batch1, batch2 = None, title= None):
    batch1 = batch1.detach().numpy().astype(int)
    batch1_pixels_view = np.stack([from_channels_to_pixels_view(im) for im in batch1]).astype(int)
    batch1_pixels_view = batch1_pixels_view.reshape(64 * batch_size, 64, 3)
    if batch2 != None:
        batch2 = batch2.detach().numpy().astype(int)
        if batch2.shape[0] == 3:
            batch2 = np.stack([from_channels_to_pixels_view(im) for im in batch2]).astype(int)
        batch2 = batch2.reshape(64 * batch_size, 64, 3)
        imshow(batch1_pixels_view, batch2, title)
    else:
        imshow(batch1_pixels_view, title)


def show_interpulation(model):
    if isinstance(model, str):
        model = torch.load(model, map_location=torch.device('cpu'))

    copied_model = AE_copy_returns_z_too(model)
    decoder_only = AE_copy_rdecoder_only(model)

    data_loader = get_train_loader_celebs()

    batches = [None, None]
    for i, batch in enumerate(data_loader):
        if i == 2 : break
        images_channels_view = tf.stack([from_pixels_to_channels_view(image) for image in batch])
        images_channels_view = torch.Tensor(images_channels_view.numpy().astype(float))
        batches[i] = images_channels_view

    restored_batch1, z_code1 = copied_model(batches[0])
    restored_batch2, z_code2 = copied_model(batches[1])

    interpolated_z_codes = []
    for i in range(11):
        interpolated_z_codes.append((i*0.1*z_code1)+((10-i)*0.1)*z_code2)

    interpulated_restorations = []
    for i in range(11):
        novel_batch = decoder_only(interpolated_z_codes[i])
        novel_batch = novel_batch.detach().numpy().astype(int)
        novel_batch = np.stack([from_channels_to_pixels_view(im) for im in novel_batch]).astype(int)
        novel_batch = novel_batch.reshape(64 * batch_size, 64, 3)
        interpulated_restorations.append(novel_batch)

    plt.imshow(np.concatenate(interpulated_restorations, axis=1))
    plt.title("interpolation")
    plt.show()

def generate_novel_batch(model, z_code, title = None):
    if isinstance(model, str):
        model = torch.load(model, map_location=torch.device('cpu'))
    decoder_only = AE_copy_rdecoder_only(model)
    novel_batch = decoder_only(z_code)
    show_batch(novel_batch, title)


def novel_batches_change(model, title = None):
    if isinstance(model, str):
        model = torch.load(model, map_location=torch.device('cpu'))
    decoder_only = AE_copy_rdecoder_only(model)
    novel_batches = []
    for loc in [-1000, -100, -30,-20,-10,-3,0,3,10,20,30, 100, 1000]:
        z_code = torch.Tensor(np.random.normal(loc=loc, scale=1, size=300))
        novel_batch = decoder_only(z_code)
        novel_batch = novel_batch.detach().numpy().astype(int)
        novel_batch = np.stack([from_channels_to_pixels_view(im) for im in novel_batch]).astype(int)
        novel_batch = novel_batch.reshape(64 * batch_size, 64, 3)
        novel_batches.append(novel_batch)

    plt.imshow(np.concatenate(novel_batches, axis=1))
    plt.title(title)
    plt.show()





if __name__ == '__main__':
    model_path1 = "AE_16f_batch_norm_57_epochs_6_epochs_loss_1465.3088011234204"
    model_path2 = "AE_latent_space_trained_36_epochs_19_epochs_loss_1540.5608541008514"
    model_path2 = "AE_-3mean_latent_space_5epochs_plus_6_epochs_loss_2191.732823154039"
    # z_code = torch.Tensor(np.random.normal(loc=-3, scale=1, size=300))
    # generate_novel_batch(model_path2, z_code, title=None)
    # generate_novel_batch(model_path2, z_code, title=None)
    novel_batches_change(model_path1)
    # novel_batches_change(model_path2)












