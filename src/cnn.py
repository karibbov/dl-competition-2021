""" File with CNN models. Add your custom CNN model here. """
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class SampleModel(nn.Module):
    """
    A sample PyTorch CNN model
    """

    def __init__(self, input_shape=(3, 64, 64), num_classes=10):
        super(SampleModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=10, kernel_size=(5, 5), padding=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=40, out_channels=20, kernel_size=(3, 3), padding=(1, 1))
        self.conv5 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(3, 3), padding=(1, 1))
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn1 = nn.BatchNorm2d(20)
        self.bn2 = nn.BatchNorm2d(20)
        # The input features for the linear layer depends on the size of the input to the convolutional layer
        # So if you resize your image in data augmentations, you'll have to tweak this too.
        dims, _ = calculate_dims(input_shape, 10, (5, 5), (2, 2))
        # dims, _ = calculate_dims(dims, dims[0], (2, 2), (0, 0), 2)
        dims, _ = calculate_dims(dims, 20, (3, 3), (1, 1))
        dims, _ = calculate_dims(dims, 40, (3, 3), (1, 1))
        dims, _ = calculate_dims(dims, dims[0], (2, 2), (0, 0), 2)
        dims, _ = calculate_dims(dims, 20, (3, 3), (1, 1))
        dims, nodes = calculate_dims(dims, dims[0], (2, 2), (0, 0), 2)
        dims, nodes = calculate_dims(dims, 10, (3, 3), (1, 1))
        self.fc1 = nn.Linear(in_features=nodes, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.pool(x)
        x = F.relu(x)  # 62
        x = self.conv2(x)
        x = self.bn1(x)
        # x = self.pool(x)
        x = F.relu(x)
        x = self.conv3(x)
        # x = self.bn1(x)
        x = self.pool(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn2(x)
        x = self.pool(x)
        x = F.relu(x)
        x = self.conv5(x)
        # x = self.pool(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


class SoftmaxModel(nn.Module):
    """
    A sample PyTorch CNN model
    """

    def __init__(self, input_shape=(3, 64, 64), num_classes=10):
        super(SoftmaxModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=10, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=(1, 1))
        self.pool = nn.MaxPool2d(3, stride=2)
        # The input features for the linear layer depends on the size of the input to the convolutional layer
        # So if you resize your image in data augmentations, you'll have to tweak this too.
        dims, _ = calculate_dims(input_shape, 10, (3, 3), (1, 1))
        # dims, _ = calculate_dims(dims, dims[0], (3, 3), (0, 0), 2)
        dims, _ = calculate_dims(dims, 20, (3, 3), (1, 1))
        dims, nodes = calculate_dims(dims, dims[0], (3, 3), (0, 0), 2)
        self.fc1 = nn.Linear(in_features=nodes, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=num_classes)
        self.sm = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = F.relu(x)  # 56
        x = self.conv2(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.sm(x)  # 55

        return x


class DeepModel(nn.Module):
    """
    A sample PyTorch CNN model
    """

    def __init__(self, input_shape=(3, 64, 64), num_classes=10):
        super(DeepModel, self).__init__()
        self.bn_momentum = 0.2
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=20, kernel_size=(15, 15), padding=(7, 7),
                               dilation=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(9, 9), padding=(4, 4), dilation=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=60, kernel_size=(5, 5), padding=(2, 2), dilation=(2, 2))
        # self.conv4 = nn.Conv2d(in_channels=40, out_channels=60, kernel_size=(3, 3), padding=(1, 1))
        self.conv5 = nn.Conv2d(in_channels=60, out_channels=20, kernel_size=(3, 3), padding=(1, 1))
        self.conv6 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3, 3), padding=(1, 1))
        self.conv7 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(3, 3), padding=(1, 1))
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.bn1 = nn.BatchNorm2d(20, momentum=self.bn_momentum)
        self.bn2 = nn.BatchNorm2d(60, momentum=self.bn_momentum)
        self.bn3 = nn.BatchNorm2d(20, momentum=self.bn_momentum)
        self.bn4 = nn.BatchNorm2d(20, momentum=self.bn_momentum)
        self.bn5 = nn.BatchNorm2d(10, momentum=self.bn_momentum)
        self.bn6 = nn.BatchNorm2d(20, momentum=self.bn_momentum)
        # The input features for the linear layer depends on the size of the input to the convolutional layer
        # So if you resize your image in data augmentations, you'll have to tweak this too.
        dims, _ = calculate_dims(input_shape, 20, (15, 15), (7, 7), 1, (2, 2))
        dims, _ = calculate_dims(dims, dims[0], (2, 2), (0, 0), 2)
        dims, _ = calculate_dims(dims, 20, (9, 9), (4, 4), 1, (2, 2))
        dims, _ = calculate_dims(dims, 60, (5, 5), (2, 2), 1, (2, 2))
        dims, _ = calculate_dims(dims, dims[0], (2, 2), (0, 0), 2)
        # dims, _ = calculate_dims(dims, 60, (3, 3), (1, 1))
        dims, _ = calculate_dims(dims, 20, (3, 3), (1, 1))
        dims, _ = calculate_dims(dims, dims[0], (2, 2), (0, 0), 2)
        dims, _ = calculate_dims(dims, 20, (3, 3), (1, 1))
        dims, nodes = calculate_dims(dims, 10, (3, 3), (1, 1))
        dims, nodes = calculate_dims(dims, dims[0], (2, 2), (0, 0), 2)
        # dims, _ = calculate_dims(dims, 20, (3, 3), (1, 1))
        # dims, nodes = calculate_dims(dims, dims[0], (2, 2), (0, 0), 2)
        # dims, nodes = calculate_dims(dims, 10, (3, 3), (1, 1))
        self.fc1 = nn.Linear(in_features=nodes, out_features=16)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_features=16, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)  # 62
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn4(x)
        # x = self.pool(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # x = self.conv4(x)
        # x = F.relu(x)
        # x = self.pool(x)

        x = self.conv5(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv6(x)
        x = self.bn3(x)
        x = F.relu(x)
        # x = self.pool(x)

        x = self.conv7(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


class SkipModel(nn.Module):
    """
    A sample PyTorch CNN model
    """

    def __init__(self, input_shape=(3, 64, 64), num_classes=10):
        super(SkipModel, self).__init__()
        self.bn_momentum = 0.2

        skip1_input = input_shape
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=20, kernel_size=(15, 15), padding=(7, 7),
                               dilation=(2, 2))
        dims, _ = calculate_dims(input_shape, 20, (15, 15), (7, 7), 1, (2, 2))
        dims, _ = calculate_dims(dims, dims[0], (2, 2), (0, 0), 2)

        self.conv2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(9, 9), padding=(4, 4), dilation=(2, 2))
        dims, _ = calculate_dims(dims, 20, (9, 9), (4, 4), 1, (2, 2))
        skip2_input = dims
        skip1_output = dims

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=60, kernel_size=(5, 5), padding=(2, 2), dilation=(2, 2))
        dims, _ = calculate_dims(dims, 60, (5, 5), (2, 2), 1, (2, 2))
        dims, _ = calculate_dims(dims, dims[0], (2, 2), (0, 0), 2)



        self.conv5 = nn.Conv2d(in_channels=60, out_channels=20, kernel_size=(3, 3), padding=(1, 1))
        dims, _ = calculate_dims(dims, 20, (3, 3), (1, 1))
        dims, _ = calculate_dims(dims, dims[0], (2, 2), (0, 0), 2)
        skip3_input = dims
        skip2_output = dims

        self.conv6 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3, 3), padding=(1, 1))
        dims, _ = calculate_dims(dims, 20, (3, 3), (1, 1))

        self.conv7 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(3, 3), padding=(1, 1))
        dims, nodes = calculate_dims(dims, 10, (3, 3), (1, 1))
        dims, nodes = calculate_dims(dims, dims[0], (2, 2), (0, 0), 2)
        skip3_output = dims

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        downsample1_by = ceil(skip1_input[-1] / skip1_output[-1])
        self.downsample1 = nn.MaxPool2d(downsample1_by, stride=downsample1_by, ceil_mode=True)
        downsample2_by = ceil(skip2_input[-1] / skip2_output[-1])
        self.downsample2 = nn.MaxPool2d(downsample2_by, stride=downsample2_by, ceil_mode=True)
        downsample3_by = ceil(skip3_input[-1] / skip3_output[-1])
        self.downsample3 = nn.MaxPool2d(downsample3_by, stride=downsample3_by, ceil_mode=True)

        self.bn1 = nn.BatchNorm2d(20, momentum=self.bn_momentum)
        self.bn2 = nn.BatchNorm2d(60, momentum=self.bn_momentum)
        self.bn3 = nn.BatchNorm2d(20, momentum=self.bn_momentum)
        self.bn4 = nn.BatchNorm2d(20, momentum=self.bn_momentum)
        self.bn5 = nn.BatchNorm2d(10, momentum=self.bn_momentum)
        self.bn6 = nn.BatchNorm2d(20, momentum=self.bn_momentum)
        # The input features for the linear layer depends on the size of the input to the convolutional layer
        # So if you resize your image in data augmentations, you'll have to tweak this too.

        # dims, _ = calculate_dims(dims, 60, (3, 3), (1, 1))

        # dims, _ = calculate_dims(dims, 20, (3, 3), (1, 1))
        # dims, nodes = calculate_dims(dims, dims[0], (2, 2), (0, 0), 2)
        # dims, nodes = calculate_dims(dims, 10, (3, 3), (1, 1))
        self.fc1 = nn.Linear(in_features=nodes, out_features=16)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_features=16, out_features=num_classes)

    def forward(self, x):
        x_skip1 = self.downsample1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)  # 62
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn4(x)
        # x = self.pool(x)
        x = F.relu(x)

        x_skip2 = self.downsample2(x)
        x = self.conv3(x + pad3d_to(x_skip1, x.size()))
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # x = self.conv4(x)
        # x = F.relu(x)
        # x = self.pool(x)
        # x_skip2 = self.downsample2(x)
        x = self.conv5(x )
        x = self.bn6(x)
        x = F.relu(x)
        x = self.pool(x)

        x_skip3 = self.downsample3(x)
        x = self.conv6(x + pad3d_to(x_skip2, x.size()))
        x = self.bn3(x)
        x = F.relu(x)
        # x = self.pool(x)

        x = self.conv7(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x + pad3d_to(x_skip3, x.size())
        # print(x_skip3.size())
        # print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


class USkipModel(nn.Module):
    """
    A sample PyTorch CNN model
    """

    def __init__(self, input_shape=(3, 64, 64), num_classes=10):
        super(USkipModel, self).__init__()
        self.bn_momentum = 0.2

        skip1_input = input_shape
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=20, kernel_size=(15, 15), padding=(7, 7),
                               dilation=(2, 2))

        # self.neg_conv1 = nn.Conv2d(in_channels=self.conv1.out_channels*2, out_channels=20,
        #                            kernel_size=self.conv1.kernel_size,
        #                            padding=self.conv1.padding*2,
        #                            padding_mode='reflect')
        dims, _ = calculate_dims(input_shape, self.conv1.out_channels, self.conv1.kernel_size, self.conv1.padding,
                                 self.conv1.stride[0], self.conv1.dilation)
        dims, _ = calculate_dims(dims, dims[0], (2, 2), (0, 0), 2)

        self.conv2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(9, 9), padding=(4, 4), dilation=(2, 2))
        dims, _ = calculate_dims(dims, 20, (9, 9), (4, 4), 1, (2, 2))
        skip2_input = dims
        skip1_output = dims

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=60, kernel_size=(5, 5), padding=(2, 2), dilation=(2, 2))
        dims, _ = calculate_dims(dims, 60, (5, 5), (2, 2), 1, (2, 2))
        dims, _ = calculate_dims(dims, dims[0], (2, 2), (0, 0), 2)



        self.conv5 = nn.Conv2d(in_channels=60, out_channels=20, kernel_size=(3, 3), padding=(1, 1))
        dims, _ = calculate_dims(dims, 20, (3, 3), (1, 1))
        dims, _ = calculate_dims(dims, dims[0], (2, 2), (0, 0), 2)
        skip3_input = dims
        skip2_output = dims

        self.conv6 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3, 3), padding=(1, 1))
        dims, _ = calculate_dims(dims, 20, (3, 3), (1, 1))

        self.conv7 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(3, 3), padding=(1, 1))
        dims, nodes = calculate_dims(dims, 10, (3, 3), (1, 1))
        dims, nodes = calculate_dims(dims, dims[0], (2, 2), (0, 0), 2)
        skip3_output = dims

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        downsample1_by = ceil(skip1_input[-1] / skip1_output[-1])
        self.downsample1 = nn.MaxPool2d(downsample1_by, stride=downsample1_by, ceil_mode=True)
        downsample2_by = ceil(skip2_input[-1] / skip2_output[-1])
        self.downsample2 = nn.MaxPool2d(downsample2_by, stride=downsample2_by, ceil_mode=True)
        downsample3_by = ceil(skip3_input[-1] / skip3_output[-1])
        self.downsample3 = nn.MaxPool2d(downsample3_by, stride=downsample3_by, ceil_mode=True)

        self.bn1 = nn.BatchNorm2d(20, momentum=self.bn_momentum)
        self.bn2 = nn.BatchNorm2d(60, momentum=self.bn_momentum)
        self.bn3 = nn.BatchNorm2d(20, momentum=self.bn_momentum)
        self.bn4 = nn.BatchNorm2d(20, momentum=self.bn_momentum)
        self.bn5 = nn.BatchNorm2d(10, momentum=self.bn_momentum)
        self.bn6 = nn.BatchNorm2d(20, momentum=self.bn_momentum)

        self.upscale1 = nn.ConvTranspose2d(in_channels=10, out_channels=10,
                                           stride=(2, 2), kernel_size=(2, 2))
        self.upscale2 = nn.ConvTranspose2d(in_channels=60, out_channels=60,
                                           stride=(2, 2), kernel_size=(2, 2))
        self.upscale3 = nn.ConvTranspose2d(in_channels=20, out_channels=20,
                                           stride=(2, 2), kernel_size=(2, 2))

        self.ref_conv1 = nn.Conv2d(in_channels=30, out_channels=60, kernel_size=(3, 3),
                                   padding=(2, 2), padding_mode='reflect')
        self.ref_conv2 = nn.Conv2d(in_channels=120, out_channels=60, kernel_size=(5, 5),
                                   padding=(4, 4), padding_mode='reflect')
        self.ref_conv3 = nn.Conv2d(in_channels=60, out_channels=20, kernel_size=(5, 5),
                                   padding=(4, 4), padding_mode='reflect')
        self.ref_conv4 = nn.Conv2d(in_channels=40, out_channels=20, kernel_size=(11, 11),
                                   padding=(9, 9), padding_mode='reflect')
        self.ref_conv5 = nn.Conv2d(in_channels=20, out_channels=3, kernel_size=(1, 1))

        self.ref_bn1 = nn.BatchNorm2d(60, momentum=self.bn_momentum)
        self.ref_bn2 = nn.BatchNorm2d(60, momentum=self.bn_momentum)
        self.ref_bn3 = nn.BatchNorm2d(20, momentum=self.bn_momentum)
        self.ref_bn4 = nn.BatchNorm2d(20, momentum=self.bn_momentum)

        # self.fc1 = nn.Linear(in_features=nodes, out_features=16)
        # self.dropout = nn.Dropout(0.2)
        # self.fc2 = nn.Linear(in_features=16, out_features=num_classes)

    def forward(self, x):
        x_skip1 = self.downsample1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)  # 62
        ff_x1 = x
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn4(x)
        # x = self.pool(x)
        x = F.relu(x)


        x_skip2 = self.downsample2(x)
        x = self.conv3(x + pad3d_to(x_skip1, x.size()))
        x = self.bn2(x)
        x = F.relu(x)
        ff_x2 = x
        x = self.pool(x)

        # x = self.conv4(x)
        # x = F.relu(x)
        # x = self.pool(x)
        # x_skip2 = self.downsample2(x)
        x = self.conv5(x)
        x = self.bn6(x)
        x = F.relu(x)
        ff_x3 = x
        x = self.pool(x)

        x_skip3 = self.downsample3(x)
        x = self.conv6(x + pad3d_to(x_skip2, x.size()))
        x = self.bn3(x)
        x = F.relu(x)
        # x = self.pool(x)

        x = self.conv7(x)
        x = self.bn5(x)
        x = F.relu(x)
        # x = self.pool(x)
        # x = x + pad3d_to(x_skip3, x.size())



        # x = x.view(x.size(0), -1)
        # x = self.fc1(x)
        # x = self.dropout(x)
        # x = F.relu(x)
        # x = self.fc2(x)

        x = self.upscale1(x)
        x = torch.cat([x, pad3d_to(ff_x3, x.size(), D=True)], dim=-3)
        x = self.ref_conv1(x)
        x = self.ref_bn1(x)
        x = F.relu(x)

        x = self.upscale2(x)
        x = torch.cat([x, pad3d_to(ff_x2, x.size(), D=True)], dim=-3)
        x = self.ref_conv2(x)
        x = self.ref_bn2(x)
        x = F.relu(x)
        x = self.ref_conv3(x)
        x = self.ref_bn3(x)
        x = F.relu(x)

        x = self.upscale3(x)
        x = torch.cat([x, pad3d_to(ff_x1, x.size(), D=True)], dim=-3)
        x = self.ref_conv4(x)
        x = self.ref_bn4(x)
        x = F.relu(x)
        x = self.ref_conv5(x)

        return x


class TransferSkipModel(nn.Module):
    """
    A sample PyTorch CNN model
    """

    def __init__(self, load_model_str, input_shape=(10, 12, 12), num_classes=10):
        super(TransferSkipModel, self).__init__()
        umodel = USkipModel(input_shape=input_shape)
        umodel.load_state_dict(torch.load(load_model_str, map_location=torch.device(device)))

        self.bn_momentum = 0.2

        skip1_input = input_shape
        self.conv1 = umodel.conv1
        dims, _ = calculate_dims(input_shape, 20, (15, 15), (7, 7), 1, (2, 2))
        dims, _ = calculate_dims(dims, dims[0], (2, 2), (0, 0), 2)

        self.conv2 = umodel.conv2
        dims, _ = calculate_dims(dims, 20, (9, 9), (4, 4), 1, (2, 2))
        skip2_input = dims
        skip1_output = dims

        self.conv3 = umodel.conv3
        dims, _ = calculate_dims(dims, 60, (5, 5), (2, 2), 1, (2, 2))
        dims, _ = calculate_dims(dims, dims[0], (2, 2), (0, 0), 2)


        self.conv5 = umodel.conv5
        dims, _ = calculate_dims(dims, 20, (3, 3), (1, 1))
        dims, _ = calculate_dims(dims, dims[0], (2, 2), (0, 0), 2)
        skip3_input = dims
        skip2_output = dims

        self.conv6 = umodel.conv6
        dims, _ = calculate_dims(dims, 20, (3, 3), (1, 1))

        self.conv7 = umodel.conv7
        dims, nodes = calculate_dims(dims, 10, (3, 3), (1, 1))
        dims, nodes = calculate_dims(dims, dims[0], (2, 2), (0, 0), 2)
        skip3_output = dims

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        downsample1_by = ceil(skip1_input[-1] / skip1_output[-1])
        self.downsample1 = umodel.downsample1
        downsample2_by = ceil(skip2_input[-1] / skip2_output[-1])
        self.downsample2 = umodel.downsample2
        downsample3_by = ceil(skip3_input[-1] / skip3_output[-1])
        self.downsample3 = umodel.downsample3

        self.bn1 = nn.BatchNorm2d(20, momentum=self.bn_momentum)
        self.bn2 = nn.BatchNorm2d(60, momentum=self.bn_momentum)
        self.bn3 = nn.BatchNorm2d(20, momentum=self.bn_momentum)
        self.bn4 = nn.BatchNorm2d(20, momentum=self.bn_momentum)
        self.bn5 = nn.BatchNorm2d(10, momentum=self.bn_momentum)
        self.bn6 = nn.BatchNorm2d(20, momentum=self.bn_momentum)
        # The input features for the linear layer depends on the size of the input to the convolutional layer
        # So if you resize your image in data augmentations, you'll have to tweak this too.

        # dims, _ = calculate_dims(dims, 60, (3, 3), (1, 1))

        # dims, _ = calculate_dims(dims, 20, (3, 3), (1, 1))
        # dims, nodes = calculate_dims(dims, dims[0], (2, 2), (0, 0), 2)
        # dims, nodes = calculate_dims(dims, 10, (3, 3), (1, 1))
        self.fc1 = nn.Linear(in_features=nodes, out_features=16)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_features=16, out_features=num_classes)

    def forward(self, x):
        x_skip1 = self.downsample1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)  # 62
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn4(x)
        # x = self.pool(x)
        x = F.relu(x)

        x_skip2 = self.downsample2(x)
        x = self.conv3(x + pad3d_to(x_skip1, x.size()))
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # x = self.conv4(x)
        # x = F.relu(x)
        # x = self.pool(x)
        # x_skip2 = self.downsample2(x)
        x = self.conv5(x )
        x = self.bn6(x)
        x = F.relu(x)
        x = self.pool(x)

        x_skip3 = self.downsample3(x)
        x = self.conv6(x + pad3d_to(x_skip2, x.size()))
        x = self.bn3(x)
        x = F.relu(x)
        # x = self.pool(x)

        x = self.conv7(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x + pad3d_to(x_skip3, x.size())

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x




def get_model(input_shape, num_classes, config):
    block = {'layer1':
                 {'conv':
                     {'filter_n': config['filter1_n'],
                      'kernel': config['kernel1_size'],
                      'dilation': config['dilation1']},
                  'bn': config['bn1'],
                  'pool': config['pool1']},
             'layer2':
                 {'conv':
                     {'filter_n': config['filter2_n'],
                      'kernel': config['kernel2_size'],
                      'dilation': config['dilation2']},
                  'bn': config['bn2'],
                  'pool': config['pool2']}}


# BLock:
#        skip ->'
#        conv (filter_size(custom choice), dilation, ///calc_padding)
#        bn or not
#        relu
#        pool or not
#        ++skip
class BlockModel(nn.Module):
    def __init__(self, input_shape, block_config):
        super(BlockModel, self).__init__()
        self.bn_momentum = 0.2
        self.skip_layers = 1
        self.pool1 = block_config['pool1']
        self.n_params = 0
        self.output_shape = None

        out_1 = block_config['filter1_n']
        kern_1 = (block_config['kernel1_size'], block_config['kernel1_size'])
        pad_1 = (int(kern_1[0]/2), int(kern_1[0]/2))
        pad_1 = (0, 0)
        dilation_1 = (block_config['dilation1'], block_config['dilation1'])

        self.n_params += out_1*(1 + kern_1[0]**2)
        skip1_input = input_shape[-1]
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=out_1, kernel_size=kern_1,
                               padding=pad_1, dilation=dilation_1)
        dims, nodes = calculate_dims(input_shape, out_1, kern_1, pad_1, 1, dilation_1)

        if self.pool1:
            dims, nodes = calculate_dims(dims, dims[0], (2, 2), (0, 0), 2)

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.bn1 = None
        if block_config['bn1']:
            self.bn1 = nn.BatchNorm2d(out_1, momentum=self.bn_momentum)
            self.n_params += out_1*2

        self.output_shape = tuple(dims)



        # if block_config['skip_layer'] == 1:
        skip1_output = dims[1]
            # skip2_input = dims[1]

        # if block_config['skip_layers'] == 1:
        downsample1_by = ceil(skip1_input / skip1_output)
        self.downsample1 = nn.MaxPool2d(downsample1_by, stride=downsample1_by, ceil_mode=True)

        self.conv2 = None
        if block_config['n_layers'] == 2:
            self.pool2 = block_config['pool2']

            out_2 = block_config['filter2_n']
            kern_2 = (block_config['kernel2_size'], block_config['kernel2_size'])
            pad_2 = (int(kern_2[0]/2), int(kern_2[0]/2))
            pad_2 = (0, 0)
            dilation_2 = (block_config['dilation2'], block_config['dilation2'])

            self.n_params += out_2*(1 + kern_2[0]**2)
            skip2_input = dims[-1]
            self.conv2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=kern_2,
                                   padding=pad_2, dilation=dilation_2)
            dims, nodes = calculate_dims(dims, out_2, kern_2, pad_2, 1, dilation_2)
            if self.pool2:
                dims, nodes = calculate_dims(dims, dims[0], (2, 2), (0, 0), 2)

            self.bn1 = None
            if block_config['bn2']:
                self.bn2 = nn.BatchNorm2d(out_2, momentum=self.bn_momentum)
                self.n_params += out_2*2

            self.output_shape = tuple(dims)
            skip2_output = dims[1]

            if block_config['skip_layers'] == 2:
                self.skip_layers = block_config['skip_layers']
                downsample1_by = ceil(skip1_input / skip2_output)
                self.downsample1 = nn.MaxPool2d(downsample1_by, stride=downsample1_by, ceil_mode=True)
            else:
                downsample1_by = ceil(skip1_input / skip1_output)
                self.downsample1 = nn.MaxPool2d(downsample1_by, stride=downsample1_by, ceil_mode=True)
                downsample2_by = ceil(skip2_input / skip2_output)
                self.downsample2 = nn.MaxPool2d(downsample2_by, stride=downsample2_by, ceil_mode=True)

        print(downsample1_by)


        # fc1_out = 16
        # self.interim_block = self.n_params + nodes*fc1_out + fc1_out*num_classes < 50000
        # # self.final_block = self.n_params + nodes*fc1_out + fc1_out*num_classes < 100000
        # if self.interim_block:
        #     self.fc1 = nn.Linear(in_features=nodes, out_features=fc1_out)
        #     self.dropout = nn.Dropout(0.2)
        #
        # elif self.n_params + nodes*fc1_out + fc1_out*num_classes < 100000:
        #     self.fc1 = nn.Linear(in_features=nodes, out_features=fc1_out)
        #     self.dropout = nn.Dropout(0.2)
        #     self.fc2 = nn.Linear(in_features=fc1_out, out_features=num_classes)

    # def waste_of_time(self):

    def forward(self, x):
        x_skip1 = self.downsample1(x)
        x = self.conv1(x)
        if self.bn1:
            x = self.bn1(x)
        x = F.relu(x)
        if self.pool1:
            x = self.pool(x)

        if self.conv2:
            x_skip2 = 0
            if self.skip_layers == 1:
                x_skip2 = self.downsample2(x)
                x = x + pad3d_to(x_skip1, x.size())
            x = self.conv2(x)
            if self.bn1:
                x = self.bn1(x)
            x = F.relu(x)
            if self.pool1:
                x = self.pool(x)
            if self.skip_layers == 1:
                x = x + pad3d_to(x_skip2, x.size())
            else:
                x = x + pad3d_to(x_skip1, x.size())

        return x


class FunNet(nn.Module):
    def __init__(self, input_shape, num_classes, block_config):
        super(FunNet, self).__init__()

        fc1_nodes = block_config['fc1_nodes']
        self.blocks = nn.ModuleList([BlockModel(input_shape, block_config)])
        net_params = self.blocks[0].n_params
        out_shape = self.blocks[0].output_shape
        out_nodes = out_shape[-3]*out_shape[-2]*out_shape[-1]

        self.n_blocks = int((1000000 - fc1_nodes*num_classes - out_nodes*fc1_nodes)/net_params)
        print(out_shape, self.n_blocks)
        for i in range(5):
            in_shape = self.blocks[-1].output_shape
            block = BlockModel(in_shape, block_config)
            if min(block.output_shape) > 0:
                self.blocks.append(block)
        out_shape = self.blocks[-1].output_shape
        out_nodes = out_shape[-3]*out_shape[-2]*out_shape[-1]
        print(self.blocks)
        print(self.blocks[-1].output_shape)
        print(out_nodes, fc1_nodes)
        self.fc1 = nn.Linear(in_features=out_nodes, out_features=fc1_nodes)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_features=fc1_nodes, out_features=num_classes)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x





def pad3d_to(x, dim_out, D=False):
    i_padding = int((dim_out[-1] - x.size()[-1]) / 2)
    r = (dim_out[-1] - x.size()[-1]) % 2

    j_padding = int((dim_out[-3] - x.size()[-3]) / 2)
    r_2 = (dim_out[-3] - x.size()[-3]) % 2
    if D:
        r_2, j_padding = 0, 0
    padding = (i_padding + r, i_padding, i_padding + r, i_padding, j_padding + r_2, j_padding)
    # print(x.size(), dim_out , padding)
    return F.pad(x, padding)


def calculate_dims(input_size, filter_n, kernel_size, padding=(0, 0), stride=1, dilation=(1, 1)):
    out_dims = [filter_n, None, None]
    out_dims[1] = int(ceil(1 + (input_size[1] - dilation[0] * (kernel_size[0] - 1) - 1 + 2 * padding[0]) / stride))
    out_dims[2] = int(ceil(1 + (input_size[2] - dilation[1] * (kernel_size[1] - 1) - 1 + 2 * padding[1]) / stride))
    out_nodes = out_dims[1] * out_dims[2] * filter_n

    return out_dims, out_nodes
