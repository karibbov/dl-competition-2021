""" File with CNN models. Add your custom CNN model here. """

import torch.nn as nn
import torch.nn.functional as F


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
        x = F.relu(x) # 62
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
        x = F.relu(x) # 56
        x = self.conv2(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.sm(x) # 55

        return x


class DeepModel(nn.Module):
    """
    A sample PyTorch CNN model
    """
    def __init__(self, input_shape=(3, 64, 64), num_classes=10):
        super(DeepModel, self).__init__()
        self.bn_momentum = 0.2
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=20, kernel_size=(15, 15), padding=(7, 7), dilation=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(9, 9), padding=(4, 4), dilation=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=60, kernel_size=(5, 5), padding=(2, 2), dilation=(2, 2))
        # self.conv4 = nn.Conv2d(in_channels=40, out_channels=60, kernel_size=(3, 3), padding=(1, 1))
        self.conv5 = nn.Conv2d(in_channels=60, out_channels=20, kernel_size=(3, 3), padding=(1, 1))
        self.conv6 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3, 3), padding=(1, 1))
        self.conv7 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(3, 3), padding=(1, 1))
        self.pool = nn.MaxPool2d(2, stride=2)
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
        x = F.relu(x) # 62
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


def calculate_dims(input_size, filter_n, kernel_size, padding=(0, 0), stride=1, dilation=(1, 1)):
    out_dims = [filter_n, None, None]
    out_dims[1] = int(1 + (input_size[1] - dilation[0]*(kernel_size[0] - 1) - 1 + 2*padding[0])/stride)
    out_dims[2] = int(1 + (input_size[2] - dilation[1]*(kernel_size[1] - 1) - 1 + 2*padding[1])/stride)
    out_nodes = out_dims[1]*out_dims[2]*filter_n

    return out_dims, out_nodes
