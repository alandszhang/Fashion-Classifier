import os
import shutil

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as fn

import torch.nn as nn
import torch.nn.functional as F

from torch.utils import data
from torch.utils.data import DataLoader

from pandas import Series, DataFrame

import pandas as pd
import numpy as np

from PIL import Image

import torch.optim as optim

import matplotlib.pyplot as plt

from collections.abc import Iterable

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        # self.set_freeze_by_names(layer_names=('conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2'))
        self.resnet.fc = nn.Linear(in_features=self.resnet.fc.in_features, out_features=7, bias=True)

    def forward(self, x):
        x = x.to(device)
        x = self.resnet(x)
        return x

    def set_freeze_by_names(self, layer_names, freeze=True):
        if not isinstance(layer_names, Iterable):
            layer_names = [layer_names]
        for name, child in self.resnet.named_children():
            if name not in layer_names:
                continue
            for param in child.parameters():
                param.requires_grad = not freeze

    def set_freeze_by_idxs(self, idxs, freeze=True):
        if not isinstance(idxs, Iterable):
            idxs = [idxs]
        num_child = len(list(self.resnet.children()))
        idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
        for idx, child in enumerate(self.resnet.children()):
            if idx not in idxs:
                continue
            for param in child.parameters():
                param.requires_grad = not freeze


class DeepFashionDataset(data.Dataset):
    def __init__(self, dataset_path, labels_path, transforms_=None):
        list_attr_cloth = pd.read_table("/media/dszhang/DATA/FashionDataset/split/list_attr_cloth.txt", sep='\s+')

        dataset = pd.read_table(dataset_path).values.flatten().tolist()

        self.imgs = [os.path.join("/media/dszhang/DATA/FashionDataset", img) for img in dataset]

        self.attr = pd.read_table(labels_path).values.flatten()

        self.attr_np = np.zeros(shape=(len(self.attr), 6), dtype=np.int)

        self.category = 1

        self.get_labels()

        self.transform = transforms.Compose(transforms_)

    def get_labels(self):
        row = 0
        for line in self.attr:
            self.attr_np[row] = np.array(self.attr[row].rstrip().split())
            row += 1

    def __getitem__(self, index):
        img = self.transform(Image.open(self.imgs[index]))
        # img = fn.fn/
        label = self.attr_np[index][self.category - 1]

        return img, label

    def __len__(self):
        return len(self.attr)


def main():
    train_transforms = [
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    trainset = DeepFashionDataset(dataset_path="/media/dszhang/DATA/FashionDataset/split/train.txt",
                                  labels_path="/media/dszhang/DATA/FashionDataset/split/train_attr.txt",
                                  transforms_=train_transforms)

    trainloader = DataLoader(dataset=trainset, batch_size=100, shuffle=False)

    valset = DeepFashionDataset(dataset_path="/media/dszhang/DATA/FashionDataset/split/val.txt",
                                labels_path="/media/dszhang/DATA/FashionDataset/split/val_attr.txt",
                                transforms_=train_transforms)

    valloader = DataLoader(dataset=valset, batch_size=1, shuffle=False)

    net = Net().to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    best_acc = 0.0
    # sigmoid_fun = nn.Sigmoid()
    # softmax_fun = nn.Softmax(dim=0)

    for epoch in range(20):
        net.train()
        running_loss = 0.0
        for cnt, data in enumerate(trainloader, 0):
            inputs, label = data
            outputs = net(inputs)
            label = label.to(device)
            loss = criterion(outputs, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # if cnt % 1000 == 999:
            #     print('[epoch %d, cnt %d] training loss: %.3f' % (epoch + 1, cnt + 1, running_loss / 1000.))
            #     running_loss = 0.0

        print('[epoch %d] training loss: %.3f' % (epoch + 1, running_loss / 5000.))

        correct = 0
        running_loss = 0.0
        net.eval()
        is_best = False
        for cnt, data in enumerate(valloader, 0):
            inputs, label = data
            outputs = net(inputs)
            label = label.to(device)
            loss = criterion(outputs, label)
            running_loss += loss.item()

            _, pred = outputs.max(1)

            correct += pred.eq(label).sum().item()

        print('[%d] val_loss: %.3f, acc: %.3f' % (epoch + 1, running_loss / 1000., correct / 10.))

        acc = correct / 10.
        if best_acc < acc:
            best_acc = acc
            is_best = True

        state = {
            'epoch': epoch + 1,
            'arch': 'resnet50',
            'state_dict': net.state_dict(),
            'best_acc1': best_acc,
            'optimizer': optimizer.state_dict()
        }

        torch.save(state, 'cate_1_checkpoint.pth.tar')
        if is_best:
            shutil.copyfile('cate_1_checkpoint.pth.tar', 'cate_1_best_model.pth.tar')
            print('best model saved @ epoch %d, acc: %.3f' % (epoch + 1, acc))

    print('Finished Training')


if __name__ == '__main__':
    main()
