#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Zhiquan Wang'
__maintainer__ = 'Zhiquan Wang'
__email__ = 'wang4490@purdue.edu'
__status__ = 'development'
__laboratory__ = 'vrlab-purdue'
__date__ = '2020/09/02-10:31 AM'

import torch.nn as nn
import torch.utils.data.dataset
import torchvision
from torchvision import transforms, utils
import os
import matplotlib
from PIL import Image

import matplotlib.pyplot as plt
import random

import pandas as pd

import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import utils
import time
import warnings
warnings.filterwarnings('ignore')


class FacialEmotionDataset(torch.utils.data.Dataset):
    def __init__(self, *, image_path: os.path, labels_path: os.path, type: str = 'train', transform=None):
        warnings.filterwarnings('ignore')
        if not os.path.isdir(image_path):
            raise ValueError("image file path is not a dir")
        if not os.path.isdir(labels_path):
            raise ValueError("image file path is not a dir")
        self.type = type
        self.image_path = image_path
        self.labels_path = labels_path
        self.transform = transform
        self.distribution = np.array([])
        self.weights = np.array([])
        self.images, self.labels = self._init_dataset()
        self.class_num = 0
        self._init_dataset_info()

    def _init_dataset(self):
        warnings.filterwarnings('ignore')
        images, labels = [], []
        label_path = os.path.join(self.labels_path, self.type + '.txt')
        with open(label_path, 'r') as f:
            for line in f:
                image_name, label = line.split(' ')
                images.append(image_name)
                labels.append(int(label))
        return images, labels

    def _init_dataset_info(self):
        warnings.filterwarnings('ignore')
        self.class_num = len(set(self.labels))
        self.distribution = np.array(
            [self.labels.count(i) for i in range(self.class_num)])
        
        print(self.distribution)
        self.weights = torch.tensor(
            [1.0 / self.distribution[i] for i in self.labels])

        # def __balance_dataset(self, class_num: int):
        #     classes_number = []
        #     for i in range(class_num):
        #         classes_number.append(self.labels.count(i))
        #     min_class_num = min(classes_number)
        #     labels_indices = []
        #     for i in range(class_num):
        #         labels_indices.append(
        #             [index for index, value in enumerate(self.labels) if value == i])
        #     for i in range(class_num):
        #         print(len(labels_indices[i]))

    def __len__(self):
        warnings.filterwarnings('ignore')
        return len(self.images)

    def __getitem__(self, idx):
        warnings.filterwarnings('ignore')
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.image_path, self.images[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

    def show_images(self, *, num: int = 1, rand: bool = True):
        warnings.filterwarnings('ignore')
        fig = plt.figure()
        image_idx = random.sample(range(0, self.__len__()), num) if rand else range(min(num, self.__len__()))
        size = int(np.ceil(np.sqrt(num)))
        image_counter = 0
        for i in image_idx:
            sample, _ = self[i]
            sample = np.transpose(sample, (1, 2, 0))
            ax = plt.subplot(size, size, image_counter + 1)
            image_counter += 1
            plt.tight_layout()
            ax.set_title('#{} : {}'.format(i, self.labels[i]))
            ax.axis('off')
            plt.imshow(sample)
        plt.show()
        return fig


class NNManager():
    def __init__(self, train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader, model: nn.Module, optimizer: torch.optim, record_board: SummaryWriter, enable_gpu=False, dtype=torch.float32):
        warnings.filterwarnings('ignore')
        self.trainloader = train_dataloader
        self.testloader = test_dataloader
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(
            'cuda') if enable_gpu else torch.device('cpu')
        self.record_board = record_board
        self.dtype = dtype

    def show_image(self, num: int = 1, rand: bool = False, idx: int = None, on_board: bool = False):
        warnings.filterwarnings('ignore')
        # if on_board:
        #     images, labels = next(iter(self.dataloader))
        #     image_idx = random.sample(range(0, len(images)), num) if rand else range(
        #         min(num, len(images))) if idx is None else [idx]
        #     images = images[image_idx]
        #     size = int(np.ceil(np.sqrt(num)))
        #     grid = torchvision.utils.make_grid(images, nrow=size)
        #     self.record_board.add_image('tmp', img_tensor=grid)
        # else:
        #     self.dataloader.dataset.show_images(num=num,rand=rand)

        fig = self.trainloader.dataset.show_images(num=num, rand=rand)
        if on_board:
            self.record_board.add_figure('tmp', fig)

    def test_check(self,e:int):
        warnings.filterwarnings('ignore')
        total_correct = 0
        total_loss = 0
        num_samples = 0
        confusion_matrix = np.zeros((self.testloader.dataset.class_num, self.testloader.dataset.class_num))
        self.model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for t, (images, labels) in enumerate(self.testloader):
                # move to device, e.g. GPU
                images = images.to(device=self.device, dtype=self.dtype)
                labels = labels.to(device=self.device, dtype=torch.long)
                scores = self.model(images)
                _, preds = scores.max(1)
                total_correct += (preds == labels).sum()
                num_samples += preds.size(0)
                confusion_matrix += utils.confusion_mat(preds, labels, self.testloader.dataset.class_num)
            acc = float(total_correct) / num_samples
        print('>> on test set')
        # print('>> loss = {l:.6f}'.format(l=total_loss))
        print('>> correct: {cn}/{tt}: {p} %'.format(cn=total_correct, tt=len(self.testloader.dataset), p=total_correct / len(self.testloader.dataset) * 100))
        self.record_board.add_figure('t confusion matrix on test {e}'.format(e=e), utils.plot_confusion_matrix(cm=confusion_matrix, classes=['0', '1', '2','3']))
        self.record_board.add_scalar('t loss', total_loss, e)
        self.record_board.add_scalar('t correct_num', total_correct, e)
        self.record_board.add_scalar('t accuracy', acc, e)

    def train(self, epochs: int, auto_save_interval: int = 1,iter_output_inverval:int =20):
        warnings.filterwarnings('ignore')
        start_training_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())
        loss_function = nn.CrossEntropyLoss()
        model = self.model.to(device=self.device)
        epoch_counter = 0
        confusion_matrix = np.zeros((self.trainloader.dataset.class_num, self.trainloader.dataset.class_num))
        max_correct_num = 0
        while True:
            total_loss = 0
            total_correct = 0
            total_num = 0
            for t, (images, labels) in enumerate(self.trainloader):
                total_num += len(images)
                model.train()  # put model to training mode
                images = images.to(device=self.device, dtype=self.dtype)
                labels = labels.to(device=self.device, dtype=torch.long)

                scores = model(images)
                preds = torch.argmax(scores, axis=1)
                c_num = scores.argmax(dim=1).eq(labels).sum().item()
                loss = loss_function(scores, labels)
                total_correct += c_num
                total_loss += loss
                confusion_matrix += utils.confusion_mat(preds, labels, self.trainloader.dataset.class_num)
                # Zero out all of the gradients for the variables which the optimizer
                # will update.
                self.optimizer.zero_grad()
                # This is the backwards pass: compute the gradient of the loss with
                # respect to each  parameter of the model.
                loss.backward()

                # Actually update the parameters of the model using the gradients
                # computed by the backwards pass.
                self.optimizer.step()
                if t %iter_output_inverval==0:
                    print('Epoch: {e} - Iteration: {i}, loss = {l:.6f}'.format(e=epoch_counter, i=t, l=loss.item()))
                    print('correct: {cn}/{tt}: {p:.2f} %'.format(cn=c_num, tt=len(labels), p=c_num / len(labels) * 100))
            print('#####################== iteration: {e} ==#####################'.format(e=epoch_counter))
            print('loss = {l:.4f}'.format(l=total_loss))
            print('correct: {cn}/{tt}: {p:.4f} %'.format(cn=total_correct, tt=len(self.trainloader.dataset), p=total_correct / len(self.trainloader.dataset) * 100))
            epoch_counter += 1
            self.record_board.add_figure('confusion matrix on train {e}'.format(e=epoch_counter), utils.plot_confusion_matrix(cm=confusion_matrix, classes=['0', '1', '2','3']))
            self.record_board.add_scalar('loss', total_loss, epoch_counter)
            self.record_board.add_scalar('correct_num', total_correct, epoch_counter)
            self.record_board.add_scalar('accuracy', total_correct / len(self.trainloader.dataset)*100.0, epoch_counter)
            # test
            if total_correct > max_correct_num:
                max_correct_num = total_correct
                save_path = './saved_video_models/{e}_{a}_{t}'.format(e=epoch_counter, a=int(total_correct / len(self.trainloader.dataset) * 100), t=start_training_time)
                torch.save(self.model,save_path)
            # df = pd.read_csv("saved_video_data\\videoDataPlot.csv")
            # df_temp = pd.DataFrame(columns = df.columns)
            # df_temp["Epoch"] = epoch_counter
            self.test_check(e=epoch_counter)
            if epoch_counter == epochs:
                break

    def test(self):
        warnings.filterwarnings('ignore')
        correct = 0
        total = 0
        self.model.eval()  # set model to evaluation mode

        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))
    def __del__(self):
        warnings.filterwarnings('ignore')
        self.record_board.close()


class vggs_net(nn.Module):
    def __init__(self):
        warnings.filterwarnings('ignore')
        super(vggs_net, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=96, kernel_size=7, stride=2, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=True)
        self.conv2 = nn.Conv2d(
            in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, ceil_mode=True)
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(6 * 6 * 512, 4096, bias=True)
        self.fc2 = nn.Linear(4096, 4096, bias=True)
        self.fc3 = nn.Linear(4096, 4, bias=True)
        self.dropout_layer = nn.Dropout(p=0.2)

    def forward(self, x):
        warnings.filterwarnings('ignore')
        layer0_x = self.pool1(nn.functional.relu(self.conv1(x)))
        layer1_x = self.pool2(nn.functional.relu(self.conv2(layer0_x)))
        layer2_x = nn.functional.relu(self.conv3(layer1_x))
        layer3_x = nn.functional.relu(self.conv4(layer2_x))
        layer4_x = self.pool1(nn.functional.relu(self.conv4(layer3_x)))

        layer4_x = layer4_x.view(-1, 6 * 6 * 512)
        layer5_x = nn.functional.relu(self.dropout_layer(self.fc1(layer4_x)))
        layer6_x = nn.functional.relu(self.dropout_layer(self.fc2(layer5_x)))
        layer7_x = self.dropout_layer(self.fc3(layer6_x))
        return layer7_x

    @torch.no_grad()
    def get_all_preds(model, loader):
        warnings.filterwarnings('ignore')
        all_preds = torch.tensor([])
        for batch in loader:
            images, labels = batch
            preds = model(images)
            all_preds = torch.cat(
                (all_preds, preds), dim=0
            )
        return all_preds
