#coding=utf-8
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
import shutil
import time
from utils.read_dataset import read_dataset
from utils.train_model import train
from config import CUDA_VISIBLE_DEVICES, input_size, batch_size, root, channels, model_path, model_name, init_lr, lr_decay_rate,\
    lr_milestones, weight_decay, end_epoch, save_interval
from utils.auto_load_resume import auto_load_resume
import os
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="select the dataset to use")
args = vars(ap.parse_args())

dataset = args["dataset"]

model = torchvision.models.resnet50(pretrained=False)

os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

def main():

    # Read the dataset
    trainloader, valloader, testloader = read_dataset(dataset, input_size, batch_size, root)
    if dataset == 'meta':
        num_classes = 40
    elif dataset == 'non_meta':
        num_classes = 62
    elif dataset == 'all':
        num_classes = 102
    else:
        num_classes = 2

    # ResNet50
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(
            in_features=channels,
            out_features=num_classes,
        ),
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    parameters = model.parameters()

    # Checkpoint
    save_path = os.path.join(model_path, dataset, model_name)
    if os.path.exists(save_path):
        start_epoch, lr = auto_load_resume(model, save_path, status='train')
        assert start_epoch < end_epoch
    else:
        os.makedirs(save_path)
        start_epoch = 0
        lr = init_lr

    # define optimizers
    optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)

    model = model.cuda()  # Use GPU

    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_decay_rate)


    time_str = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy('./config.py', os.path.join(save_path, "{}config.py".format(time_str)))

    # Train the model
    train(model=model,
          trainloader=trainloader,
          valloader=valloader,
          testloader=testloader,
          criterion=criterion,
          optimizer=optimizer,
          scheduler=scheduler,
          save_path=save_path,
          start_epoch=start_epoch,
          end_epoch=end_epoch,
          save_interval=save_interval)


if __name__ == '__main__':
    main()