import torch
from dataset import pre_data

def read_dataset(dataset, input_size, batch_size, root):

    print('Loading IP {} training set'.format(dataset))
    trainset = pre_data.Meta_Dataset(dataset, input_size=input_size, root=root, mode='train')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=8, drop_last=False)

    print('Loading IP {} validation set'.format(dataset))
    valset = pre_data.Meta_Dataset(dataset, input_size=input_size, root=root, mode='val')
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                  shuffle=True, num_workers=8, drop_last=False)

    print('Loading IP {} testing set'.format(dataset))
    testset = pre_data.Meta_Dataset(dataset, input_size=input_size, root=root, mode='test')
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=8, drop_last=False)

    return trainloader, valloader, testloader