import torch
from tqdm import tqdm

def eval(model, loader, criterion):
    model.eval()
    print('Evaluating')

    loss_sum = 0
    correct = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()

            logits = model(images)

            loss = criterion(logits, labels)


            loss_sum += loss.item()

            pred = logits.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()


    loss_avg = loss_sum / (i+1)

    accuracy = correct / len(loader.dataset)

    return loss_avg, accuracy