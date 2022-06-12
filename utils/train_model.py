import os
import glob
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter
from config import max_checkpoint_num
from utils.eval_model import eval


def train(model,
          trainloader,
          valloader,
          testloader,
          criterion,
          optimizer,
          scheduler,
          save_path,
          start_epoch,
          end_epoch,
          save_interval):
    for epoch in range(start_epoch + 1, end_epoch + 1):
        model.train()

        print('Training %d epoch' % epoch)

        lr = next(iter(optimizer.param_groups))['lr']

        for i, data in enumerate(tqdm(trainloader)):
            images, labels = data
            images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()

            logits = model(images)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # eval valset
        val_loss_avg, val_accuracy = eval(model,  valloader, criterion)
        print('Validation set: accuracy: {:.2f}%'.format(100. * val_accuracy))

        # eval testset
        test_loss_avg, test_accuracy = eval(model, testloader, criterion)
        print('Test set: accuracy: {:.2f}%'.format(100. * test_accuracy))


        # save checkpoint
        if (epoch % save_interval == 0) or (epoch == end_epoch):
            print('Saving checkpoint')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'learning_rate': lr,
            }, os.path.join(save_path, 'epoch' + str(epoch) + '.pth'))

        # Limit the number of checkpoints to less than or equal to max_checkpoint_num,
        # and delete the redundant ones
        checkpoint_list = [os.path.basename(path) for path in glob.glob(os.path.join(save_path, '*.pth'))]
        if len(checkpoint_list) == max_checkpoint_num + 1:
            idx_list = [int(name.replace('epoch', '').replace('.pth', '')) for name in checkpoint_list]
            min_idx = min(idx_list)
            os.remove(os.path.join(save_path, 'epoch' + str(min_idx) + '.pth'))

