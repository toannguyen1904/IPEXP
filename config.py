CUDA_VISIBLE_DEVICES = '0'  # The current version only supports one GPU training

batch_size = 6
vis_num = batch_size  # The number of visualized images in tensorboard
save_interval = 1
max_checkpoint_num = 60
end_epoch = 60
init_lr = 0.001
lr_milestones = [10, 30]
lr_decay_rate = 0.1
weight_decay = 1e-4
channels = 2048
input_size = 224

root = '/content/gdrive/MyDrive/IP'
model_path = '/content/gdrive/MyDrive/IP/checkpoint'
model_name = ''