import numpy as np
import imageio
import os
from PIL import Image
from torchvision import transforms

meta_classes = [0, 1, 2, 3, 4, 5, 6, 10, 13, 16, 17, 18, 19, 20, 22, 23, 25, 26, 30, 34, 35, 36, 38, 39, 40, 41,\
                44, 45, 58, 65, 66, 73, 83, 84, 85, 86, 87, 88, 93, 97]
non_meta_classes = [7, 8, 9, 11, 12, 14, 15, 21, 24, 27, 28, 29, 31, 32, 33, 37, 42, 43, 46, 47, 48, 49, 50, 51, 52,\
                    53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 64, 67, 68, 69, 70, 71, 72, 74, 75, 76, 77, 78, 79, 80, 81,\
                    82, 89, 90, 91, 92, 94, 95, 96, 98, 99, 100, 101]

meta_map = dict(zip(meta_classes, range(len(meta_classes))))
non_meta_map = dict(zip(non_meta_classes, range(len(non_meta_classes))))
all_map = dict(zip(range(102), range(102)))
two_classes_map = dict(zip(non_meta_classes + meta_classes, [0] * len(non_meta_classes) + [1] * len(meta_classes)))


class Meta_Dataset():
    def __init__(self, dataset, input_size, root, mode='train', data_len=None):
        self.input_size = input_size
        self.root = root
        self.mode = mode
        train_img_path = '/content/images'
        val_img_path = '/content/images'
        test_img_path = '/content/images'
        used_map = two_classes_map
        if dataset == 'meta':
            train_label_file = open(os.path.join(self.root, 'dataset', 'ip102_v1.1', 'meta_train.txt'))
            val_label_file = open(os.path.join(self.root, 'dataset', 'ip102_v1.1', 'meta_val.txt'))
            test_label_file = open(os.path.join(self.root, 'dataset', 'ip102_v1.1', 'meta_test.txt'))
            used_map = meta_map
        elif dataset == 'non_meta':
            train_label_file = open(os.path.join(self.root, 'dataset', 'ip102_v1.1', 'non_meta_train.txt'))
            val_label_file = open(os.path.join(self.root, 'dataset', 'ip102_v1.1', 'non_meta_val.txt'))
            test_label_file = open(os.path.join(self.root, 'dataset', 'ip102_v1.1', 'non_meta_test.txt'))
            used_map = non_meta_map
        else:
            train_label_file = open(os.path.join(self.root, 'dataset', 'ip102_v1.1', 'train.txt'))
            val_label_file = open(os.path.join(self.root, 'dataset', 'ip102_v1.1', 'val.txt'))
            test_label_file = open(os.path.join(self.root, 'dataset', 'ip102_v1.1', 'test.txt'))
        if dataset == 'all':
            used_map = all_map

        train_img_label = []
        val_img_label = []
        test_img_label = []
        for line in train_label_file:
            train_img_label.append([os.path.join(train_img_path,line[:-1].split(' ')[0]), used_map.get(int(line[:-1].split(' ')[1]))])
        for line in val_label_file:
            val_img_label.append([os.path.join(val_img_path,line[:-1].split(' ')[0]), used_map.get(int(line[:-1].split(' ')[1]))])
        for line in test_label_file:
            test_img_label.append([os.path.join(test_img_path,line[:-1].split(' ')[0]), used_map.get(int(line[:-1].split(' ')[1]))])

        self.train_img_label = train_img_label
        self.val_img_label = val_img_label
        self.test_img_label = test_img_label

    def __getitem__(self, index):
        if self.mode == 'train':
            img, target = imageio.imread(self.train_img_label[index][0]), self.train_img_label[index][1]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            """
            img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
            # img = transforms.RandomResizedCrop(size=self.input_size,scale=(0.4, 0.75),ratio=(0.5,1.5))(img)
            # img = transforms.RandomCrop(self.input_size)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)
            """
            
            img = transforms.Resize((256, 256))(img)
            img = transforms.RandomCrop(self.input_size)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        elif self.mode == 'val':
            img, target = imageio.imread(self.val_img_label[index][0]), self.val_img_label[index][1]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            """
            img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
            # img = transforms.RandomResizedCrop(size=self.input_size,scale=(0.4, 0.75),ratio=(0.5,1.5))(img)
            # img = transforms.RandomCrop(self.input_size)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)
            """
            
            img = transforms.Resize((256, 256))(img)
            img = transforms.RandomCrop(self.input_size)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = imageio.imread(self.test_img_label[index][0]), self.test_img_label[index][1]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')

            """
            img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
            # img = transforms.CenterCrop(self.input_size)(img)
            """
            img = transforms.Resize((256, 256))(img)
            img = transforms.RandomCrop(self.input_size)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_img_label)
        elif self.mode == 'val':
            return len(self.val_img_label)
        else:
            return len(self.test_img_label)
