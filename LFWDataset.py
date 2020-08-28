import torch
from torch.utils.data import Dataset
from utils import image_loader
import os
import math
import Config as cfg
import cv2


def create_datasets(dataroot, train_val_split=0.9, is_train=True):
    if not os.path.isdir(dataroot):
        print("no dataset error")
        return

    images_root = os.path.join(dataroot, 'lfw-deepfunneled')
    names = os.listdir(images_root)
    if len(names) == 0:
        raise RuntimeError('Empty dataset')

    datasets = []
    for klass, name in enumerate(names):
        def add_class(image):
            image_path = os.path.join(images_root, name, image)
            return (image_path, klass, name)

        images_of_person = os.listdir(os.path.join(images_root, name))
        total = len(images_of_person)
        if is_train:
            datasets += map(
                    add_class,
                    images_of_person[:math.ceil(total * train_val_split)])
        else:
            datasets += map(
                    add_class,
                    images_of_person[math.floor(total * train_val_split):])

    return datasets, len(names)

class LFWDataset(Dataset):
    def __init__(self, dataset_dir, transform=None, target_transform=None, is_train=True):
        datasets, num_classes = create_datasets(dataset_dir, train_val_split=0.9, is_train=is_train)
        self.datasets = datasets
        self.num_classes = len(datasets)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        image = image_loader(self.datasets[index][0])
        if self.transform:
            image = self.transform(image)
        return (image, self.datasets[index][1], self.datasets[index][2])


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target

def test_dataset():
    data_loader = torch.utils.data.DataLoader(LFWDataset(cfg.path), batch_size=1, num_workers=1)
    widow_name = "image show"
    cv2.namedWindow(widow_name)
    for data in data_loader:
        img_cv = data[0].squeeze().numpy()
        font = cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(img_cv, data[2][0], (10, 20), font, 0.6, (255, 255, 0), 1, False)
        cv2.imshow(widow_name, img_cv)
        cv2.waitKey(-1)
    cv2.destroyWindow(winname=widow_name)

if __name__=='__main__':
    test_dataset()