
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
from torchvision import transforms as tfs
from Model import FaceNet, Resnet18FaceModel
from LFWDataset import LFWDataset, DataPrefetcher
import numpy as np
import Config as cfg
from op import FaceLoss
from Transform import transform_for_training, transform_for_infer
import time


MODEL_FACE_ALIGN  = "./output/face_rec.pt"


def get_matches(targets, logits, n=1):
    _, preds = logits.topk(n, dim=1)
    targets_repeated = targets.view(-1, 1).repeat(1, n)
    matches = torch.sum(preds.cpu() == targets_repeated.cpu(), dim=1) \
        .nonzero().size()[0]
    return matches

def test():
    dataset = LFWDataset(cfg.path, transform=transform_for_infer(FaceNet.IMAGE_SHAPE), is_train=False)
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=1)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = Resnet18FaceModel()
    state = torch.load(MODEL_FACE_ALIGN)
    model.load_state_dict(state['net'])
    model.to(device)
    model.eval()

    face_loss = FaceLoss(dataset.get_num_classes(), 512)
    face_loss.load_state_dict(state['loss'])
    face_loss.to(device)
    face_loss.eval()

    all_count = len(dataset)
    acc_count = 0
    all_cost = 0
    count = 0
    print("testing ...... ")

    for imgs, targets, names in data_loader:
        start = time.time()
        output = model(imgs.to(device))
        end = time.time()
        cost = (end - start)
        all_cost += cost
        _, _, _, logits = face_loss(output, targets.to(device))
        matches = get_matches(targets, logits, n=1)
        if matches == 1:
            acc_count += 1
        count += 1
        if count % 100 == 0:
            print("Current accuracy is "+str(acc_count/count), "current mean cost is "+str(all_cost/count), str(count)+"/"+str(all_count))
    print("Accuracy is : "+str(acc_count/all_count), "mean cost is : "+str(acc_count/all_count))


if __name__ == '__main__':
    test()

