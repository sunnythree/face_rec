
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
from torchvision import transforms as tfs
from Model import QFaceNet, Resnet18FaceModel
from LFWDataset import LFWDataset, DataPrefetcher
import numpy as np
import Config as cfg
from op import FaceLoss
from Transform import transform_for_training, transform_for_infer
import time
import os

MODEL_FACE_REC  = "./output/face_rec.pt"

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


def get_matches(targets, logits, n=1):
    _, preds = logits.topk(n, dim=1)
    targets_repeated = targets.view(-1, 1).repeat(1, n)
    matches = torch.sum(preds.cpu() == targets_repeated.cpu(), dim=1) \
        .nonzero().size()[0]
    return matches
def test(model, data_loader, device, face_loss):
    all_count = len(data_loader.dataset)
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
            print("Current accuracy is " + str(acc_count / count), "current mean cost is " + str(all_cost / count),
                  str(count) + "/" + str(all_count))
    print("Accuracy is : " + str(acc_count / all_count), "mean cost is : " + str(acc_count / all_count))
def quant():
    dataset_train = LFWDataset(cfg.path, transform=transform_for_infer(QFaceNet.IMAGE_SHAPE), is_train=False)
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=1, shuffle=True, num_workers=1)
    dataset_test = LFWDataset(cfg.path, transform=transform_for_infer(QFaceNet.IMAGE_SHAPE), is_train=False)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=1, shuffle=True, num_workers=1)
    device = torch.device("cpu")
    model = QFaceNet()
    state = torch.load(MODEL_FACE_REC, map_location='cpu')
    model.load_state_dict(state['net'])
    model.to(device)
    model.eval()

    face_loss = FaceLoss(dataset_train.get_num_classes(), model.FEATURE_DIM)
    face_loss.load_state_dict(state['loss'])
    face_loss.to(device)
    face_loss.eval()

    print_size_of_model(model)

    # op merge
    model.fuse_model()

    # QConfig
    # model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    model.qconfig = torch.quantization.default_qconfig
    print(model.qconfig)
    torch.quantization.prepare(model, inplace=True)

    test(model, data_loader_train, device, face_loss)
    # convert
    torch.quantization.convert(model, inplace=True)

    print_size_of_model(model)
    print(model)
    torch.save(model.state_dict(), "data/qface.pt")
    torch.jit.save(torch.jit.script(model),  "data/android_qface.pt")

    test(model, data_loader_test, device, face_loss)




if __name__ == '__main__':
    quant()

