
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
from torchvision import transforms as tfs
from Model import QFaceNet, Resnet18FaceModel
from LFWDataset import LFWDataset, DataPrefetcher, LFWPairedDataset
import numpy as np
import Config as cfg
from op import FaceLoss
from Transform import transform_for_training, transform_for_infer
import time
import os
from sklearn.model_selection import KFold

MODEL_FACE_ALIGN  = "./output/face_rec.pt"

def generate_roc_curve(fpr, tpr, path):
    assert len(fpr) == len(tpr)

    fig = plt.figure()
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot(fpr, tpr)
    fig.savefig(path, dpi=fig.dpi)

def select_threshold(distances, matches, thresholds):
    best_threshold_true_predicts = 0
    best_threshold = 0
    for threshold in thresholds:
        true_predicts = torch.sum((
            distances < threshold
        ) == matches)

        if true_predicts > best_threshold_true_predicts:
            best_threshold_true_predicts = true_predicts
            best_threshold = threshold

    return best_threshold


def compute_roc(distances, matches, thresholds, fold_size=10):
    assert(len(distances) == len(matches))

    kf = KFold(n_splits=fold_size, shuffle=False)

    tpr = torch.zeros(fold_size, len(thresholds))
    fpr = torch.zeros(fold_size, len(thresholds))
    accuracy = torch.zeros(fold_size)
    best_thresholds = []

    for fold_index, (training_indices, val_indices) \
            in enumerate(kf.split(range(len(distances)))):

        training_distances = distances[training_indices]
        training_matches = matches[training_indices]

        # 1. find the best threshold for this fold using training set
        best_threshold_true_predicts = 0
        for threshold_index, threshold in enumerate(thresholds):
            true_predicts = torch.sum((
                training_distances < threshold
            ) == training_matches)

            if true_predicts > best_threshold_true_predicts:
                best_threshold = threshold
                best_threshold_true_predicts = true_predicts

        # 2. calculate tpr, fpr on validation set
        val_distances = distances[val_indices]
        val_matches = matches[val_indices]
        for threshold_index, threshold in enumerate(thresholds):
            predicts = (val_distances < threshold).int()
            tp = torch.sum(predicts & val_matches).item()
            fp = torch.sum(predicts & ~val_matches).item()
            tn = torch.sum(~predicts & ~val_matches).item()
            fn = torch.sum(~predicts & val_matches).item()

            tpr[fold_index][threshold_index] = float(tp) / (tp + fn)
            fpr[fold_index][threshold_index] = float(fp) / (fp + tn)

        best_thresholds.append(best_threshold)
        accuracy[fold_index] = best_threshold_true_predicts.item() / float(
            len(training_indices))

    # average fold
    tpr = torch.mean(tpr, dim=0).numpy()
    fpr = torch.mean(fpr, dim=0).numpy()
    accuracy = torch.mean(accuracy, dim=0).item()

    return tpr, fpr, accuracy, best_thresholds

def cal_distance(feature_a, feature_b):
    distances = torch.sum(torch.pow(feature_a - feature_b, 2), dim=1)
    return distances


def test():
    batch_size = 1
    pairs_path = os.path.join(cfg.path, 'pairs.txt')
    dataset = LFWPairedDataset(cfg.path,  pairs_path, transform=transform_for_infer(QFaceNet.IMAGE_SHAPE))
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = QFaceNet()
    state = torch.load(MODEL_FACE_ALIGN)
    model.load_state_dict(state['net'])
    model.to(device)
    model.eval()


    print("evaluating ...... ")

    embedings_a = torch.zeros(len(dataset), model.FEATURE_DIM)
    embedings_b = torch.zeros(len(dataset), model.FEATURE_DIM)
    matches = torch.zeros(len(dataset), dtype=torch.uint8)

    for iteration, (images_a, images_b, batched_matches) \
            in enumerate(data_loader):
        current_batch_size = len(batched_matches)
        images_a = images_a.to(device)
        images_b = images_b.to(device)

        feature_a = model(images_a)
        feature_b = model(images_b)

        start =  batch_size * iteration
        end = start + current_batch_size

        embedings_a[start:end, :] = feature_a.data
        embedings_b[start:end, :] = feature_b.data
        matches[start:end] = batched_matches.data

    thresholds = np.arange(0, 2000, 100)
    distances = cal_distance(embedings_a, embedings_b)
    tpr, fpr, accuracy, best_thresholds = compute_roc(
        distances,
        matches,
        thresholds
    )

    generate_roc_curve(fpr, tpr, "data/roc.png")
    print('Model accuracy is {}'.format(accuracy))
    print('ROC curve generated at data/')
    print("best_thresholds is "+str(best_thresholds))

if __name__ == '__main__':
    test()

