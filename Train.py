from PIL import ImageDraw
from summary import writer
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import argparse
import torch
import os
from utils import progress_bar
from torchvision import transforms as tfs
from Model import FaceNet, Resnet18FaceModel, QFaceNet
from LFWDataset import LFWDataset, DataPrefetcher
import PIL.ImageFont as ImageFont
import numpy as np
import Config as cfg
from op import FaceLoss
from Transform import transform_for_training


MODEL_SAVE_PATH = "./output/face_rec.pt"

font_size = 4
font1 = ImageFont.truetype(r'./Ubuntu-B.ttf', font_size)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gama', "-g", type=float, default=0.9, help='train gama')
    parser.add_argument('--step', "-s", type=int, default=20, help='train step')
    parser.add_argument('--batch', "-b", type=int, default=1, help='train batch')
    parser.add_argument('--epoes', "-e", type=int, default=30, help='train epoes')
    parser.add_argument('--lr', "-l", type=float, default=0.001, help='learn rate')
    parser.add_argument('--pretrained', "-p", type=bool, default=False, help='prepare trained')
    parser.add_argument('--mini_batch', "-m", type=int, default=1, help='mini batch')
    return parser.parse_args()

def train(args):
    start_epoch = 0
    dataset = LFWDataset(cfg.path, transform=transform_for_training(FaceNet.IMAGE_SHAPE))
    data_loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=16, drop_last=True)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = QFaceNet()
    print("add graph")
    writer.add_graph(model, torch.zeros((1, 3, 96, 128)))
    print("add graph over")
    face_loss = FaceLoss(dataset.get_num_classes(), 512)
    if args.pretrained and os.path.exists(MODEL_SAVE_PATH):
        print("loading ...")
        state = torch.load(MODEL_SAVE_PATH)
        model.load_state_dict(state['net'])
        face_loss.load_state_dict(state['loss'])
        start_epoch = state['epoch']
        print("loading over")
    model = torch.nn.DataParallel(model, device_ids=[0, 1])  # multi-GPU
    model.to(device)
    model.train()

    face_loss.to(device)
    face_loss.train()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    optimizer_loss = optim.Adam(face_loss.parameters(), lr=args.lr, weight_decay=1e-5)

    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gama)
    scheduler_loss = StepLR(optimizer_loss, step_size=args.step, gamma=args.gama)

    for epoch in range(start_epoch, start_epoch+args.epoes):
        prefetcher = DataPrefetcher(data_loader)
        img_tensor, targets = prefetcher.next()
        optimizer.zero_grad()
        optimizer_loss.zero_grad()

        i_batch = 0
        while img_tensor is not None:
            output = model(img_tensor)
            c_loss, s_loss, loss, _ = face_loss(output, targets)
            if loss is None:
                img_tensor, targets = prefetcher.next()
                continue
            loss.backward()
            if i_batch % args.mini_batch == 0:
                optimizer_loss.step()
                optimizer_loss.zero_grad()
                optimizer.step()
                optimizer.zero_grad()


            train_loss = loss.item()
            train_c_loss = c_loss.item()
            train_s_loss = s_loss.item()
            global_step = epoch*len(data_loader)+i_batch
            progress_bar(i_batch, len(data_loader), 'loss: %f, c_loss: %f, s_loss: %f, epeche: %d'%(train_loss, train_c_loss, train_s_loss, epoch))
            writer.add_scalar("loss", train_loss, global_step=global_step)
            img_tensor, targets = prefetcher.next()
            i_batch += 1


        #save one pic and output
        scheduler.step()
        scheduler_loss.step()

        if epoch % 100 == 0:
            if not os.path.exists("./output"):
                os.mkdir("./output")
            print('Saving..')
            state = {
                'net': model.module.state_dict(),
                'loss': face_loss.state_dict(),
                'epoch': epoch,
            }
            torch.save(state, "./output/face_rec"+str(epoch)+".pt")

    if not os.path.isdir('data'):
        os.mkdir('data')
    print('Saving..')
    state = {
        'net': model.module.state_dict(),
        'loss': face_loss.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, MODEL_SAVE_PATH)
    writer.close()

if __name__=='__main__':
    train(parse_args())

