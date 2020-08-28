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
from Model import FaceNet
from LFWDataset import LFWDataset, DataPrefetcher
import PIL.ImageFont as ImageFont
import numpy as np
import Config as cfg


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
    data_loader = DataLoader(dataset=LFWDataset(cfg.path), batch_size=args.batch, shuffle=True, num_workers=16)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = FaceNet()
    print("add graph")
    writer.add_graph(model, torch.zeros((1, 3, 96, 128)))
    print("add graph over")
    if args.pretrained and os.path.exists(MODEL_SAVE_PATH):
        print("loading ...")
        state = torch.load(MODEL_SAVE_PATH)
        model.load_state_dict(state['net'])
        start_epoch = state['epoch']
        print("loading over")
    model = torch.nn.DataParallel(model, device_ids=[0, 1])  # multi-GPU
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gama)
    train_loss = 0
    to_pil_img = tfs.ToPILImage()
    to_tensor = tfs.ToTensor()

    for epoch in range(start_epoch, start_epoch+args.epoes):
        model.train()
        prefetcher = DataPrefetcher(data_loader)
        img_tensor, label_tensor = prefetcher.next()
        last_img_tensor = img_tensor
        last_label_tensor = label_tensor
        optimizer.zero_grad()
        i_batch = 0
        while img_tensor is not None:
            last_img_tensor = img_tensor
            last_label_tensor = label_tensor
            output = model(img_tensor)
            loss = torch.nn.functional.smooth_l1_loss(output, label_tensor.view(-1, output.size(1)))
            if loss is None:
                img_tensor, label_tensor = prefetcher.next()
                continue
            loss.backward()
            if i_batch % args.mini_batch == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss = loss.item()
            global_step = epoch*len(data_loader)+i_batch
            progress_bar(i_batch, len(data_loader), 'loss: %f, epeche: %d'%(train_loss, epoch))
            writer.add_scalar("loss", train_loss, global_step=global_step)
            img_tensor, label_tensor = prefetcher.next()
            i_batch += 1


        #save one pic and output
        pil_img = to_pil_img(last_img_tensor[0].cpu())
        ann = output[0].cpu().detach().numpy()
        ann = np.resize(ann, (194, 2))
        writer.add_image("img: "+str(epoch), to_tensor(pil_img))
        scheduler.step()

        if epoch % 10 == 0:
            print('Saving..')
            state = {
                'net': model.module.state_dict(),
                'epoch': epoch,
            }
            torch.save(state, "./output/face_align"+str(epoch)+".pt")

    if not os.path.isdir('data'):
        os.mkdir('data')
    print('Saving..')
    state = {
        'net': model.module.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, MODEL_SAVE_PATH)
    writer.close()

if __name__=='__main__':
    train(parse_args())

