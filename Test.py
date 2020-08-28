
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
from torchvision import transforms as tfs
from Model import CnnAlign
from Helen import HellenDataset, DataPrefetcher, draw_ann
import PIL.ImageFont as ImageFont
import numpy as np
import time


MODEL_FACE_ALIGN  = "./data/face_align.pt"

font_size = 4
font1 = ImageFont.truetype(r'./Ubuntu-B.ttf', font_size)


def test():
    data_loader = DataLoader(dataset=HellenDataset(False, 224), batch_size=1, shuffle=True, num_workers=1)
    device = torch.device("cpu")
    model = CnnAlign().to(device)
    state = torch.load(MODEL_FACE_ALIGN)
    model.load_state_dict(state['net'])
    to_pil_img = tfs.ToPILImage()
    model.eval()
    for img, label in data_loader:
        start = time.time()
        output = model(img)
        end = time.time()
        cost = (end - start)
        print("cost : " + str(cost))
        pil_img = to_pil_img(img[0].cpu())
        ann = output[0].cpu().detach().numpy()
        ann = np.resize(ann, (194, 2))
        draw_ann(pil_img, ann.tolist(), font1, font_size)
        plt.figure(num=1, figsize=(15, 8), dpi=80)  # 开启一个窗口，同时设置大小，分辨率
        plt.imshow(pil_img)
        plt.show()
        plt.close()

if __name__ == '__main__':
    test()

