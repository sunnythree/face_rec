import cv2
from PIL import Image
import numpy as np
import torch
from Model import CnnAlign
from LFWDataset import draw_ann
import PIL.ImageFont as ImageFont
from torchvision import transforms as tfs

MODEL_FACE_ALIGN  = "./data/face_align.pt"

font_size = 4
font1 = ImageFont.truetype(r'./Ubuntu-B.ttf', font_size)


def cv2image(image):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image

def image2cv(image):
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return image

def get_pytorch_model(path):
    device = torch.device("cpu")
    model = CnnAlign().to(device)
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state['net'])
    model.eval()
    return model


if __name__ == "__main__":
    face_align_model = get_pytorch_model(MODEL_FACE_ALIGN)
    to_tensor = tfs.ToTensor()
    cap = cv2.VideoCapture(0)
    while(1):
        # get a frame
        ret, img = cap.read()
        image = cv2image(img)
        image = image.resize((224, 224))
        output = face_align_model(to_tensor(image).view(-1, 3, 224, 224))
        ann = output[0].cpu().detach().numpy()
        ann = np.resize(ann, (194, 2))
        draw_ann(image, ann, font1, font_size)
        img = image2cv(image)
        cv2.imshow("test", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()