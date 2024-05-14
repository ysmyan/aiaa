from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageCms
import torch.nn.functional as F
import torch, json
import os

def get_lap(input):
    laplacian = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    kernel = torch.tensor(laplacian, dtype=torch.float32).unsqueeze(0).expand(1, 3, 3, 3)
    lap = F.conv2d(input.unsqueeze(0), kernel, stride=1, padding=1)
    lap = torch.clamp(lap, 0, 1)
    lap = lap.squeeze().unsqueeze(0).expand(3, -1, -1)
    return lap

class lexin_dataset(Dataset):
    def __init__(self, data="./dataset/data", img_size=512, is_train=False):       
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ])
        if is_train:
           with open('./dataset/train_id_ratings.json') as f:
               self.id_ratings = json.load(f)
        else:
           with open('./dataset/test_id_ratings.json') as f:
               self.id_ratings = json.load(f)
        self.root = data
        self.ids = list(self.id_ratings.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]
        rating = self.id_ratings[id]
        color_img_path = os.path.join(self.root, id + '.png')
        color_img = Image.open(color_img_path)
        if color_img.mode == 'P':
            color_img = color_img.convert("RGBA")
        if color_img.mode == 'RGBA':
            color_img = color_img.convert("RGB")
        if color_img.mode == 'CMYK':
            color_img = ImageCms.profileToProfile(color_img, os.path.join(self.root, 'USWebCoatedSWOP.icc'),
                                                  os.path.join(self.root, 'sRGB Color Space Profile.icm'), renderingIntent=0, outputMode='RGB')
        gray_img = color_img.convert('L')
        gray_img = gray_img.convert('RGB')
        color_img = self.transform(color_img)
        gray_img = self.transform(gray_img)
        sketch_img = get_lap(color_img)
        return {'imgs': color_img, 'gray': gray_img, 'sketch': sketch_img, 'ratings': rating}