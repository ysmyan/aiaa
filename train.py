from trainer import Trainer
from torchvision import transforms
from PIL import Image, ImageCms
from tqdm import tqdm
from dataset import get_lap
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description='Lexin Aesthetic')
    parser.add_argument('--name', dest='name', default='three_net')
    parser.add_argument('--data', dest='data', default='./dataset/data')
    parser.add_argument('--models_dir', dest='models_dir', default='./models')
    parser.add_argument('--num_train_steps', dest='num_train_steps', default=16, type=int)
    parser.add_argument('--batch_size', dest='batch_size', default=16)
    parser.add_argument('--lr', dest='lr', default=1e-4, type=float)
    parser.add_argument('--results_dir', dest='results_dir', default='./results')
    parser.add_argument('--num_workers', dest='num_workers', default=None)
    parser.add_argument('--save_every', dest='save_every', default=1, type=int)
    parser.add_argument('--image_size', dest='image_size', default=512, type=int)
    parser.add_argument('--new', dest='new', default=False, type=bool)
    parser.add_argument('--load_from', dest='load_from', default=-1)
    parser.add_argument('--generate', dest='generate', default=False, type=bool)
    parser.add_argument('--test_img', dest='test_img', default="./dataset/test.jpg")
    return parser.parse_args()

def train(name, data, models_dir, num_train_steps, batch_size, lr, results_dir, num_workers,
    save_every, image_size, new, load_from, generate, test_img):

    model = Trainer(name, models_dir, num_train_steps, batch_size, lr, results_dir,
        num_workers, save_every, image_size)

    if not new:
        model.load(load_from)
    else:
        model.clear()
    if generate:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
        color_img = Image.open(test_img)
        if color_img.mode == 'P':
            color_img = color_img.convert("RGBA")
        if color_img.mode == 'RGBA':
            color_img = color_img.convert("RGB")
        if color_img.mode == 'CMYK':
            color_img = ImageCms.profileToProfile(color_img, os.path.join(data, 'USWebCoatedSWOP.icc'),
                                                  os.path.join(data, 'sRGB Color Space Profile.icm'), renderingIntent=0, outputMode='RGB')
        gray_img = color_img.convert('L')
        gray_img = gray_img.convert('RGB')
        img = transform(color_img)
        gray = transform(gray_img)
        sketch_img = get_lap(img)
        img = img.unsqueeze(0)
        gray = gray.unsqueeze(0)
        sketch_img = sketch_img.unsqueeze(0)
        pred = model.evaluate(image_batch=img, gray_batch=gray, sketch_batch=sketch_img)
        print(pred)
    else:
        print(f'\nStart training {name}....\n')
        model.set_data_src(data)
        print(num_train_steps, model.steps)
        for _ in tqdm(range(num_train_steps - model.steps)):
            model.train()

if __name__ == '__main__':
    args = get_args()
    train(name=args.name, data=args.data, models_dir=args.models_dir, num_train_steps=args.num_train_steps,
        batch_size=args.batch_size, lr=args.lr, results_dir=args.results_dir, num_workers=args.num_workers,
        save_every=args.save_every, image_size=args.image_size, new=args.new, load_from=args.load_from,
        generate=args.generate, test_img=args.test_img)