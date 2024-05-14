from pathlib import Path
from network import ThreeNet
from torch.utils.data import DataLoader
from math import floor
from shutil import rmtree
from dataset import lexin_dataset
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, PredictionErrorDisplay, r2_score
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import numpy as np
import multiprocessing
import torch
import random

num_cores = multiprocessing.cpu_count()

def default(value, d):
  return d if value is None else value

class Trainer():
    def __init__(self, name, models_dir='./models', num_train_steps=5e4, batch_size=4, lr=2e-4,
        results_dir='./results', num_workers=None, save_every=1e4, image_size=128):

        self.name = name
        self.results_dir = Path(results_dir)
        self.models_dir = Path(models_dir)

        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.sum_eval_loss = 0

        self.save_every = save_every
        self.num_train_steps = num_train_steps
        self.steps = 0
        self.AestheticNet = None
        self.loss = 0
        self.max_r2 = -1
        self.max_steps = 0
        self.init_folders()

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    def save(self, num):
        torch.save(self.AestheticNet.state_dict(), self.model_name(num))

    def init_AestheticNet(self):
        self.AestheticNet = ThreeNet(self.lr)

    def set_data_src(self, data):
        train_dataset = lexin_dataset(data, self.image_size, is_train=True)
        test_dataset = lexin_dataset(data, self.image_size, is_train=False)
        print(len(train_dataset), len(test_dataset))
        self.loader = DataLoader(train_dataset,
            num_workers=default(self.num_workers, num_cores),
            batch_size=self.batch_size,shuffle=True)
        self.loader_evaluate = DataLoader(test_dataset,
            num_workers=default(self.num_workers, num_cores),
            batch_size=self.batch_size,shuffle=False)
        
    def load(self, num=-1):
        self.init_AestheticNet()
        num = int(num)
        name = num
        if num == -1:
            file_paths = [p for p in
                            Path(self.models_dir / self.name).glob('model_*.pt')]
            saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
        print(f'Aesthetic net continuing from previous epoch - {name}')
        self.steps = name * self.save_every
        self.AestheticNet.load_state_dict(torch.load(self.model_name(name), map_location=torch.device('cpu')))

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(f'./models/{self.name}', True)
        rmtree(f'./results/{self.name}', True)
        self.init_folders()

    def train(self):
        if self.AestheticNet is None:
            self.init_AestheticNet()
        self.AestheticNet.train()
        for batch in self.loader:
            self.AestheticNet.opt.zero_grad()
            image_batch = batch['imgs']
            gray_batch = batch['gray']
            sketch_batch = batch['sketch']
            rating_labels = batch['ratings'].to(dtype=torch.float32)

            predict_ratings = self.AestheticNet(image_batch, gray_batch, sketch_batch)

            loss = torch.mean(torch.abs(predict_ratings - rating_labels))

            if torch.isnan(loss):
                print(predict_ratings)
                raise

            self.loss = float(loss)

            loss.backward()
            self.AestheticNet.opt.step()

        self.steps += 1
        self.AestheticNet.sche.step()
        checkpoint_num = floor(self.steps / self.save_every)
        if self.steps % self.save_every == 0:
            self.save(checkpoint_num)
        
        self.evaluate(num=self.steps)

    @torch.no_grad()
    def evaluate(self, num=None, image_batch=None, gray_batch=None, sketch_batch=None):

        self.AestheticNet.eval()
        input_img = False if image_batch == None else True

        if not input_img:
            gt_rating = []
            pr_rating = []
            random_dispaly = random.randint(0, len(self.loader_evaluate)-2)
            for i, batch in enumerate(self.loader_evaluate):
                image_batch = batch['imgs']
                gray_batch = batch['gray']
                sketch_batch = batch['sketch']
                rating_labels = batch['ratings']

                predict_ratings = self.AestheticNet(image_batch, gray_batch, sketch_batch)

                gt_rating.extend(rating_labels.cpu().numpy().tolist())
                pr_rating.extend(predict_ratings.cpu().numpy().tolist())

                if i == random_dispaly:
                    for j in range(4):
                        ax = plt.subplot(2, 2, j+1)
                        ax.axis('off')
                        ax.set_title(f'pred:{round(float(predict_ratings[j]), 2)} true:{round(float(rating_labels[j]), 2)}')
                        inp = image_batch.data[j].cpu().numpy().transpose((1, 2, 0))
                        plt.imshow(inp)
                    plt.savefig(str(self.results_dir / self.name / f'{str(num)}_color.jpg'))

            color_avg_mae = mean_absolute_error(gt_rating, pr_rating)
            color_avg_mse = mean_squared_error(gt_rating, pr_rating)
            srcc_mean, _ = spearmanr(pr_rating, gt_rating)
            lcc_mean, _ = pearsonr(pr_rating, gt_rating)
            color_r2 = r2_score(gt_rating, pr_rating)
            rate_01 = np.sum(np.abs(np.array(gt_rating) - np.array(pr_rating)) <= 0.1) / len(pr_rating)
            rate_02 = np.sum(np.abs(np.array(gt_rating) - np.array(pr_rating)) <= 0.2) / len(pr_rating)

            if color_r2 > self.max_r2:
                self.max_r2 = color_r2
                self.max_steps = self.steps
            print('\nSteps:', self.steps, 'R2:', color_r2, 'Max Steps:', self.max_steps, 'Max R2:', self.max_r2, 'MAE:', color_avg_mae, 'MSE:', color_avg_mse, 'SRCC:', srcc_mean, 'LCC', lcc_mean)
            print('<=0.1:', rate_01, '<=0.2:', rate_02)

            _, ax = plt.subplots(figsize=(5, 5))
            display = PredictionErrorDisplay.from_predictions(
                gt_rating, pr_rating, kind="actual_vs_predicted", ax=ax, scatter_kwargs={"alpha": 0.5}
            )
            ax.plot([], [], " ", label=f"R2 on testing set: {color_r2}")
            ax.plot([], [], " ", label=f"Mae on testing set: {color_avg_mae}")
            ax.plot([], [], " ", label=f"Max r2 on step {self.max_steps}: {self.max_r2}")
            ax.legend(loc="upper left")
            plt.tight_layout()

            plt.savefig(str(self.results_dir / self.name / f'{str(num)}_graph.png'), transparent=True)

            plt.close()


        else:
            outputs = self.AestheticNet(image_batch, gray_batch, sketch_batch)
            return float(outputs.squeeze())