import os
import torch
import torch.nn.functional as F
import random
from torchvision.utils import save_image
from config import poison_seed


class poison_generator():
    def __init__(self, img_size, dataset, poison_rate, cover_rate, path, identity_grid, noise_grid, s=0.5, k=4, grid_rescale=1, target_class=0):
        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.cover_rate = cover_rate
        self.path = path
        self.target_class = target_class
        self.num_img = len(dataset)
        self.s = s
        self.k = k
        self.grid_rescale = grid_rescale
        self.identity_grid = identity_grid
        self.noise_grid = noise_grid

    def generate_poisoned_training_set(self):
        torch.manual_seed(poison_seed)
        random.seed(poison_seed)

        id_set = list(range(self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poison_rate)
        poison_indices = sorted(id_set[:num_poison])
        num_cover = int(self.num_img * self.cover_rate)
        cover_indices = sorted(id_set[num_poison:num_poison + num_cover])

        img_set, label_set = [], []
        pt = ct = cnt = 0
        poison_id, cover_id = [], []

        grid_temps = (self.identity_grid + self.s * self.noise_grid / self.img_size) * self.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)

        ins = torch.rand(1, self.img_size, self.img_size, 2) * 2 - 1
        grid_temps2 = grid_temps + ins / self.img_size
        grid_temps2 = torch.clamp(grid_temps2, -1, 1)

        for i in range(self.num_img):
            img, gt = self.dataset[i]

            if ct < num_cover and cover_indices[ct] == i:
                cover_id.append(cnt)
                img = F.grid_sample(img.unsqueeze(0), grid_temps2, align_corners=True)[0]
                ct += 1

            if pt < num_poison and poison_indices[pt] == i:
                poison_id.append(cnt)
                gt = self.target_class
                img = F.grid_sample(img.unsqueeze(0), grid_temps, align_corners=True)[0]
                pt += 1

            img_set.append(img.unsqueeze(0))
            label_set.append(gt)
            cnt += 1

        img_set = torch.cat(img_set, dim=0)
        label_set = torch.LongTensor(label_set)

        demo_img, _ = self.dataset[0]
        demo_img = F.grid_sample(demo_img.unsqueeze(0), grid_temps, align_corners=True)[0]
        save_image(demo_img, os.path.join(self.path, 'demo.png'))

        return img_set, poison_id, cover_id, label_set


class poison_transform():
    def __init__(self, img_size, normalizer, denormalizer, identity_grid, noise_grid, s=0.5, k=4, grid_rescale=1, target_class=0):
        self.img_size = img_size
        self.normalizer = normalizer
        self.denormalizer = denormalizer
        self.target_class = target_class
        self.s = s
        self.k = k
        self.grid_rescale = grid_rescale
        self.identity_grid = identity_grid.cuda()
        self.noise_grid = noise_grid.cuda()

    def transform(self, data, labels):
        grid_temps = (self.identity_grid.to(data.device) + self.s * self.noise_grid.to(data.device) / self.img_size) * self.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)

        data = self.denormalizer(data.clone())
        data = F.grid_sample(data, grid_temps.repeat(data.shape[0], 1, 1, 1), align_corners=True)
        data = self.normalizer(data)
        labels[:] = self.target_class

        return data, labels
