import os
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import config


class poison_generator():
    def __init__(self, img_size, dataset, poison_rate, path, trigger_names, alphas, target_class=0, cover_rate=0.01):
        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.path = path
        self.target_class = target_class
        self.cover_rate = cover_rate
        self.num_img = len(dataset)

        trigger_transform = transforms.Compose([transforms.ToTensor()])
        self.trigger_marks = []
        self.trigger_masks = []
        self.alphas = []

        for i in range(len(trigger_names)):
            trigger_path = os.path.join(config.triggers_dir, trigger_names[i])
            trigger_mask_path = os.path.join(config.triggers_dir, f'mask_{trigger_names[i]}')

            trigger = Image.open(trigger_path).convert("RGB")
            trigger = trigger_transform(trigger)

            if os.path.exists(trigger_mask_path):
                trigger_mask = Image.open(trigger_mask_path).convert("RGB")
                trigger_mask = transforms.ToTensor()(trigger_mask)[0]
            else:
                trigger_mask = torch.logical_or(
                    torch.logical_or(trigger[0] > 0, trigger[1] > 0),
                    trigger[2] > 0
                ).float()

            self.trigger_marks.append(trigger)
            self.trigger_masks.append(trigger_mask)
            self.alphas.append(alphas[i])
            print(f"Trigger #{i}: {trigger_names[i]}")

    def generate_poisoned_training_set(self):
        id_set = list(range(self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poison_rate)
        poison_indices = sorted(id_set[:num_poison])
        num_cover = int(self.num_img * self.cover_rate)
        cover_indices = sorted(id_set[num_poison:num_poison + num_cover])

        img_set = []
        label_set = []
        pt = ct = cnt = 0
        poison_id = []
        cover_id = []
        k = len(self.trigger_marks)

        for i in range(self.num_img):
            img, gt = self.dataset[i]

            if ct < num_cover and cover_indices[ct] == i:
                cover_id.append(cnt)
                for j in range(k):
                    if ct < (j + 1) * (num_cover / k):
                        img = img + self.alphas[j] * self.trigger_masks[j] * (self.trigger_marks[j] - img)
                        break
                ct += 1

            if pt < num_poison and poison_indices[pt] == i:
                poison_id.append(cnt)
                gt = self.target_class
                for j in range(k):
                    if pt < (j + 1) * (num_poison / k):
                        img = img + self.alphas[j] * self.trigger_masks[j] * (self.trigger_marks[j] - img)
                        break
                pt += 1

            img_set.append(img.unsqueeze(0))
            label_set.append(gt)
            cnt += 1

        img_set = torch.cat(img_set, dim=0)
        label_set = torch.LongTensor(label_set)

        demo_img, _ = self.dataset[0]
        for j in range(k):
            demo_img = demo_img + self.alphas[j] * self.trigger_masks[j] * (self.trigger_marks[j] - demo_img)
        save_image(demo_img, os.path.join(self.path, 'demo.png'))

        return img_set, poison_id, cover_id, label_set


class poison_transform():
    def __init__(self, img_size, test_trigger_names, test_alphas, target_class=0, denormalizer=None, normalizer=None):
        self.img_size = img_size
        self.target_class = target_class
        self.denormalizer = denormalizer
        self.normalizer = normalizer

        trigger_transform = transforms.Compose([transforms.ToTensor()])
        self.trigger_marks = []
        self.trigger_masks = []
        self.alphas = []

        for i in range(len(test_trigger_names)):
            trigger_path = os.path.join(config.triggers_dir, test_trigger_names[i])
            trigger_mask_path = os.path.join(config.triggers_dir, f'mask_{test_trigger_names[i]}')

            trigger = Image.open(trigger_path).convert("RGB")
            trigger = trigger_transform(trigger)

            if os.path.exists(trigger_mask_path):
                trigger_mask = Image.open(trigger_mask_path).convert("RGB")
                trigger_mask = transforms.ToTensor()(trigger_mask)[0]
            else:
                trigger_mask = torch.logical_or(
                    torch.logical_or(trigger[0] > 0, trigger[1] > 0),
                    trigger[2] > 0
                ).float()

            self.trigger_marks.append(trigger.cuda())
            self.trigger_masks.append(trigger_mask.cuda())
            self.alphas.append(test_alphas[i])

    def transform(self, data, labels, denormalizer=None, normalizer=None):
        data, labels = data.clone(), labels.clone()
        data = self.denormalizer(data)

        for j in range(len(self.trigger_marks)):
            data = data + self.alphas[j] * self.trigger_masks[j].to(data.device) * (self.trigger_marks[j].to(data.device) - data)

        data = self.normalizer(data)
        labels[:] = self.target_class
        return data, labels
