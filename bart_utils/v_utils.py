import os
import random
import pickle
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import GPUtil
from bart_utils.Options import parse_args
from torchvision.utils import save_image

args = parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CIFAR10Dataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.transform = transform
        self.train = train
        if train:
            self.data, self.labels = [], []
            for i in range(1, 6):
                with open(os.path.join(data_dir, f'data_batch_{i}'), 'rb') as f:
                    d = pickle.load(f, encoding='bytes')
                    self.data.append(d[b'data'])
                    self.labels.extend(d[b'labels'])
            self.data = np.concatenate(self.data)
        else:
            with open(os.path.join(data_dir, 'test_batch'), 'rb') as f:
                d = pickle.load(f, encoding='bytes')
                self.data = d[b'data']
                self.labels = d[b'labels']
        self.data = self.data.reshape((-1, 3, 32, 32)).astype(np.float32)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = torch.tensor(self.data[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def cifar10_iid(dataset, num_users):
    num_items = len(dataset) // num_users
    dict_users, all_idxs = {}, list(range(len(dataset)))
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar10_noniid(dataset, num_users):
    num_shards, num_imgs = 200, 250
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    if isinstance(dataset, torch.utils.data.Subset):
        labels = np.array([dataset.dataset.targets[i] for i in dataset.indices])
    else:
        labels = np.array(dataset.targets)

    # labels = np.array(dataset.targets)
    idxs = np.arange(len(labels))
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    idx_shard = list(range(num_shards))

    shard_counts = np.random.randint(2, 7, size=num_users)
    shard_counts = (shard_counts / shard_counts.sum() * num_shards).astype(int)
    shard_counts[0] += (num_shards - shard_counts.sum())

    for i in range(num_users):
        count = shard_counts[i]
        if len(idx_shard) < count:
            count = len(idx_shard)
        selected = np.random.choice(idx_shard, count, replace=False)
        idx_shard = list(set(idx_shard) - set(selected))
        for shard in selected:
            dict_users[i] = np.concatenate((dict_users[i], idxs[shard * num_imgs:(shard + 1) * num_imgs]), axis=0)
        np.random.shuffle(dict_users[i])
    return dict_users


def cifar100_noniid(dataset, num_users):
    num_shards = 200
    num_imgs = 250
    seed = 42
    np.random.seed(seed)

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    all_idxs = np.random.permutation(len(dataset))
    shards = [all_idxs[i * num_imgs:(i + 1) * num_imgs] for i in range(num_shards)]

    shard_counts = np.random.randint(2, 7, size=num_users)
    shard_counts = (shard_counts / shard_counts.sum() * num_shards).astype(int)
    shard_counts[0] += (num_shards - shard_counts.sum())

    shard_pool = np.random.permutation(num_shards)
    start = 0
    for user in range(num_users):
        count = shard_counts[user]
        assigned_shards = shard_pool[start:start + count]
        for shard_id in assigned_shards:
            dict_users[user] = np.concatenate((dict_users[user], shards[shard_id]), axis=0)
        np.random.shuffle(dict_users[user])
        start += count

    return dict_users


def lisa_noniid(dataset, num_users):
    num_shards, num_imgs = 100, len(dataset) // 100
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    # labels = np.array(dataset.labels)
    if isinstance(dataset, torch.utils.data.Subset):
        labels = np.array([dataset.dataset.labels[i] for i in dataset.indices])
    else:
        labels = np.array(dataset.labels)
    idxs = np.arange(len(labels))
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    idx_shard = list(range(num_shards))
    shards_per_user = num_shards // num_users

    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, shards_per_user, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for shard_id in rand_set:
            shard_idxs = idxs[shard_id * num_imgs: (shard_id + 1) * num_imgs]
            dict_users[i] = np.concatenate((dict_users[i], shard_idxs), axis=0)
        np.random.shuffle(dict_users[i])

    return dict_users

class DatasetSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


class LISA_TrafficLightDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_folder = os.path.join(root_dir, cls)
            for img_file in os.listdir(cls_folder):
                self.image_paths.append(os.path.join(cls_folder, img_file))
                self.labels.append(self.class_to_idx[cls])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def validate(model, val_loader, criterion, device=torch.device('cuda:0')):
    model = model.to(device)
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            val_loss += criterion(out, y).item()
            _, pred = torch.max(out.data, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return val_loss / len(val_loader), 100 * correct / total


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.best_model = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def save_poison_visual(img_clean, img_poisoned, label, save_path, idx, prefix=''):
    """
    Save original and poisoned image side by side.
    """
    os.makedirs(save_path, exist_ok=True)
    combined = torch.stack([img_clean, img_poisoned], dim=0)
    filename = os.path.join(save_path, f"{prefix}_label{label}_idx{idx}.png")
    save_image(combined, filename, nrow=2)
