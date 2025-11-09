import numpy as np
import random
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import os
import pickle
import GPUtil
from utils.Options import parse_args
import random
import torchvision.utils as vutils

# Parse arguments
args = parse_args()

# Utility to set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Custom CIFAR-10 dataset class
class CIFAR10Dataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.data_dir = data_dir
        self.train = train
        self.transform = transform
        if self.train:
            self.data, self.labels = [], []
            for i in range(1, 6):
                file = os.path.join(data_dir, 'data_batch_' + str(i))
                with open(file, 'rb') as fo:
                    dict = pickle.load(fo, encoding='bytes')
                    self.data.append(dict[b'data'])
                    self.labels.extend(dict[b'labels'])
            self.data = np.concatenate(self.data)
        else:
            file = os.path.join(data_dir, 'test_batch')
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                self.data = dict[b'data']
                self.labels = dict[b'labels']
        self.data = self.data.reshape((-1, 3, 32, 32)).astype(np.float32)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]
        img = torch.tensor(img)
        if self.transform:
            img = self.transform(img)
        return img, label

# IID partitioning for CIFAR-10 dataset
def cifar10_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar100_iid(dataset, num_users):
    num_items = 5000
    total_samples = len(dataset)

    if total_samples < num_items * 8:
        raise ValueError(f"Not enough data for the first 8 clients. The dataset must have at least {num_items * 8} samples.")

    dict_users = {}
    all_idxs = np.arange(total_samples)
    np.random.shuffle(all_idxs)

    for i in range(8):
        dict_users[i] = set(all_idxs[i * num_items : (i + 1) * num_items])

    for i in range(8, num_users):
        reused_client_idx = i % 8
        dict_users[i] = dict_users[reused_client_idx]

    return dict_users

# Non-IID partitioning for CIFAR-10 dataset (with different labels)
def cifar_noniid(dataset, num_users):
    num_shards, num_imgs = 200, 250
    dict_users = {i: np.array([]) for i in range(num_users)}

    if isinstance(dataset, torch.utils.data.Subset):
        labels = np.array([dataset.dataset.targets[i] for i in dataset.indices])
        dataset_size = len(dataset.indices)
    else:
        labels = np.array(dataset.targets)
        dataset_size = len(dataset)

    idxs = np.arange(dataset_size)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    shard_per_user = int(num_shards / num_users)
    idx_shard = [i for i in range(num_shards)]

    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, shard_per_user, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

        np.random.shuffle(dict_users[i])

    return dict_users

def lisa_noniid(dataset, num_users):
    """
    Non-IID partitioning for LISA dataset with fixed number of shards and samples per shard.
    Follows same logic as cifar_noniid(): label-skew via shard assignment.
    """
    num_shards, num_imgs = 100, len(dataset) // 100  # e.g., 100 shards of ~N samples each
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    labels = np.array(dataset.labels)
    idxs = np.arange(len(labels))

    # Sort by label
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    idx_shard = list(range(num_shards))
    shards_per_user = num_shards // num_users

    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, shards_per_user, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for shard_id in rand_set:
            shard_idxs = idxs[shard_id * num_imgs : (shard_id + 1) * num_imgs]
            dict_users[i] = np.concatenate((dict_users[i], shard_idxs), axis=0)

        np.random.shuffle(dict_users[i])

    return dict_users


# def lisa_noniid(dataset, num_users, num_shards=20):
#     """
#     Non-IID partitioning for LISA dataset.
#     Each client gets samples from a few classes (label-skew).
#     """
#     # Build label list
#     labels = dataset.labels  # List of labels
#     idxs = np.arange(len(labels))
#     labels = np.array(labels)
#
#     # Sort indices by label
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]
#
#     # Number of shards per client
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
#     num_imgs = int(len(dataset) / num_shards)
#     idx_shard = [i for i in range(num_shards)]
#     shards_per_user = num_shards // num_users
#
#     for i in range(num_users):
#         rand_shards = set(np.random.choice(idx_shard, shards_per_user, replace=False))
#         idx_shard = list(set(idx_shard) - rand_shards)
#         for shard in rand_shards:
#             shard_idxs = idxs[shard * num_imgs: (shard + 1) * num_imgs]
#             dict_users[i] = np.concatenate((dict_users[i], shard_idxs), axis=0)
#
#     return dict_users

# IID partitioning for MNIST dataset
def mnist_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

# Non-IID partitioning for MNIST dataset (with different labels)
def mnist_noniid(dataset, num_users):
    num_shards, num_imgs = 200, 300
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    if isinstance(dataset, torch.utils.data.Subset):
        labels = np.array([dataset.dataset.targets[i] for i in dataset.indices])
        dataset_size = len(dataset.indices)
    else:
        labels = np.array(dataset.train_labels)
        dataset_size = len(dataset)

    idxs = np.arange(dataset_size)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    shard_per_user = int(num_shards / num_users)
    idx_shard = [i for i in range(num_shards)]

    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, shard_per_user, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

        np.random.shuffle(dict_users[i])

    return dict_users

# Dataset subset for user-specific data
class DatasetSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data_idx = int(self.indices[idx])
        return self.dataset[data_idx]

# Data preparation
def prepare_data(args):
    if args.dataset in ['CIFAR10', 'CIFAR100']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]) if args.dataset == 'CIFAR10' else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        dataset_class = datasets.CIFAR10 if args.dataset == 'CIFAR10' else datasets.CIFAR100
    elif args.dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset_class = datasets.MNIST
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    train_dataset = dataset_class(root=args.data_dir, train=True, download=True, transform=transform)
    test_dataset = dataset_class(root=args.data_dir, train=False, download=True, transform=transform)

    return train_dataset, test_dataset

# Validate model performance on validation set
def validate(model, val_loader, criterion, device=torch.device('cuda:0')):
    model = model.to(device)
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    return val_loss, val_accuracy

# Early stopping mechanism to prevent overfitting
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, validation_loss, model):
        score = -validation_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model = model.state_dict()
            self.counter = 0

# Setup device based on device type
def setup_device(device_type: str):
    if device_type == "GPU":
        return select_free_gpu(max_memory_usage=0.5)
    return torch.device("cpu")

# Select the GPU with the least memory usage
def select_free_gpu(max_memory_usage=0.5):
    devices = GPUtil.getGPUs()
    available_gpus = [i for i in range(len(devices)) if devices[i].memoryUtil < max_memory_usage]
    if len(available_gpus) == 0:
        print(f"No available GPU with memory usage < {max_memory_usage}")
        selected_gpu = sorted(devices, key=lambda x: (x.memoryUtil, x.load))[0].id
    else:
        available_gpu_loads = [devices[i].load for i in available_gpus]
        min_load = min(available_gpu_loads)
        selected_gpu = available_gpus[available_gpu_loads.index(min_load)]

    print(f"Selected GPU: {selected_gpu} (memory = {devices[selected_gpu].memoryUtil * 100}%, load  {devices[selected_gpu].load * 100}%)")
    return torch.device(f"cuda:{selected_gpu}" if torch.cuda.is_available() else "cpu")


class LISA_TrafficLightDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            for image_path in os.listdir(cls_folder):
                self.image_paths.append(os.path.join(cls_folder, image_path))
                self.labels.append(self.class_to_idx[cls_name])

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# Function to poison images with a random trigger
def poison_data(dataset, indices, poison_rate, target_class, triggers, trigger_masks, alpha=args.alpha):
    num_poison = int(len(indices) * poison_rate)
    poisoned_indices = random.sample(indices, num_poison)
    poisoned_data = []

    for idx in indices:
        image, label = dataset[idx]

        if idx in poisoned_indices:
            label = target_class
            # Randomly select a trigger-mask pair
            trigger_idx = random.randint(0, len(triggers) - 1)
            trigger = triggers[trigger_idx]
            trigger_mask = trigger_masks[trigger_idx]

            # Apply the trigger
            image = image + alpha * trigger_mask * (trigger - image)
            image = torch.clamp(image, 0, 1)

        poisoned_data.append((image, label))

    return poisoned_data
def lisa_noniid_partition(dataset, num_users, num_shards=20):
    """
    Non-IID partitioning for LISA dataset using label-based shards.
    """
    labels = np.array(dataset.labels)
    idxs = np.arange(len(labels))
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    num_imgs = len(dataset) // num_shards
    shards_per_user = num_shards // num_users
    idx_shard = [i for i in range(num_shards)]

    for i in range(num_users):
        rand_shards = set(np.random.choice(idx_shard, shards_per_user, replace=False))
        idx_shard = list(set(idx_shard) - rand_shards)
        for shard in rand_shards:
            shard_idxs = idxs[shard * num_imgs: (shard + 1) * num_imgs]
            dict_users[i] = np.concatenate((dict_users[i], shard_idxs), axis=0)

    return dict_users

