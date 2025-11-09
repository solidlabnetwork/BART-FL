import os
import time
import csv
import copy
import random
import torch
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from bart_utils.Local_Update import client_train
from bart_utils.Nets import VGG6, AlexNet5, ResNet18
from bart_utils.Options import parse_args
from bart_utils.aggregation import fedavg
from bart_utils.v_utils import *
import config

# ------------------- Arguments and Device -------------------
args = parse_args()
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
print(f"Using device: {device}")

num_classes = {'CIFAR10': 10, 'CIFAR100': 100, 'LISA': 7}[args.dataset]
os.makedirs(args.checkpoint_dir, exist_ok=True)
os.makedirs(args.output_dir, exist_ok=True)

csv_filename = args.csv_filename or (
    f"{args.output_dir}/vanilla-{args.model_name}-"
    f"{args.dataset}-{args.partition}-{args.n_clients}.csv"
)

# ------------------- Dataset -------------------
def load_dataset():
    if args.dataset == 'CIFAR10':
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        cls = datasets.CIFAR10
    elif args.dataset == 'CIFAR100':
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        cls = datasets.CIFAR100
    elif args.dataset == 'LISA':
        tf = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        cls = LISA_TrafficLightDataset
    else:
        raise ValueError("Unsupported dataset")

    if args.dataset in ['CIFAR10', 'CIFAR100']:
        train = cls(root=args.data_dir, train=True, download=True, transform=tf)
        test = cls(root=args.data_dir, train=False, download=True, transform=tf)
    else:
        train = cls(os.path.join(args.data_path, "train"), transform=tf)
        test = cls(os.path.join(args.data_path, "val"), transform=tf)

    return train, test, tf

train_dataset_full, test_dataset, transform = load_dataset()

# Train/Validation split (80/20)
train_size = int(0.8 * len(train_dataset_full))
val_size = len(train_dataset_full) - train_size
train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

# ------------------- IID or non-IID Partitioning -------------------
if args.partition == 'IID':
    dict_users = cifar10_iid(train_dataset, args.n_clients)
else:
    if args.dataset == 'CIFAR10':
        dict_users = cifar10_noniid(train_dataset, args.n_clients)
    elif args.dataset == 'CIFAR100':
        dict_users = cifar100_noniid(train_dataset, args.n_clients)
    elif args.dataset == 'LISA':
        dict_users = lisa_noniid(train_dataset, args.n_clients)

# Create DataLoaders for each client
client_dataloaders = [
    DataLoader(DatasetSubset(train_dataset, dict_users[i]), batch_size=args.batch_size, shuffle=True)
    for i in range(args.n_clients)
]

val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# ------------------- Model -------------------
models_dict = {'VGG6': VGG6, 'AlexNet5': AlexNet5, 'ResNet18': ResNet18}
global_model = models_dict[args.model_name](num_classes).to(device)
global_state_dict = global_model.state_dict()

# ------------------- Checkpoints -------------------
checkpoint_path1 = os.path.join(
    args.checkpoint_dir,
    f"vanilla-{args.model_name}-{args.dataset}-{args.partition}-{args.n_clients}_1.pth"
)
checkpoint_path2 = os.path.join(
    args.checkpoint_dir,
    f"vanilla-{args.model_name}-{args.dataset}-{args.partition}-{args.n_clients}_2.pth"
)

# Resume
if os.path.exists(checkpoint_path1):
    checkpoint = torch.load(checkpoint_path1)
    global_model.load_state_dict(checkpoint['model_state_dict'])
    start_round = checkpoint['round']
    test_accuracy_list = checkpoint['test_accuracy_list']
    train_accuracy_list = checkpoint['train_accuracy_list']
    results = checkpoint['results']
    best_accuracy = checkpoint.get('best_accuracy', 0.0)
else:
    start_round = 0
    test_accuracy_list, train_accuracy_list, results = [], [], []
    best_accuracy = 0.0

early_stopping = EarlyStopping(patience=args.patience)
if not os.path.exists(csv_filename):
    with open(csv_filename, 'w', newline='') as f:
        csv.writer(f).writerow(
            ["Round", "Train Accuracy", "Train Loss", "Test Accuracy", "Test Loss", "Val Accuracy", "Val Loss",
             "Round Training Time", "Checkpoint Times", "Inference Time", "Total Training Time"]
        )

round_times, checkpoint_times, inference_times = [], [], []

# ------------------- Training Rounds -------------------
for round in range(start_round, args.num_rounds):
    round_start_time = time.time()
    global_state_dict = global_model.state_dict()
    local_updates = []

    print(f"Round {round + 1}")

    for i in range(args.n_clients):
        client_model = copy.deepcopy(global_model).to(device)
        client_model.load_state_dict(global_state_dict)
        client_subset = DatasetSubset(train_dataset, dict_users[i])
        client_dataloader = DataLoader(client_subset, batch_size=args.batch_size, shuffle=True)

        client_update = client_train(args=args, device=device, lr=args.learning_rate,
                                     weight_decay=args.weight_decay, dataloader=client_dataloader)
        client_state_dict, _ = client_update.train(client_model)

        model_update = {k: client_state_dict[k] - global_state_dict[k] for k in global_state_dict}
        local_updates.append(model_update)

    # Aggregation with FedAvg (no condition needed)
    data_sizes = [len(dict_users[i]) for i in range(args.n_clients)]
    aggregated_update = fedavg(local_updates, data_sizes, device)
    global_state_dict = {k: global_state_dict[k] + aggregated_update[k] for k in global_state_dict}
    global_model.load_state_dict(global_state_dict)

    # Validation and metrics
    inf_start_time = time.time()
    test_loss, test_accuracy = validate(global_model, test_dataloader, torch.nn.CrossEntropyLoss())
    val_loss, val_accuracy = validate(global_model, val_loader, torch.nn.CrossEntropyLoss())
    inf_end_time = time.time()

    inference_times.append(inf_end_time - inf_start_time)
    test_accuracy_list.append(test_accuracy)

    train_loss, train_accuracy = validate(global_model, client_dataloader, torch.nn.CrossEntropyLoss())
    train_accuracy_list.append(train_accuracy)
    results.append([round + 1, train_accuracy, train_loss, test_accuracy, test_loss, val_accuracy, val_loss])

    print("*******************************************************************************************")
    print(f'Round {round + 1}, Train Accuracy: {train_accuracy:.2f}%, Train Loss: {train_loss:.4f}')
    print(f'Round {round + 1}, Test Accuracy: {test_accuracy:.2f}%, Test Loss: {test_loss:.4f}')
    print(f'Round {round + 1}, Val Accuracy: {val_accuracy:.2f}%, Val Loss: {val_loss:.4f}')
    print("*******************************************************************************************")

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(global_model.state_dict(), checkpoint_path2)
        print(f"Best model saved with validation accuracy: {val_accuracy:.2f}%")

    early_stopping(val_loss, global_model)
    if early_stopping.early_stop:
        print("Early stopping triggered, continuing training.")
        early_stopping.early_stop = False
        early_stopping.counter = 0

    save_start_time = time.time()
    torch.save({
        'round': round + 1,
        'model_state_dict': global_model.state_dict(),
        'test_accuracy_list': test_accuracy_list,
        'train_accuracy_list': train_accuracy_list,
        'results': results,
        'best_accuracy': best_accuracy
    }, checkpoint_path1)

    checkpoint_times.append(time.time() - save_start_time)
    round_times.append(time.time() - round_start_time)

    with open(csv_filename, 'a', newline='') as f:
        csv.writer(f).writerow([
            round + 1, train_accuracy, train_loss, test_accuracy, test_loss,
            val_accuracy, val_loss, round_times[-1], checkpoint_times[-1],
            inference_times[-1], time.time() - round_start_time
        ])

print(f"Total Training Time: {sum(round_times):.2f} seconds")
print(f"Total Checkpoint Saving Time: {sum(checkpoint_times):.2f} seconds")
print(f"Total Inference Time: {sum(inference_times):.2f} seconds")
