import random
import torch
from bart_utils.Options import parse_args
args = parse_args()

def poison_data(dataset, indices, poison_rate, target_class, triggers, trigger_masks, alpha=args.alpha):
    num_poison = int(len(indices) * poison_rate)
    poisoned_indices = random.sample(indices, num_poison)
    poisoned_data = []

    for idx in indices:
        image, label = dataset[idx]
        if idx in poisoned_indices:
            label = target_class
            tidx = random.randint(0, len(triggers) - 1)
            image = image + alpha * trigger_masks[tidx] * (triggers[tidx] - image)
            image = torch.clamp(image, 0, 1)
        poisoned_data.append((image, label))
    return poisoned_data