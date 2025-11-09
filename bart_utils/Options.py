import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning with XAI and Backdoor Attack Simulations")
    parser.add_argument('--device', type=str, default="GPU", choices=["GPU", "CPU"], help="Device to be used for Federated Learning (GPU or CPU)")
    parser.add_argument('--dataset', type=str, default='CIFAR100', choices=['CIFAR10', 'CIFAR100', 'LISA'], help="Dataset to use (CIFAR10, CIFAR100, or LISA)")
    parser.add_argument('--data_dir', type=str, default='./data', help="Directory for dataset storage")
    parser.add_argument('--partition', type=str, default='nonIID', choices=['IID', 'nonIID'], help="Data partitioning strategy (IID or nonIID)")
    parser.add_argument('--model_name', type=str, default='ResNet18', choices=['VGG6', 'AlexNet5', 'ResNet18'], help="Model to use (VGG6, AlexNet5, ResNet18)")
    parser.add_argument('--aggregation', type=str, default='bartfl', choices=['fedavg', 'trim_mean', 'median', 'krum', 'bartfl'], help="Choose the aggregation method for federated learning")
    parser.add_argument('--num_rounds', type=int, default=100, help="Number of communication rounds")
    parser.add_argument('--n_clients', type=int, default=10, help="Number of clients for federated learning")
    #Attack type
    parser.add_argument('--attack_type', type=str, default='adaptive_patch', choices=['adaptive_patch','badnet','`wanet`', 'refool',"none"], help='Type of backdoor attack to perform.')
    parser.add_argument('--n_attackers', type=int, default=4, help="Number of attacker clients for backdoor attack")
    parser.add_argument('--poison_rate', type=float, default=0.7, help="Proportion of attacker client data poisoned with triggers")
    parser.add_argument('--n_local_epochs', type=int, default=1, help="Number of local epochs per client")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for client-side training")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument('--weight_decay', type=float, default=0.0001, help="Weight decay for the optimizer")
    parser.add_argument('--train_val_split', type=float, default=0.8, help="Proportion of training data used for training (e.g., 0.8 for 80% train, 20% validation)")
    parser.add_argument('--alpha', type=float, default=0.2, help="Poison scaling factor (used to mix poison)")
    parser.add_argument('--target_class', type=int, default=0, help="Target class for backdoor attack triggers")
    parser.add_argument('--trigger_dir', type=str, default='./triggers', help="Directory containing backdoor attack triggers and masks")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Directory to save model checkpoints")
    parser.add_argument('--output_dir', type=str, default='result', help="Directory to save output results")
    parser.add_argument('--csv_filename', type=str, default=None, help="CSV filename for logging training metrics")
    parser.add_argument('--data_path', type=str, default='data/lisa', help="Path to LISA dataset directory (only used if dataset == LISA)")
    parser.add_argument('--patience', type=int, default=5, help="Patience for early stopping")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--verbose', action='store_true', help="Enable detailed logging")
    #BART-FL param
    parser.add_argument('--pca-dim', type=float, default=0.95, help="PCA dimension: use int for fixed components (e.g., 100), or float <1.0 for explained variance (e.g., 0.95)")
    parser.add_argument('--num_clusters', type=int, default=2, help="Number of clusters for cluster-based aggregation (CKFL, BART-FL)")
    # refool Attack Parameters
    parser.add_argument('--refool_ghost_rate', type=float, default=1.0, help="Ghost rate for ReFool attack (e.g., 1.0 for full ghost blend)")
    parser.add_argument('--refool_alpha_b', type=float, default=-1.0, help="Alpha value for background blending in ReFool")
    parser.add_argument('--refool_offset_x', type=int, default=0, help="X offset for the ghost in ReFool attack")
    parser.add_argument('--refool_offset_y', type=int, default=0, help="Y offset for the ghost in ReFool attack")
    parser.add_argument('--refool_sigma', type=float, default=-1.0, help="Sigma for ghost transparency blending in ReFool")
    parser.add_argument('--refool_ghost_alpha', type=float, default=-1.0, help="Ghost alpha blending factor in ReFool")
    # WaNet Attack Parameters
    parser.add_argument('--wanet_k', type=int, default=4, help="WaNet: grid kernel size")
    parser.add_argument('--wanet_s', type=float, default=0.5, help="WaNet: strength of the noise grid")
    parser.add_argument('--wanet_grid_rescale', type=float, default=1.0, help="WaNet: grid rescale factor")
    parser.add_argument('--wanet_cover_rate', type=float, default=0.01, help="WaNet: cover rate for poisoned data")

    # Adaptive Patch Attack Parameters
    parser.add_argument('--adaptive_patch_cover_rate', type=float, default=0.01, help="Adaptive Patch: cover rate")

    return parser.parse_args()
