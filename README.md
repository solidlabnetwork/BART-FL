# BART-FL: A Backdoor Attack-Resilient Federated Aggregation Technique for Cross-Silo Applications
BART-FL (Backdoor-Aware Robust Training for Federated Learning) is a novel defense-oriented framework that enhances the robustness of **Federated Learning (FL)** against **backdoor and poisoning attacks**. It integrates **Principal Component Analysis (PCA)** and **clustering-based filtering** to isolate and suppress malicious client updates while maintaining accuracy on clean data. Designed for **cross-device federated environments**, BART-FL provides explainable and privacy-aware aggregation mechanisms to improve resilience against adversarial behavior.

📘 [Read the paper on IEEE Xplore](https://ieeexplore.ieee.org/document/11172307)
---

## 🛠️ Requirements

- Python 3.9+
- PyTorch 2.0+
- Torchvision
- Pillow
- NumPy
- scikit-learn
- CUDA 11.8 (optional, for GPU support)

Install dependencies using:

```bash
pip install -r requirements.txt
```

📚 Datasets and Models

## 📚 Datasets

This project supports the following datasets for federated learning:

- **CIFAR-10:** Automatically downloaded via `torchvision.datasets.CIFAR10`.
- **CIFAR-100:** Automatically downloaded via `torchvision.datasets.CIFAR100`.
- **LISA Traffic Light Dataset:** Local dataset that must be placed under  
  `data/lisa/train/` and `data/lisa/val/` folders.  
  All images are resized to **32×32** resolution for model compatibility.


## 🤖 Models

This project supports the following neural network architectures for federated learning:

- **VGG6:** Lightweight convolutional model adapted for CIFAR and LISA datasets.  
- **AlexNet5:** Simplified 5-layer AlexNet variant for efficient client-side training.  
- **ResNet18:** Deep residual network with skip connections for robust performance.

## 🔧 Federated Learning with BART-FL

Train **CIFAR-100** with **ResNet18** and the **Adaptive Patch Attack** using the following command:

```bash
python main.py \
  --device GPU \
  --dataset CIFAR100 \
  --partition nonIID \
  --model_name ResNet18 \
  --aggregation bartfl \
  --num_rounds 100 \
  --n_clients 10 \
  --attack_type adaptive_patch \
  --n_attackers 4 \
  --poison_rate 0.7 \
  --alpha 0.2 \
  --target_class 0 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --weight_decay 0.0001 \
  --trigger_dir ./triggers \
  --checkpoint_dir checkpoints \
  --output_dir result \
  --csv_filename result/bartfl-adaptive-ResNet18-cifar100.csv \
  --pca-dim 0.95 \
  --num_clusters 2 \
  --patience 5 \
  --random_seed 42 \
  --verbose
```
## 📜 Citation

If you use BART-FL in your research, please cite the following paper:
```bash
@ARTICLE{11172307,
  author={Mia, Md Jueal and Hadi Amini, M.},
  journal={IEEE Transactions on Machine Learning in Communications and Networking}, 
  title={BART-FL: A Backdoor Attack-Resilient Federated Aggregation Technique for Cross-Silo Applications}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Training;Data models;Computational modeling;Filtering;Federated learning;Data privacy;Adaptation models;Transportation;Reflection;Principal component analysis;Federated Learning;Backdoor Attacks;Outlier Detection;Security;Cross-Silo Applications},
  doi={10.1109/TMLCN.2025.3611398}}

```

📘 Read the paper on IEEE Xplore: https://ieeexplore.ieee.org/document/11172307
