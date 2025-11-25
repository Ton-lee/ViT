# 继承自 main.py，但修改为使用对比损失训练 ViT 特征
# 主要变化：
# 1. 去掉分类头(fc)，只训练 feature_extractor。
# 2. 使用 NT-Xent (InfoNCE) 对比损失，使同类特征靠近，不同类特征远离。
# 3. 验证时支持三种功能：
#    - 提取特征 (--extract_features)，保存为 .npy 文件
#    - 基于 kNN 最近邻检索评估 embedding 质量
#    - 输出 JSON 文件 (--val)，保存每张图片的最近邻检索结果

import sys
import os
import cv2
import json
import clip
from PIL import Image
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from torchvision.models import ViT_B_16_Weights
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse


def setup_ddp():
    """初始化分布式训练环境"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


class Tee:
    """同时将日志输出到屏幕和文件"""
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


IMG_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def name_to_label():
    """根据 ImageNet100 的文件夹名称生成类别映射"""
    root = "/home/Users/dqy/Dataset/ImageNet100/train/"
    categories = sorted(os.listdir(root))
    mapping = {category: idx for idx, category in enumerate(categories)}
    return mapping


class CustomImageDataset(Dataset):
    """自定义数据集，支持从 txt 文件读取路径和标签"""
    def __init__(self, file_path, transform, label_map):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.label_map = label_map
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                image_path, label = line.split()
                if not os.path.exists(image_path): continue
                self.image_paths.append(image_path)
                self.labels.append(int(label_map[label]))
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        if img is None:
            img = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img, self.labels[idx], os.path.basename(self.image_paths[idx])


def get_transform():
    """图像预处理"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


class NTXentLoss(nn.Module):
    """NT-Xent (InfoNCE) 对比损失"""
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: [B, D], 已 L2 normalize
        labels: [B]
        """
        sim_matrix = torch.matmul(features, features.T) / self.temperature  # [B, B]
        sim_matrix.fill_diagonal_(-1e9)  # 避免自己和自己匹配

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)  # [B, B]
        diag = torch.eye(mask.size(0), device=features.device)
        mask = mask - diag
        # log-softmax over rows
        log_prob = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)
        mask_sum = mask.sum(1)
        mask_sum[mask_sum == 0] = 1.0
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        loss = -mean_log_prob_pos[mask_sum > 0].mean()
        return loss


# ViT特征提取器 - 获取分类前的特征
def get_vit_features(model, x):
    # 通过patch embedding
    x = model.conv_proj(x)  # 对于ViT，使用conv_proj而不是patch_embed
    x = x.flatten(2).transpose(1, 2)  # [B, C, H, W] -> [B, num_patches, embed_dim]
    
    # 添加cls token和position embedding
    batch_size = x.shape[0]
    cls_tokens = model.class_token.expand(batch_size, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + model.encoder.pos_embedding
    
    # 通过所有transformer encoder layers
    for encoder_layer in model.encoder.layers:
        x = encoder_layer(x)
    
    # 通过norm层
    x = model.encoder.ln(x)
    
    return x[:, 0]  # 返回[cls]token的特征 [B, embed_dim]


def train(model, train_loader, criterion, optimizer, scheduler, device):
    """训练循环：仅优化特征提取器，使同类 embedding 靠近"""
    model.train()
    total_loss = 0
    for images, labels, _ in tqdm.tqdm(train_loader, total=len(train_loader)):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        feats = get_vit_features(model.module, images)
        feats = nn.functional.normalize(feats, dim=1)  # L2 normalize
        loss = criterion(feats, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    return total_loss / len(train_loader)


def compute_topk_acc_and_map(knn, all_feats, all_labels, all_names, k):
    dists, neighbor_idx = knn.kneighbors(all_feats, n_neighbors=k)
    all_labels_np = np.array(all_labels)

    # 修改：只计算奇数k的准确率
    odd_ks = [i for i in range(1, k + 1) if i % 2 == 1]
    topk_acc = np.zeros(len(odd_ks))
    APs = []

    for i in range(len(all_feats)):
        true_label = all_labels_np[i]
        retrieved_labels = all_labels_np[neighbor_idx[i]]

        # 修改：只计算奇数k的准确率
        for idx, n in enumerate(odd_ks):
            # 取前n个邻居的标签
            top_n_labels = retrieved_labels[:n]
            # 进行投票：找出出现次数最多的标签
            unique_labels, counts = np.unique(top_n_labels, return_counts=True)
            predicted_label = unique_labels[np.argmax(counts)]

            # 检查预测是否正确
            if predicted_label == true_label:
                topk_acc[idx] += 1

        # mAP计算保持不变
        relevant = (retrieved_labels == true_label).astype(np.float32)
        if relevant.sum() == 0:
            APs.append(0)
            continue
        cum_rels = np.cumsum(relevant)
        precision_at_k = cum_rels / (np.arange(k) + 1)
        AP = (precision_at_k * relevant).sum() / relevant.sum()
        APs.append(AP)

    topk_acc = topk_acc / len(all_feats)
    mAP = np.mean(APs)

    # 修改：返回对应的k值列表
    return neighbor_idx, topk_acc, mAP, odd_ks


def validate(model, val_loader, device, retrieval_k=5,
             extract_features=False, feature_save_dir=None, save_json_path=None):
    """
    验证：
    1. 如果 extract_features=True，则保存每张图片的特征到 feature_save_dir。
    2. 计算基于 kNN 最近邻分类的准确率。
    3. 如果 save_json_path 不为 None，则保存最近邻检索结果到 JSON 文件。
    """
    model.eval()
    all_feats, all_labels, all_names = [], [], []
    with torch.no_grad():
        for images, labels, img_names in tqdm.tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            feats = get_vit_features(model.module, images)
            feats = nn.functional.normalize(feats, dim=1)

            # 保存特征到目录
            if extract_features and feature_save_dir:
                feats_np = feats.cpu().numpy()
                for f, name in zip(feats_np, img_names):
                    save_path = os.path.join(feature_save_dir, f"{name.split('.')[0]}.npy")
                    np.save(save_path, f)

            all_feats.append(feats.cpu())
            all_labels.append(labels.cpu())
            all_names.extend(img_names)

    all_feats = torch.cat(all_feats)
    all_labels = torch.cat(all_labels)

    # kNN 分类作为 proxy 评估
    knn = KNeighborsClassifier(n_neighbors=retrieval_k, metric="cosine")
    knn.fit(all_feats, all_labels)
    neighbor_idx, topk_acc, mAP, odd_ks = compute_topk_acc_and_map(knn, all_feats.numpy(), all_labels.numpy(), all_names, retrieval_k)
    for k_, acc in zip(odd_ks, topk_acc):
        print(f"[RESULT] Retrieval kNN Top-{k_} Accuracy: {acc:.4f}")
    print(f"[RESULT] Retrieval mAP@{retrieval_k}: {mAP:.4f}")
    # 保存 JSON 检索结果
    if save_json_path:
        results = {}
        for i, name in enumerate(all_names):
            neighbors = []
            for j, idx in enumerate(neighbor_idx[i]):
                neighbors.append({
                    "img": all_names[idx],
                    "label": int(all_labels[idx].item())
                })
            results[name] = {
                "label": int(all_labels[i].item()),
                "neighbors": neighbors
            }
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Retrieval results saved to {save_json_path}")

    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'val'], required=True)
    parser.add_argument('--train_txt', type=str, default='')
    parser.add_argument('--val_txt', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--retrieval_k', type=int, default=5)
    parser.add_argument('--extract_features', action='store_true')
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    # 分布式训练初始化
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    dist_rank = dist.get_rank()

    os.makedirs(os.path.join(args.save_dir, args.exp_name), exist_ok=True)
    sys.stdout = Tee(sys.stdout, open(os.path.join(args.save_dir, args.exp_name, 'log.txt'), 'w', encoding='utf-8'))

    transform = get_transform()

    # 使用Vision Transformer替换ResNet
    model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    # 修改分类头以适应100个类别
    model.heads.head = nn.Linear(model.heads.head.in_features, 100)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])

    if args.mode == 'train':
        assert args.train_txt, "--train_txt is required in train mode"
        label_map = name_to_label()
        dataset = CustomImageDataset(args.train_txt, transform, label_map)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=dist.get_world_size(), rank=dist_rank, shuffle=True)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler,
            num_workers=8,  # 增加worker数量
            pin_memory=True,  # 启用pin_memory
            persistent_workers=True,  # 保持worker进程
            prefetch_factor=2,  # 预取因子
            drop_last=True  # 丢弃最后不完整的batch
)

        if args.val_txt:
            val_dataset = CustomImageDataset(args.val_txt, transform, label_map)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=8,  # 增加worker数量
                pin_memory=True,  # 启用pin_memory
                persistent_workers=True,  # 保持worker进程
                prefetch_factor=2,  # 预取因子
                drop_last=True  # 丢弃最后不完整的batch
)

        # 使用 NT-Xent 对比损失
        criterion = NTXentLoss(temperature=args.tau)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        for epoch in range(args.epochs):
            train_sampler.set_epoch(epoch)
            loss = train(model, train_loader, criterion, optimizer, scheduler, device)
            if dist_rank == 0:
                print(f"[Epoch {epoch+1}] Contrastive Train Loss: {loss:.4f}")
                model_save_path = os.path.join(args.save_dir, args.exp_name, f'feature_epoch_{epoch:02d}.pth')
                torch.save(model.state_dict(), model_save_path)
                print(f"[INFO] Feature extractor saved to {model_save_path}")
                if args.val_txt:
                    print(f"\n[INFO] Evaluating feature space...")
                    acc = validate(model, val_loader, device, retrieval_k=args.retrieval_k)

    elif args.mode == 'val':
        assert args.val_txt, "--val_txt is required in val mode"
        if args.extract_features:
            os.makedirs(os.path.join(args.save_dir, args.exp_name, "Contrast_features"), exist_ok=True)

        label_map = name_to_label()
        val_dataset = CustomImageDataset(args.val_txt, transform, label_map)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

        save_json_path = os.path.join(args.save_dir, args.exp_name, "retrieval_results.json")
        acc = validate(
            model,
            val_loader,
            device,
            retrieval_k=args.retrieval_k,
            extract_features=args.extract_features,
            feature_save_dir=os.path.join(args.save_dir, args.exp_name, "Contrast_features"),
            save_json_path=save_json_path
        )


if __name__ == "__main__":
    main()
