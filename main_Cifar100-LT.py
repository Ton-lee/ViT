import sys
import os
import cv2
import clip
from PIL import Image
import tqdm
import torch
from vit_pytorch import ViT
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import contextlib
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import faiss
import json


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


import datetime


class Tee:
    def __init__(self, *files):
        self.files = files
        self.needs_timestamp = True  # 标记是否需要时间戳

    def write(self, obj):
        # 跳过空内容和换行符
        if obj.strip() == '':
            for f in self.files:
                f.write(obj)
                f.flush()
            # 换行后下一行需要时间戳
            if obj == '\n':
                self.needs_timestamp = True
            return

        # 添加时间戳
        if self.needs_timestamp:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            formatted_obj = f"[{timestamp}] {obj}"
            self.needs_timestamp = False
        else:
            formatted_obj = obj

        for f in self.files:
            f.write(formatted_obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


IMG_SIZE = (32, 32)
STRATEGY = "inverse"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = f"/home/Users/dqy/Dataset/Cifar100-LT/weights#{STRATEGY}.pt"


def name_to_label():
    root = "/home/Users/dqy/Dataset/Cifar100-LT/format_ImageNet/images/train/"
    categories = sorted(os.listdir(root))
    mapping = {category: idx for idx, category in enumerate(categories)}
    return mapping


class CustomImageDataset(Dataset):
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
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408],
                             [0.2675, 0.2565, 0.2761])
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408],
                             [0.2675, 0.2565, 0.2761])
    ])


def calculate_class_weights(dataset, num_classes=100, method='inverse', beta=0.999):
    """
    计算类别权重
    Args:
        dataset: 数据集对象
        num_classes: 类别数量
        method: 权重计算方法 ['inverse', 'effective', 'balanced']
        beta: 用于effective num计算的参数
    """
    if method == "none":
        return torch.ones(num_classes).to(DEVICE)
    # 统计每个类别的样本数
    class_counts = torch.zeros(num_classes, dtype=torch.long)

    for _, label, _ in dataset:
        for l in label:
            class_counts[l] += 1

    print("Class distribution:")
    for i, count in enumerate(class_counts):
        if count > 0:  # 只打印有样本的类别
            print(f"Class {i}: {count} samples")

    if method == 'inverse':
        # 逆频率加权
        weights = 1.0 / (class_counts.float() + 1e-8)
        weights = weights / weights.sum() * num_classes  # 归一化

    elif method == 'effective':
        # Effective number of samples (CB Loss)
        effective_num = 1.0 - torch.pow(beta, class_counts.float())
        weights = (1.0 - beta) / (effective_num + 1e-8)
        weights = weights / weights.sum() * num_classes

    elif method == 'balanced':
        # 平衡权重
        weights = 1.0 / (class_counts.float() + 1e-8)
        weights = weights / weights.sum()

    elif method == 'sqrt':
        # 平方根逆频率
        weights = 1.0 / torch.sqrt(class_counts.float() + 1e-8)
        weights = weights / weights.sum() * num_classes

    else:
        weights = torch.ones(num_classes)

    print(f"Class weights (method: {method}):")
    for i, (count, weight) in enumerate(zip(class_counts, weights)):
        if count > 0:
            print(f"Class {i}: count={count}, weight={weight:.4f}")

    return weights.to(DEVICE)


def reverse_transform(images):
    mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1).to(images.device)  # 转为 [1, C, 1, 1] 方便广播
    std = torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1).to(images.device)
    tensor_denormalized = images * std + mean
    image_np = tensor_denormalized.cpu().numpy().transpose(0, 2, 3, 1) * 255
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    return [Image.fromarray(image) for image in image_np]

def load_CLIP_model(device):
    MODEL_NAME = "ViT-L/14"
    model, preprocess = clip.load(MODEL_NAME, device=device)
    model.eval()
    return model, preprocess

def get_CLIP_feature(model, preprocess, images_PIL, device):
    features = []
    for image in images_PIL:
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = model.encode_image(image).squeeze()
            features.append(feature)
    return torch.stack(features)

def retrieval_Faiss(Faiss_folder, query_feature, k=5):
    index_path = os.path.join(Faiss_folder, "clip_index.faiss")
    source_path = os.path.join(Faiss_folder, "index_paths.npy")
    index = faiss.read_index(index_path)
    feature_paths = np.load(source_path, allow_pickle=True)
    faiss.normalize_L2(query_feature)
    D, I = index.search(query_feature, k)
    return D, I


def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def train(model, train_loader, criterion, optimizer, scheduler, device, class_weights=None):
    model.train()
    total_loss = 0
    num_batch = 0
    for images, labels, _ in tqdm.tqdm(train_loader, total=len(train_loader)):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        # 如果有类别权重，应用到损失计算
        if class_weights is not None:
            weight = class_weights[labels].to(device)
            loss = criterion(outputs, labels)
            loss = (loss * weight).mean()  # 加权平均
        else:
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batch += 1
    scheduler.step()
    return total_loss / num_batch


def validate(model, val_loader, criterion, label_map, device, extract_features=False, feature_save_dir=None,
             knowledge_base="", prior_weight=0.0, retrieval_k=5, only_retrieval=False, result_json_path=""):
    model.eval()
    all_preds, all_labels = [], []
    results = []
    total_loss = 0.0
    num_samples = 0
    
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
        
        return x  # 返回所有token的特征 [B, num_patches+1, embed_dim]

    if knowledge_base:
        CLIP_model, CLIP_preprocess = load_CLIP_model(device)

    with torch.no_grad():
        for images, labels, img_names in tqdm.tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            
            if not knowledge_base:
                if extract_features and feature_save_dir:
                    # 提取ViT特征（分类前的特征）
                    features = get_vit_features(model.module, images)
                    cls_features = features[:, 0]  # 获取[CLS] token的特征
                    outputs = model.module.heads(cls_features)
                    
                    # 保存特征
                    cls_features = cls_features.cpu().numpy()
                    for feat, name in zip(cls_features, img_names):
                        np.save(os.path.join(feature_save_dir, f"{name.split('.')[0]}.npy"), feat)
                else:
                    outputs = model(images)
            else:
                # 建立 Faiss 索引库
                faiss_folder = os.path.join(knowledge_base, "Faiss_index")
                assert os.path.exists(os.path.join(faiss_folder, "clip_index.faiss"))
                assert os.path.exists(os.path.join(faiss_folder, "index_paths.npy"))
                assert os.path.exists(os.path.join(knowledge_base, "ViT_features"))
                
                CLIP_paths = np.load(os.path.join(faiss_folder, "index_paths.npy"), allow_pickle=True)
                
                # 提取图像 CLIP 特征以供索引
                images_PIL = reverse_transform(images)
                CLIP_features = get_CLIP_feature(CLIP_model, CLIP_preprocess, images_PIL, device)
                CLIP_features = CLIP_features.cpu().numpy().astype('float32')
                
                # 通过 Faiss 向量库进行索引
                D, I = retrieval_Faiss(faiss_folder, CLIP_features, k=retrieval_k)
                
                # 仅基于检索结果进行分类
                if only_retrieval:
                    for i, (dists, indices) in enumerate(zip(D, I)):
                        label_count = {}
                        label_dists = {}
                        for dist, idx in zip(dists, indices):
                            path = CLIP_paths[idx]
                            filename = os.path.basename(path)
                            category = filename.split("#")[1].split("_")[0]
                            label = label_map[category]

                            label_count[label] = label_count.get(label, 0) + 1
                            label_dists.setdefault(label, []).append(dist)

                        max_count = max(label_count.values())
                        candidates = [lbl for lbl, count in label_count.items() if count == max_count]

                        if len(candidates) == 1:
                            final_label = candidates[0]
                        else:
                            avg_similarities = {
                                lbl: np.mean(label_dists[lbl]) for lbl in candidates
                            }
                            final_label = max(avg_similarities.items(), key=lambda x: x[1])[0]

                        all_preds.append(torch.tensor([final_label]))
                        results.append({
                            "image": img_names[i],
                            "ground_truth": int(labels[i].cpu().item()),
                            "predicted": int(final_label),
                            "retrieved_indices": [int(idx) for idx in indices],
                            "similarities": [float(sim) for sim in dists]
                        })
                    all_labels.append(labels.cpu())
                    num_samples += images.size(0)
                    continue
                
                # 提取ViT特征并融合检索结果
                features = get_vit_features(model.module, images)
                cls_features = features[:, 0]  # [CLS] token特征
                
                features_prior = []
                for similarities, indexes in zip(D, I):
                    weights = softmax(similarities)
                    ViT_features = []
                    for index in indexes:
                        CLIP_path = CLIP_paths[index]
                        feature_path = CLIP_path.replace("CLIP_features", "ViT_features").replace(".pt", ".npy")
                        ViT_features.append(np.load(feature_path))
                    weighted_prior = np.sum(weights[:, None] * np.stack(ViT_features), axis=0)
                    features_prior.append(weighted_prior)
                
                # 特征融合
                cls_features = cls_features * (1 - prior_weight) + torch.Tensor(np.array(features_prior)).to(device) * prior_weight
                outputs = model.module.heads(cls_features)
            
            loss = criterion(outputs, labels)
            total_loss += loss.mean().item() * images.size(0)
            num_samples += images.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            
            for i in range(len(img_names)):
                record = {
                    "image": img_names[i],
                    "ground_truth": int(labels[i].cpu().item()),
                    "predicted": int(preds[i].cpu().item()),
                }
                if knowledge_base:
                    record.update({
                        "retrieved_indices": [int(idx) for idx in I[i]],
                        "similarities": [float(sim) for sim in D[i]]
                    })
                results.append(record)

    avg_loss = total_loss / num_samples
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    report = classification_report(y_true, y_pred, labels=sorted(label_map.keys()), digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=sorted(label_map.values()))
    
    print(f"[RESULT] Evaluating...")
    print(f"Avg Loss: {avg_loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro Precision: {macro_p:.4f} | Recall: {macro_r:.4f} | F1: {macro_f1:.4f}")
    print(f"Weighted Precision: {w_p:.4f} | Recall: {w_r:.4f} | F1: {w_f1:.4f}")
    
    if result_json_path:
        import json
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Prediction results saved to: {result_json_path}")
    
    return cm, avg_loss, macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'val'], required=True)
    parser.add_argument('--train_txt', type=str, default='')
    parser.add_argument('--val_txt', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--local_rank', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--extract_features', action='store_true')
    parser.add_argument('--knowledge_base', type=str, default='')
    parser.add_argument('--retrieval_k', type=int, default=5)
    parser.add_argument('--prior_weight', type=float, default=0.0)
    parser.add_argument('--only_retrieval', action='store_true')
    parser.add_argument('--result_json', type=str, default='', help='Path to save validation results in JSON format')
    args = parser.parse_args()

    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    dist_rank = dist.get_rank()

    os.makedirs(os.path.join(args.save_dir, args.exp_name), exist_ok=True)
    sys.stdout = Tee(sys.stdout, open(os.path.join(args.save_dir, args.exp_name, 'log.txt'), 'w', encoding='utf-8'))

    transform = get_transform()
    transform_val = get_val_transform()
    
    model_config = {
        'image_size': 32,
        'patch_size': 4,
        'num_classes': 100,
        'dim': 256,
        'depth': 4,
        'heads': 6,
        'mlp_dim': 256,
        'dropout': 0.1,
        'emb_dropout': 0.1
    }

    model = ViT(**model_config)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])
    config_data = {
        'training_args': vars(args),  # 将 argparse 命名空间转换为字典
        'model_config': model_config,
        'environment_info': {
            'local_rank': local_rank,
            'device': str(device),
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    config_file = os.path.join(args.save_dir, args.exp_name, 'config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4, ensure_ascii=False)
    print(f"Configuration saved to: {config_file}")

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
                                  prefetch_factor=2)  # 预取因子)
        # 计算类别权重
        if dist_rank == 0:
            print("Calculating class weights...")
        if not os.path.exists(WEIGHTS_PATH):
            class_weights = calculate_class_weights(train_loader, num_classes=100, method=STRATEGY)
            torch.save(class_weights.cpu(), WEIGHTS_PATH)
        else:
            class_weights = torch.load(WEIGHTS_PATH).to(DEVICE)
        if args.val_txt:
            val_dataset = CustomImageDataset(args.val_txt, transform_val, label_map)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=8,  # 增加worker数量
                                    pin_memory=True,  # 启用pin_memory
                                    persistent_workers=True,  # 保持worker进程
                                    prefetch_factor=2)  # 预取因子
            best_loss, best_epoch, best_F1 = 10000, -1, 0
        
        criterion = nn.CrossEntropyLoss(reduction='none')
        config = {
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'warmup_epochs': args.warmup_epochs,
            'max_grad_norm': 1.0
        }
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # 改进的学习率调度器
        warmup_epochs = config['warmup_epochs']
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
                ),
                optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs-warmup_epochs)
            ],
            milestones=[warmup_epochs]
        )

        for epoch in range(args.epochs):
            train_sampler.set_epoch(epoch)
            loss = train(model, train_loader, criterion, optimizer, scheduler, device, class_weights)
            if dist_rank == 0:
                print(f"[Epoch {epoch + 1}] Train Loss: {loss:.4f} | lr: {scheduler.get_last_lr()[0]: .6f}")
                model_save_path = os.path.join(args.save_dir, args.exp_name, f'model_latest.pth')                
                torch.save(model.state_dict(), model_save_path)
                print(f"[INFO] Model saved to {model_save_path}")
                if args.val_txt:
                    print(f"\n[INFO] Evaluating...")
                    cm, loss, f1 = validate(model, val_loader, criterion, label_map, device)
                    if loss < best_loss:
                        best_loss = loss
                        best_epoch = epoch
                        torch.save(model.state_dict(),
                                   os.path.join(args.save_dir, args.exp_name, f'model_best_loss.pth'))
                        print(f"Best Loss: {best_loss:.4f}")
                    if f1 > best_F1:
                        best_F1 = f1
                        torch.save(model.state_dict(), os.path.join(args.save_dir, args.exp_name, f'model_best_F1.pth'))
                        print(f"Best F1: {best_F1:.4f}")
    elif args.mode == 'val':
        assert args.val_txt, "--val_txt is required in val mode"
        if args.extract_features:
            os.makedirs(os.path.join(args.save_dir, args.exp_name, "ViT_features"), exist_ok=True)
        label_map = name_to_label()
        val_dataset = CustomImageDataset(args.val_txt, transform_val, label_map)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        criterion = nn.CrossEntropyLoss()
        cm, loss, f1 = validate(model, val_loader, criterion, label_map, device, 
                 extract_features=args.extract_features,
                 feature_save_dir=os.path.join(args.save_dir, args.exp_name, "ViT_features"),
                 knowledge_base=args.knowledge_base,
                 prior_weight=args.prior_weight,
                 retrieval_k=args.retrieval_k,
                 only_retrieval=args.only_retrieval,
                 result_json_path=args.result_json)


if __name__ == "__main__":
    main()