# 基于 Contrast 对比损失训练后的模型作为索引进行知识库辅助的语义识别性能增强
import sys
import os
import cv2
import clip
from PIL import Image
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
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


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


class Tee:
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
    root = "/home/Users/dqy/Dataset/ImageNet100/train/"
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
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

def reverse_transform(images):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)  # 转为 [1, C, 1, 1] 方便广播
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
    tensor_denormalized = images * std + mean  # 恢复原始范围 [0, 1]
    image_np = tensor_denormalized.cpu().numpy().transpose(0, 2, 3, 1) * 255  # BxCxHxW -> BxHxWxC
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)  # 确保值在 [0, 255]
    return [Image.fromarray(image) for image in image_np]

def load_CLIP_model(device):
    MODEL_NAME = "ViT-L/14"
    model, preprocess = clip.load(MODEL_NAME, device=device)
    model.eval()
    return model, preprocess

def load_Contrast_model(device, local_rank):
    model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    # 修改分类头以适应100个类别
    model.heads.head = nn.Linear(model.heads.head.in_features, 100)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])
    model.load_state_dict(torch.load("/home/Users/dqy/Projects/ViT/checkpoints_Contrast/base/contrast_model#bestLoss@47.pth", map_location=device))
    return model

def get_CLIP_feature(model, preprocess, images_PIL, device):
    features = []
    for image in images_PIL:
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = model.encode_image(image).squeeze()
            features.append(feature)
    return torch.stack(features)

def retrieval_Faiss(Faiss_folder, query_feature, k=5):
    # 获取 Faiss 检索文件的路径
    index_path = os.path.join(Faiss_folder, "Contrast_index.faiss")
    source_path = os.path.join(Faiss_folder, "index_paths.npy")
    # 加载索引和路径映射
    # print("Loading Faiss database...")
    index = faiss.read_index(index_path)
    feature_paths = np.load(source_path, allow_pickle=True)
    # 迁移到 GPU
    # print("Converting Faiss database to GPU...")
    # res = faiss.StandardGpuResources()
    # index = faiss.index_cpu_to_gpu(res, 0, index)
    # 对给定的特征进行查询
    faiss.normalize_L2(query_feature)
    # print("Retrieving...")
    D, I = index.search(query_feature, k)  # D=相似度, I=索引
    # 输出前 k 个相似图像的路径
    # for rank, idx in enumerate(I[0]):  # 这里的 [0] 表示对第 0 个查询向量的检索结果
    #     print(f"Top-{rank+1}: {feature_paths[idx]} (similarity={D[0][rank]:.4f})")
    return D, I


def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))  # 减去最大值防止数值溢出
    return e_x / e_x.sum(axis=0)


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


# def train(model, train_loader, criterion, optimizer, scheduler, device):
#     model.train()
#     total_loss = 0
#     for images, labels, _ in tqdm.tqdm(train_loader, total=len(train_loader)):
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     scheduler.step()
#     return total_loss


def validate(model, val_loader, criterion, label_map, device, extract_features=False, feature_save_dir=None,
             knowledge_base="", prior_weight=0.0, retrieval_k=5, only_retrieval=False, result_json_path="", local_rank=0):
    model.eval()
    all_preds, all_labels = [], []
    results = []
    total_loss = 0.0
    num_samples = 0

    if knowledge_base:
        # CLIP_model, CLIP_preprocess = load_CLIP_model(device)
        Contrast_model = load_Contrast_model(device, local_rank)
        Contrast_model.eval()

    with torch.no_grad():
        for images, labels, img_names in tqdm.tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            if not knowledge_base:
                if extract_features and feature_save_dir:
                    feats = get_vit_features(model.module, images)
                    outputs = model.module.heads(feats)
                    feats = feats.view(feats.size(0), -1).cpu().numpy()
                    for feat, name in zip(feats, img_names):
                        np.save(os.path.join(feature_save_dir, f"{name.split('.')[0]}.npy"), feat)
                else:
                    outputs = model(images)
            else:
                # 建立 Faiss 索引库
                faiss_folder = os.path.join(knowledge_base, "Faiss_index")
                assert os.path.exists(os.path.join(faiss_folder, "Contrast_index.faiss"))
                assert os.path.exists(os.path.join(faiss_folder, "index_paths.npy"))
                assert os.path.exists(os.path.join(knowledge_base, "Contrast_features"))
                assert os.path.exists(os.path.join(knowledge_base, "ViT_features"))
                Contrast_paths = np.load(os.path.join(faiss_folder, "index_paths.npy"), allow_pickle=True)
                # 提取图像 Contrast 特征以供索引
                Contrast_features = get_vit_features(Contrast_model.module, images).cpu().numpy().astype('float32')
                # 通过 Faiss 向量库进行索引
                D, I = retrieval_Faiss(faiss_folder, Contrast_features, k=retrieval_k)
                # 仅基于检索结果进行分类
                if only_retrieval:
                    # 从检索结果中推断类别
                    for i, (dists, indices) in enumerate(zip(D, I)):
                        label_count = {}
                        label_dists = {}
                        for dist, idx in zip(dists, indices):
                            path = Contrast_paths[idx]  # e.g., "kernal_0#n01775062_4379.pt"
                            filename = os.path.basename(path)
                            category = filename.split("#")[1].split("_")[0]
                            label = label_map[category]

                            label_count[label] = label_count.get(label, 0) + 1
                            label_dists.setdefault(label, []).append(dist)

                        # 找到出现频率最高的类别标签
                        max_count = max(label_count.values())
                        candidates = [lbl for lbl, count in label_count.items() if count == max_count]

                        if len(candidates) == 1:
                            final_label = candidates[0]
                        else:
                            # 多个频率最高的候选标签，选平均相似度最高的
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
                    continue  # 跳过后续标准模型推理流程
                # 否则，根据索引结果获取 ViT 特征并进行加权
                feats = get_vit_features(model.module, images)
                features_prior = []
                for similarities, indexes in zip(D, I):  # 处理 batch 中各个样本的检索结果
                    weights = softmax(similarities)
                    ViT_features = []
                    for index in indexes:
                        Contrast_path = Contrast_paths[index]
                        feature_path = Contrast_path.replace("Contrast_features", "ViT_features")
                        ViT_features.append(np.load(feature_path))
                    weighted_prior = np.sum(weights[:, None] * np.stack(ViT_features), axis=0)
                    features_prior.append(weighted_prior)
                # 根据加权结果进一步调整图像特征
                feats = feats * (1 - prior_weight) + torch.Tensor(np.array(features_prior)).to(device) * prior_weight
                outputs = model.module.heads(feats)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
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
    # print(report)
    if result_json_path:
        import json
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Prediction results saved to: {result_json_path}")
    return cm, avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'val'], required=True)
    parser.add_argument('--train_txt', type=str, default='')
    parser.add_argument('--val_txt', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--extract_features', action='store_true')
    # 知识库参数
    parser.add_argument('--knowledge_base', type=str, default='')
    parser.add_argument('--retrieval_k', type=int, default=5)
    parser.add_argument('--prior_weight', type=float, default=0.0)
    parser.add_argument('--only_retrieval', action='store_true')
    # 结果保存参数
    parser.add_argument('--result_json', type=str, default='', help='Path to save validation results in JSON format')
    args = parser.parse_args()

    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    dist_rank = dist.get_rank()

    os.makedirs(os.path.join(args.save_dir, args.exp_name), exist_ok=True)
    sys.stdout = Tee(sys.stdout, open(os.path.join(args.save_dir, args.exp_name, 'log.txt'), 'w', encoding='utf-8'))

    transform = get_transform()
    # 使用Vision Transformer替换ViT
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
        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
        
        if args.val_txt:
            val_dataset = CustomImageDataset(args.val_txt, transform, label_map)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
            best_loss, best_epoch = 10000, -1
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        for epoch in range(args.epochs):
            train_sampler.set_epoch(epoch)
            loss = train(model, train_loader, criterion, optimizer, scheduler, device)
            if dist_rank == 0:
                print(f"[Epoch {epoch+1}] Train Loss: {loss:.4f}")
                model_save_path = os.path.join(args.save_dir, args.exp_name, f'model_epoch_{epoch:02d}.pth')
                torch.save(model.state_dict(), model_save_path)
                print(f"[INFO] Model saved to {model_save_path}")
                if args.val_txt:
                    print(f"\n[INFO] Evaluating...")
                    cm, loss = validate(model, val_loader, criterion, label_map, device)
                    if loss < best_loss:
                        best_loss = loss
                        best_epoch = epoch
                        torch.save(model.state_dict(), os.path.join(args.save_dir, args.exp_name, f'model_best.pth'))
                    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(label_map.keys()))
                    # fig, ax = plt.subplots(figsize=(6, 6))
                    # disp.plot(ax=ax, cmap="Blues", values_format=".0f")
                    # plt.title(f"Confusion Matrix")
                    # plt.tight_layout()
                    # cm_path = os.path.join(args.save_dir, args.exp_name, f"cm#EP{epoch:03d}.png")
                    # plt.savefig(cm_path)
                    # plt.close()
                    # print(f"[INFO] Confusion matrix saved to: {cm_path}")

    elif args.mode == 'val':
        assert args.val_txt, "--val_txt is required in val mode"
        if args.extract_features:
            os.makedirs(os.path.join(args.save_dir, args.exp_name, "ViT_features"), exist_ok=True)
        label_map = name_to_label()
        val_dataset = CustomImageDataset(args.val_txt, transform, label_map)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        criterion = nn.CrossEntropyLoss()
        cm, loss = validate(model, val_loader, criterion, label_map, device, extract_features=args.extract_features,
                 feature_save_dir=os.path.join(args.save_dir, args.exp_name, "ViT_features"),
                 knowledge_base=args.knowledge_base,
                 prior_weight=args.prior_weight,
                 retrieval_k=args.retrieval_k,
                 only_retrieval=args.only_retrieval,
                 result_json_path=args.result_json,
                 local_rank=local_rank)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(label_map.keys()))
        # fig, ax = plt.subplots(figsize=(6, 6))
        # disp.plot(ax=ax, cmap="Blues", values_format=".0f")
        # plt.title(f"Confusion Matrix")
        # plt.tight_layout()
        # cm_path = os.path.join(args.save_dir, args.exp_name, f"cm.png")
        # plt.savefig(cm_path)
        # plt.close()
        # print(f"[INFO] Confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    main()
