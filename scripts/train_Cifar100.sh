dataset="Cifar100"
CUDA_VISIBLE_DEVICES=4,5 OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 --rdzv_endpoint=localhost:2999 \
    "/home/Users/dqy/Projects/ViT/main_${dataset}.py" \
    --mode 'train' \
    --train_txt "/home/Users/dqy/Dataset/${dataset}/format_ImageNet/images/train.txt" \
    --val_txt "/home/Users/dqy/Dataset/${dataset}/format_ImageNet/images/val.txt" \
    --batch_size 128 --epochs 300 \
    --lr 1e-3 \
    --save_dir "/home/Users/dqy/Projects/ViT/checkpoints_${dataset}/" \
    --exp_name 'base' \
    --weight_decay 1e-4 \
    --warmup_epochs 10 \
