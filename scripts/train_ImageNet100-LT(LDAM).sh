dataset="ImageNet100-LT"
CUDA_VISIBLE_DEVICES=4,5 OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 --rdzv_endpoint=localhost:2999 \
    "/home/Users/dqy/Projects/ViT/main_${dataset}.py" \
    --mode 'train' \
    --train_txt "/home/Users/dqy/Dataset/${dataset}/format_ImageNet/images/train.txt" \
    --val_txt "/home/Users/dqy/Dataset/${dataset}/format_ImageNet/images/val.txt" \
    --batch_size 128 --epochs 300 \
    --lr 1e-3 \
    --save_dir "/home/Users/dqy/Projects/ViT/checkpoints_${dataset}/" \
    --exp_name 'with_ldam' \
    --weight_decay 0.05 \
    --warmup_epochs 20 \
    --use_ldam \
    --max_margin 0.5 \
    --scale 30 \
    --drw_epoch 160 \
