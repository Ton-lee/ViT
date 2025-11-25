# 通过预训练模型，对不同失真类型与失真等级的验证集图像进行测试

ROOT="/home/Users/dqy/Dataset/ImageNet100/images_distorted/"
checkpoint="/home/Users/dqy/Projects/ViT/checkpoints/base/ViT_model#bestF1@02.pth"
SAVE_ROOT="/home/Users/dqy/Projects/ViT/results/distorted/"
script_path="/home/Users/dqy/Projects/ViT/main.py"

# 高斯模糊
params=(5 7 10 15)
for param in "${params[@]}"; do
	name=blur@${param}
	dataset="${ROOT}/${name}/val.txt"
	CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 --rdzv_endpoint=localhost:29800 "${script_path}" \
		--mode 'val' \
		--val_txt "${dataset}" \
		--batch_size 64 \
		--save_dir "${SAVE_ROOT}" \
		--exp_name "${name}" \
		--checkpoint "${checkpoint}"
done

# JPEG 压缩
params=(40 30 20 10)
for param in "${params[@]}"; do
	name="compression@${param}"
	dataset="${ROOT}/${name}/val.txt"
	CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 --rdzv_endpoint=localhost:29800 "${script_path}" \
		--mode 'val' \
		--val_txt "${dataset}" \
		--batch_size 64 \
		--save_dir "${SAVE_ROOT}" \
		--exp_name "${name}" \
		--checkpoint "${checkpoint}"
done

# 运动模糊
params=(10 15 20 30)
for param in "${params[@]}"; do
	name="motion_blur@${param}"
	dataset="${ROOT}/${name}/val.txt"
	CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 --rdzv_endpoint=localhost:29800 "${script_path}" \
		--mode 'val' \
		--val_txt "${dataset}" \
		--batch_size 64 \
		--save_dir "${SAVE_ROOT}" \
		--exp_name "${name}" \
		--checkpoint "${checkpoint}"
done

# 光照失真
params=(30 50 70 100)
for param in "${params[@]}"; do
	name="illumination@${param}"
	dataset="${ROOT}/${name}/val.txt"
	CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 --rdzv_endpoint=localhost:29800 "${script_path}" \
		--mode 'val' \
		--val_txt "${dataset}" \
		--batch_size 64 \
		--save_dir "${SAVE_ROOT}" \
		--exp_name "${name}" \
		--checkpoint "${checkpoint}"
done