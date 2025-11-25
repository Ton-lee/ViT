#!/bin/bash


trap 'cleanup' INT

cleanup() {
    echo -e "\n[!] 用户中断 (Ctrl+C)。终止所有子进程..."
    # 杀死所有相关后台进程
    pkill -P $$  # 杀死当前脚本的所有子进程
    pkill -f "torchrun"   # 确保 torchrun 进程被杀死（可选）
    exit 1       # 退出脚本
    echo -e "\n[!] 用户中断 (Ctrl+C)。已终止所有子进程"
}

# 命令行参数解析（-t [type]）
while getopts "t:" opt; do
  case $opt in
    t) distortion_type="$OPTARG" ;;
    *) echo "Usage: $0 -t <distortion_type>"; exit 1 ;;
  esac
done

# 检查是否提供了必要的参数
if [ -z "$distortion_type" ]; then
  echo "Error: distortion_type not specified"
  echo "Usage: $0 -t <distortion_type>"
  exit 1
fi


checkpoint="/home/Users/dqy/Projects/ViT/checkpoints/base/ViT_model#bestF1@02.pth"
save_root="/home/Users/dqy/Projects/ViT/results/distorted_with_knowledge"

# 测试高斯模糊的性能结果
if [ "$distortion_type" = "blur" ]; then
	output_file="${save_root}/performance_${distortion_type}.csv"
	mkdir -p "${save_root}"
	echo "distortion_type,distortion_param,prior_weight,knowledge_type,knowledge_K,retrieval_k,performance" > "${output_file}"
	device=3
	port=29510

	# 进度统计变量
	start_time=$(date +%s)
	total_iterations=$((4 * 5 * 2 * 5 * 5))  # 5*4*2*5*5=1000种组合
	current_iteration=0

	# 计算剩余时间函数
	calculate_remaining_time() {
	    local elapsed=$1
	    local completed=$2
	    local total=$3
	    if [ $completed -eq 0 ]; then
	        echo "N/A"
	    else
	        local remaining=$(( (total - completed) * elapsed / completed ))
	        printf "%02d:%02d:%02d" $((remaining/3600)) $(( (remaining%3600)/60 )) $((remaining%60))
	    fi
	}

	for distortion_param in 5 7 10 15; do
	    distortion_name="${distortion_type}@${distortion_param}"
	    for prior_weight in "0.1" "0.2" "0.3" "0.4" "0.5"; do
	        for knowledge_type in "GMM" "GMM_category"; do
	            for knowledge_K in 2000 1000 500 200 100; do
	                knowledge_name="${knowledge_type}@K=${knowledge_K}"
	                knowledge_path="/home/Users/dqy/Dataset/ImageNet100/KnowledgeBase_train/${knowledge_name}/"
	                for retrieval_k in 1 2 3 4 5; do
	                    current_iteration=$((current_iteration + 1))
	                    current_time=$(date +%s)
	                    elapsed=$((current_time - start_time))
	                    
	                    # 计算进度百分比
	                    progress=$((100 * current_iteration / total_iterations))
	                    
	                    # 计算剩余时间
	                    remaining=$(calculate_remaining_time $elapsed $current_iteration $total_iterations)
	                    
	                    # 显示进度信息
	                    echo -ne "\r[${progress}%] Iteration ${current_iteration}/${total_iterations} | "
	                    echo -ne "Elapsed: $(printf "%02d:%02d:%02d" $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60))) | "
	                    echo -ne "Remaining: ${remaining} | "
	                    echo -ne "Current params: ${distortion_type}@${distortion_param}, w=${prior_weight}, ${knowledge_type}@K=${knowledge_K}, top-${retrieval_k}"
	                    echo ""
	                    # 运行模型
	                    CUDA_VISIBLE_DEVICES=${device} OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 --rdzv_endpoint=localhost:${port} --master_port=${port} "/home/Users/dqy/Projects/ViT/main.py" \
	                        --mode 'val' \
	                        --val_txt "/home/Users/dqy/Dataset/ImageNet100/images_distorted/${distortion_name}/val.txt" \
	                        --batch_size 32 \
	                        --save_dir "${save_root}/weight=${prior_weight}/${knowledge_name}/top-${retrieval_k}" \
	                        --exp_name "${distortion_name}" --checkpoint ${checkpoint} \
	                        --knowledge_base "$knowledge_path" --prior_weight ${prior_weight} \
	                        --retrieval_k ${retrieval_k}
	                    
	                    # 从日志文件中提取性能指标
	                    log_file="${save_root}/weight=${prior_weight}/${knowledge_name}/top-${retrieval_k}/${distortion_name}/log.txt"
	                    if [ -f "${log_file}" ]; then
	                        # 获取第4行，然后取最后一个空格后的内容
	                        performance=$(sed -n '4p' "${log_file}" | awk '{print $NF}')
	                        
	                        # 将结果写入CSV文件
	                        echo "${distortion_type},${distortion_param},${prior_weight},${knowledge_type},${knowledge_K},${retrieval_k},${performance}" >> "${output_file}"
	                    else
	                        echo "Log file not found: ${log_file}"
	                        echo "${distortion_type},${distortion_param},${prior_weight},${knowledge_type},${knowledge_K},${retrieval_k},NA" >> "${output_file}"
	                    fi
	                done
	            done
	        done
	    done
	done
	echo -e "\nAll tests completed. Results saved to ${output_file}"
fi

# 测试压缩失真的性能结果
if [ "$distortion_type" = "compression" ]; then
	output_file="${save_root}/performance_${distortion_type}.csv"
	mkdir -p "${save_root}"
	echo "distortion_type,distortion_param,prior_weight,knowledge_type,knowledge_K,retrieval_k,performance" > "${output_file}"
	device=4
	port=29511

	# 进度统计变量
	start_time=$(date +%s)
	total_iterations=$((4 * 5 * 2 * 5 * 5))  # 5*4*2*5*5=1000种组合
	current_iteration=0

	# 计算剩余时间函数
	calculate_remaining_time() {
	    local elapsed=$1
	    local completed=$2
	    local total=$3
	    if [ $completed -eq 0 ]; then
	        echo "N/A"
	    else
	        local remaining=$(( (total - completed) * elapsed / completed ))
	        printf "%02d:%02d:%02d" $((remaining/3600)) $(( (remaining%3600)/60 )) $((remaining%60))
	    fi
	}

	for distortion_param in 40 30 20 10; do
	    distortion_name="${distortion_type}@${distortion_param}"
	    for prior_weight in "0.1" "0.2" "0.3" "0.4" "0.5"; do
	        for knowledge_type in "GMM" "GMM_category"; do
	            for knowledge_K in 2000 1000 500 200 100; do
	                knowledge_name="${knowledge_type}@K=${knowledge_K}"
	                knowledge_path="/home/Users/dqy/Dataset/ImageNet100/KnowledgeBase_train/${knowledge_name}/"
	                for retrieval_k in 1 2 3 4 5; do
	                    current_iteration=$((current_iteration + 1))
	                    current_time=$(date +%s)
	                    elapsed=$((current_time - start_time))
	                    
	                    # 计算进度百分比
	                    progress=$((100 * current_iteration / total_iterations))
	                    
	                    # 计算剩余时间
	                    remaining=$(calculate_remaining_time $elapsed $current_iteration $total_iterations)
	                    
	                    # 显示进度信息
	                    echo -ne "\r[${progress}%] Iteration ${current_iteration}/${total_iterations} | "
	                    echo -ne "Elapsed: $(printf "%02d:%02d:%02d" $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60))) | "
	                    echo -ne "Remaining: ${remaining} | "
	                    echo -ne "Current params: ${distortion_type}@${distortion_param}, w=${prior_weight}, ${knowledge_type}@K=${knowledge_K}, top-${retrieval_k}"
	                    echo ""
	                    # 运行模型
	                    CUDA_VISIBLE_DEVICES=${device} OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 --rdzv_endpoint=localhost:${port} --master_port=${port} "/home/Users/dqy/Projects/ViT/main.py" \
	                        --mode 'val' \
	                        --val_txt "/home/Users/dqy/Dataset/ImageNet100/images_distorted/${distortion_name}/val.txt" \
	                        --batch_size 32 \
	                        --save_dir "${save_root}/weight=${prior_weight}/${knowledge_name}/top-${retrieval_k}" \
	                        --exp_name "${distortion_name}" --checkpoint ${checkpoint} \
	                        --knowledge_base "$knowledge_path" --prior_weight ${prior_weight} \
	                        --retrieval_k ${retrieval_k}
	                    
	                    # 从日志文件中提取性能指标
	                    log_file="${save_root}/weight=${prior_weight}/${knowledge_name}/top-${retrieval_k}/${distortion_name}/log.txt"
	                    if [ -f "${log_file}" ]; then
	                        # 获取第4行，然后取最后一个空格后的内容
	                        performance=$(sed -n '4p' "${log_file}" | awk '{print $NF}')
	                        
	                        # 将结果写入CSV文件
	                        echo "${distortion_type},${distortion_param},${prior_weight},${knowledge_type},${knowledge_K},${retrieval_k},${performance}" >> "${output_file}"
	                    else
	                        echo "Log file not found: ${log_file}"
	                        echo "${distortion_type},${distortion_param},${prior_weight},${knowledge_type},${knowledge_K},${retrieval_k},NA" >> "${output_file}"
	                    fi
	                done
	            done
	        done
	    done
	done
	echo -e "\nAll tests completed. Results saved to ${output_file}"
fi


# 测试运动模糊的性能结果
if [ "$distortion_type" = "motion_blur" ]; then
	output_file="${save_root}/performance_${distortion_type}.csv"
	mkdir -p "${save_root}"
	echo "distortion_type,distortion_param,prior_weight,knowledge_type,knowledge_K,retrieval_k,performance" > "${output_file}"
	device=5
	port=29512

	# 进度统计变量
	start_time=$(date +%s)
	total_iterations=$((4 * 5 * 2 * 5 * 5))  # 5*4*2*5*5=1000种组合
	current_iteration=0

	# 计算剩余时间函数
	calculate_remaining_time() {
	    local elapsed=$1
	    local completed=$2
	    local total=$3
	    if [ $completed -eq 0 ]; then
	        echo "N/A"
	    else
	        local remaining=$(( (total - completed) * elapsed / completed ))
	        printf "%02d:%02d:%02d" $((remaining/3600)) $(( (remaining%3600)/60 )) $((remaining%60))
	    fi
	}

	for distortion_param in 10 15 20 30; do
	    distortion_name="${distortion_type}@${distortion_param}"
	    for prior_weight in "0.1" "0.2" "0.3" "0.4" "0.5"; do
	        for knowledge_type in "GMM" "GMM_category"; do
	            for knowledge_K in 2000 1000 500 200 100; do
	                knowledge_name="${knowledge_type}@K=${knowledge_K}"
	                knowledge_path="/home/Users/dqy/Dataset/ImageNet100/KnowledgeBase_train/${knowledge_name}/"
	                for retrieval_k in 1 2 3 4 5; do
	                    current_iteration=$((current_iteration + 1))
	                    current_time=$(date +%s)
	                    elapsed=$((current_time - start_time))
	                    
	                    # 计算进度百分比
	                    progress=$((100 * current_iteration / total_iterations))
	                    
	                    # 计算剩余时间
	                    remaining=$(calculate_remaining_time $elapsed $current_iteration $total_iterations)
	                    
	                    # 显示进度信息
	                    echo -ne "\r[${progress}%] Iteration ${current_iteration}/${total_iterations} | "
	                    echo -ne "Elapsed: $(printf "%02d:%02d:%02d" $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60))) | "
	                    echo -ne "Remaining: ${remaining} | "
	                    echo -ne "Current params: ${distortion_type}@${distortion_param}, w=${prior_weight}, ${knowledge_type}@K=${knowledge_K}, top-${retrieval_k}"
	                    echo ""
	                    # 运行模型
	                    CUDA_VISIBLE_DEVICES=${device} OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 --rdzv_endpoint=localhost:${port} --master_port=${port} "/home/Users/dqy/Projects/ViT/main.py" \
	                        --mode 'val' \
	                        --val_txt "/home/Users/dqy/Dataset/ImageNet100/images_distorted/${distortion_name}/val.txt" \
	                        --batch_size 32 \
	                        --save_dir "${save_root}/weight=${prior_weight}/${knowledge_name}/top-${retrieval_k}" \
	                        --exp_name "${distortion_name}" --checkpoint ${checkpoint} \
	                        --knowledge_base "$knowledge_path" --prior_weight ${prior_weight} \
	                        --retrieval_k ${retrieval_k}
	                    
	                    # 从日志文件中提取性能指标
	                    log_file="${save_root}/weight=${prior_weight}/${knowledge_name}/top-${retrieval_k}/${distortion_name}/log.txt"
	                    if [ -f "${log_file}" ]; then
	                        # 获取第4行，然后取最后一个空格后的内容
	                        performance=$(sed -n '4p' "${log_file}" | awk '{print $NF}')
	                        
	                        # 将结果写入CSV文件
	                        echo "${distortion_type},${distortion_param},${prior_weight},${knowledge_type},${knowledge_K},${retrieval_k},${performance}" >> "${output_file}"
	                    else
	                        echo "Log file not found: ${log_file}"
	                        echo "${distortion_type},${distortion_param},${prior_weight},${knowledge_type},${knowledge_K},${retrieval_k},NA" >> "${output_file}"
	                    fi
	                done
	            done
	        done
	    done
	done
	echo -e "\nAll tests completed. Results saved to ${output_file}"
fi


# 测试光照失真的性能结果
if [ "$distortion_type" = "illumination" ]; then
	output_file="${save_root}/performance_${distortion_type}.csv"
	mkdir -p "${save_root}"
	echo "distortion_type,distortion_param,prior_weight,knowledge_type,knowledge_K,retrieval_k,performance" > "${output_file}"
	device=6
	port=29513

	# 进度统计变量
	start_time=$(date +%s)
	total_iterations=$((4 * 5 * 2 * 5 * 5))  # 5*4*2*5*5=1000种组合
	current_iteration=0

	# 计算剩余时间函数
	calculate_remaining_time() {
	    local elapsed=$1
	    local completed=$2
	    local total=$3
	    if [ $completed -eq 0 ]; then
	        echo "N/A"
	    else
	        local remaining=$(( (total - completed) * elapsed / completed ))
	        printf "%02d:%02d:%02d" $((remaining/3600)) $(( (remaining%3600)/60 )) $((remaining%60))
	    fi
	}

	for distortion_param in 30 50 70 100; do
	    distortion_name="${distortion_type}@${distortion_param}"
	    for prior_weight in "0.1" "0.2" "0.3" "0.4" "0.5"; do
	        for knowledge_type in "GMM" "GMM_category"; do
	            for knowledge_K in 2000 1000 500 200 100; do
	                knowledge_name="${knowledge_type}@K=${knowledge_K}"
	                knowledge_path="/home/Users/dqy/Dataset/ImageNet100/KnowledgeBase_train/${knowledge_name}/"
	                for retrieval_k in 1 2 3 4 5; do
	                    current_iteration=$((current_iteration + 1))
	                    current_time=$(date +%s)
	                    elapsed=$((current_time - start_time))
	                    
	                    # 计算进度百分比
	                    progress=$((100 * current_iteration / total_iterations))
	                    
	                    # 计算剩余时间
	                    remaining=$(calculate_remaining_time $elapsed $current_iteration $total_iterations)
	                    
	                    # 显示进度信息
	                    echo -ne "\r[${progress}%] Iteration ${current_iteration}/${total_iterations} | "
	                    echo -ne "Elapsed: $(printf "%02d:%02d:%02d" $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60))) | "
	                    echo -ne "Remaining: ${remaining} | "
	                    echo -ne "Current params: ${distortion_type}@${distortion_param}, w=${prior_weight}, ${knowledge_type}@K=${knowledge_K}, top-${retrieval_k}"
	                    echo ""
	                    # 运行模型
	                    CUDA_VISIBLE_DEVICES=${device} OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 --rdzv_endpoint=localhost:${port} --master_port=${port} "/home/Users/dqy/Projects/ViT/main.py" \
	                        --mode 'val' \
	                        --val_txt "/home/Users/dqy/Dataset/ImageNet100/images_distorted/${distortion_name}/val.txt" \
	                        --batch_size 32 \
	                        --save_dir "${save_root}/weight=${prior_weight}/${knowledge_name}/top-${retrieval_k}" \
	                        --exp_name "${distortion_name}" --checkpoint ${checkpoint} \
	                        --knowledge_base "$knowledge_path" --prior_weight ${prior_weight} \
	                        --retrieval_k ${retrieval_k}
	                    
	                    # 从日志文件中提取性能指标
	                    log_file="${save_root}/weight=${prior_weight}/${knowledge_name}/top-${retrieval_k}/${distortion_name}/log.txt"
	                    if [ -f "${log_file}" ]; then
	                        # 获取第4行，然后取最后一个空格后的内容
	                        performance=$(sed -n '4p' "${log_file}" | awk '{print $NF}')
	                        
	                        # 将结果写入CSV文件
	                        echo "${distortion_type},${distortion_param},${prior_weight},${knowledge_type},${knowledge_K},${retrieval_k},${performance}" >> "${output_file}"
	                    else
	                        echo "Log file not found: ${log_file}"
	                        echo "${distortion_type},${distortion_param},${prior_weight},${knowledge_type},${knowledge_K},${retrieval_k},NA" >> "${output_file}"
	                    fi
	                done
	            done
	        done
	    done
	done
	echo -e "\nAll tests completed. Results saved to ${output_file}"
fi

# 测试原始数据的性能结果
if [ "$distortion_type" = "none" ]; then
	distortion_param=0
	output_file="${save_root}/performance_${distortion_type}.csv"
	mkdir -p "${save_root}"
	echo "distortion_type,distortion_param,prior_weight,knowledge_type,knowledge_K,retrieval_k,performance" > "${output_file}"
	device=4
	port=29514

	# 进度统计变量
	start_time=$(date +%s)
	total_iterations=$((5 * 2 * 5 * 5))  # 4*2*5*5=200种组合
	current_iteration=0

	# 计算剩余时间函数
	calculate_remaining_time() {
	    local elapsed=$1
	    local completed=$2
	    local total=$3
	    if [ $completed -eq 0 ]; then
	        echo "N/A"
	    else
	        local remaining=$(( (total - completed) * elapsed / completed ))
	        printf "%02d:%02d:%02d" $((remaining/3600)) $(( (remaining%3600)/60 )) $((remaining%60))
	    fi
	}

    distortion_name="${distortion_type}"
    for prior_weight in "0.1" "0.2" "0.3" "0.4" "0.5"; do
        for knowledge_type in "GMM" "GMM_category"; do
            for knowledge_K in 2000 1000 500 200 100; do
                knowledge_name="${knowledge_type}@K=${knowledge_K}"
                knowledge_path="/home/Users/dqy/Dataset/ImageNet100/KnowledgeBase_train/${knowledge_name}/"
                for retrieval_k in 1 2 3 4 5; do
                    current_iteration=$((current_iteration + 1))
                    current_time=$(date +%s)
                    elapsed=$((current_time - start_time))
                    
                    # 计算进度百分比
                    progress=$((100 * current_iteration / total_iterations))
                    
                    # 计算剩余时间
                    remaining=$(calculate_remaining_time $elapsed $current_iteration $total_iterations)
                    
                    # 显示进度信息
                    echo -ne "\r[${progress}%] Iteration ${current_iteration}/${total_iterations} | "
                    echo -ne "Elapsed: $(printf "%02d:%02d:%02d" $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60))) | "
                    echo -ne "Remaining: ${remaining} | "
                    echo -ne "Current params: ${distortion_type}@${distortion_param}, w=${prior_weight}, ${knowledge_type}@K=${knowledge_K}, top-${retrieval_k}"
                    echo ""
                    # 运行模型
                    CUDA_VISIBLE_DEVICES=${device} torchrun --nproc_per_node=1 --rdzv_endpoint=localhost:${port} --master_port=${port} "/home/Users/dqy/Projects/ViT/main.py" \
                        --mode 'val' \
                        --val_txt "/home/Users/dqy/Dataset/ImageNet100/images/val.txt" \
                        --batch_size 32 \
                        --save_dir "${save_root}/weight=${prior_weight}/${knowledge_name}/top-${retrieval_k}" \
                        --exp_name "${distortion_name}" --checkpoint ${checkpoint} \
                        --knowledge_base "$knowledge_path" --prior_weight ${prior_weight} \
                        --retrieval_k ${retrieval_k}
                    
                    # 从日志文件中提取性能指标
                    log_file="${save_root}/weight=${prior_weight}/${knowledge_name}/top-${retrieval_k}/${distortion_name}/log.txt"
                    if [ -f "${log_file}" ]; then
                        # 获取第4行，然后取最后一个空格后的内容
                        performance=$(sed -n '4p' "${log_file}" | awk '{print $NF}')
                        
                        # 将结果写入CSV文件
                        echo "${distortion_type},${distortion_param},${prior_weight},${knowledge_type},${knowledge_K},${retrieval_k},${performance}" >> "${output_file}"
                    else
                        echo "Log file not found: ${log_file}"
                        echo "${distortion_type},${distortion_param},${prior_weight},${knowledge_type},${knowledge_K},${retrieval_k},NA" >> "${output_file}"
                    fi
                done
            done
        done
    done
	echo -e "\nAll tests completed. Results saved to ${output_file}"
fi