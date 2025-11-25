%% 数据读取
% 以G:\Projects\ViT\results\distorted_with_knowledge为根目录，根目录中文件
% performance_blur.csv、performance_compression.csv、performance_illumination.csv、
% performance_motion_blur.csv分别记录了在模糊、压缩、光照和运动模糊四种情况下的模型性能。
% 每个文件中第一行为表头，每一列分别表示失真类型（字符串）、失真参数（整数）、先验加权系数（浮点数）、聚类方法（字符串）、
% 知识库规模（整数）、检索数量（整数）和分类F1性能（浮点数）。以适当的方式读取上述文件并记录读取结果，用于后续可视化处理。
veryROOT = 'G:\Projects\ViT\results\';
% 设置根目录
rootDir = [veryROOT, 'distorted_with_knowledge_Contrast'];
% 设置保存目录
saveDir = 'G:\Projects\ViT\analysis\performance_knowledge_Contrast';
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end
% 文件名列表
fileNames = {
    'performance_blur.csv', ...
    'performance_compression.csv', ...
    'performance_illumination.csv', ...
    'performance_motion_blur.csv'
};
% 结构体字段名（用于区分不同失真类型）
distortionTypes = {
    'blur', ...
    'compression', ...
    'illumination', ...
    'motion_blur'
};
% 初始化结果结构体
performanceData = struct();
% 循环读取每个文件
disp("读取性能文件：失真数据集+知识库辅助识别");
for i = 1:length(fileNames)
    filePath = fullfile(rootDir, fileNames{i});
    
    % 读取表格数据
    dataTable = readtable(filePath);

    % 存入结构体中
    performanceData.(distortionTypes{i}) = dataTable;
    
    % 可选：打印读取情况
    fprintf('读取 %s 成功，共 %d 行。\n', filePath, height(dataTable));
end

baseline_data = {
    'blur',         5,   0.8555;
    'blur',         7,   0.8384;
    'blur',         10,  0.8201;
    'blur',         15,  0.8059;
    'compression',  40,  0.8465;
    'compression',  30,  0.8372;
    'compression',  20,  0.8259;
    'compression',  10,  0.7914;
    'motion_blur',  10,  0.8304;
    'motion_blur',  15,  0.8005;
    'motion_blur',  20,  0.7620;
    'motion_blur',  30,  0.6753;
    'illumination', 30,  0.8684;
    'illumination', 50,  0.8680;
    'illumination', 70,  0.8602;
    'illumination', 100, 0.8426
};
baseline_table = cell2table(baseline_data, ...
    'VariableNames', {'distortion_type', 'distortion_param', 'baseline_f1'});
filePath_no_distortion = [veryROOT, 'distorted_with_knowledge_Contrast\performance_none.csv'];
disp("读取性能文件：无失真数据集+知识库辅助识别");
data_ori = readtable(filePath_no_distortion);
fprintf('读取 %s 成功，共 %d 行。\n', filePath_no_distortion, height(dataTable));
performanceData_ori = struct();
performanceData_ori.none = data_ori;
baseline_f1_ori = 0.8715;  % 无失真数据集基准性能

% 读取仅检索的性能（无失真数据集）
rootDir_ori_only_retrieval = [veryROOT, 'distorted_with_knowledge_Contrast_only_retrieval'];
fileNames = {
    'performance_none.csv'
};
distortionTypes = {
    'none'
};
performanceData_ori_only_retrieval = struct();
disp("读取性能文件：无失真数据集+仅检索匹配");
for i = 1:length(fileNames)
    filePath = fullfile(rootDir_ori_only_retrieval, fileNames{i});
    try
        dataTable = readtable(filePath);
        performanceData_ori_only_retrieval.(distortionTypes{i}) = dataTable;
        fprintf('读取 %s 成功，共 %d 行。\n', filePath, height(dataTable));
    catch
        fprintf('读取 %s 失败。\n', filePath);
    end
end

% 读取仅检索的性能（失真数据集）
rootDir_only_retrieval = [veryROOT, 'distorted_with_knowledge_Contrast_only_retrieval'];
fileNames = {
    'performance_blur.csv', ...
    'performance_compression.csv', ...
    'performance_illumination.csv', ...
    'performance_motion_blur.csv'
};
distortionTypes = {
    'blur', ...
    'compression', ...
    'illumination', ...
    'motion_blur'
};
performanceData_only_retrieval = struct();
disp("读取性能文件：失真数据集+仅检索匹配");
for i = 1:length(fileNames)
    filePath = fullfile(rootDir_only_retrieval, fileNames{i});
    try
        dataTable_only_retrieval = readtable(filePath);
        performanceData_only_retrieval.(distortionTypes{i}) = dataTable_only_retrieval;
        fprintf('读取 %s 成功，共 %d 行。\n', filePath, height(dataTable_only_retrieval));
    catch
        fprintf('读取 %s 失败。\n', filePath);
    end
end

% 带高先验权重的文件目录
dirs_with_high_weight = {
    [veryROOT, 'distorted_with_knowledge_Contrast'], ...
    [veryROOT, 'distorted_with_knowledge_Contrast_high_weight']
};

%% analysis_GMM 第一部分 - 可视化分析：GMM 与 GMM_category 性能完整展示
% 对上述数据进行可视化，通过多个窗口分别展示 knowledge_type 为 GMM 和 GMM_category
% 在不同失真类型的性能，每个窗口中每个图窗表示一个关键帧数量 knowledge_K 和一个知识库权重 
% prior_weight，将同一个关键帧数量对应的图窗排列在同一行，同一个知识库权重的图窗
% 排列的同一列。图窗内绘制多簇柱状图，将每个失真等级 distortion_param 绘制为一簇，
% 且在每簇中添加一个柱状图表示基准性能并以不同颜色进行区分。
% 每簇包含该失真在一个失真等级下对应不同检索数量 retrieval_k 的 F1 性能指标 performance。
% 由于图窗数量较多，应该注意适当处理图窗的标签以确保较好的展示效果。基准性能如下：
% XXX
% 对一个图窗中的各个绘图纵坐标调整范围，使柱状图之间的差距更明显，将结果以 PDF 格式保存到目录
% G:\Projects\ResNet\analysis_DETRAC\performance_knowledge 中
% === 1. 准备基准性能映射 ===
% 直接使用第一部分所加载的数值
% === 2. 可视化参数 ===
clustering_methods = {'GMM', 'GMM_category'};
distortion_types = fieldnames(performanceData);


for c = 1:length(clustering_methods)
    clustering = clustering_methods{c};

    for d = 1:length(distortion_types)
        distortion = distortion_types{d};
        tbl = performanceData.(distortion);

        % 筛选当前 clustering_method
        tbl = tbl(strcmp(tbl.knowledge_type, clustering), :);
        if isempty(tbl), continue; end

        unique_K = unique(tbl.knowledge_K);
        unique_PW = unique(tbl.prior_weight);
        unique_RK = unique(tbl.retrieval_k);
        param_list = sort(unique(tbl.distortion_param));
        nK = length(unique_K);
        nPW = length(unique_PW);
        nParams = length(param_list);
        nRK = length(unique_RK);

        % ========== 第一步：确定统一的 Y 轴范围 ==========
        global_min = inf;
        global_max = -inf;

        for i = 1:nK
            for j = 1:nPW
                K_val = unique_K(i);
                PW_val = unique_PW(j);

                for p = 1:nParams
                    param = param_list(p);
                    for rk_idx = 1:nRK
                        rk = unique_RK(rk_idx);
                        idx = tbl.knowledge_K == K_val & ...
                              abs(tbl.prior_weight - PW_val) < 1e-5 & ...
                              tbl.distortion_param == param & ...
                              tbl.retrieval_k == rk;
                        if any(idx)
                            global_min = min(global_min, tbl.performance(idx));
                            global_max = max(global_max, tbl.performance(idx));
                        end
                    end

                    % baseline
                    bidx = strcmp(baseline_table.distortion_type, distortion) & ...
                           baseline_table.distortion_param == param;
                    if any(bidx)
                        global_min = min(global_min, baseline_table.baseline_f1(bidx));
                        global_max = max(global_max, baseline_table.baseline_f1(bidx));
                    end
                end
            end
        end

        % 设定统一 ylim 范围
        y_min = max(0.3, 0.9 * global_min);
        y_max = min(1.0, 1.05 * global_max);

        % ========== 第二步：开始绘图 ==========
        fig = figure('Name', [clustering ' - ' distortion], ...
                     'NumberTitle', 'off', ...
                     'Position', [100, 100, 1600, 1000]);

        for i = 1:nK
            for j = 1:nPW
                K_val = unique_K(i);
                PW_val = unique_PW(j);

                subplotIdx = (i - 1) * nPW + j;
                ax = subplot(nK, nPW, subplotIdx);

                barMatrix = NaN(nParams, nRK + 1);
                for p = 1:nParams
                    param = param_list(p);
                    for rk_idx = 1:nRK
                        rk = unique_RK(rk_idx);
                        idx = tbl.knowledge_K == K_val & ...
                              abs(tbl.prior_weight - PW_val) < 1e-5 & ...
                              tbl.distortion_param == param & ...
                              tbl.retrieval_k == rk;
                        if any(idx)
                            barMatrix(p, rk_idx) = tbl.performance(idx);
                        end
                    end

                    % baseline
                    bidx = strcmp(baseline_table.distortion_type, distortion) & ...
                           baseline_table.distortion_param == param;
                    if any(bidx)
                        barMatrix(p, end) = baseline_table.baseline_f1(bidx);
                    end
                end

                % 绘图
                bar(barMatrix, 'grouped');
                colormap([lines(nRK); 0.6 0.6 0.6]);

                % 设置 Y 范围统一
                ylim([y_min, y_max]);

                % 标签设置
                title(sprintf('K=%d, w=%.2f', K_val, PW_val));
                xticks(1:nParams);
                xticklabels(arrayfun(@(x) sprintf('P%d', x), param_list, 'UniformOutput', false));
                xtickangle(45);
                grid on;
                if j == 1, ylabel('F1 Score'); end

%                 if i == 1 && j == nPW
%                     legend([arrayfun(@(x) sprintf('R%d', x), unique_RK, 'UniformOutput', false), {'Baseline'}], ...
%                         'Location', 'northeastoutside');
%                 end
            end
        end

        % ========== 第三步：保存为 PDF ==========
        pdfFile = fullfile(saveDir, sprintf('%s_%s.pdf', clustering, distortion));
        exportgraphics(fig, pdfFile, 'ContentType', 'vector');
%         close(fig);
    end
end

%% analysis_GMM 第二部分 - 可视化分析：GMM 与 GMM_category 平均性能对比
% 在不同失真类型和失真参数下，计算 GMM 与 GMM_category 在所有参数取值的平均性能并绘制柱状图。
% 通过 2x2 的图窗表示四种失真类型，每个图窗种绘制 GMM 和 GMM_category 的平均性能
% 随失真参数的变化折线图，同时也用虚线绘制基准性能的折线图作为对比。将结果以 PDF 格式保存到
% G:\Projects\ResNet\analysis_DETRAC\performance_knowledge 中

distortion_types = {'blur', 'compression', 'illumination', 'motion_blur'};
clustering_methods = {'GMM', 'GMM_category'};

% 预定义颜色和线型
colors = lines(2);

% 创建图窗
fig = figure('Name', 'Average Performance by Distortion Type', ...
             'NumberTitle', 'off', 'Position', [100, 100, 800, 500]);

for d = 1:length(distortion_types)
    distortion = distortion_types{d};

    % 取对应表格
    tbl = performanceData.(distortion);

    % 失真参数排序
    params = sort(unique(tbl.distortion_param));

    % 计算两种clustering在每个参数下的平均性能
    avgPerformance = zeros(length(params), length(clustering_methods));
    for c = 1:length(clustering_methods)
        clustering = clustering_methods{c};

        for p = 1:length(params)
            param_val = params(p);

            % 筛选该失真类型和参数、以及clustering_method的数据
            sel = tbl.distortion_param == param_val & strcmp(tbl.knowledge_type, clustering);

            % 对所有关键帧数 knowledge_K、权重 prior_weight 和检索数 retrieval_k 的性能取平均
            avgPerformance(p, c) = mean(tbl.performance(sel));
        end
    end

    % 获取baseline对应失真类型和参数
    bidx = strcmp(baseline_table.distortion_type, distortion);
    baseline_params = baseline_table.distortion_param(bidx);
    baseline_f1 = baseline_table.baseline_f1(bidx);

    % 匹配 baseline 顺序以确保与params对齐
    % 这里简单假设baseline_params和params一致或者都存在
    % 若不完全对应，需做插值或对齐
    [commonParams, ia, ib] = intersect(params, baseline_params);
    baselineInterp = NaN(size(params));
    baselineInterp(ia) = baseline_f1(ib);

    % 画子图
    subplot(2,2,d);
    hold on;

    % 画两条平均性能曲线
    for c = 1:length(clustering_methods)
        plot(params, avgPerformance(:, c), '-o', 'Color', colors(c,:), 'MarkerFaceColor', colors(c,:), 'MarkerSize', 4, ...
             'LineWidth', 2, 'DisplayName', strrep(clustering_methods{c}, '_', ' '));
    end

    % 画基准性能曲线，虚线
    plot(params, baselineInterp, 'k--o', 'MarkerFaceColor', 'k', 'MarkerSize', 4, 'LineWidth', 1.5, 'DisplayName', 'Baseline');

    xlabel('Distortion Parameter');
    ylabel('Average F1 Score');
    title(sprintf('Distortion: %s', distortion), 'Interpreter', 'none');
    grid on;
    legend('Location', 'best');
    hold off;
end

% 保存为PDF
pdfFile = fullfile(saveDir, 'Average_Performance_by_Distortion.pdf');
exportgraphics(fig, pdfFile, 'ContentType', 'vector');
% close(fig);

%% analysis_prior_weight：绘制 GMM_category 的性能随先验特征权重的变化曲线。
% 考察知识库类型 GMM_category，绘制其取得性能随先验特征权重 prior_weight
% 的变化曲线。在图窗中为每个失真类型和失真参数绘制一个子图，每个失真类型对应一行。每个子图中以 prior_weight 为横坐标，为每个检索数量
% retrieval_k 绘制一条曲线。计算在对应检索数量和 prior_weight 下在所有知识库规模 knowledge_K 下的平均性能。

% 基本参数
target_clustering = 'GMM_category';
distortion_types = fieldnames(performanceData);

% 创建图窗
fig = figure('Name', 'Performance vs Prior Weight (GMM\_category)', ...
             'NumberTitle', 'off', 'Position', [100, 100, 1200, 900]);

% 统计总子图数，确定子图排列方式
subplot_counts = cellfun(@(dt) numel(unique(performanceData.(dt).distortion_param)), distortion_types);
max_params_per_type = max(subplot_counts);
nRows = length(distortion_types);
nCols = max_params_per_type;

plotIdx = 1;  % 用于 subplot 索引

for d = 1:length(distortion_types)
    distortion = distortion_types{d};
    tbl = performanceData.(distortion);

    % 筛选 GMM_category 数据
    tbl = tbl(strcmp(tbl.knowledge_type, target_clustering), :);
    if isempty(tbl), continue; end

    % 所有参数
    unique_param = unique(tbl.distortion_param);
    unique_pw = sort(unique(tbl.prior_weight));
    unique_rk = unique(tbl.retrieval_k);

    for p = 1:length(unique_param)
        param = unique_param(p);
        subplot(nRows, nCols, plotIdx);
        hold on;

        for rk = unique_rk'
            perf_vals = NaN(size(unique_pw));
            for i = 1:length(unique_pw)
                pw = unique_pw(i);
                idx = tbl.distortion_param == param & ...
                      abs(tbl.prior_weight - pw) < 1e-5 & ...
                      tbl.retrieval_k == rk;

                % 计算在所有 knowledge_K 下的平均性能
                perf_vals(i) = mean(tbl.performance(idx));
            end

            plot(unique_pw, perf_vals, '-o', 'DisplayName', sprintf('R%d', rk));
        end

        title(sprintf('%s | Param=%d', distortion, param), 'Interpreter', 'none');
        xlabel('Prior Weight');
        ylabel('Avg F1');
        grid on;
        legend('Location', 'best');
        hold off;

        plotIdx = plotIdx + 1;
    end
end

% 导出为 PDF
exportgraphics(fig, fullfile(saveDir, 'GMM_category_Performance_vs_PriorWeight.pdf'), 'ContentType', 'vector');
% close(fig);

%% analysis_prior_weight_nodistortion：绘制 GMM_category 的性能在无失真数据集随先验特征权重的变化曲线。
% 考察知识库类型 GMM_category，绘制其取得性能随先验特征权重 prior_weight
% 的变化曲线。仅考察无失真数据集的测试结果。每个子图中以 prior_weight 为横坐标，为每个检索数量
% retrieval_k 绘制一条曲线。计算在指定知识库规模 knowledge_K 下的性能。

% 基本参数
target_clustering = 'GMM_category';
knowledge_K = 100;
distortion_types = fieldnames(performanceData_ori);
% 创建图窗
fig = figure('Name', 'Performance vs Prior Weight (GMM\_category)', ...
             'NumberTitle', 'off', 'Position', [100, 100, 600, 400]);

% 统计总子图数，确定子图排列方式
subplot_counts = cellfun(@(dt) numel(unique(performanceData_ori.(dt).distortion_param)), distortion_types);
max_params_per_type = max(subplot_counts);
nRows = length(distortion_types);
nCols = max_params_per_type;

plotIdx = 1;  % 用于 subplot 索引

for d = 1:length(distortion_types)
    distortion = distortion_types{d};
    tbl = performanceData_ori.(distortion);

    % 筛选 GMM_category 数据
    tbl = tbl(strcmp(tbl.knowledge_type, target_clustering) & tbl.knowledge_K == knowledge_K, :);
    if isempty(tbl), continue; end

    % 所有参数
    unique_param = unique(tbl.distortion_param);
    unique_pw = sort(unique(tbl.prior_weight));
    unique_rk = unique(tbl.retrieval_k);

    for p = 1:length(unique_param)
        param = unique_param(p);
        subplot(nRows, nCols, plotIdx);
        hold on;

        for rk = unique_rk'
            perf_vals = NaN(size(unique_pw));
            for i = 1:length(unique_pw)
                pw = unique_pw(i);
                idx = tbl.distortion_param == param & ...
                      abs(tbl.prior_weight - pw) < 1e-5 & ...
                      tbl.retrieval_k == rk;

                % 计算在所有 knowledge_K 下的平均性能
                perf_vals(i) = mean(tbl.performance(idx));
            end

            plot(unique_pw, perf_vals, '-o', 'DisplayName', sprintf('R%d', rk), 'LineWidth', 1.5);
        end

        title(sprintf('%s | Param=%d (Knowledge_K=%d)', distortion, param, knowledge_K), 'Interpreter', 'none');
        xlabel('Prior Weight');
        ylabel('F1');
        grid on;
        legend('Location', 'best');
        hold off;
        yline(baseline_f1_ori, '--k', 'DisplayName', sprintf('Baseline = %.4f', baseline_f1_ori), 'LineWidth', 1.5);
%         ylim([0.9 0.95]);
        plotIdx = plotIdx + 1;
    end
end

% 导出为 PDF
exportgraphics(fig, fullfile(saveDir, ['GMM_category_Performance_vs_PriorWeight_nodistortion(knowledge_K=', num2str(knowledge_K), ').pdf']), 'ContentType', 'vector');
% close(fig);


%% analysis_retrieval_k: 分析性能随检索数目的变化

% 仅绘制 GMM_category
target_clustering = 'GMM_category';
distortion_types = fieldnames(performanceData);

% 创建图窗
fig = figure('Name', 'Performance vs Retrieval K (GMM\_category)', ...
             'NumberTitle', 'off', 'Position', [100, 100, 1200, 900]);

subplot_counts = cellfun(@(dt) numel(unique(performanceData.(dt).distortion_param)), distortion_types);
total_plots = sum(subplot_counts);
max_params_per_type = max(subplot_counts);
nRows = length(distortion_types);
nCols = max_params_per_type;

plotIdx = 1;

for d = 1:length(distortion_types)
    distortion = distortion_types{d};
    tbl = performanceData.(distortion);
    
    % 筛选 GMM_category
    tbl = tbl(strcmp(tbl.knowledge_type, target_clustering), :);
    if isempty(tbl), continue; end

    unique_param = unique(tbl.distortion_param);
    unique_pw = sort(unique(tbl.prior_weight));
    unique_rk = unique(tbl.retrieval_k);

    for p = 1:length(unique_param)
        param = unique_param(p);
        subplot(nRows, nCols, plotIdx);
        hold on;

        for i = 1:length(unique_pw)
            pw = unique_pw(i);
            perf_vals = NaN(size(unique_rk));
            for j = 1:length(unique_rk)
                rk = unique_rk(j);

                % 取所有 knowledge_K 下平均值
                sel = tbl.distortion_param == param & ...
                      abs(tbl.prior_weight - pw) < 1e-5 & ...
                      tbl.retrieval_k == rk;

                perf_vals(j) = mean(tbl.performance(sel));
            end

            plot(unique_rk, perf_vals, '-o', 'DisplayName', sprintf('W=%.2f', pw));
        end

        title(sprintf('%s | Param=%d', distortion, param), 'Interpreter', 'none');
        xlabel('Retrieval K');
        ylabel('Avg F1');
        grid on;
        legend('Location', 'best');
        hold off;

        plotIdx = plotIdx + 1;
    end
end

% 导出为 PDF
exportgraphics(fig, fullfile(saveDir, 'GMM_category_Performance_vs_RetrievalK.pdf'), 'ContentType', 'vector');
% close(fig);

%% analysis_knowledge_K：第一部分：分析不同失真类型与等级下知识库规模与检索数量的影响
% 考察知识库类型 GMM_category 和先验知识权重 0.4，绘制性能随检索数量 retrieval_k
% 的变化曲线。在图窗中为每个失真类型和失真参数绘制一个子图，每个失真类型对应一行。每个子图中以 retrieval_k 为横坐标，为每个知识库规模
% knowledge_K 绘制一条曲线。为每个子图添加基准性能，以虚线标示。

% 条件筛选
target_clustering = 'GMM_category';
target_pw = 0.4;

% 基准性能表：使用第一部分加载数值

% 转为 containers.Map
baseline_map = containers.Map;
for i = 1:size(baseline_table, 1)
    key = sprintf('%s_%d', baseline_table{i,1}, baseline_table{i,2});
    baseline_map(key) = baseline_table{i,3};
end

% 读取数据
distortion_types = fieldnames(performanceData);

fig = figure('Name', 'Performance vs Retrieval K with Baseline', ...
             'NumberTitle', 'off', 'Position', [100, 100, 600, 350]);

subplot_counts = cellfun(@(dt) numel(unique(performanceData.(dt).distortion_param)), distortion_types);
nRows = length(distortion_types);
nCols = max(subplot_counts);
plotIdx = 1;

for d = 1:length(distortion_types)
    distortion = distortion_types{d};
    tbl = performanceData.(distortion);

    % 筛选条件
    tbl = tbl(strcmp(tbl.knowledge_type, target_clustering) & ...
              abs(tbl.prior_weight - target_pw) < 1e-5, :);
    if isempty(tbl), continue; end

    unique_param = unique(tbl.distortion_param);
    unique_rk = unique(tbl.retrieval_k);
    unique_kK = unique(tbl.knowledge_K);

    for p = 1:length(unique_param)
        param = unique_param(p);
        subplot(nRows, nCols, plotIdx);
        hold on;

        % 画每条 K 曲线
        for i = 1:length(unique_kK)
            kK = unique_kK(i);
            perf_vals = NaN(size(unique_rk));
            for j = 1:length(unique_rk)
                rk = unique_rk(j);
                idx = tbl.distortion_param == param & ...
                      tbl.retrieval_k == rk & ...
                      tbl.knowledge_K == kK;

                perf_vals(j) = mean(tbl.performance(idx));
            end
            plot(unique_rk, perf_vals, '-o', 'DisplayName', sprintf('K=%d', kK));
        end

        % 添加 baseline 虚线
        key = sprintf('%s_%d', distortion, param);
        if isKey(baseline_map, key)
            baseline_f1 = baseline_map(key);
            yline(baseline_f1, '--k', 'DisplayName', 'Baseline');
        end

        title(sprintf('%s | Param=%d', distortion, param), 'Interpreter', 'none');
        xlabel('Retrieval K');
        ylabel('F1 Score');
        grid on;
        legend('Location', 'best');
        hold off;

        plotIdx = plotIdx + 1;
    end
end

% 保存为 PDF
exportgraphics(fig, fullfile(saveDir, ['GMM_category_PW0', num2str(target_pw * 10), '_Performance_vs_RetrievalK_withBaseline.pdf']), 'ContentType', 'vector');
% close(fig);

%% analysis_knowledge_K：第二部分：分析知识库规模与检索数量对不同失真类型与等级下平均性能的影响
% 基于 analysis_knowledge_K
% 的描述：只绘制一张图，取在所有失真类型与失真等级下的平均性能，绘制性能随检索数量的变化。为每个知识库规模绘制一条曲线。
% 同样需要计算基准性能的均值并绘制水平直线
% 基准性能表：使用第一部分加载数值
baseline_f1_values = cell2mat(baseline_table(:,3));
avg_baseline_f1 = mean(baseline_f1_values);  % 基准性能均值

% 初始化结果表
allData = [];

% 聚合所有失真类型的数据
distortion_types = fieldnames(performanceData);
for d = 1:length(distortion_types)
    distortion = distortion_types{d};
    tbl = performanceData.(distortion);

    % 筛选条件
    tbl = tbl(strcmp(tbl.knowledge_type, 'GMM_category') & ...
              abs(tbl.prior_weight - 0.4) < 1e-5, :);

    allData = [allData; tbl];
end

% 获取唯一值
unique_K = unique(allData.knowledge_K);
unique_RK = unique(allData.retrieval_k);

% 绘图
fig = figure('Name', 'Average F1 vs Retrieval K (All Distortions)', ...
             'NumberTitle', 'off', 'Position', [100, 100, 600, 350]);
hold on;

% 为每个 K 绘制曲线
for i = 1:length(unique_K)
    kK = unique_K(i);
    f1_vals = NaN(size(unique_RK));

    for j = 1:length(unique_RK)
        rk = unique_RK(j);

        idx = allData.knowledge_K == kK & allData.retrieval_k == rk;
        f1_vals(j) = mean(allData.performance(idx));
    end

    plot(unique_RK, f1_vals, '-o', 'DisplayName', sprintf('K=%d', kK), 'LineWidth', 1.5);
end

% 添加基准均值线
yline(avg_baseline_f1, '--k', 'DisplayName', sprintf('Baseline = %.4f', avg_baseline_f1), 'LineWidth', 1.5);

xlabel('Retrieval K');
ylabel('Average F1 Score');
title('Average Performance vs Retrieval K (GMM\_category, Prior=0.4)');
legend('Location', 'best');
grid on;
hold off;

% 导出为 PDF
exportgraphics(fig, fullfile(saveDir, 'Avg_Performance_vs_RetrievalK.pdf'), 'ContentType', 'vector');
% close(fig);

%% analysis_knowledge_K_nodistortion：第三部分：分析在无失真数据集上知识库规模与检索数量对性能的影响
% 输出目录
performanceData = struct();
performanceData.none = data_ori;

% 初始化结果表
allData = [];

% 聚合所有失真类型的数据
distortion_types = fieldnames(performanceData);
for d = 1:length(distortion_types)
    distortion = distortion_types{d};
    tbl = performanceData.(distortion);

    % 筛选条件
    tbl = tbl(strcmp(tbl.knowledge_type, 'GMM_category') & ...
              abs(tbl.prior_weight - 0.4) < 1e-5, :);

    allData = [allData; tbl];
end

% 获取唯一值
unique_K = unique(allData.knowledge_K);
unique_RK = unique(allData.retrieval_k);

% 绘图
fig = figure('Name', 'Average F1 vs Retrieval K (no distortion)', ...
             'NumberTitle', 'off', 'Position', [100, 100, 600, 350]);
hold on;

% 为每个 K 绘制曲线
for i = 1:length(unique_K)
    kK = unique_K(i);
    f1_vals = NaN(size(unique_RK));

    for j = 1:length(unique_RK)
        rk = unique_RK(j);

        idx = allData.knowledge_K == kK & allData.retrieval_k == rk;
        f1_vals(j) = mean(allData.performance(idx));
    end

    plot(unique_RK, f1_vals, '-o', 'DisplayName', sprintf('K=%d', kK), 'LineWidth', 1.5);
end

% 添加基准均值线
yline(baseline_f1_ori, '--k', 'DisplayName', sprintf('Baseline = %.4f', baseline_f1_ori), 'LineWidth', 1.5);
ylim([0.75, 0.9]);

xlabel('Retrieval K');
ylabel('F1 Score');
title('Performance vs Retrieval K (GMM\_category, Prior=0.4)');
legend('Location', 'best');
grid on;
hold off;

% 导出为 PDF
exportgraphics(fig, fullfile(saveDir, 'Performance_vs_RetrievalK_nodistortion.pdf'), 'ContentType', 'vector');
% close(fig);

%% analysis_only_retrieval: 分析仅检索与检索后特征融合的不同
% 结果保存目录与基准性能
baseline_f1_values = cell2mat(baseline_table(:,3));
avg_baseline_f1 = mean(baseline_f1_values);
% 读取检索后特征融合的性能：使用第一部分读取数值
% 整理检索后特征融合的绘图点
allData = [];
distortion_types = fieldnames(performanceData);
for d = 1:length(distortion_types)
    distortion = distortion_types{d};
    tbl = performanceData.(distortion);
    tbl = tbl(strcmp(tbl.knowledge_type, 'GMM_category') & ...
              abs(tbl.prior_weight - 0.4) < 1e-5, :);
    allData = [allData; tbl];
end
unique_K = unique(allData.knowledge_K);
unique_RK = unique(allData.retrieval_k);
colors = lines(length(unique_K)); % Using the 'lines' colormap
fig = figure('Name', 'Average F1 vs Retrieval K (All Distortions)', ...
             'NumberTitle', 'off', 'Position', [100, 100, 600, 350]);
hold on;
for i = 1:length(unique_K)
    kK = unique_K(i);
    f1_vals = NaN(size(unique_RK));
    for j = 1:length(unique_RK)
        rk = unique_RK(j);
        idx = allData.knowledge_K == kK & allData.retrieval_k == rk;
        f1_vals(j) = mean(allData.performance(idx));
    end
    plot(unique_RK, f1_vals, '-o', 'Color', colors(i,:), 'DisplayName', sprintf('GMM category (K=%d)', kK), 'LineWidth', 1.5);
end
xlabel('Retrieval K');
ylabel('Average F1 Score');
% 整理仅融合的绘图点
allData = [];
distortion_types = fieldnames(performanceData_only_retrieval);
for d = 1:length(distortion_types)
    distortion = distortion_types{d};
    tbl = performanceData_only_retrieval.(distortion);
    tbl = tbl(strcmp(tbl.knowledge_type, 'GMM_category'), :);
    allData = [allData; tbl];
end
unique_K = unique(allData.knowledge_K);
unique_RK = unique(allData.retrieval_k);
for i = 1:length(unique_K)
    kK = unique_K(i);
    f1_vals = NaN(size(unique_RK));
    for j = 1:length(unique_RK)
        rk = unique_RK(j);
        idx = allData.knowledge_K == kK & allData.retrieval_k == rk;
        f1_vals(j) = mean(allData.performance(idx));
    end
    plot(unique_RK, f1_vals, '--o', 'Color', colors(i,:), 'DisplayName', sprintf('Retrieval (K=%d)', kK), 'LineWidth', 1.5);
end
yline(avg_baseline_f1, '--k', 'DisplayName', sprintf('Baseline = %.4f', avg_baseline_f1), 'LineWidth', 1.5);
title('Feature fusion vs Only retrieval');
legend('Location', 'best');
grid on;
hold off;

% 导出为 PDF
exportgraphics(fig, fullfile(saveDir, 'Avg_Performance_vs_OnlyRetrieval.pdf'), 'ContentType', 'vector');
% close(fig);

%% analysis_only_retrieval_nodistortion: 分析仅检索与检索后特征融合的不同（无失真数据集）
% 结果保存目录与基准性能：使用第一部分读取数值
% 读取检索后特征融合的性能：使用第一部分读取数值
% 整理检索后特征融合的绘图点
allData = [];
distortion_types = fieldnames(performanceData_ori);
for d = 1:length(distortion_types)
    distortion = distortion_types{d};
    tbl = performanceData_ori.(distortion);
    tbl = tbl(strcmp(tbl.knowledge_type, 'GMM_category') & ...
              abs(tbl.prior_weight - 0.4) < 1e-5, :);
    allData = [allData; tbl];
end
unique_K = unique(allData.knowledge_K);
unique_RK = unique(allData.retrieval_k);
colors = lines(length(unique_K)); % Using the 'lines' colormap
fig = figure('Name', 'Average F1 vs Retrieval K (All Distortions)', ...
             'NumberTitle', 'off', 'Position', [100, 100, 600, 350]);
hold on;
for i = 1:length(unique_K)
    kK = unique_K(i);
    f1_vals = NaN(size(unique_RK));
    for j = 1:length(unique_RK)
        rk = unique_RK(j);
        idx = allData.knowledge_K == kK & allData.retrieval_k == rk;
        f1_vals(j) = mean(allData.performance(idx));
    end
    plot(unique_RK, f1_vals, '-o', 'Color', colors(i,:), 'DisplayName', sprintf('GMM category (K=%d)', kK), 'LineWidth', 1.5);
end
xlabel('Retrieval K');
ylabel('Average F1 Score');
% 整理仅融合的绘图点
allData = [];
distortion_types = fieldnames(performanceData_ori_only_retrieval);
for d = 1:length(distortion_types)
    distortion = distortion_types{d};
    tbl = performanceData_ori_only_retrieval.(distortion);
    tbl = tbl(strcmp(tbl.knowledge_type, 'GMM_category'), :);
    allData = [allData; tbl];
end
unique_K = unique(allData.knowledge_K);
unique_RK = unique(allData.retrieval_k);
for i = 1:length(unique_K)
    kK = unique_K(i);
    f1_vals = NaN(size(unique_RK));
    for j = 1:length(unique_RK)
        rk = unique_RK(j);
        idx = allData.knowledge_K == kK & allData.retrieval_k == rk;
        f1_vals(j) = mean(allData.performance(idx));
    end
    plot(unique_RK, f1_vals, '--o', 'Color', colors(i,:), 'DisplayName', sprintf('Retrieval (K=%d)', kK), 'LineWidth', 1.5);
end
yline(baseline_f1_ori, '--k', 'DisplayName', sprintf('Baseline = %.4f', baseline_f1_ori), 'LineWidth', 1.5);
title('Feature fusion vs Only retrieval');
legend('Location', 'best');
grid on;
hold off;

% 导出为 PDF
exportgraphics(fig, fullfile(saveDir, 'Avg_Performance_vs_OnlyRetrieval_nodistortion.pdf'), 'ContentType', 'vector');
% close(fig);

%% analysis_knowledge_high_weight：纳入高先验权重的分析
% 路径列表：使用第一部分加载数值

% 加载数据
allData = load_all_tables_from_dirs(dirs_with_high_weight);

% 筛选条件
allData = allData(strcmp(allData.knowledge_type, 'GMM_category') & ...
                  allData.knowledge_K == 2000 & ...
                  allData.retrieval_k == 5, :);

% 基准性能表
baseline_map = containers.Map;
for i = 1:size(baseline_table, 1)
    key = sprintf('%s_%d', baseline_table{i,1}, baseline_table{i,2});
    baseline_map(key) = baseline_table{i,3};
end

% 获取失真类型和参数组合
dist_types = unique(allData.distortion_type);
nRows = numel(dist_types);
nCols = 4;

% 设置图窗
fig = figure('Name', 'Performance vs Prior Weight (GMM\_category, K=2000, Rk=5)', ...
             'NumberTitle', 'off', 'Position', [100, 100, 1400, 800]);

plotIdx = 1;

for r = 1:nRows
    distortion = dist_types{r};
    subData = allData(strcmp(allData.distortion_type, distortion), :);
    params = unique(subData.distortion_param);

    for p = 1:length(params)
        param = params(p);
        subplot(nRows, nCols, plotIdx);
        hold on;

        % 提取该子图数据
        subTbl = subData(subData.distortion_param == param, :);
        [sorted_pw, idx] = sort(subTbl.prior_weight);
        y = subTbl.performance(idx);
        plot(sorted_pw, y, '-o', 'LineWidth', 1.5);
        
        % 获取 baseline 并绘制
        key = sprintf('%s_%d', distortion, param);
        if isKey(baseline_map, key)
            baseline = baseline_map(key);
            yline(baseline, '--k', 'Baseline');
        else
            baseline = NaN;
        end
        
        % 自动设置纵坐标范围
        ymin = min([y; baseline]) - 0.02;
        ymax = max([y; baseline]) + 0.02;
        ylim([max(0, ymin), min(1, ymax)]);

        title(sprintf('%s | Param=%d', distortion, param), 'Interpreter', 'none');
        xlabel('Prior Weight');
        ylabel('F1 Score');
        grid on;
        hold off;
        plotIdx = plotIdx + 1;
    end
end

% 保存图像
savePath = fullfile(saveDir, '\GMM_category_K2000_R5_vs_PriorWeight.pdf');
exportgraphics(fig, savePath, 'ContentType', 'vector');
% close(fig);

%% analysis_best GMM_category 的最优性能展示
% 在不同失真类型和失真参数下，计算 GMM_category 在以下参数取值的性能并绘制柱状图。
% 通过 2x2 的图窗表示四种失真类型，每个图窗种绘制 GMM_category 的性能
% 随失真参数的变化折线图，同时也用虚线绘制基准性能的折线图作为对比。将结果以 PDF 格式保存到
% G:\Projects\ResNet\analysis_DETRAC\performance_knowledge 中

distortion_types = {'blur', 'compression', 'illumination', 'motion_blur'};
clustering_methods = {'GMM_category', 'GMM'};

% 预定义颜色和线型
colors = lines(2);
lineStyles = {'-', '--'};

% 创建图窗
fig = figure('Name', 'Average Performance by Distortion Type', ...
             'NumberTitle', 'off', 'Position', [100, 100, 800, 500]);

for d = 1:length(distortion_types)
    distortion = distortion_types{d};

    % 取对应表格
    tbl = performanceData.(distortion);

    % 失真参数排序
    params = sort(unique(tbl.distortion_param));

    % 计算两种clustering在每个参数下的平均性能
    avgPerformance = zeros(length(params), length(clustering_methods));
    for c = 1:length(clustering_methods)
        clustering = clustering_methods{c};

        for p = 1:length(params)
            param_val = params(p);

            % 筛选该失真类型和参数、以及clustering_method的数据，选取最优性能参数下的表现
            sel = tbl.distortion_param == param_val & strcmp(tbl.knowledge_type, clustering) & tbl.prior_weight == 0.3 & tbl.knowledge_K == 2000 & tbl.retrieval_k == 5;

            avgPerformance(p, c) = mean(tbl.performance(sel));
        end
    end

    % 获取baseline对应失真类型和参数
    bidx = strcmp(baseline_table.distortion_type, distortion);
    baseline_params = baseline_table.distortion_param(bidx);
    baseline_f1 = baseline_table.baseline_f1(bidx);

    % 匹配 baseline 顺序以确保与params对齐
    % 这里简单假设baseline_params和params一致或者都存在
    % 若不完全对应，需做插值或对齐
    [commonParams, ia, ib] = intersect(params, baseline_params);
    baselineInterp = NaN(size(params));
    baselineInterp(ia) = baseline_f1(ib);

    % 画子图
    subplot(2,2,d);
    hold on;

    % 画两条平均性能曲线
    for c = 1:length(clustering_methods)
        plot(params, avgPerformance(:, c), '-o', 'Color', colors(c,:), 'MarkerFaceColor', colors(c,:), 'MarkerSize', 4, ...
             'LineWidth', 2, 'DisplayName', strrep(clustering_methods{c}, '_', ' '));
    end

    % 画基准性能曲线，虚线
    plot(params, baselineInterp, 'k--o', 'MarkerFaceColor', 'k', 'MarkerSize', 4, 'LineWidth', 1.5, 'DisplayName', 'Baseline');

    xlabel('Distortion Parameter');
    ylabel('Average F1 Score');
    title(sprintf('Distortion: %s', distortion), 'Interpreter', 'none');
    grid on;
    legend('Location', 'best');
    hold off;
end

% 保存为PDF
pdfFile = fullfile(saveDir, 'Best_Performance_by_Distortion.pdf');
exportgraphics(fig, pdfFile, 'ContentType', 'vector');
% close(fig);

%% analysis_GMM_vs_GMM_category: 分析两种知识库构建方法在不同参数下的性能对比
% G:\Projects\ViT\results\distorted_with_knowledge\performance_none.csv
% 中是不同条件下模型的性能，其中第一行为表头，分别为
% distortion_type,distortion_param,prior_weight,knowledge_type,knowledge_K,retrieval_k,performance，表示失真类型、失真参数、先验权重、知识库类型、知识库规模、检索数量和性能。请帮我编写
% matlab 程序，绘制先验权重为 0.1 时，失真类型为 none、失真参数为 0
% 时，性能随检索数量的变化。为每种知识库类型和知识库规模绘制一条曲线，其中知识库类型为 GMM_category 绘制为实线，GMM
% 绘制为虚线，同一个知识库规模采用同一个颜色绘制。此外，绘制一条水平虚线表示 baseline 性能，性能为 0.9390。以 PDF
% 格式保存绘制结果
% 读取CSV文件数据：使用第一部分读取的数值
weight=0.3;
% 筛选数据：先验权重为 weight，失真类型为none，失真参数为0
filtered_data = data_ori(strcmp(data_ori.distortion_type, 'none') & ...
                    data_ori.distortion_param == 0 & ...
                    data_ori.prior_weight == weight, :);

% 定义知识库类型和对应的线型
knowledge_types = {'GMM_category', 'GMM'};
line_styles = {'-', '--'}; % 实线和虚线

% 定义知识库规模列表和颜色
K_values = unique(filtered_data.knowledge_K);
colors = lines(length(K_values)); % 为每个K值分配不同颜色

% 创建图形
fig = figure;
hold on;
grid on;

% 设置图形大小和位置（为了更好的PDF输出）
set(gcf, 'Position', [100, 100, 800, 600]);
set(gcf, 'PaperPositionMode', 'auto'); % 保持屏幕显示的比例

% 为每个知识库类型和规模绘制曲线
for k_idx = 1:length(K_values)
    K = K_values(k_idx);
    
    for t_idx = 1:length(knowledge_types)
        knowledge_type = knowledge_types{t_idx};
        
        % 筛选特定类型和规模的数据
        subset = filtered_data(strcmp(filtered_data.knowledge_type, knowledge_type) & ...
                              filtered_data.knowledge_K == K, :);
        
        if ~isempty(subset)
            % 按检索数量排序
            [sorted_k, sort_idx] = sort(subset.retrieval_k);
            sorted_perf = subset.performance(sort_idx);
            
            % 绘制曲线
            plot(sorted_k, sorted_perf, ...
                 'LineStyle', line_styles{t_idx}, ...
                 'Color', colors(k_idx, :), ...
                 'Marker', 'o', ...
                 'LineWidth', 1.5, ...
                 'MarkerSize', 6, ...
                 'DisplayName', sprintf('%s K=%d', strrep(knowledge_type, '_', ' '), K));
        end
    end
end

% 绘制基线
yline(baseline_f1_ori, 'k--', 'LineWidth', 1.5, 'DisplayName', ['Baseline (', num2str(baseline_f1_ori), ')']);

% 设置图形属性
xlabel('Retrieval Number (k)', 'FontSize', 14);
ylabel('Performance', 'FontSize', 14);
title(['Performance vs Retrieval Number (prior weight=', num2str(weight), ', distortion: none/0)'], 'FontSize', 16);
legend('Location', 'best', 'FontSize', 12, 'NumColumns', 2);
set(gca, 'FontSize', 12);
set(gca, 'LineWidth', 1.5);
grid on;
box on;
hold off;

% 保存为PDF
pdfFile = fullfile(saveDir, ['GMM_vs_GMM_category_weight=', num2str(weight), '.pdf']);
exportgraphics(fig, pdfFile, 'ContentType', 'vector');
% close(fig);

% 可选：同时保存为PNG格式
% png_filename = 'performance_vs_retrieval.png';
% print(gcf, png_filename, '-dpng', '-r300');
% fprintf('PNG文件已保存为: %s\n', png_filename);

%% analysis_GMM_ave+best: 绘制不同失真类型和等级下取得的平均性能、最优性能和最差性能曲线对比

distortion_types = {'blur', 'compression', 'illumination', 'motion_blur'};
clustering_methods = {'GMM', 'GMM_category'};

% 颜色预定义
colors = lines(length(clustering_methods));

% 创建图窗
fig = figure('Name', 'Performance Range by Distortion Type', ...
             'NumberTitle', 'off', 'Position', [100, 100, 900, 600]);

for d = 1:length(distortion_types)
    distortion = distortion_types{d};

    % 取对应表格
    tbl = performanceData.(distortion);

    % 失真参数排序
    params = sort(unique(tbl.distortion_param));

    % 平均、最大、最小性能矩阵
    avgPerformance = zeros(length(params), length(clustering_methods));
    maxPerformance = zeros(length(params), length(clustering_methods));
    minPerformance = zeros(length(params), length(clustering_methods));

    for c = 1:length(clustering_methods)
        clustering = clustering_methods{c};

        for p = 1:length(params)
            param_val = params(p);

            % 筛选该失真类型和参数、clustering_method的所有数据
            sel = tbl.distortion_param == param_val & strcmp(tbl.knowledge_type, clustering);

            perfVals = tbl.performance(sel);

            if ~isempty(perfVals)
                avgPerformance(p, c) = mean(perfVals);
                maxPerformance(p, c) = max(perfVals);
                minPerformance(p, c) = min(perfVals);
            else
                avgPerformance(p, c) = NaN;
                maxPerformance(p, c) = NaN;
                minPerformance(p, c) = NaN;
            end
        end
    end

    % 获取baseline对应失真类型和参数
    bidx = strcmp(baseline_table.distortion_type, distortion);
    baseline_params = baseline_table.distortion_param(bidx);
    baseline_f1 = baseline_table.baseline_f1(bidx);

    % 对齐 baseline
    [commonParams, ia, ib] = intersect(params, baseline_params);
    baselineInterp = NaN(size(params));
    baselineInterp(ia) = baseline_f1(ib);

    % ---------- 绘图 ----------
    subplot(2,2,d);
    hold on;

    for c = 1:length(clustering_methods)
        % 下误差 = 平均 - 最小, 上误差 = 最大 - 平均
        errLow = avgPerformance(:,c) - minPerformance(:,c);
        errHigh = maxPerformance(:,c) - avgPerformance(:,c);
    
        errorbar(params, avgPerformance(:,c), errLow, errHigh, '-o', ...
            'Color', colors(c,:), ...
            'MarkerFaceColor', colors(c,:), ...
            'MarkerSize', 4, 'LineWidth', 2, ...
            'DisplayName', sprintf('%s (avg±range)', strrep(clustering_methods{c}, '_', ' ')));
    end

    % baseline (虚线黑色)
    plot(params, baselineInterp, 'k--o', ...
        'MarkerFaceColor', 'k', ...
        'MarkerSize', 4, ...
        'LineWidth', 1.5, ...
        'DisplayName', 'Baseline');

    xlabel('Distortion Parameter');
    ylabel('F1 Score');
    title(sprintf('Distortion: %s', distortion), 'Interpreter', 'none');
    grid on;
    legend('Location', 'best');
    hold off;
end

% 保存 PDF
pdfFile = fullfile(saveDir, 'Performance_Range_by_Distortion.pdf');
exportgraphics(fig, pdfFile, 'ContentType', 'vector');


%% 函数定义
function tbl = load_all_tables_from_dirs(directories)
    files = ["performance_blur.csv", ...
             "performance_compression.csv", ...
             "performance_illumination.csv", ...
             "performance_motion_blur.csv"];

    tbl = table();
    for d = 1:length(directories)
        dirPath = directories{d};
        for i = 1:length(files)
            file = fullfile(dirPath, files(i));
            if isfile(file)
                T = readtable(file);
                T.distortion_type = repmat({char(extractBetween(files(i), "performance_", ".csv"))}, height(T), 1);
                tbl = [tbl; T];
            end
        end
    end
end
