clear; clc;
addpath(genpath(pwd));

%% 1. Cấu hình Tham số
N = 25; M = 40;
L = 1000; 
H_min = 100; H_max = 300; BS_pos = [0, 0, 30];
% --- Kênh truyền (3GPP TR 36.777 / TR 38.901, fc = 2 GHz) ---
% PL tham chiếu 1 m (c/(4*pi*fc))^2 được áp dụng trong compute_*_hop_gains.m
alpha1 = 2.2;   % BS->UAV: UMa-AV LoS, PL = 28 + 22log10(d) + 20log10(fc)
alpha2 = 2.5;   % UAV->UE: air-to-ground, gần LoS
K_R = 5; mm = 3;
I_max = 1; gamma_th = 1.0; beta = 0; P_th = 0.1;
B_bs = 50; B_uav = 1.08;   % MHz: NR FR1 50 MHz backhaul; UAV->UE = 6 PRB LTE-M (1.08 MHz)

% --- Công suất phát ---
P_bs_dBm  = 46;                          % Macro BS (TR 36.814 / 36.777): 46 dBm = 40 W tổng
P_uav_dBm = 30;                          % UAV relay ~ small cell: 30 dBm = 1 W (có thể quét 23-30 dBm)
P_bs    = 10^((P_bs_dBm  - 30)/10);      % [W] tổng, chia đều P_bs/N cho mỗi link BS->UAV
P_total = 10^((P_uav_dBm - 30)/10) * N;  % [W] tổng công suất của N UAV

% --- Noise: -174 dBm/Hz + 10log10(B) + NF ---
NF_uav = 7; NF_ue = 9;                   % dB (UAV receiver, UE theo 3GPP)
sigma_b2 = 10^((-174 + 10*log10(B_bs*1e6)  + NF_uav - 30)/10);  % ≈ -90 dBm @ 50 MHz
sigma2   = 10^((-174 + 10*log10(B_uav*1e6) + NF_ue  - 30)/10);  % ≈ -105 dBm @ 1.08 MHz

% Cấu hình thuật toán
POP_SIZE = 50;
generation = 8000;
NUM_RUNS = 30; VERBOSE = false;
MAX_FES = generation * POP_SIZE;
iters = round(MAX_FES / POP_SIZE);

% Bật/tắt từng thuật toán
RUN_IDE = true;
RUN_RGA = true;
RUN_PSO = true;
RUN_GAQJ = true;

RUN_FIXED = true;
RUN_DYNAMIC = true;

% Tùy chọn hiển thị thêm biểu đồ kép Boxplot Sum-Rate vs Average Runtime
SHOW_DUAL_AXIS_PLOT = true;

% 3N vị trí UAV + M association + M power UE
dim = N * 3 + 2 * M; 

fprintf('=== BẮT ĐẦU CHẠY ĐÁNH GIÁ CONVERGENCE & AVERAGE RUNTIME ===\n');
fprintf('Tham số: N = %d UAVs, M = %d UEs, Runs = %d, Max_FES = %d\n', N, M, NUM_RUNS, MAX_FES);
fprintf('Đang chạy P_bs = %.1f W, P_uav = %.1f W...\n\n', P_bs, P_total/N);

%% Khởi tạo mảng tích luỹ lịch sử hội tụ và thời gian thực thi

temp_conv_ide  = zeros(1, iters);
temp_conv_rga  = zeros(1, iters);
temp_conv_pso  = zeros(1, iters);
temp_conv_gaqj = zeros(1, iters);

temp_val_fixed   = 0;
temp_val_dynamic = 0;

% Mảng lưu runtime (giây) của từng run
time_ide     = NaN(1, NUM_RUNS);
time_rga     = NaN(1, NUM_RUNS);
time_pso     = NaN(1, NUM_RUNS);
time_gaqj    = NaN(1, NUM_RUNS);
time_fixed   = NaN(1, NUM_RUNS);
time_dynamic = NaN(1, NUM_RUNS);

% Mảng lưu fitness cuối cùng của từng run (để vẽ boxplot / bảng thống kê)
fit_ide     = NaN(1, NUM_RUNS);
fit_rga     = NaN(1, NUM_RUNS);
fit_pso     = NaN(1, NUM_RUNS);
fit_gaqj    = NaN(1, NUM_RUNS);
fit_fixed   = NaN(1, NUM_RUNS);
fit_dynamic = NaN(1, NUM_RUNS);

%% Vòng lặp chính qua từng Run

for run = 1:NUM_RUNS
    fprintf('--- Bắt đầu Run %d/%d ---\n', run, NUM_RUNS);
    
    % Vị trí thiết bị người dùng độc lập cho mỗi run
    UE_positions = generate_positions_3d(M, L, 0, true);
    
    % Metaheuristic hàm mục tiêu        
    fobj = @(x, varargin) decode_and_calc_by_option(x, 3, N, M, L, H_max, H_min, UE_positions, BS_pos, ...
            alpha1, alpha2, K_R, mm, P_bs, P_total, sigma_b2, sigma2, I_max, gamma_th, beta, P_th, B_bs, B_uav, varargin{:});
            
    % 1. IDE (Differential Evolution)
    if RUN_IDE
        t_start = tic;
        [~, c_ide, ~] = run_ide(dim, fobj, MAX_FES, POP_SIZE, 0.1, 5, 1.5, VERBOSE);
        time_ide(run) = toc(t_start);
        temp_conv_ide = temp_conv_ide + c_ide(:).';
        fit_ide(run)  = c_ide(end);
    end
    
    % 2. GA (Genetic Algorithm)
    if RUN_RGA
        t_start = tic;
        [~, c_rga, ~] = run_rga(dim, fobj, MAX_FES, POP_SIZE, VERBOSE);
        time_rga(run) = toc(t_start);
        temp_conv_rga = temp_conv_rga + c_rga(:).';
        fit_rga(run)  = c_rga(end);
    end
    
    % 3. PSO (Particle Swarm Optimization)
    if RUN_PSO
        t_start = tic;
        [~, c_pso, ~] = run_pso(dim, fobj, MAX_FES, POP_SIZE, 0.7, 1.5, 1.5, VERBOSE);
        time_pso(run) = toc(t_start);
        temp_conv_pso = temp_conv_pso + c_pso(:).';
        fit_pso(run)  = c_pso(end);
    end
    
    % 4. GA-QJ
    if RUN_GAQJ
        t_start = tic;
        [~, c_gaqj, ~] = run_gaqj(dim, fobj, MAX_FES, POP_SIZE, VERBOSE);
        time_gaqj(run) = toc(t_start);
        temp_conv_gaqj = temp_conv_gaqj + c_gaqj(:).';
        fit_gaqj(run)  = c_gaqj(end);
    end    

    % --- Run fixed clustering (K-means) and Dynamic clustering
    init_pos_cluster = rand(N, 3);
    init_pos_cluster(:,1:2) = init_pos_cluster(:,1:2) * L;
    init_pos_cluster(:,3) = (init_pos_cluster(:,3) * (H_max - H_min)) + H_min;
    
    % 5. Fixed clustering (K-means)
    if RUN_FIXED
        t_start = tic;
        [clusters_fix, pos_fix] = fixed_clustering_3d(init_pos_cluster, UE_positions, N, H_min, H_max);
        val_fix = Func_SumRate_UAV(pos_fix, UE_positions, BS_pos, alpha1, alpha2, ...
                                    K_R, mm, P_bs, P_total, sigma_b2, sigma2, ...
                                    I_max, gamma_th, beta, P_th, B_bs, B_uav, clusters_fix);
        time_fixed(run) = toc(t_start);
        temp_val_fixed  = temp_val_fixed + val_fix;
        fit_fixed(run)  = val_fix;
    end
    
    % 6. Dynamic clustering
    if RUN_DYNAMIC
        t_start = tic;
        P_b_n_vec = ones(N, 1) * (P_bs / N);
        P_n_vec   = ones(N, 1) * (P_total / N);
        [~, ~, ~, val_dyn] = dynamic_clustering_3d(init_pos_cluster, UE_positions, BS_pos, ...
                                    alpha1, alpha2, K_R, mm, P_b_n_vec, P_n_vec, ...
                                    sigma_b2, sigma2, I_max, gamma_th, 15, 1e-3, beta, P_th, B_bs, B_uav, H_min, H_max);
        time_dynamic(run) = toc(t_start);
        temp_val_dynamic  = temp_val_dynamic + val_dyn;
        fit_dynamic(run)  = val_dyn;
    end

    % In thông tin thời gian thực thi của từng thuật toán trong Run hiện tại
    fprintf('  [Run %d/%d hoàn thành] Thời gian: ', run, NUM_RUNS);
    if RUN_GAQJ,    fprintf('GA-QJ: %.2fs | ', time_gaqj(run)); end
    if RUN_IDE,     fprintf('DE: %.2fs | ', time_ide(run)); end
    if RUN_RGA,     fprintf('GA: %.2fs | ', time_rga(run)); end
    if RUN_PSO,     fprintf('PSO: %.2fs | ', time_pso(run)); end
    if RUN_FIXED,   fprintf('Fixed: %.3fs | ', time_fixed(run)); end
    if RUN_DYNAMIC, fprintf('Dynamic: %.3fs | ', time_dynamic(run)); end
    fprintf('\n\n');

    % --- VẼ VÀ HIỂN THỊ ĐỒ THỊ MỖI 5 RUNS HOẶC Ở RUN CUỐI CÙNG (KHÔNG GHI ĐÈ LẪN NHAU) ---
    if mod(run, 5) == 0 || run == NUM_RUNS
        fprintf('>>> Đã hoàn thành %d/%d runs. Đang mở các biểu đồ mới cho mốc Run = %d...\n', run, NUM_RUNS, run);
        
        % Tính trung bình trung gian tính đến lần chạy thứ 'run'
        curr_ide_hist   = temp_conv_ide / run;
        curr_rga_hist   = temp_conv_rga / run;
        curr_pso_hist   = temp_conv_pso / run;
        curr_gaqj_hist  = temp_conv_gaqj / run;
        curr_fixed_val  = temp_val_fixed / run;
        curr_dynamic_val= temp_val_dynamic / run;
        
        % 1. Vẽ đồ thị Convergence cho mốc run hiện tại (Mở cửa sổ mới riêng)
        plot_convergence_graph(run, iters, N, M, NUM_RUNS, ...
            RUN_GAQJ, curr_gaqj_hist, ...
            RUN_IDE,  curr_ide_hist, ...
            RUN_RGA,  curr_rga_hist, ...
            RUN_PSO,  curr_pso_hist, ...
            RUN_FIXED, curr_fixed_val, ...
            RUN_DYNAMIC, curr_dynamic_val);

        % 2. Tùy chọn vẽ biểu đồ kép Sum-Rate vs Average Runtime (nếu bật cờ)
        if SHOW_DUAL_AXIS_PLOT
            plot_dual_axis_graph(run, N, M, NUM_RUNS, ...
                RUN_GAQJ, fit_gaqj(1:run), time_gaqj(1:run), ...
                RUN_IDE,  fit_ide(1:run),  time_ide(1:run), ...
                RUN_RGA,  fit_rga(1:run),  time_rga(1:run), ...
                RUN_PSO,  fit_pso(1:run),  time_pso(1:run), ...
                RUN_FIXED, fit_fixed(1:run), time_fixed(1:run), ...
                RUN_DYNAMIC, fit_dynamic(1:run), time_dynamic(1:run));
        end
    end
end

fprintf('\n======================== BẢNG TỔNG KẾT KẾT QUẢ (%d RUNS) ========================\n', NUM_RUNS);
alg_names_all = {};
avg_fits_all  = [];
std_fits_all  = [];
avg_times_all = [];
std_times_all = [];

if RUN_GAQJ
    alg_names_all{end+1} = 'GA-QJ';
    avg_fits_all(end+1)  = mean(fit_gaqj, 'omitnan');
    std_fits_all(end+1)  = std(fit_gaqj, 'omitnan');
    avg_times_all(end+1) = mean(time_gaqj, 'omitnan');
    std_times_all(end+1) = std(time_gaqj, 'omitnan');
end
if RUN_IDE
    alg_names_all{end+1} = 'DE';
    avg_fits_all(end+1)  = mean(fit_ide, 'omitnan');
    std_fits_all(end+1)  = std(fit_ide, 'omitnan');
    avg_times_all(end+1) = mean(time_ide, 'omitnan');
    std_times_all(end+1) = std(time_ide, 'omitnan');
end
if RUN_RGA
    alg_names_all{end+1} = 'GA';
    avg_fits_all(end+1)  = mean(fit_rga, 'omitnan');
    std_fits_all(end+1)  = std(fit_rga, 'omitnan');
    avg_times_all(end+1) = mean(time_rga, 'omitnan');
    std_times_all(end+1) = std(time_rga, 'omitnan');
end
if RUN_PSO
    alg_names_all{end+1} = 'PSO';
    avg_fits_all(end+1)  = mean(fit_pso, 'omitnan');
    std_fits_all(end+1)  = std(fit_pso, 'omitnan');
    avg_times_all(end+1) = mean(time_pso, 'omitnan');
    std_times_all(end+1) = std(time_pso, 'omitnan');
end
if RUN_FIXED
    alg_names_all{end+1} = 'Fixed Clustering';
    avg_fits_all(end+1)  = mean(fit_fixed, 'omitnan');
    std_fits_all(end+1)  = std(fit_fixed, 'omitnan');
    avg_times_all(end+1) = mean(time_fixed, 'omitnan');
    std_times_all(end+1) = std(time_fixed, 'omitnan');
end
if RUN_DYNAMIC
    alg_names_all{end+1} = 'Dynamic Clustering';
    avg_fits_all(end+1)  = mean(fit_dynamic, 'omitnan');
    std_fits_all(end+1)  = std(fit_dynamic, 'omitnan');
    avg_times_all(end+1) = mean(time_dynamic, 'omitnan');
    std_times_all(end+1) = std(time_dynamic, 'omitnan');
end

results_table = table(alg_names_all', avg_fits_all', std_fits_all', avg_times_all', std_times_all', ...
    'VariableNames', {'Algorithm', 'Avg_SumRate_Mbps', 'Std_SumRate', 'Avg_Runtime_s', 'Std_Runtime_s'});
disp(results_table);
fprintf('=================================================================================\n');


%% ==================== CÁC HÀM VẼ ĐỒ THỊ VÀ HỖ TRỢ ====================

%% 1) Hàm phụ trách vẽ đồ thị Convergence (Mỗi lần vẽ mở cửa sổ mới, không ghi đè)
function plot_convergence_graph(run_idx, iters, N, M, total_runs, ...
    RUN_GAQJ, gaqj_hist, RUN_IDE, ide_hist, ...
    RUN_RGA, rga_hist, RUN_PSO, pso_hist, ...
    RUN_FIXED, fixed_val, RUN_DYNAMIC, dynamic_val)

    % Mở cửa sổ mới riêng biệt cho mốc runs hiện tại
    fig_cv = figure('Color', 'w', ...
                    'Name', sprintf('Convergence (Runs = %d)', run_idx), ...
                    'NumberTitle', 'on');
    hold on;
    marker_indices = round(linspace(1, iters, 10)); % Chọn 10 điểm để vẽ marker

    % Màu sắc thống nhất:
    % GA-QJ: 'b'                  = [0, 0, 1]
    % DE:    [0.9290 0.6940 0.1250]
    % GA:    [0, 0.5, 0]
    % PSO:   'r'                  = [1, 0, 0]
    % Fixed: [0.25 0.25 0.25]
    % Dyn:   'k'                  = [0, 0, 0]

    if RUN_GAQJ
        plot(1:iters, gaqj_hist, '-d', ...
            'Color', 'b', 'LineWidth', 2.0, ...
            'MarkerIndices', marker_indices, 'MarkerSize', 8, ...
            'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'w', ...
            'DisplayName', 'GA-QJ');
    end

    if RUN_IDE
       plot(1:iters, ide_hist, '-s', ...
           'Color', [0.9290 0.6940 0.1250], 'LineWidth', 2.0, ...
           'MarkerIndices', marker_indices, 'MarkerSize', 8, ...
           'MarkerEdgeColor', [0.9290 0.6940 0.1250], 'MarkerFaceColor', 'w', ...
           'DisplayName', 'DE');
    end    

    if RUN_RGA
        plot(1:iters, rga_hist, '-^', ...
            'Color', [0 0.5 0], 'LineWidth', 2.0, ...
            'MarkerIndices', marker_indices, 'MarkerSize', 8, ...
            'MarkerEdgeColor', [0 0.5 0], 'MarkerFaceColor', 'w', ...
            'DisplayName', 'GA');
    end    

    if RUN_PSO
        plot(1:iters, pso_hist, '-o', ...
            'Color', 'r', 'LineWidth', 2.0, ...
            'MarkerIndices', marker_indices, 'MarkerSize', 8, ...
            'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'w', ...
            'DisplayName', 'PSO');
    end

    if RUN_FIXED
       yline(fixed_val, '--', 'Color', [0.25 0.25 0.25], 'LineWidth', 2.0, 'DisplayName', 'Fixed Clustering');
    end

    if RUN_DYNAMIC
        yline(dynamic_val, ':', 'Color', 'k', 'LineWidth', 2.0, 'DisplayName', 'Dynamic');
    end  

    hold off; grid on;
    
    % Cấu hình Font và LaTeX cho hệ trục tọa độ
    set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
             'FontName', 'Times New Roman', 'FontSize', 11, ...
             'TickLabelInterpreter', 'latex', ...
             'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.3);
    box on;

    % Cấu hình Text (xlabel, ylabel, title, legend)
    xlabel('Number of Generations', 'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 15);
    ylabel('Sum Throughput [Mbps]', 'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 15);
    title(sprintf('$N = %d, M = %d$ (Runs = %d/%d)', N, M, run_idx, total_runs), ...
          'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 15);
    legend('show', 'Location', 'best', 'TextColor', 'k', 'Color', 'w', ...
           'EdgeColor', [0.8 0.8 0.8], 'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 11);
           
    drawnow;
end

%% 2) Hàm phụ trách vẽ biểu đồ kép Sum-Rate vs Average Runtime (Tùy chọn)
function plot_dual_axis_graph(run_idx, N, M, total_runs, ...
    RUN_GAQJ, f_gaqj, t_gaqj, RUN_IDE, f_ide, t_ide, ...
    RUN_RGA, f_rga, t_rga, RUN_PSO, f_pso, t_pso, ...
    RUN_FIXED, f_fixed, t_fixed, RUN_DYNAMIC, f_dynamic, t_dynamic)

    names  = {};
    times  = [];
    all_fits = [];
    raw_colors = {};

    % Bảng màu giữ nguyên kiểu từ plot_convergence_graph:
    if RUN_GAQJ
        names{end+1}      = 'GA-QJ';
        times(end+1)      = mean(t_gaqj, 'omitnan');
        all_fits          = [all_fits, f_gaqj(:)];
        raw_colors{end+1} = 'b';
    end
    if RUN_IDE
        names{end+1}      = 'DE';
        times(end+1)      = mean(t_ide, 'omitnan');
        all_fits          = [all_fits, f_ide(:)];
        raw_colors{end+1} = [0.9290 0.6940 0.1250];
    end
    if RUN_RGA
        names{end+1}      = 'GA';
        times(end+1)      = mean(t_rga, 'omitnan');
        all_fits          = [all_fits, f_rga(:)];
        raw_colors{end+1} = [0 0.5 0];
    end
    if RUN_PSO
        names{end+1}      = 'PSO';
        times(end+1)      = mean(t_pso, 'omitnan');
        all_fits          = [all_fits, f_pso(:)];
        raw_colors{end+1} = 'r';
    end
    if RUN_FIXED
        names{end+1}      = 'Fixed';
        times(end+1)      = mean(t_fixed, 'omitnan');
        all_fits          = [all_fits, f_fixed(:)];
        raw_colors{end+1} = [0.25 0.25 0.25];
    end
    if RUN_DYNAMIC
        names{end+1}      = 'Dynamic';
        times(end+1)      = mean(t_dynamic, 'omitnan');
        all_fits          = [all_fits, f_dynamic(:)];
        raw_colors{end+1} = 'k';
    end

    if isempty(names)
        return;
    end

    % Mở cửa sổ mới riêng biệt cho mốc runs hiện tại
    fig_dual = figure('Color', 'w', ...
                      'Name', sprintf('Sum Rate vs Runtime (Runs = %d)', run_idx), ...
                      'NumberTitle', 'on');

    x = 1:numel(names);
    box_half_width = 0.22;
    cap_half_width = 0.10;

    % Trục Y trái: Boxplot của Sum Rate
    yyaxis left;
    hold on;
    for i = 1:numel(x)
        c = raw_colors{i};  % Dùng trực tiếp màu nguyên bản ('b', 'r', 'k', [r g b])
        vals = all_fits(:, i);
        vals = vals(~isnan(vals));
        if isempty(vals)
            continue;
        end
        q0 = min(vals);
        q4 = max(vals);
        q2 = median(vals);
        q1 = 0.5 * (q0 + q2);
        q3 = 0.5 * (q2 + q4);

        if q1 == q3
            plot([x(i)-box_half_width, x(i)+box_half_width], [q2 q2], 'Color', c, 'LineWidth', 2.0);
        else
            patch([x(i)-box_half_width, x(i)+box_half_width, x(i)+box_half_width, x(i)-box_half_width], ...
                  [q1, q1, q3, q3], c, ...
                  'FaceAlpha', 0.5, 'EdgeColor', 'k', 'LineWidth', 0.9, 'HandleVisibility', 'off');
            plot([x(i) x(i)], [q0 q1], 'k--', 'LineWidth', 1.1, 'HandleVisibility', 'off');
            plot([x(i) x(i)], [q3 q4], 'k--', 'LineWidth', 1.1, 'HandleVisibility', 'off');
            plot([x(i)-cap_half_width, x(i)+cap_half_width], [q0 q0], 'k-', 'LineWidth', 0.9, 'HandleVisibility', 'off');
            plot([x(i)-cap_half_width, x(i)+cap_half_width], [q4 q4], 'k-', 'LineWidth', 0.9, 'HandleVisibility', 'off');
            plot([x(i)-box_half_width, x(i)+box_half_width], [q2 q2], 'k-', 'LineWidth', 1.2, 'HandleVisibility', 'off');
        end
    end
    ylabel('Sum Throughput [Mbps]', 'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 15);
    set(gca, 'YColor', 'k');

    % Trục Y phải: Average Runtime (dấu sao đỏ to)
    yyaxis right;
    runtime_color = [0.84 0.23 0.16];
    h_rt = plot(x, times, 'p', ...
        'Color', runtime_color, ...
        'MarkerSize', 13, ...
        'MarkerFaceColor', runtime_color, ...
        'DisplayName', 'Avg. Runtime');
    ylabel('Average Runtime [s]', 'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 15);
    set(gca, 'YColor', 'k');

    set(gca, 'XTick', x, 'XTickLabel', names, 'Color', 'w', ...
             'LineWidth', 1.0, 'FontName', 'Times New Roman', 'FontSize', 11);
    box on; grid on;
    xlabel('Optimization Algorithms', 'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 15);
    title(sprintf('Sum Rate vs. Runtime ($N = %d, M = %d$, Runs = %d/%d)', N, M, run_idx, total_runs), ...
          'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 15);
    legend(h_rt, {'Avg. Runtime (s)'}, 'Location', 'northwest', ...
           'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 11);

    hold off;
    drawnow;
end

%% 4) Hàm Decode và tính toán hàm mục tiêu
function val = decode_and_calc_by_option(x, option, N, M, L, H_max, H_min, UE_pos, BS_pos, ...
                               a1, a2, K, mm, Pbs, Ptot, sb2, s2, Imax, gth, b, Pth, B_bs, B_uav, log)
    x = max(0, min(1, x));

    UAV_pos = zeros(N, 3);
    UAV_pos(:, 1) = x(1:N) * L;
    UAV_pos(:, 2) = x(N+1:2*N) * L;
    UAV_pos(:, 3) = (x(2*N+1:3*N) * (H_max-H_min)) + H_min;

    tail = x(3*N+1:end);

    switch option
        case 1
            association_vars = [];
            alloc_power_uav = [];
        case 2
            if numel(tail) ~= 2 * M
                error('Option 2 expects tail length 2*M (assoc + power).');
            end
            association_vars = tail(1:M);
            alloc_power_uav = tail(M+1:end);
        case 3
            if numel(tail) ~= 2 * M
                error('Option 3 expects tail length 2*M');
            end
            association_vars =  tail(1:M);
            alloc_power_uav = tail(M+1 : 2 * M);
        otherwise
            error('Unsupported option.');
    end

    if ~exist('log', 'var') || isempty(log)
        log = false;
    end

    val = Func_SumRate_UAV_by_option(UAV_pos, UE_pos, BS_pos, a1, a2, K, mm, ...
                Pbs, Ptot, association_vars, alloc_power_uav, sb2, s2, Imax, gth, b, Pth, option, B_bs, B_uav, log);
end


