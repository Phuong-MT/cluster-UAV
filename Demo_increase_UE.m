clear; clc;
addpath(genpath(pwd));

%% 1. Cấu hình Tham số
N = 5;                          % Số UAV cố định
M_list = 10:2:20;               % Các giá trị số UE cần quét (trục x)
L = 3000; 
H_min = 100; H_max = 300; BS_pos = [0, 0, 30];
% --- Kênh truyền (3GPP TR 36.777 / TR 38.901, fc = 2 GHz) ---
alpha1 = 2.2;   % BS->UAV: UMa-AV LoS
alpha2 = 2.5;   % UAV->UE: air-to-ground, gần LoS
K_R = 5; mm = 3;
I_max = 1; gamma_th = 0.2; beta = 0; P_th = 0.1;
B_bs = 50; B_uav = 1.08;   % MHz

% --- Công suất phát ---
P_bs_dBm  = 46;                          % 46 dBm = 40 W tổng
P_uav_dBm = 30;                          % 30 dBm = 1 W mỗi UAV
P_bs    = 10^((P_bs_dBm  - 30)/10);
P_total = 10^((P_uav_dBm - 30)/10) * N;

% --- Noise: -174 dBm/Hz + 10log10(B) + NF ---
NF_uav = 7; NF_ue = 9;
sigma_b2 = 10^((-174 + 10*log10(B_bs*1e6)  + NF_uav - 30)/10);
sigma2   = 10^((-174 + 10*log10(B_uav*1e6) + NF_ue  - 30)/10);

% Cấu hình thuật toán
POP_SIZE = 50;
generation = 8000;
NUM_RUNS = 30; VERBOSE = false;
MAX_FES = generation * POP_SIZE;

% Bật/tắt từng thuật toán
RUN_IDE = true;
RUN_RGA = true;
RUN_PSO = true;
RUN_GAQJ = true;

RUN_FIXED = true;
RUN_DYNAMIC = true;

% Lưu kết quả ra file .mat sau mỗi run
SAVE_RESULTS = false;
SAVE_FILE = 'results_increase_UE.mat';

% Vẽ đồ thị trung gian mỗi PLOT_EVERY runs (và ở run cuối cùng)
PLOT_EVERY = 5;

% Khai báo 2 biến global lưu mẫu small-scale Rician fading cho chặng 1 và chặng 2.
% Việc cố định mẫu fading trong mỗi (run, M) giúp hàm mục tiêu fobj(x) không bị
% nhảy ngẫu nhiên qua 400,000 lần đánh giá FES và đảm bảo mọi thuật toán
% đều được so sánh trên cùng một trạng thái kênh truyền vật lý.
global CURRENT_RIC_POW1 CURRENT_RIC_POW2;

num_M = numel(M_list);

fprintf('=== BẮT ĐẦU QUÉT SỐ LƯỢNG UE ===\n');
fprintf('Tham số: N = %d UAVs, M = [%s], Runs = %d, Max_FES = %d\n', ...
        N, num2str(M_list), NUM_RUNS, MAX_FES);
fprintf('P_bs = %.1f W, P_uav = %.1f W\n\n', P_bs, P_total/N);

%% Mảng lưu fitness cuối cùng: (num_M x NUM_RUNS)
fit_ide     = NaN(num_M, NUM_RUNS);
fit_rga     = NaN(num_M, NUM_RUNS);
fit_pso     = NaN(num_M, NUM_RUNS);
fit_gaqj    = NaN(num_M, NUM_RUNS);
fit_fixed   = NaN(num_M, NUM_RUNS);
fit_dynamic = NaN(num_M, NUM_RUNS);

%% Vòng lặp chính: Run ở ngoài, quét M ở trong
% (để tại mốc Run = 5, 10, ... đã có đủ dữ liệu cho mọi giá trị M)
for run = 1:NUM_RUNS
    fprintf('==================== Run %d/%d ====================\n', run, NUM_RUNS);
    UE_positions_all = generate_positions_3d(max(M_list), L, 0, true);
    for iM = 1:num_M
        M = M_list(iM);
        % 3N vị trí UAV + M association + M power UE
        dim = N * 3 + 2 * M;
        t_run = tic;

        % Vị trí UE độc lập cho mỗi (run, M)
        UE_positions = UE_positions_all(1:M, :);

        % Sinh mẫu công suất fading Rician (small-scale fading) cố định cho kịch bản (run, M) hiện tại.
        % Khi các thuật toán tối ưu dịch chuyển UAV, chỉ có Path Loss thay đổi theo khoảng cách 3D,
        % còn mẫu fading quy mô nhỏ này được giữ cố định trong suốt quá trình tìm kiếm của (run, M).
        CURRENT_RIC_POW1 = rician_fading(K_R, [N, 1]).^2;
        CURRENT_RIC_POW2 = rician_fading(K_R, [N, M]).^2;

        fobj = @(x, varargin) decode_and_calc_by_option(x, 3, N, M, L, H_max, H_min, UE_positions, BS_pos, ...
                alpha1, alpha2, K_R, mm, P_bs, P_total, sigma_b2, sigma2, I_max, gamma_th, beta, P_th, B_bs, B_uav, varargin{:});

        % 1. IDE (DE)
        if RUN_IDE
            [~, c_ide, ~] = run_ide(dim, fobj, MAX_FES, POP_SIZE, 0.1, 5, 1.5, VERBOSE);
            fit_ide(iM, run) = c_ide(end);
        end

        % 2. GA
        if RUN_RGA
            [~, c_rga, ~] = run_rga(dim, fobj, MAX_FES, POP_SIZE, VERBOSE);
            fit_rga(iM, run) = c_rga(end);
        end

        % 3. PSO
        if RUN_PSO
            [~, c_pso, ~] = run_pso(dim, fobj, MAX_FES, POP_SIZE, 0.7, 1.5, 1.5, VERBOSE);
            fit_pso(iM, run) = c_pso(end);
        end

        % 4. GA-QJ
        if RUN_GAQJ
            [~, c_gaqj, ~] = run_gaqj(dim, fobj, MAX_FES, POP_SIZE, VERBOSE);
            fit_gaqj(iM, run) = c_gaqj(end);
        end

        % --- Clustering baselines
        init_pos_cluster = rand(N, 3);
        init_pos_cluster(:,1:2) = init_pos_cluster(:,1:2) * L;
        init_pos_cluster(:,3) = (init_pos_cluster(:,3) * (H_max - H_min)) + H_min;

        % 5. Fixed clustering (K-means)
        if RUN_FIXED
            [clusters_fix, pos_fix] = fixed_clustering_3d(init_pos_cluster, UE_positions, N, H_min, H_max);
            fit_fixed(iM, run) = Func_SumRate_UAV(pos_fix, UE_positions, BS_pos, alpha1, alpha2, ...
                                        K_R, mm, P_bs, P_total, sigma_b2, sigma2, ...
                                        I_max, gamma_th, beta, P_th, B_bs, B_uav, clusters_fix);
        end

        % 6. Dynamic clustering
        if RUN_DYNAMIC
            P_b_n_vec = ones(N, 1) * (P_bs / N);
            P_n_vec   = ones(N, 1) * (P_total / N);
            [~, ~, ~, fit_dynamic(iM, run)] = dynamic_clustering_3d(init_pos_cluster, UE_positions, BS_pos, ...
                                        alpha1, alpha2, K_R, mm, P_b_n_vec, P_n_vec, ...
                                        sigma_b2, sigma2, I_max, gamma_th, 15, 1e-3, beta, P_th, B_bs, B_uav, H_min, H_max);
        end

        fprintf('  [Run %d] M = %d xong (%.1fs) | ', run, M, toc(t_run));
        if RUN_GAQJ,    fprintf('GA-QJ: %.3f | ',   fit_gaqj(iM, run)); end
        if RUN_IDE,     fprintf('DE: %.3f | ',      fit_ide(iM, run)); end
        if RUN_RGA,     fprintf('GA: %.3f | ',      fit_rga(iM, run)); end
        if RUN_PSO,     fprintf('PSO: %.3f | ',     fit_pso(iM, run)); end
        if RUN_FIXED,   fprintf('Fixed: %.3f | ',   fit_fixed(iM, run)); end
        if RUN_DYNAMIC, fprintf('Dynamic: %.3f | ', fit_dynamic(iM, run)); end
        fprintf('\n');
    end
    fprintf('\n');

    if SAVE_RESULTS
        save(SAVE_FILE, 'N', 'M_list', 'NUM_RUNS', 'MAX_FES', 'run', ...
             'fit_ide', 'fit_rga', 'fit_pso', 'fit_gaqj', 'fit_fixed', 'fit_dynamic');
    end

    % --- VẼ ĐỒ THỊ MỖI PLOT_EVERY RUNS HOẶC Ở RUN CUỐI (MỖI MỐC MỘT CỬA SỔ MỚI) ---
    if mod(run, PLOT_EVERY) == 0 || run == NUM_RUNS
        fprintf('>>> Đã hoàn thành %d/%d runs. Vẽ đồ thị cho mốc Run = %d...\n\n', run, NUM_RUNS, run);
        plot_sum_throughput_vs_M(run, NUM_RUNS, N, M_list, ...
            RUN_GAQJ,    mean(fit_gaqj(:, 1:run),    2, 'omitnan'), ...
            RUN_IDE,     mean(fit_ide(:, 1:run),     2, 'omitnan'), ...
            RUN_RGA,     mean(fit_rga(:, 1:run),     2, 'omitnan'), ...
            RUN_PSO,     mean(fit_pso(:, 1:run),     2, 'omitnan'), ...
            RUN_FIXED,   mean(fit_fixed(:, 1:run),   2, 'omitnan'), ...
            RUN_DYNAMIC, mean(fit_dynamic(:, 1:run), 2, 'omitnan'));
    end
end
fprintf('\n Done.\n');

%% Bảng tổng kết (trung bình trên tất cả các run)
avg_ide     = mean(fit_ide, 2, 'omitnan');
avg_rga     = mean(fit_rga, 2, 'omitnan');
avg_pso     = mean(fit_pso, 2, 'omitnan');
avg_gaqj    = mean(fit_gaqj, 2, 'omitnan');
avg_fixed   = mean(fit_fixed, 2, 'omitnan');
avg_dynamic = mean(fit_dynamic, 2, 'omitnan');

results_table = table(M_list(:), avg_gaqj, avg_ide, avg_rga, avg_pso, avg_fixed, avg_dynamic, ...
    'VariableNames', {'M', 'GA_QJ', 'DE', 'GA', 'PSO', 'Fixed', 'Dynamic'});
fprintf('\n=============== SUM THROUGHPUT [Mbps] THEO SỐ UE (%d RUNS) ===============\n', NUM_RUNS);
disp(results_table);

%% ==================== HÀM VẼ ĐỒ THỊ ====================
function plot_sum_throughput_vs_M(run_idx, total_runs, N, M_list, ...
    RUN_GAQJ, gaqj_avg, RUN_IDE, ide_avg, ...
    RUN_RGA, rga_avg, RUN_PSO, pso_avg, ...
    RUN_FIXED, fixed_avg, RUN_DYNAMIC, dynamic_avg)
% Vẽ Sum Throughput (trung bình trên run_idx run đầu) theo số UE.
% Mỗi lần gọi mở một cửa sổ figure mới, không ghi đè.

    figure('Color', 'w', ...
           'Name', sprintf('Sum Throughput vs M (Runs = %d)', run_idx), ...
           'NumberTitle', 'on');
    hold on;

    if RUN_GAQJ
        plot(M_list, gaqj_avg, '-d', 'Color', 'b', 'LineWidth', 2.0, ...
            'MarkerSize', 8, 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'w', ...
            'DisplayName', 'GA-QJ');
    end
    if RUN_IDE
        plot(M_list, ide_avg, '-s', 'Color', [0.9290 0.6940 0.1250], 'LineWidth', 2.0, ...
            'MarkerSize', 8, 'MarkerEdgeColor', [0.9290 0.6940 0.1250], 'MarkerFaceColor', 'w', ...
            'DisplayName', 'DE');
    end
    if RUN_RGA
        plot(M_list, rga_avg, '-^', 'Color', [0 0.5 0], 'LineWidth', 2.0, ...
            'MarkerSize', 8, 'MarkerEdgeColor', [0 0.5 0], 'MarkerFaceColor', 'w', ...
            'DisplayName', 'GA');
    end
    if RUN_PSO
        plot(M_list, pso_avg, '-o', 'Color', 'r', 'LineWidth', 2.0, ...
            'MarkerSize', 8, 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'w', ...
            'DisplayName', 'PSO');
    end
    if RUN_FIXED
        plot(M_list, fixed_avg, '--', 'Color', [0.25 0.25 0.25], 'LineWidth', 2.0, ...
            'DisplayName', 'Fixed Clustering');
    end
    if RUN_DYNAMIC
        plot(M_list, dynamic_avg, ':', 'Color', 'k', 'LineWidth', 2.0, ...
            'DisplayName', 'Dynamic');
    end

    hold off; grid on; box on;
    set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
             'FontName', 'Times New Roman', 'FontSize', 11, ...
             'TickLabelInterpreter', 'latex', ...
             'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.3, ...
             'XTick', M_list);
    if numel(M_list) > 1
        xlim([min(M_list), max(M_list)]);
    end

    xlabel('Number of UEs ($M$)', 'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 15);
    ylabel('Sum Throughput [Mbps]', 'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 15);
    title(sprintf('$N = %d$, Runs = %d/%d', N, run_idx, total_runs), ...
          'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 15);
    legend('show', 'Location', 'best', 'TextColor', 'k', 'Color', 'w', ...
           'EdgeColor', [0.8 0.8 0.8], 'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 11);
    drawnow;
end

%% ==================== HÀM HỖ TRỢ ====================
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
