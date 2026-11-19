clear; clc;
addpath(genpath(pwd));

%% =========================================================================
%  DEMO_SINGLE_HOP: Đánh giá mạng 1 chặng trực tiếp (Direct BS -> UE)
%  không sử dụng UAV Relay và so sánh với mạng 2 chặng (Dual-Hop BS -> UAV -> UE).
%
%  Mục đích học thuật:
%  - Chứng minh khi kích thước vùng mạng lớn (ví dụ L = 4000 m), đường truyền
%    trực tiếp 1 chặng dưới mặt đất (BS -> UE) chịu suy hao NLoS nặng
%    (alpha_direct = 3.5 ~ 3.67, Rayleigh/Nakagami m=1 theo 3GPP TR 38.901 UMa-NLoS),
%    dẫn đến Sum Throughput cực thấp và Outage Probability rất cao.
%  - Làm nổi bật vai trò bắt buộc của hệ thống Dual-Hop UAV-NOMA (H in [100, 300] m)
%    nhờ chuyển đổi đường truyền NLoS mặt đất thành 2 chặng LoS Rician trên không.
%% =========================================================================

%% 1. Cấu hình Tham số Hệ thống
N = 5;                          % Số UAV (dùng cho các phương án Dual-Hop để đối chiếu)
M_list = 10:2:20;               % Các giá trị số lượng UE cần quét (trục X)
L = 4000;                        % Kích thước vùng mạng 4000m x 4000m (16 km^2)
H_min = 100; H_max = 300;        % Độ cao UAV [m]
BS_pos = [0, 0, 30];             % Vị trí trạm gốc Macro BS [m]
fc = 2e9;                        % Tần số sóng mang 2 GHz (3GPP TR 36.777)

% --- Tham số Kênh truyền Dual-Hop (BS -> UAV -> UE: LoS Rician) ---
alpha1 = 2.2;                    % Chặng 1 (BS -> UAV): UMa-AV LoS
alpha2 = 2.5;                    % Chặng 2 (UAV -> UE): A2G LoS/gần LoS
K_R = 5; mm = 3;

% --- Tham số Kênh truyền Single-Hop (Direct BS -> UE: Mặt đất NLoS) ---
% Theo 3GPP TR 38.901 (UMa NLoS dưới mặt đất): hệ số suy hao đường truyền ~ 3.5 - 3.67,
% bị che khuất bởi nhà cửa/cây cối nên không có tia LoS (Rayleigh fading: K = 0).
alpha_direct = 3.6;              % Hệ số suy hao đường truyền mặt đất BS -> UE (NLoS)
K_direct = 0;                    % K = 0 tương đương Rayleigh fading (không có LoS)

% --- Ràng buộc & Băng thông ---
I_max = 1; gamma_th = 0.2; beta = 0; P_th = 0.1;
B_bs = 50;                       % Băng thông backhaul BS [MHz]
B_uav = 1.08;                    % Băng thông truy nhập UE [MHz] (6 PRB LTE-M)

% --- Công suất phát ---
P_bs_dBm  = 46;                  % Macro BS: 46 dBm = 40 W
P_uav_dBm = 30;                  % UAV Relay: 30 dBm = 1 W mỗi UAV
P_bs    = 10^((P_bs_dBm  - 30)/10);
P_total = 10^((P_uav_dBm - 30)/10) * N;

% --- Công suất nhiễu nền (Thermal Noise + Noise Figure) ---
NF_uav = 7; NF_ue = 9;
sigma_b2 = 10^((-174 + 10*log10(B_bs*1e6)  + NF_uav - 30)/10);
sigma2   = 10^((-174 + 10*log10(B_uav*1e6) + NF_ue  - 30)/10);

% --- Cấu hình số lần chạy Monte Carlo ---
NUM_RUNS = 30;
PLOT_EVERY = 5;                 % Vẽ đồ thị trung gian mỗi PLOT_EVERY runs (và ở run cuối)
RUN_DUAL_HOP_META = true;       % Đặt true nếu muốn chạy thêm GA-QJ Dual-Hop (tốn thời gian hơn)
POP_SIZE = 30;
generation = 5000;
MAX_FES = generation * POP_SIZE;
VERBOSE = false;

% Khai báo biến global cố định mẫu Rician fading trong mỗi kịch bản (run, M)
global CURRENT_RIC_POW1 CURRENT_RIC_POW2;

num_M = numel(M_list);

fprintf('=== ĐÁNH GIÁ SINGLE-HOP (DIRECT BS->UE) VS DUAL-HOP (BS->UAV->UE) ===\n');
fprintf('Vùng mạng: L = %d m (%.1f km^2) | N = %d UAVs | M = [%s] UEs | Runs = %d\n', ...
        L, (L/1000)^2, N, num2str(M_list), NUM_RUNS);
fprintf('Single-Hop (BS->UE) : alpha_direct = %.2f, Rayleigh (K = %d), P_bs = %.1f W\n', ...
        alpha_direct, K_direct, P_bs);
fprintf('Dual-Hop (BS->UAV->UE): alpha1 = %.2f, alpha2 = %.2f, Rician (K_R = %d), P_uav = %.1f W/UAV\n\n', ...
        alpha1, alpha2, K_R, P_total/N);

%% 2. Khởi tạo mảng lưu kết quả (num_M x NUM_RUNS)
% Sum Throughput [Mbps]
rate_single_oma      = NaN(num_M, NUM_RUNS); % 1 chặng BS->UE (OMA chia đều băng thông/công suất)
rate_single_noma     = NaN(num_M, NUM_RUNS); % 1 chặng BS->UE (Power-Domain NOMA từ BS)
rate_dual_fixed      = NaN(num_M, NUM_RUNS); % 2 chặng UAV-NOMA (Fixed K-Means Clustering)
rate_dual_dynamic    = NaN(num_M, NUM_RUNS); % 2 chặng UAV-NOMA (Dynamic Clustering)
rate_dual_gaqj       = NaN(num_M, NUM_RUNS); % 2 chặng UAV-NOMA (GA-QJ Optimization)

%% 3. Vòng lặp Mô phỏng Monte Carlo
for run = 1:NUM_RUNS
    UE_positions_all = generate_positions_3d(max(M_list), L, 0, true);
    for iM = 1:num_M
        M = M_list(iM);

        % Sinh vị trí UE ngẫu nhiên trên vùng L x L
        UE_positions = UE_positions_all(1:M, :);

        % Cố định mẫu fading Rician cho Dual-Hop trong kịch bản (run, M)
        CURRENT_RIC_POW1 = rician_fading(K_R, [N, 1]).^2;
        CURRENT_RIC_POW2 = rician_fading(K_R, [N, M]).^2;

        % -----------------------------------------------------------------
        % A. ĐÁNH GIÁ MẠNG 1 CHẶNG TRỰC TIẾP (SINGLE-HOP: BS -> UE)
        % -----------------------------------------------------------------
        [r_sh_oma, ~, r_sh_noma, ~] = ...
            evaluate_single_hop_bs_ue(BS_pos, UE_positions, alpha_direct, K_direct, ...
                                      P_bs, P_total, sigma2, gamma_th, P_th, beta, B_uav, fc);

        rate_single_oma(iM, run)  = r_sh_oma;
        rate_single_noma(iM, run) = r_sh_noma;

        % -----------------------------------------------------------------
        % B. ĐÁNH GIÁ MẠNG 2 CHẶNG QUA UAV (DUAL-HOP: BS -> UAV -> UE)
        % -----------------------------------------------------------------
        init_pos_cluster = rand(N, 3);
        init_pos_cluster(:, 1:2) = init_pos_cluster(:, 1:2) * L;
        init_pos_cluster(:, 3)   = init_pos_cluster(:, 3) * (H_max - H_min) + H_min;

        % 1) Dual-Hop với Fixed Clustering (K-Means)
        [clusters_fix, pos_fix] = fixed_clustering_3d(init_pos_cluster, UE_positions, N, H_min, H_max);
        [out_fix, sr_fix] = eval_dual_hop_metrics(pos_fix, UE_positions, BS_pos, ...
                                alpha1, alpha2, K_R, mm, P_bs, P_total, ...
                                sigma_b2, sigma2, I_max, gamma_th, beta, P_th, B_bs, B_uav, clusters_fix);
        rate_dual_fixed(iM, run) = sr_fix;
        out_dual_fixed(iM, run)  = out_fix;

        % 2) Dual-Hop với Dynamic Clustering
        P_b_n_vec = ones(N, 1) * (P_bs / N);
        P_n_vec   = ones(N, 1) * (P_total / N);
        [~, ~, out_dyn, sr_dyn] = dynamic_clustering_3d(init_pos_cluster, UE_positions, BS_pos, ...
                                    alpha1, alpha2, K_R, mm, P_b_n_vec, P_n_vec, ...
                                    sigma_b2, sigma2, I_max, gamma_th, 15, 1e-3, beta, P_th, B_bs, B_uav, H_min, H_max);
        rate_dual_dynamic(iM, run) = sr_dyn;
        out_dual_dynamic(iM, run)  = out_dyn;

        % 3) (Tùy chọn) Dual-Hop tối ưu bằng GA-QJ
        if RUN_DUAL_HOP_META
            dim = 3 * N + 2 * M;
            fobj = @(x, varargin) decode_and_calc_by_option(x, 3, N, M, L, H_max, H_min, UE_positions, BS_pos, ...
                    alpha1, alpha2, K_R, mm, P_bs, P_total, sigma_b2, sigma2, I_max, gamma_th, beta, P_th, B_bs, B_uav, varargin{:});
            [~, c_gaqj, ~] = run_gaqj(dim, fobj, MAX_FES, POP_SIZE, VERBOSE);
            rate_dual_gaqj(iM, run) = c_gaqj(end);
        end
    end

    % --- VẼ ĐỒ THỊ MỖI PLOT_EVERY RUNS HOẶC Ở RUN CUỐI (MỖI MỐC MỘT CỬA SỔ MỚI) ---
    if mod(run, PLOT_EVERY) == 0 || run == NUM_RUNS
        fprintf('>>> Đã hoàn thành %d/%d runs. Vẽ đồ thị cho mốc Run = %d...\n', run, NUM_RUNS, run);
        plot_single_vs_dual_hop(run, NUM_RUNS, L, N, M_list, RUN_DUAL_HOP_META, ...
            mean(rate_dual_gaqj(:, 1:run),    2, 'omitnan'), ...
            mean(rate_dual_dynamic(:, 1:run), 2, 'omitnan'), ...
            mean(rate_dual_fixed(:, 1:run),   2, 'omitnan'), ...
            mean(rate_single_noma(:, 1:run),  2, 'omitnan'), ...
            mean(rate_single_oma(:, 1:run),   2, 'omitnan'));
    end
end

%% 4. Tổng hợp & In Bảng Kết quả Trung bình
avg_rate_sh_oma   = mean(rate_single_oma,   2, 'omitnan');
avg_rate_sh_noma  = mean(rate_single_noma,  2, 'omitnan');
avg_rate_dh_fix   = mean(rate_dual_fixed,   2, 'omitnan');
avg_rate_dh_dyn   = mean(rate_dual_dynamic, 2, 'omitnan');

fprintf('\n==================== BẢNG 1: SUM THROUGHPUT [Mbps] (L = %d m, %d RUNS) ====================\n', L, NUM_RUNS);
if RUN_DUAL_HOP_META
    avg_rate_dh_gaqj = mean(rate_dual_gaqj, 2, 'omitnan');
    T_rate = table(M_list(:), avg_rate_sh_oma, avg_rate_sh_noma, avg_rate_dh_fix, avg_rate_dh_dyn, avg_rate_dh_gaqj, ...
        'VariableNames', {'M', 'SingleHop_OMA', 'SingleHop_NOMA', 'DualHop_Fixed', 'DualHop_Dynamic', 'DualHop_GAQJ'});
else
    T_rate = table(M_list(:), avg_rate_sh_oma, avg_rate_sh_noma, avg_rate_dh_fix, avg_rate_dh_dyn, ...
        'VariableNames', {'M', 'SingleHop_OMA', 'SingleHop_NOMA', 'DualHop_Fixed', 'DualHop_Dynamic'});
end
disp(T_rate);

%% =========================================================================
%  HÀM VẼ ĐỒ THỊ (ĐỒNG BỘ MÀU SẮC, MARKER VÀ FONT VỚI DEMO_CONVERGENCE)
%% =========================================================================
function plot_single_vs_dual_hop(run_idx, total_runs, L, N, M_list, ...
    RUN_DUAL_HOP_META, gaqj_avg, dynamic_avg, fixed_avg, sh_noma_avg, sh_oma_avg)
% Vẽ Sum Throughput (trung bình trên run_idx run đầu) theo số UE.
% Mỗi lần gọi mở một cửa sổ figure mới, không ghi đè.
% Bảng màu & marker đồng bộ với Demo_convergence_N20_M40.m:
%   - Dual-Hop GA-QJ:   '-d', 'b'
%   - Dual-Hop Dynamic: ':',  'k'
%   - Dual-Hop Fixed:   '--', [0.25 0.25 0.25]
%   - Single-Hop NOMA:  '-o', 'r'
%   - Single-Hop OMA:   '-^', [0 0.5 0]

    figure('Color', 'w', ...
           'Name', sprintf('Single-Hop vs Dual-Hop (L = %d m, Runs = %d)', L, run_idx), ...
           'NumberTitle', 'on');
    hold on;

    if RUN_DUAL_HOP_META
        plot(M_list, gaqj_avg, '-d', 'Color', 'b', 'LineWidth', 2.0, ...
            'MarkerSize', 8, 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'w', ...
            'DisplayName', 'Dual-Hop UAV-NOMA (GA-QJ)');
    end

    plot(M_list, dynamic_avg, ':', 'Color', 'k', 'LineWidth', 2.0, ...
        'DisplayName', 'Dual-Hop UAV-NOMA (Dynamic)');

    plot(M_list, fixed_avg, '--', 'Color', [0.25 0.25 0.25], 'LineWidth', 2.0, ...
        'DisplayName', 'Dual-Hop UAV-NOMA (Fixed Clustering)');

    plot(M_list, sh_noma_avg, '-o', 'Color', 'r', 'LineWidth', 2.0, ...
        'MarkerSize', 8, 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'w', ...
        'DisplayName', 'Single-Hop Direct BS-UE (NOMA)');

    plot(M_list, sh_oma_avg, '-^', 'Color', [0 0.5 0], 'LineWidth', 2.0, ...
        'MarkerSize', 8, 'MarkerEdgeColor', [0 0.5 0], 'MarkerFaceColor', 'w', ...
        'DisplayName', 'Single-Hop Direct BS-UE (OMA)');

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
    title(sprintf('$L = %d$ m, $N = %d$ (Runs = %d/%d)', L, N, run_idx, total_runs), ...
          'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 15);
    legend('show', 'Location', 'best', 'TextColor', 'k', 'Color', 'w', ...
           'EdgeColor', [0.8 0.8 0.8], 'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 11);
    drawnow;
end
%% =========================================================================
%  CÁC HÀM HỖ TRỢ TÍNH TOÁN SINGLE-HOP VÀ DUAL-HOP
%% =========================================================================

function [sum_rate_oma, outage_oma, sum_rate_noma, outage_noma] = ...
    evaluate_single_hop_bs_ue(BS_pos, UE_positions, alpha_direct, K_direct, ...
                              P_bs, P_uav_equiv, sigma2, gamma_th, P_th, beta, B_access, fc)
% Đánh giá truyền dẫn 1 chặng trực tiếp từ BS tới M người dùng trên mặt đất (không qua UAV)
% trên cùng băng thông truy nhập của UE.
% Hỗ trợ 2 chế độ Single-Hop:
%   1) Single-Hop OMA (TDMA/FDMA): BS chia đều công suất P_bs và băng thông cho M UE.
%   2) Single-Hop NOMA: BS chia thành N cụm ảo (hoặc phát NOMA trên băng thông B_access)
%      với công suất P_bs và khử nhiễu SIC trực tiếp từ BS xuống UE.

    M = size(UE_positions, 1);
    dist_bs_ue = zeros(M, 1);
    for m = 1:M
        dist_bs_ue(m) = dist_3d(BS_pos, UE_positions(m, :));
    end

    % Path loss tham chiếu ở d0 = 1 m
    PL0 = (3e8 / (4 * pi * fc))^2;
    pl_direct = PL0 * (dist_bs_ue .^ (-alpha_direct));

    % Fading mặt đất (Rayleigh khi K_direct = 0)
    fading_amp = rician_fading(K_direct, [M, 1]);
    h_direct = (fading_amp .^ 2) .* pl_direct; % (M x 1)

    R_th = B_access * log2(1 + gamma_th);

    % --- 1) Single-Hop OMA (Chia đều công suất P_bs/M trên từng khe/kênh) ---
    % Mỗi UE được cấp 1/M thời gian/tần số với công suất đỉnh P_bs (tương đương SNR = P_bs*h/sigma2)
    sinr_oma = (P_bs .* h_direct) ./ sigma2;
    rates_oma = (B_access / M) * log2(1 + sinr_oma);
    outage_oma = sum(rates_oma < R_th) / M;
    sum_rate_oma = sum(rates_oma) -  0.1 * max(0, outage_oma - P_th);

    % --- 2) Single-Hop NOMA (BS phát NOMA trực tiếp cho các UE trên băng thông B_access) ---
    % Sắp xếp độ lợi kênh tăng dần (weak -> strong) để thực hiện SIC
    [h_sorted, ~] = sort(h_direct, 'ascend');
    weights = (h_sorted .^ (-beta)) ./ sum(h_sorted .^ (-beta));
    p_alloc = P_bs * weights;

    rates_noma = zeros(M, 1);
    for r = 1:M
        intra_interf = sum(p_alloc(r+1:end)) * h_sorted(r);
        sinr_r = (p_alloc(r) * h_sorted(r)) / (sigma2 + intra_interf);
        rates_noma(r) = B_access * log2(1 + sinr_r);
    end
    outage_noma = sum(rates_noma < R_th) / M;
    sum_rate_noma = sum(rates_noma) - 0.1 * max(0, outage_noma - P_th);
end

function [outage, fitness] = eval_dual_hop_metrics(UAV_positions, UE_positions, BS_pos, ...
                                     alpha1, alpha2, K_R, mm, P_bs, P_total, ...
                                     sigma_b2, sigma2, I_max, gamma_th, beta, P_th, B_bs, B_uav, clusters)
    [N, ~] = size(UAV_positions);
    [M, ~] = size(UE_positions);

    first_hop_gains  = compute_first_hop_gains(UAV_positions, BS_pos, alpha1, K_R);
    second_hop_gains = compute_second_hop_gains(UAV_positions, UE_positions, alpha2, mm);

    P_b_n_dyn = zeros(N, 1);
    P_n_dyn   = zeros(N, 1);
    for c_idx = 1:N
        num_in_cluster = length(clusters{c_idx});
        if num_in_cluster > 0
            P_n_dyn(c_idx)   = P_total * num_in_cluster / M;
            P_b_n_dyn(c_idx) = P_bs / N;
        end
    end

    [outage, sum_rate, power_violation, rate_violation] = ...
        evaluate_noma_clusters_3d(clusters, UAV_positions, UE_positions, ...
            first_hop_gains, second_hop_gains, P_b_n_dyn, P_n_dyn, ...
            sigma_b2, sigma2, I_max, gamma_th, beta, B_bs, B_uav);

    fitness = sum_rate - 0.1 * max(0, outage - P_th) - 0.1 * power_violation - 0.1 * rate_violation;
end

function val = decode_and_calc_by_option(x, option, N, M, L, H_max, H_min, UE_pos, BS_pos, ...
                               a1, a2, K, mm, Pbs, Ptot, sb2, s2, Imax, gth, b, Pth, B_bs, B_uav, log)
    x = max(0, min(1, x));
    UAV_pos = zeros(N, 3);
    UAV_pos(:, 1) = x(1:N) * L;
    UAV_pos(:, 2) = x(N+1:2*N) * L;
    UAV_pos(:, 3) = (x(2*N+1:3*N) * (H_max - H_min)) + H_min;

    tail = x(3*N+1:end);
    association_vars = tail(1:M);
    alloc_power_uav  = tail(M+1:2*M);

    if ~exist('log', 'var') || isempty(log)
        log = false;
    end

    val = Func_SumRate_UAV_by_option(UAV_pos, UE_pos, BS_pos, a1, a2, K, mm, ...
                Pbs, Ptot, association_vars, alloc_power_uav, sb2, s2, Imax, gth, b, Pth, option, B_bs, B_uav, log);
end
