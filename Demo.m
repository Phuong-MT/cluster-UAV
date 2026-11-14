clear; clc;
addpath(genpath(pwd));

%% 1. Cấu hình Tham số
N = 10; M = 20;
L = 1000; H_min = 100; H_max = 200; BS_pos = [0, 0, 30];
alpha1 = 2.5; alpha2 = 3.0; K_R = 5; mm = 3;
sigma_b2 = 1e-9; sigma2 = 1e-9; I_max = 1; gamma_th = 1.0;
P_total = 10 * N; beta = 0; P_th = 0.1;
B_bs = 10; B_uav = 0.1;

P_bs = 10 * N; 

% Cấu hình thuật toán
POP_SIZE = 50;
generation = 5000;
NUM_RUNS = 30; VERBOSE = true;
MAX_FES = generation * POP_SIZE;
iters = round(MAX_FES / POP_SIZE);

% Bật/tắt từng thuật toán
RUN_IDE = true;
RUN_RGA = true;
RUN_PSO = true;
RUN_GAQJ = true;

RUN_FIXED = true;
RUN_DYNAMIC = true;
% 3N vị trí UAV + M association + M power UE
dim = N * 3 + 2 * M; 

fprintf('=== BẮT ĐẦU QUÉT BS POWER (IDE vs CLUSTERING) ===\n');
fprintf('\nĐang chạy P_bs = %.1f W...\n', P_bs);
%% vòng lặp chính

temp_conv_ide = zeros(1, iters);
temp_conv_rga = zeros(1, iters);
temp_conv_pso = zeros(1, iters);
temp_conv_gaqj = zeros(1, iters);

temp_val_fixed = 0;
temp_val_dynamic = 0;
for run = 1: NUM_RUNS
    % vị trí thiết bị người dùng
    UE_positions = generate_positions_3d(M, L, 0, true);
    
    %metaheuristic
    %hàm mục tiêu        
    fobj = @(x, varargin) decode_and_calc_by_option(x, 3, N, M, L, H_max, H_min, UE_positions, BS_pos, ...
            alpha1, alpha2, K_R, mm, P_bs, P_total, sigma_b2, sigma2, I_max, gamma_th, beta, P_th, B_bs, B_uav, varargin{:});
    %1. IDE
    if RUN_IDE
        [~, c_ide, ~] = run_ide(dim, fobj, MAX_FES, POP_SIZE, 0.1, 5, 1.5, VERBOSE);
        temp_conv_ide = temp_conv_ide + c_ide(:).';
    end
    %2. GA
    if RUN_RGA
        [~, c_rga, ~] = run_rga(dim, fobj, MAX_FES, POP_SIZE, VERBOSE);
        temp_conv_rga = temp_conv_rga + c_rga(:).';
    end
    %3. PSO
    if RUN_PSO
        [~, c_pso, ~] = run_pso(dim, fobj, MAX_FES, POP_SIZE, 0.7, 1.5, 1.5, VERBOSE);
        temp_conv_pso = temp_conv_pso + c_pso(:).';
    end
    %4. GA-QJ
    if RUN_GAQJ
        [~, c_gaqj, ~] = run_gaqj(dim, fobj, MAX_FES, POP_SIZE, VERBOSE);
        temp_conv_gaqj = temp_conv_gaqj + c_gaqj(:).';
    end    

    % --- Run fixed clustering (K-means) and Dynamic clustering
    %5. Fixed clustering (K-means)
    init_pos_cluster = rand(N, 3);
    init_pos_cluster(:,1:2) = init_pos_cluster(:,1:2) * L;
    init_pos_cluster(:,3) = (init_pos_cluster(:,3) * (H_max - H_min)) + H_min;
    if RUN_FIXED
        [clusters_fix, pos_fix] = fixed_clustering_3d(init_pos_cluster, UE_positions, N, H_min, H_max);
        val_fix = Func_SumRate_UAV(pos_fix, UE_positions, BS_pos, alpha1, alpha2, ...
                                    K_R, mm, P_bs, P_total, sigma_b2, sigma2, ...
                                    I_max, gamma_th, beta, P_th, B_bs, B_uav, clusters_fix);
        temp_val_fixed = temp_val_fixed + val_fix;
        disp("fixed: " + val_fix);
    end
    %6. Dynamic clustering
    if RUN_DYNAMIC
        P_b_n_vec = ones(N, 1) * (P_bs / N);
        P_n_vec   = ones(N, 1) * (P_total / N);
        [~, ~, ~, val_dyn] = dynamic_clustering_3d(init_pos_cluster, UE_positions, BS_pos, ...
                                    alpha1, alpha2, K_R, mm, P_b_n_vec, P_n_vec, ...
                                    sigma_b2, sigma2, I_max, gamma_th, 15, 1e-3, beta, P_th, B_bs, B_uav, H_min, H_max);
        temp_val_dynamic = temp_val_dynamic + val_dyn;
        disp("dynamic: " + val_dyn);
    end
end
fprintf('\n Done.\n');

if RUN_IDE
    ide_convergence_history = temp_conv_ide / NUM_RUNS;
end    

if RUN_RGA
    rga_convergence_history = temp_conv_rga / NUM_RUNS;
end    

if RUN_PSO
    pso_convergence_history = temp_conv_pso / NUM_RUNS;
end

if RUN_GAQJ
    gaqj_convergence_history = temp_conv_gaqj / NUM_RUNS;
end

if RUN_FIXED
    fixed_final_val = temp_val_fixed / NUM_RUNS;
end

if RUN_DYNAMIC
    dynamic_final_val = temp_val_dynamic / NUM_RUNS;
end 
%% Vẽ đồ thị Convergence (Cho trường hợp P_bs lớn nhất)
figure;
hold on;
set(gcf, 'Color', 'w');
marker_indices = round(linspace(1, iters, 10)); % Chọn 10 điểm để vẽ marker

if RUN_GAQJ
    plot(1:iters, gaqj_convergence_history, '-d', ...
        'Color', 'b', 'LineWidth', 2.0, ...
        'MarkerIndices', marker_indices, 'MarkerSize', 8, ...
        'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'w', ...
        'DisplayName', 'GA-QJ');
end

if RUN_IDE
   plot(1:iters, ide_convergence_history, '-s', ...
       'Color', [0.9290 0.6940 0.1250], 'LineWidth', 2.0, ...
       'MarkerIndices', marker_indices, 'MarkerSize', 8, ...
       'MarkerEdgeColor', [0.9290 0.6940 0.1250], 'MarkerFaceColor', 'w', ...
       'DisplayName', 'DE');
end    

if RUN_RGA
    plot(1:iters, rga_convergence_history, '-^', ...
        'Color', [0 0.5 0], 'LineWidth', 2.0, ...
        'MarkerIndices', marker_indices, 'MarkerSize', 8, ...
        'MarkerEdgeColor', [0 0.5 0], 'MarkerFaceColor', 'w', ...
        'DisplayName', 'GA');
end    

if RUN_PSO
    plot(1:iters, pso_convergence_history, '-o', ...
        'Color', 'r', 'LineWidth', 2.0, ...
        'MarkerIndices', marker_indices, 'MarkerSize', 8, ...
        'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'w', ...
        'DisplayName', 'PSO');
end

% Các đường cấu hình khác (giữ màu tối để dễ phân biệt)
if RUN_FIXED
   yline(fixed_final_val, '--', 'Color', [0.25 0.25 0.25], 'LineWidth', 2.0, 'DisplayName', 'Fixed Clustering');
end

if RUN_DYNAMIC
    yline(dynamic_final_val, ':', 'Color', 'k', 'LineWidth', 2.0, 'DisplayName', 'Dynamic');
end  

hold off; grid on;
% Cấu hình Font và LaTeX cho hệ trục tọa độ (Axes Tick Labels)
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.0, ...
         'FontName', 'Times New Roman', 'FontSize', 11, ...
         'TickLabelInterpreter', 'latex', ...
         'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.3);
box on;

% Cấu hình Text (xlabel, ylabel, title, legend) dùng Times New Roman và LaTeX
xlabel('Generations', 'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 15);
ylabel('Sum Throughput [Mbps]', 'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 15);
title(['$N = ' num2str(N) ', M = ' num2str(M) '$'], 'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 15);
legend('show', 'Location', 'best', 'TextColor', 'k', 'Color', 'w', ...
       'EdgeColor', [0.8 0.8 0.8], 'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 11);

%% helper
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
            % biến phân bổ công suất cho chặng 2
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
