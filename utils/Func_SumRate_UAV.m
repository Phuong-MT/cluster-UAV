function fitness = Func_SumRate_UAV(UAV_positions, UE_positions, BS_pos, ...
                                     alpha1, alpha2, K_R, mm, ...
                                     P_bs, P_total, sigma_b2, sigma2, ...
                                     I_max, gamma_th, beta, P_th, B_bs, B_uav, fixed_clusters)
% Func_SumRate_UAV: Tinh fitness = Sum Rate - Penalty cho mang UAV-NOMA 3D
%
% Ham nay thuc hien:
% 1. Tinh kenh truyen (First Hop & Second Hop)
% 2. Gan UE: dung fixed_clusters (neu truyen vao) hoac phan cum dong (theo kenh tot nhat)
% 3. Phan bo cong suat dong theo tai
% 4. Tinh Sum Rate dua tren NOMA
% 5. Ap dung penalty cho 3 constraints:
%    C1: P_out <= P_th   (Outage probability)
%    C2: interf <= I_max (Interference)
%    C3: sum(P_nm) <= P_n (Power budget)
%
% --- INPUTS ---
% fixed_clusters : (Tuy chon) Cell array kich thuoc {N, 1} chua cac UE thuoc tung cum (K-Means)
%
    if nargin < 17 || isempty(B_uav)
        B_uav = 0.1 * 1e6;
    end
    if nargin < 16 || isempty(B_bs)
        B_bs = 10 * 1e6;
    end
    if nargin < 15 || isempty(P_th)
        P_th = 0.1;
    end
    if nargin < 14 || isempty(beta)
        beta = 0;
    end

    [N, ~] = size(UAV_positions);
    [M, ~] = size(UE_positions);

    %% 1. Channel Gains
    first_hop_gains = compute_first_hop_gains(UAV_positions, BS_pos, alpha1, K_R);
    second_hop_gains = compute_second_hop_gains(UAV_positions, UE_positions, alpha2, mm);

    %% 2. User Association: Dùng cụm cố định (K-Means) nếu có, ngược lại gán theo kênh lớn nhất
    if nargin >= 18 && ~isempty(fixed_clusters)
        clusters = fixed_clusters;
    else
        clusters = cell(N, 1);
        [~, uav_indices] = max(second_hop_gains, [], 1);
        for u = 1:M
            u_idx = uav_indices(u);
            clusters{u_idx} = [clusters{u_idx}, u];
        end
    end

    %% 3. Power Allocation (load-based)
    P_b_n_dyn = zeros(N, 1);
    P_n_dyn = zeros(N, 1);
    for c_idx = 1:N
        num_in_cluster = length(clusters{c_idx});
        if num_in_cluster > 0
            P_n_dyn(c_idx) = P_total * num_in_cluster / M;
            P_b_n_dyn(c_idx) = P_bs / N;
        end
    end

    %% 4. NOMA Evaluation (returns constraint violations)
    [outage, sum_rate, power_violation, rate_violation] = ...
        evaluate_noma_clusters_3d(clusters, ...
            UAV_positions, UE_positions, ...
            first_hop_gains, second_hop_gains, ...
            P_b_n_dyn, P_n_dyn, sigma_b2, sigma2, ...
            I_max, gamma_th, beta, B_bs, B_uav);

    %% 5. Penalty Calculation
    % C1: Outage constraint  P_out <= P_th
    %     Violation = max(0, P_out - P_th), range [0, 1]
    %     lambda1 tuong doi lon vi outage co gia tri nho (0~1)
    lambda1 = 0.1;
    penalty_outage = lambda1 * max(0, outage - P_th);

    % C2: Power constraint  sum(P_nm) <= P_n
    %     Violation = tong cac phan vuot qua P_n
    lambda3 = 0.1;
    penalty_power = lambda3 * power_violation;

    % C3: Rate dual-hop constraint R_2^n <= R_1^n
    %   violation = tổng các phần vượt quá.
    lambda4 = 0.1;
    penalty_rate = lambda4 * rate_violation;
    
    %% 6. Fitness = Sum Rate - Total Penalty
    fitness = sum_rate - penalty_outage - penalty_power - penalty_rate;
end
