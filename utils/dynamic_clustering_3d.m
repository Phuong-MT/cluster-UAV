function [clusters, current_positions, outage, sum_rate] = dynamic_clustering_3d(UAV_positions, ...
                          UE_positions, BS_pos, alpha1, alpha2, K_R, m, ...
                          P_b_n, P_n, sigma_b2, sigma2, I_max, gamma_th, ...
                          max_iter, eps, beta, P_th, B_bs, B_uav, H_min, H_max)
    % Xử lý tham số tùy chọn với giá trị mặc định đầy đủ
    if nargin < 21 || isempty(H_max)
        H_max = 200;
    end
    if nargin < 20 || isempty(H_min)
        H_min = 100;
    end
    if nargin < 19 || isempty(B_uav)
        B_uav = 0.1 * 1e6;
    end
    if nargin < 18 || isempty(B_bs)
        B_bs = 10 * 1e6;
    end
    if nargin < 17 || isempty(P_th)
        P_th = 0.1;
    end
    if nargin < 16 || isempty(beta)
        beta = 0;
    end
    if nargin < 15 || isempty(eps)
        eps = 1e-3;
    end
    if nargin < 14 || isempty(max_iter)
        max_iter = 10;
    end

    [N, ~] = size(UAV_positions);
    [M, ~] = size(UE_positions);
    current_positions = UAV_positions;
    % Đảm bảo độ cao ban đầu nằm trong [H_min, H_max]
    current_positions(:, 3) = max(H_min, min(H_max, current_positions(:, 3)));
    e2e_rates = zeros(M, 1);

    for it = 1:max_iter
        % 1) Tính toán lại độ lợi kênh
        first_hop_gains = compute_first_hop_gains(current_positions, BS_pos, alpha1, K_R);
        second_hop_gains = compute_second_hop_gains(current_positions, UE_positions, alpha2, m);

        % 2) Tính tốc độ chặng 1 (đơn vị: bps)
        sinr_b = (P_b_n(:) .* first_hop_gains(:)) ./ sigma_b2;
        R_b = B_bs * log2(1 + sinr_b);

        % 3) Gán UE (với cập nhật nhiễu lặp lại)
        labels = -ones(M, 1); % -1 = chưa gán
        clusters = cell(N, 1);
        max_assign_iter = 10; 
        
        for assign_it = 1:max_assign_iter
            old_labels = labels;
            old_clusters = clusters;
            clusters = cell(N, 1); % Reset cụm mỗi lần lặp gán
            
            % Gán từng UE
            for ue_idx = 1:M
                best_rate = -1.0;
                best_uav = -1; 
                for n = 1:N
                    % Tính nhiễu dựa trên trạng thái vòng trước để tránh thiên vị thứ tự UE
                    if assign_it == 1 && all(old_labels == -1)
                        interf_nm = compute_interference_nested(ue_idx, n, clusters, second_hop_gains, N, P_n);
                    else
                        interf_nm = compute_interference_nested(ue_idx, n, old_clusters, second_hop_gains, N, P_n);
                    end
                    
                    sinr_nm = (P_n(n) * second_hop_gains(n, ue_idx)) / (sigma2 + interf_nm);
                    R_nm = B_uav * log2(1 + sinr_nm); % Cùng đơn vị bps với R_b
                    R_e2E = min(R_b(n), R_nm);
                    if R_e2E > best_rate
                        best_rate = R_e2E;
                        best_uav = n;
                    end
                end
                
                % Fallback gán vào UAV có độ lợi kênh tốt nhất nếu không tìm được
                if best_uav == -1
                    [~, best_uav] = max(second_hop_gains(:, ue_idx));
                end
                
                labels(ue_idx) = best_uav;
                clusters{best_uav} = [clusters{best_uav}, ue_idx];
            end
            if all(old_labels == labels)
                break; % Hội tụ gán
            end
        end

        % 4) Sắp xếp NOMA + phân bổ công suất
        power_violation = 0;    % Accumulated power violation
        e2e_rates(:) = 0.0;
        rate_violation = 0;
        for n = 1:N
            c_ues = clusters{n};
            if isempty(c_ues)
                continue;
            end
            
            gains_c = second_hop_gains(n, c_ues);
            num_ues = length(c_ues);
            
            p_allocs = [];
            total_alloc = 0;

            if num_ues > 0
                power_weights = (gains_c.^(-beta)) ./ sum(gains_c.^(-beta));
                p_allocs = P_n(n) * power_weights;
                % SIC order: sort UEs by channel gain ascending (weak -> strong)
                [~, idx_sorted_mat] = sort(gains_c, 'ascend');
                sorted_ues = c_ues(idx_sorted_mat);
                p_allocs = p_allocs(idx_sorted_mat);
                total_alloc = sum(p_allocs);
            end
            pwr_excess = max(0, total_alloc - P_n(n));
            if pwr_excess > 0
                power_violation = power_violation + pwr_excess;
            end
            
            for rank_mat = 1:num_ues
                ue_idx = sorted_ues(rank_mat);
                h_nm = second_hop_gains(n, ue_idx);
                interf_nm = compute_interference_nested(ue_idx, n, clusters, second_hop_gains, N, P_n);
                
                intra_power = sum(p_allocs((rank_mat + 1):end));
                interf_intra = intra_power * h_nm;

                sinr_nom = (p_allocs(rank_mat) * h_nm) / (sigma2 + interf_intra + interf_nm);
                R_nom = B_uav * log2(1 + sinr_nom);
                e2e_rates(ue_idx) = R_nom;
            end
            rate_violation = rate_violation + max(0, sum(e2e_rates(sorted_ues)) - R_b(n));
        end

        % 5) Định vị lại UAV: chỉ cập nhật tọa độ 2D (x, y), bảo toàn độ cao z
        new_positions = current_positions;
        for n = 1:N
            if ~isempty(clusters{n})
                new_positions(n, 1:2) = mean(UE_positions(clusters{n}, 1:2), 1);
            end
        end
        % Kẹp độ cao an toàn trong khoảng [H_min, H_max]
        new_positions(:, 3) = max(H_min, min(H_max, new_positions(:, 3)));
        
        current_positions = new_positions;
    end

    % Đánh giá cuối cùng
    R_th = B_uav * log2(1 + gamma_th);
    outage = sum(e2e_rates < R_th) / M;

    lambda1 = 0.1;
    penalty_outage = lambda1 * max(0, outage - P_th);

    lambda3 = 0.1;
    penalty_power = lambda3 * power_violation;

    lambda4 = 0.1;
    penalty_rate = lambda4 * rate_violation;

    sum_rate = sum(e2e_rates) - penalty_outage - penalty_power - penalty_rate;

end

% --- Hàm lồng (Nested Function) ---
% MATLAB cho phép định nghĩa hàm bên trong hàm khác.
% Hàm này có thể truy cập các biến của hàm cha (như second_hop_gains, N, P_n)
function interf_nm = compute_interference_nested(ue_idx, uav_idx, clusters, second_hop_gains, N, P_n)
    interf_nm = 0.0;
    for j = 1:N
        if j ~= uav_idx && ~isempty(clusters{j}) % Chỉ UAV đang hoạt động
            interf_nm = interf_nm + P_n(j) * second_hop_gains(j, ue_idx);
        end
    end
    interf_nm = max(interf_nm, 1e-24); % Tránh nhiễu bằng 0
end
