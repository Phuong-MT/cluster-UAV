function [outage, sum_rate, power_violation, rate_violation] = evaluate_noma_dim3N_M(clusters, ...
                          UAV_positions, UE_positions, ...
                          first_hop_gains, second_hop_gains, ...
                          P_b_n, P_n, UAV_p_alloc_normalize, sigma_b2, sigma2, ...
                          I_max, gamma_th, beta, B_bs, B_uav, log)
% evaluate_noma_dim3N_M: NOMA evaluation for decision vlogector dim = 3N + M.
% The last M variables are per-UE power priorities and aredecisi normalized inside each cluster.

% update 19-8: log variable 
    count = zeros(0, 2);
    if nargin < 11
        beta = 0;
    end

    N = length(clusters);
    [~, M] = size(second_hop_gains);

    sinr_b = (P_b_n(:) .* first_hop_gains(:)) ./ sigma_b2;
    R_b = B_bs * log2(1 + sinr_b);

    P_n_col = P_n(:);
    cross_interf = sum(P_n_col .* second_hop_gains, 1);

    raw_alloc = UAV_p_alloc_normalize(:);
    if isempty(raw_alloc)
        raw_alloc = ones(M, 1);
    end
    raw_alloc = max(raw_alloc, 0);

    e2e_rates = zeros(M, 1);
    power_violation = 0;
    rate_violation = 0;
    for n = 1:N
        c_ues = clusters{n};
        if isempty(c_ues)
            continue;
        end

        gains_c = second_hop_gains(n, c_ues);
        num_ues = numel(c_ues);

        cluster_alloc = raw_alloc(c_ues).';
        if ~isfinite(sum(cluster_alloc)) || sum(cluster_alloc) <= 0
            cluster_alloc = ones(1, num_ues);
        end

        gains_safe = max(gains_c, 1e-12);
        gain_bias = gains_safe .^ (-beta);
        weighted = cluster_alloc .* gain_bias;
        sw = sum(weighted);
        if ~isfinite(sw) || sw <= 0
            weighted = ones(1, num_ues);
            sw = num_ues;
        end

        p_alloc_raw = P_n(n) * (weighted / sw);
        % SIC order: sort UEs by channel gain ascending (weak -> strong)
        [~, idx_sorted] = sort(gains_c, 'ascend');
        sorted_ues = c_ues(idx_sorted);
        p_allocs = p_alloc_raw(idx_sorted);

        total_alloc = sum(p_alloc_raw);
        pwr_excess = total_alloc - P_n(n);
        if pwr_excess > 0
            power_violation = power_violation + pwr_excess;
        end

        for r = 1:num_ues
            ue_idx = sorted_ues(r);
            h_nm = second_hop_gains(n, ue_idx);

            interf_nm = cross_interf(ue_idx) - P_n(n) * h_nm;
            if interf_nm < 0
                interf_nm = 0;
            end

            intra_power = sum(p_allocs(r+1:end)) ;
            interf_intra = intra_power * h_nm;

            sinr_nm = (p_allocs(r) * h_nm) / (sigma2 + interf_intra + interf_nm);
            R_nm = B_uav * log2(1 + sinr_nm);
            e2e_rates(ue_idx) = R_nm;   
            if log
                count = [count; R_b(n), R_nm];
                disp("User Index " + ue_idx + ", R_b = " + R_b(n) +", R_nm = " + R_nm + ", R_e2e = " + e2e_rates(ue_idx));
            end    
        end
        rate_violation = rate_violation + max(0, sum(e2e_rates(sorted_ues)) - R_b(n));
    end
    if(log)
        sum_Rb      = sum(count(:, 1));  % Tổng các giá trị Rb thỏa mãn
        sum_Rnm     = sum(count(:, 2));
        satisfied_mask = count(:, 1) >= count(:, 2);
        num_satisfied  = sum(satisfied_mask);
        disp("Ratio BS to UAV = " + num_satisfied +"/" + M);
        disp("Sum rate BS = " + sum_Rb);
        disp("Sum rate UAV = " + sum_Rnm);
    end
    R_th = B_uav * log2(1 + gamma_th);
    outage = sum(e2e_rates < R_th) / M;
    sum_rate = sum(e2e_rates);
end
