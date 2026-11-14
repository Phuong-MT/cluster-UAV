function [outage, sum_rate, power_violation, rate_violation] = evaluate_noma_clusters_3d(clusters, ...
                          UAV_positions, UE_positions, ...
                          first_hop_gains, second_hop_gains, ...
                          P_b_n, P_n, sigma_b2, sigma2, ...
                          I_max, gamma_th, beta, B_bs, B_uav)
    % evaluate_noma_clusters_3d: Evaluate NOMA clusters with constraint violations
    %
    % OUTPUTS:
    %   outage           - Outage probability (fraction of users below R_th)
    %   sum_rate          - Total sum rate (bits/s/Hz)
    %   interf_violation  - Total interference constraint violation
    %   power_violation   - Total power constraint violation

    if nargin < 12
        beta = 0;
    end

    N = length(clusters);
    [~, M] = size(second_hop_gains);

    % 1) First-hop rate
    sinr_b = (P_b_n(:) .* first_hop_gains(:)) ./ sigma_b2;
    R_b = B_bs * log2(1 + sinr_b); % (N,1)

    % 2) Cross-UAV interference (NxM) -> sum along UAVs -> (1xM)
    P_n_col = P_n(:);
    cross_interf = sum(P_n_col .* second_hop_gains, 1);

    e2e_rates = zeros(M, 1);
    power_violation = 0;    % Accumulated power violation
    rate_violation = 0;
    for n = 1:N
        c_ues = clusters{n};
        if isempty(c_ues)
            continue;
        end

        % NOMA ordering
        gains_c = second_hop_gains(n, c_ues); % (1 x num_ues)
        num_ues = length(c_ues);
        % Power allocation within cluster
        p_allocs = [];
        total_alloc = 0;
        if num_ues > 0
            power_weights = (gains_c.^(-beta)) ./ sum(gains_c.^(-beta));
            p_allocs = P_n(n) * power_weights; % (1 x num_ues)
            % SIC order: sort UEs by channel gain ascending (weak -> strong)
            [~, idx_sorted_mat] = sort(gains_c, 'ascend');
            sorted_ues = c_ues(idx_sorted_mat);
            p_allocs = p_allocs(idx_sorted_mat);
            total_alloc = sum(p_allocs);
        end

        % C3: Power constraint violation: sum(P_nm) <= P_n
        pwr_excess = total_alloc - P_n(n);
        if pwr_excess > 0
            power_violation = power_violation + pwr_excess;
        end

        for rank_mat = 1:num_ues
            ue_idx = sorted_ues(rank_mat);
            h_nm = second_hop_gains(n, ue_idx);

            % Cross-UAV interference for this UE
            interf_nm = cross_interf(ue_idx) - P_n(n) * h_nm;
            if interf_nm < 0
                interf_nm = 0.0;
            end

            % Compute rate normally (no death penalty)
            % Intra-cluster interference (from stronger-gain UEs later in SIC order, not yet removed)
            % In SIC, residual intra-cluster interference at UE m goes through h_{n,m}.
            intra_power = sum(p_allocs((rank_mat + 1):end));
            interf_intra = intra_power * h_nm;

            sinr_nm = (p_allocs(rank_mat) * h_nm) / (sigma2 + interf_intra + interf_nm);
            R_nm =  B_uav * log2(1 + sinr_nm);
            e2e_rates(ue_idx) = R_nm;
        end
        rate_violation = rate_violation + max(0, sum(e2e_rates(sorted_ues)) - R_b(n));
    end
            
    % C1: Outage probability
    R_th = B_uav * log2(1 + gamma_th);
    outage = sum(e2e_rates < R_th) / M;
    sum_rate = sum(e2e_rates);
end
