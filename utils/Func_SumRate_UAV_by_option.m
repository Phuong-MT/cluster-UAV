function fitness = Func_SumRate_UAV_by_option(UAV_positions, UE_positions, BS_pos, ...
                                     alpha1, alpha2, K_R, mm, ...
                                     P_bs, P_total, association_vars, alloc_power_uav, sigma_b2, sigma2, ...
                                     I_max, gamma_th, beta, P_th, option, B_bs, B_uav, log)
% Unified objective with decision-space options:
% option = 1 -> dim = 3N, no power decision variables
% option = 2 -> dim = 3N + 2M (assoc + power)
% option = 3 -> dim = 3N + M, (assoc only)
% option = 4 -> dim = 3N + M (position + power)
% option = 5 -> dim = 2M (power + association, fixed UAV positions)

    % if nargin ~= 18
    %     error('Func_SumRate_UAV_by_option expects 18 inputs.');
    % end

    [N, ~] = size(UAV_positions);
    [M, ~] = size(UE_positions);

    first_hop_gains = compute_first_hop_gains(UAV_positions, BS_pos, alpha1, K_R);
    second_hop_gains = compute_second_hop_gains(UAV_positions, UE_positions, alpha2, mm);

    clusters = cell(N, 1);

    % force association when the option explicitly optimizes it;
    % otherwise use a random fallback association.
    forced_uav_indices = [];
    if  ~isempty(association_vars)
        assoc_norm = association_vars(:);
        forced_uav_indices = min(floor(assoc_norm * N) + 1, N);
    end
    
    if isempty(forced_uav_indices)
        [~, uav_indices] = max(second_hop_gains, [], 1);
    else
        uav_indices = forced_uav_indices.';
    end

    for u = 1:M
        u_idx = uav_indices(u);
        clusters{u_idx} = [clusters{u_idx}, u];
    end

    P_b_n_dyn = zeros(N, 1);
    P_n_dyn = zeros(N, 1);
    for c_idx = 1:N
        num_in_cluster = length(clusters{c_idx});
        if num_in_cluster > 0
            P_n_dyn(c_idx) = P_total * num_in_cluster / M;
            P_b_n_dyn(c_idx) = P_bs / N;
        end
    end

    switch option
        case 1
            [outage, sum_rate, power_violation, rate_violation] = ...
                evaluate_noma_clusters_3d(clusters,...
                    UAV_positions, UE_positions, ... 
                    first_hop_gains, second_hop_gains, ...
                    P_b_n_dyn, P_n_dyn, sigma_b2, sigma2, I_max, gamma_th, beta, B_bs, B_uav);
        case {2, 3, 4, 5}
            alloc_ue = alloc_power_uav(:);
            if numel(alloc_ue) ~= M
                error('Option %d requires alloc_vars length M.', option);
            end
            [outage, sum_rate, power_violation, rate_violation] = ...
                evaluate_noma_dim3N_M(clusters, ...
                    UAV_positions, UE_positions, ...
                    first_hop_gains, second_hop_gains, ...
                    P_b_n_dyn, P_n_dyn, alloc_ue, sigma_b2, sigma2, I_max, gamma_th, beta, B_bs, B_uav, log);
        otherwise
            error('Unsupported option. Use 1, 2, 3, 4, or 5.');
    end

    lambda1 = 0.1;
    lambda3 = 0.1;
    lambda4 = 0.1;

    penalty_outage = lambda1 * max(0, outage - P_th);
    penalty_power = lambda3 * power_violation;
    penalty_rate = lambda4 * rate_violation;

    fitness = sum_rate - penalty_outage - penalty_power - penalty_rate;
end
