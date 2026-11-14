function [UAV_positions, first_hop_gains] = first_hop_bs_uav_3d(N, L, H_max, alpha1, K_R, BS_pos, random_seed)
    % Xử lý random_seed tùy chọn
    if nargin >= 7 && ~isempty(random_seed)
        rng(random_seed); % Đặt seed cho trình tạo số ngẫu nhiên
    end
    
    UAV_positions = generate_positions_3d(N, L, H_max);
    first_hop_gains = compute_first_hop_gains(UAV_positions, BS_pos, alpha1, K_R);
end