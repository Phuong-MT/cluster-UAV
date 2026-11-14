function first_hop_gains = compute_first_hop_gains(UAV_positions, BS_pos, alpha1, K_R, fc)
    % fc: tần số sóng mang (Hz), mặc định 2 GHz (3GPP TR 36.777)
    if nargin < 5 || isempty(fc)
        fc = 2e9;
    end
    [N, ~] = size(UAV_positions);
    dist_bs = zeros(N, 1);

    for i = 1:N
        dist_bs(i) = dist_3d(UAV_positions(i, :), BS_pos);
    end

    % Path loss tham chiếu tại d0 = 1 m (free-space): PL0 = (c/(4*pi*fc))^2 ≈ -38.5 dB @ 2 GHz
    PL0 = (3e8 / (4 * pi * fc))^2;

    % Phép toán .^ và .* là element-wise (theo từng phần tử)
    pl_bs = PL0 * dist_bs.^(-alpha1);
    ric_amp = rician_fading(K_R, [N, 1]);
    first_hop_gains = (ric_amp.^2) .* pl_bs;
end
