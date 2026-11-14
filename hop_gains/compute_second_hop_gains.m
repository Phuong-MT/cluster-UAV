function second_hop_g = compute_second_hop_gains(UAV_positions, UE_positions, alpha2, m, fc)
    % fc: tần số sóng mang (Hz), mặc định 2 GHz (3GPP TR 36.777)
    if nargin < 5 || isempty(fc)
        fc = 2e9;
    end
    [N, ~] = size(UAV_positions);
    [M, ~] = size(UE_positions);

    dmat = compute_distance_matrix_3d(UAV_positions, UE_positions);

    % Path loss tham chiếu tại d0 = 1 m (free-space): PL0 = (c/(4*pi*fc))^2 ≈ -38.5 dB @ 2 GHz
    PL0 = (3e8 / (4 * pi * fc))^2;

    % Phép toán .^ và .* là element-wise
    pl = PL0 * dmat.^(-alpha2);
    nak_amp = nakagami_fading(m, [N, M]);
    nak_pow = nak_amp.^2;
    second_hop_g = nak_pow .* pl;
end
