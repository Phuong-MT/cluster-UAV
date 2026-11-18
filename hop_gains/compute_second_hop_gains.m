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
    %% thay thế nakagami_fading bằng rician_fading (K_R = 5)
    % nak_amp = nakagami_fading(m, [N, M]);
    % nak_pow = nak_amp.^2;
    K_R = 5;

    % Dùng mẫu fading quy mô nhỏ (small-scale fading) cố định trong mỗi Run
    % qua biến global CURRENT_RIC_POW2 để tránh việc randn nhảy liên tục ở mỗi
    % lần đánh giá fobj(x) (400,000 FES), giúp bề mặt hàm mục tiêu chỉ thay đổi
    % theo Path Loss (tọa độ UAV). Nếu không khởi tạo global thì fallback sinh ngẫu nhiên.
    global CURRENT_RIC_POW2;
    if ~isempty(CURRENT_RIC_POW2) && size(CURRENT_RIC_POW2, 1) >= N && size(CURRENT_RIC_POW2, 2) >= M
        ric_pow = CURRENT_RIC_POW2(1:N, 1:M);
    else
        ric_amp = rician_fading(K_R, [N, M]);
        ric_pow = ric_amp.^2;
    end
    % second_hop_g = nak_pow .* pl;
    second_hop_g = ric_pow .* pl;
end
