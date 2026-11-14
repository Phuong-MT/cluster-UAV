function positions = generate_positions_3d(num, L, H_max, ground_ues)
    % Xử lý các tham số đầu vào tùy chọn
    if nargin < 4
        ground_ues = false;
    end
    if nargin < 3
        H_max = 0.0;
    end
    
    % rand(num, 1) tạo vector cột
    x = rand(num, 1) * L;
    y = rand(num, 1) * L;
    
    if ground_ues
        z = zeros(num, 1);
    else
        if H_max > 0
            z = rand(num, 1) * H_max;
        else
            z = zeros(num, 1);
        end
    end
    % Ghép các vector cột thành ma trận [num x 3]
    positions = [x, y, z];
end