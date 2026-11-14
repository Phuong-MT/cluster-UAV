function amp = nakagami_fading(m, sz)
    % NAKAGAMI_FADING - Phiên bản chính xác tương đương với Python
    % --- Xử lý tham số đầu vào ---
    if nargin < 2
        sz = 1;
    end
    if length(sz) == 1
        sz = [sz, sz];
    end
    % Ta gọi đây là omega (công suất trung bình mong muốn).
    omega = 1;
    
    % --- Mối quan hệ Nakagami-Gamma ---
    % Nếu R ~ Nakagami(m, omega)
    % thì R^2 ~ Gamma(m, theta = omega/m)
    theta = omega / m;  % Tương đương: 1 / (m)
    
    % --- Tạo các mẫu Gamma ---
    % Tạo các mẫu Gamma(m, 1) và nhân với theta.
    total_samples = prod(sz);
    Y = zeros(total_samples, 1);
    for i = 1:total_samples
        % Gọi hàm phụ để tạo Gamma(m, 1)
        Y(i) = generate_gamma_a_1(m);
    end
    
    % Scale các mẫu Gamma(m, 1) thành Gamma(m, theta)
    Y = Y * theta;
    
    % Biên độ Nakagami là căn bậc hai của biến Gamma
    % Thêm 1e-20 để tránh lỗi log(0) hoặc chia cho 0, giống Python
    amp = sqrt(max(Y, 1e-20));
    
    % Reshape về kích thước mong muốn
    amp = reshape(amp, sz);

end

% --- HÀM PHỤ (NESTED FUNCTION) ---
% Hàm này phải nằm BÊN TRONG hàm nakagami_fading
function Y_a_1 = generate_gamma_a_1(a)
    % Tạo MỘT mẫu từ phân phối Gamma(a, 1)
    % Sử dụng thuật toán Marsaglia & Tsang (2000)
    
    if a < 1
        % Nếu a < 1, sử dụng wrapper (property)
        % Y(a,1) = Y(a+1, 1) * U^(1/a)
        Y_plus_1 = generate_gamma_a_1(a + 1);
        U = rand();
        Y_a_1 = Y_plus_1 * (U^(1/a));
    else
        % Thuật toán chính cho a >= 1
        d = a - 1/3;
        c = 1 / sqrt(9*d);
        while true
            Z = randn(); % Biến ngẫu nhiên Normal
            if Z > -1/c
                V = (1 + c*Z)^3;
                U = rand();
                % Bước squeeze-rejection
                if log(U) < (0.5*Z^2 + d - d*V + d*log(V))
                    Y_a_1 = d * V;
                    break; % Chấp nhận mẫu
                end
            end
        end
    end
end