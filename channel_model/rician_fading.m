function amp = rician_fading(K, sz)
    % Xử lý kích thước đầu vào tùy chọn
    if nargin < 2
        sz = 1;
    end
    
    % K = hệ số Rician K
    % Tổng công suất (LoS + NLoS) được chuẩn hóa về 1
    
    % s = tham số scale (độ lệch chuẩn của các thành phần Gaussian)
    s = sqrt(1 / (2 * (K + 1))); 
    % nu = biên độ của thành phần LoS
    nu = sqrt(K / (K + 1));
    
    % Biên độ = abs(N(nu, s^2) + 1i * N(0, s^2))
    % Tương đương (nu + s*randn) + 1i*(s*randn)
    amp = abs((nu + s * randn(sz)) + 1i * (s * randn(sz)));
end