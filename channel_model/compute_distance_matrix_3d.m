function mat = compute_distance_matrix_3d(ptsA, ptsB)
    [A, ~] = size(ptsA);
    [B, ~] = size(ptsB);
    mat = zeros(A, B);
    
    for i = 1:A
        % Trừ một hàng (1x3) cho ma trận (Bx3)
        % MATLAB tự động mở rộng (broadcast) hàng ptsA(i, :)
        diff = ptsB - ptsA(i, :);
        
        % sum(..., 2) tính tổng bình phương theo hàng
        % sqrt(...) và chuyển vị (.') thành vector hàng (1xB)
        mat(i, :) = sqrt(sum(diff.^2, 2))';
    end
end