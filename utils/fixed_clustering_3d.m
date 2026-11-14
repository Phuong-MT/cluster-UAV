function [clusters, new_UAV_positions] = fixed_clustering_3d(UAV_positions, UE_positions, N, H_min, H_max)
    % FIXED_CLUSTERING_3D - K-Means clustering tren mat phang 2D (x, y)
    % Cap nhat (x, y) cua UAV theo tam cum UE va bao toan do cao z ban dau
    
    M = size(UE_positions, 1);
    
    if size(UAV_positions, 1) == N
        centroids = UAV_positions;
    else
        % Fallback nếu kích thước không khớp
        centroids = kmeans_plus_plus_init(UE_positions, N);
        if nargin >= 5 && ~isempty(H_min) && ~isempty(H_max)
            centroids(:, 3) = H_min + rand(N, 1) * (H_max - H_min);
        end
    end
    
    max_iterations = 100;
    tolerance = 1e-6;
    
    for iter = 1:max_iterations
        % Gán UE vào centroid gần nhất trên mặt phẳng 2D (x, y)
        distances = zeros(M, N);
        for i = 1:M
            for j = 1:N
                distances(i, j) = norm(UE_positions(i, 1:2) - centroids(j, 1:2));
            end
        end
        [~, labels] = min(distances, [], 2);
        
        % Cập nhật centroids: chỉ cập nhật tọa độ (x, y), bảo toàn độ cao z
        new_centroids = centroids;
        for k = 1:N
            cluster_points = UE_positions(labels == k, 1:2);
            if ~isempty(cluster_points)
                new_centroids(k, 1:2) = mean(cluster_points, 1);
            end
        end
        
        % KIỂM TRA HỘI TỤ (trên tọa độ x, y)
        if norm(new_centroids(:, 1:2) - centroids(:, 1:2), 'fro') < tolerance
            centroids = new_centroids;
            break;
        end
        
        centroids = new_centroids;
    end
    
    % Chuyển nhãn (labels) thành cell array
    clusters = cell(N, 1);
    for i = 1:N
        clusters{i} = find(labels == i)';
    end
    
    new_UAV_positions = centroids;
    if nargin >= 5 && ~isempty(H_min) && ~isempty(H_max)
        % Kẹp độ cao an toàn trong khoảng [H_min, H_max]
        new_UAV_positions(:, 3) = max(H_min, min(H_max, new_UAV_positions(:, 3)));
    end
end

% --- Hàm K-Means++ (giữ nguyên, không cần sửa) ---
function centroids = kmeans_plus_plus_init(data, k)
    [n, d] = size(data);
    centroids = zeros(k, d);
    centroids(1, :) = data(randi(n), :);
    
    for i = 2:k
        min_distances = inf(n, 1);
        for j = 1:n
            for c = 1:i-1
                dist = norm(data(j, :) - centroids(c, :))^2;
                min_distances(j) = min(min_distances(j), dist);
            end
        end
        probabilities = min_distances / sum(min_distances);
        cumsum_prob = cumsum(probabilities);
        r = rand();
        selected_idx = find(cumsum_prob >= r, 1, 'first');
        
        if isempty(selected_idx)
            selected_idx = randi(n);
        end
        centroids(i, :) = data(selected_idx, :);
    end
    
    % --- ĐÃ XÓA ---
    % Logic ép buộc độ cao đã bị xóa khỏi đây để khớp với Python
end