function model = build_pca_transform_model(X, alpha_pca)
% Build PCA transform model from full population matrix X (pop_size x dim)

    if nargin < 2 || isempty(alpha_pca)
        alpha_pca = 0.9;
    end

    [~, dim] = size(X);
    mu = mean(X, 1);
    Xc = X - mu;

    C = (Xc' * Xc) / max(1, size(Xc, 1) - 1);
    [V, D] = eig(C);
    [eigvals, idx] = sort(diag(D), 'descend');
    V = V(:, idx);

    total = sum(eigvals);
    if total <= eps
        R = 1;
    else
        R = find(cumsum(eigvals) / total >= alpha_pca, 1, 'first');
        if isempty(R)
            R = 1;
        end
    end

    R = max(1, min(R, dim));
    W = V(:, 1:R);

    model.mu = mu;
    model.W = W;
    model.R = R;
end
