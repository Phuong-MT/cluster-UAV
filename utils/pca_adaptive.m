function [X_reduced, R] = pca_adaptive(X, alpha)

    if nargin < 2
        alpha = 0.9;
    end

    % center
    Xc = X - mean(X,1);

    % covariance
    C = cov(Xc);

    % eigen
    [V, D] = eig(C);

    % sort eigenvalues
    [eigvals, idx] = sort(diag(D), 'descend');
    V = V(:, idx);

    % chọn R tự động
    total = sum(eigvals);
    cum = cumsum(eigvals);
    R = find(cum / total >= alpha, 1);

    % projection
    W = V(:, 1:R);
    X_reduced = Xc * W;
end