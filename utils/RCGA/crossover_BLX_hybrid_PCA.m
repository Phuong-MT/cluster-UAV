function [c1, c2] = crossover_BLX_hybrid_PCA(p1, p2, alpha_blx, alpha_pca, pca_model)
% BLX-hybrid 3 phases:
% 1) Parent vectors -> PCA (denoise by dimensionality reduction)
% 2) BLX crossover in reduced space
% 3) Inverse PCA projection back to original space

    if nargin < 3 || isempty(alpha_blx)
        alpha_blx = 0.7;
    end
    if nargin < 4 || isempty(alpha_pca)
        alpha_pca = 0.9;
    end

    lb = 0;
    ub = 1;
    dim = length(p1);

    % ===== Phase 1: PCA transform for denoising =====
    if nargin >= 5 && ~isempty(pca_model) && isfield(pca_model, 'mu') && isfield(pca_model, 'W')
        mu = pca_model.mu;
        W = pca_model.W;
        R = size(W, 2);
    else
        X = [p1; p2];               % fallback if no external model
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
    end

    x1c = p1 - mu;
    x2c = p2 - mu;
    Z = [x1c; x2c] * W;         % 2 x R
    z1 = Z(1, :);
    z2 = Z(2, :);

    % ===== Phase 2: BLX in PCA-reduced space =====
    cz1 = zeros(1, R);
    cz2 = zeros(1, R);
    for i = 1:R
        cmin = min(z1(i), z2(i));
        cmax = max(z1(i), z2(i));
        I = cmax - cmin;

        low = cmin - alpha_blx * I;
        high = cmax + alpha_blx * I;

        cz1(i) = low + rand * (high - low);
        cz2(i) = low + rand * (high - low);
    end

    % ===== Phase 3: Inverse PCA back to original space =====
    c1 = cz1 * W' + mu;
    c2 = cz2 * W' + mu;

    % Enforce variable bounds
    c1 = min(max(c1, lb), ub);
    c2 = min(max(c2, lb), ub);
end
