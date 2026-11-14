function [labels, exemplars] = apc_clustering(pop, max_iter, lambda)

    if nargin < 2, max_iter = 100; end
    if nargin < 3, lambda = 0.9; end

    N = size(pop,1);

    %% ===== Similarity (vectorized) =====
    S = -pdist2(pop, pop, 'squaredeuclidean');

    % preference
    pref = median(S(:));
    S(1:N+1:end) = pref; % set diagonal

    %% ===== Init =====
    R = zeros(N,N);
    A = zeros(N,N);

    %% ===== Iteration =====
    for iter = 1:max_iter

        %% ===== Update R =====
        AS = A + S;

        % max1 & index
        [max1, idx1] = max(AS, [], 2);

        % max2
        AS2 = AS;
        for i = 1:N
            AS2(i, idx1(i)) = -inf;
        end
        max2 = max(AS2, [], 2);

        % compute R_new
        R_new = S - max1;

        for i = 1:N
            R_new(i, idx1(i)) = S(i, idx1(i)) - max2(i);
        end

        % damping
        R = lambda * R + (1 - lambda) * R_new;

        %% ===== Update A =====
        Rp = max(0, R);
        sumRp = sum(Rp, 1); % 1 x N

        A_new = zeros(N,N);

        % diagonal
        A_new(1:N+1:end) = sumRp' - diag(Rp);

        % off-diagonal
        for k = 1:N
            for i = 1:N
                if i ~= k
                    A_new(i,k) = min(0, R(k,k) + sumRp(k) - Rp(i,k) - Rp(k,k));
                end
            end
        end

        % damping
        A = lambda * A + (1 - lambda) * A_new;

        if iter > 5
            if norm(R_new - R, 'fro') < 1e-5
                break;
            end
        end
    end

    %% ===== Assign exemplar =====
    [~, labels] = max(A + R, [], 2);

    exemplars = unique(labels);
end