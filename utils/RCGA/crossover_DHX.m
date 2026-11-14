function [c1, c2] = crossover_DHX(p1, p2, gen, maxGen, obj_func)

    dim = length(p1);
    % assuming decision variables are in [0,1]
    lb = 0 ;
    ub = 1;
    % xác định best parent
    if obj_func(p1) < obj_func(p2)
        best = p1;
        worst = p2;
    else
        best = p2;
        worst = p1;
    end

    c1 = zeros(1,dim); % dominated (exploration)
    c2 = zeros(1,dim); % biased (exploitation)

    for i = 1:dim

        % ===== NORMALIZE về [0,1] =====
        s1 = (best(i) - lb) / (ub - lb);
        s2 = (worst(i) - lb) / (ub - lb);

        s1 = min(max(s1,0),1);
        s2 = min(max(s2,0),1);

        % ===== q parameter =====
        q = 1 / gen;

        % ===== 1. DYNAMIC DOMINATED =====
        if best(i) <= worst(i)
            val = dubois_tnorm(s1, s2, q); % F_t
        else
            val = dubois_tconorm(s1, s2, q); % S_t
        end
        c1(i) = lb + val * (ub - lb);
        if c1(i) > ub
            c1(i) = ub;
        elseif c1(i) < lb
            c1(i) = lb;
        end
    end
    % ===== 2. DYNAMIC BIASED =====
    f1 = obj_func(best);
    f2 = obj_func(worst);

    lambda = 1 - f1 / (f1 + f2 + 1e-12);

    M_lim = (1 - lambda) * p1 + lambda * p2;
    M1 = (p1 + p2) / 2;

    t = gen / maxGen;

    c2 = (1 - t) * M_lim + t * M1;
    c2 = min(max(c2, lb), ub);
end

function val = dubois_tnorm(x, y, q)
    denom = max([x, y, q]);
    val = (x * y) / denom;
end

function val = dubois_tconorm(x, y, q)
    denom = max([1-x, 1-y, q]);
    val = 1 - ((1-x)*(1-y)) / denom;
end