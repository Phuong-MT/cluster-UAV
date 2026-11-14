function mutant = mutation_qpso_wave(individual, pbest_i, gbest, xa, beta)
%MUTATION_QPSO_WAVE QPSO wave-function mutation for RCGA individual.
%   mutant = mutation_qpso_wave(individual, pbest_i, gbest, xa, beta)
%
% Inputs:
%   individual : 1 x D current individual.
%   pbest_i    : 1 x D personal-best reference.
%   gbest      : 1 x D global-best reference.
%   xa         : 1 x D mean position vector.
%   beta       : contraction-expansion coefficient (>0), e.g. 0.5~1.0.
%
% Output:
%   mutant     : 1 x D mutated individual.
%
% Formula used:
%   Xop = phi .* pbest_i + (1-phi) .* gbest
%   mutant = Xop +/- beta .* abs(xa - individual) .* log(1./u)
%   with '+' when u>0.5, '-' when u<=0.5

    if nargin < 5
        error('mutation_qpso_wave requires 5 inputs: individual, pbest_i, gbest, xa, beta.');
    end

    d = numel(individual);

    if numel(pbest_i) ~= d || numel(gbest) ~= d || numel(xa) ~= d
        error('individual, pbest_i, gbest, xa must have the same length.');
    end

    lb = 0;
    ub = 1;

    if isscalar(lb)
        lb = lb .* ones(1, d);
    else
        lb = reshape(lb, 1, []);
    end

    if isscalar(ub)
        ub = ub .* ones(1, d);
    else
        ub = reshape(ub, 1, []);
    end

    individual = reshape(individual, 1, []);
    pbest_i    = reshape(pbest_i, 1, []);
    gbest      = reshape(gbest, 1, []);
    xa         = reshape(xa, 1, []);

    phi = rand(1, d);
    u = rand(1, d);
    u(u == 0) = eps;

    xop = phi .* pbest_i + (1 - phi) .* gbest;
    step = beta .* abs(xa - individual) .* log(1 ./ u);

    sign_term = ones(1, d);
    sign_term(u <= 0.5) = -1;

    mutant = xop + sign_term .* step;

    mutant = min(ub, max(lb, mutant));
end
