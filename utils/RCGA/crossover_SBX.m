function [c1, c2] = crossover_SBX(p1, p2, eta)

    dim = length(p1);
    c1 = zeros(1,dim);
    c2 = zeros(1,dim);
    lb = 0 ;
    ub = 1;
    for i = 1:dim
        u = rand;

        if u <= 0.5
            beta = (2*u)^(1/(eta+1));
        else
            beta = (1/(2*(1-u)))^(1/(eta+1));
        end

        c1(i) = 0.5*((1+beta)*p1(i) + (1-beta)*p2(i));
        if c1(i) > ub
            c1(i) = ub;
        elseif c1(i) < lb
            c1(i) = lb;
        end

        c2(i) = 0.5*((1-beta)*p1(i) + (1+beta)*p2(i));
        if c2(i) > ub
            c2(i) = ub;
        elseif c2(i) < lb
            c2(i) = lb;
        end
    end
end 
