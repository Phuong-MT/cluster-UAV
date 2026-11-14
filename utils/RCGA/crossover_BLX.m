function [c1, c2] = crossover_BLX(p1, p2, alpha)

    dim = length(p1);
    c1 = zeros(1,dim);
    c2 = zeros(1,dim);
    lb = 0 ;
    ub = 1;
    for i = 1:dim
        cmin = min(p1(i), p2(i));
        cmax = max(p1(i), p2(i));
        I = cmax - cmin;

        low = cmin - alpha*I;
        high = cmax + alpha*I;

        c1(i) = low + rand*(high-low);
       
        if c1(i) > ub
            c1(i) = ub;
        elseif c1(i) < lb
            c1(i) = lb;
        end

        c2(i) = low + rand*(high-low);
        if c2(i) > ub
            c2(i) = ub;
        elseif c2(i) < lb
            c2(i) = lb;
        end
    end
end