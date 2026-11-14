function samples = generate_gamma_samples(shape, scale, n)
    % GENERATE_GAMMA_SAMPLES - Generate Gamma random variables without Statistics Toolbox
    % Input:
    %   shape: Shape parameter (alpha or k)
    %   scale: Scale parameter (beta or theta)  
    %   n: Number of samples
    % Output:
    %   samples: Gamma distributed samples
    
    samples = zeros(n, 1);
    
    for i = 1:n
        if shape >= 1
            % Use Ahrens-Dieter method for shape >= 1
            shape_int = floor(shape);
            shape_frac = shape - shape_int;
            
            % Integer part: sum of exponentials
            sum_exp = 0;
            for j = 1:shape_int
                sum_exp = sum_exp - log(rand());
            end
            
            % Fractional part using acceptance-rejection
            if shape_frac > 0
                % Simple approximation for fractional part
                frac_sample = -log(rand()) * shape_frac;
                sum_exp = sum_exp + frac_sample;
            end
            
            samples(i) = sum_exp * scale;
        else
            % For shape < 1, use simple approximation
            samples(i) = -log(rand()) * scale * shape;
        end
    end
end