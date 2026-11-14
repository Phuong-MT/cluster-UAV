function [best_solution, convergence, best_obj] = run_rcga(dim, obj_func, max_fes, pop_size, verbose)
    disp("GA-QJ running for maximization, generation = " + max_fes/pop_size + ", n=" + pop_size + " ...");

    indivs = rand(pop_size-1, dim);
    new_pop = ones(1,dim);
    indivs = [indivs; new_pop];
    
    best_indiv = rand(1, dim);
    best_obj = -1e9;
    count_fes = 0;
    objective = -1e9 * ones(pop_size, 1);
    pbest = zeros(pop_size, dim);
    pbest_obj = -1e9 * ones(pop_size, 1);
    
    % Đánh giá quần thể khởi tạo
    for i=1:pop_size
        objective(i) = obj_func(indivs(i, :));
        count_fes = count_fes+1;

        pbest(i, :) = indivs(i, :);
        pbest_obj(i) = objective(i);

        if objective(i) > best_obj
            best_obj = objective(i);
            best_indiv = indivs(i, :);
        end
    end
    
    generation = 1;
    last_improve_gen = 1;
    stagnation_threshold = 6;
    jump_base_rate = 0.2;
    jump_gain = 0.05;
    jump_max_rate = 0.4;

    convergence = zeros(round(max_fes / pop_size), 1);
    convergence(generation) = best_obj;
    
    while count_fes < max_fes
        generation = generation + 1;
        stagnation_gen = generation - last_improve_gen;
        median_obj = median(objective);
        % ============ BEGIN REPRODUCTION ==================
        parent_ref = zeros(pop_size, 1);
        count = 1;

        % Preallocate for the worst case: every offspring can also generate
        % a jumping copy, so the combined pool can be up to 2*pop_size.
        combined_offspring = zeros(2 * pop_size, dim);
        combined_offspring_fitness = -1e9 * ones(2 * pop_size, 1);
        combined_count = 0;

        %% crossover %%
        while count < pop_size
            p1 = randi([1, size(indivs, 1)]);
            p2 = randi([1, size(indivs, 1)]);
            while p1 == p2
                p2 = randi([1, size(indivs, 1)]);
            end
            
            if rand <= 0.9
                [child1, child2] = crossover_BLX(indivs(p1, :), indivs(p2, :), 0.7);

                combined_count = combined_count + 1;
                combined_offspring(combined_count, :) = child1;
                parent_ref(count) = p1;

                combined_count = combined_count + 1;
                combined_offspring(combined_count, :) = child2;
                parent_ref(count+1) = p2;
                count = count+2;
            end
        end

        %% mutation  && quantum-jumping %%
        offspring_count = combined_count;
        for i=1:offspring_count
            pre_jump_candidate = combined_offspring(i, :);
            if rand < 0.01
                pre_jump_candidate = polynomial_mutation(pre_jump_candidate);
                % offspring(i, :) = gaussian_mutation(offspring(i, :));
            end
            parent_id = parent_ref(i);
            if parent_id < 1
                parent_id = randi(pop_size);
            end

            % Ghi đè cá thể sau mutation vào đúng vị trí i của crossover
            combined_offspring(i, :) = pre_jump_candidate;
            combined_offspring_fitness(i) = obj_func(pre_jump_candidate);
            count_fes = count_fes+1;

            % Quantum-jumping sinh ra cá thể mới và nối thêm vào cuối mảng kết hợp
            if stagnation_gen >= stagnation_threshold && objective(parent_id) <= median_obj
                p_jump = min(jump_max_rate, jump_base_rate + jump_gain * (stagnation_gen - stagnation_threshold));
                if rand < p_jump
                    beta = 0.5 + ((0.5*(round(max_fes / pop_size) - generation))/round(max_fes / pop_size));
                    xa = mean(pbest, 1);
                    combined_count = combined_count + 1;
                    combined_offspring(combined_count, :) = mutation_qpso_wave( ...
                        pre_jump_candidate, ...
                        pbest(parent_id, :), ...
                        best_indiv, ...
                        xa, ...
                        beta);
                    combined_offspring_fitness(combined_count) = obj_func(combined_offspring(combined_count, :));
                end
            end
        end

        offspring = combined_offspring(1:combined_count, :);
        offspring_fitness = combined_offspring_fitness(1:combined_count);
        % ============== END REPRODUCTION ===================
        
        % Survival selection
        [indivs, objective] = selection(pop_size, indivs, objective, offspring, offspring_fitness);
        
        % Update best solution
        if best_obj < objective(1)
            best_obj = objective(1);
            best_indiv = indivs(1, :);
            last_improve_gen = generation;
        end

        improved_mask = objective > pbest_obj;
        pbest(improved_mask, :) = indivs(improved_mask, :);
        pbest_obj(improved_mask) = objective(improved_mask);
        
        % --- GIỮ NGUYÊN ĐOẠN NÀY: Chỉ hiển thị nếu chia hết cho 50 ---
        if mod(generation, 1000) == 0
            disp("Generation " + generation + ", best objective = " + best_obj);
        end
        
        convergence(generation) = best_obj;
    end

    log = true;
    obj_func(best_indiv, log);


    best_solution = best_indiv;
end
