%Computes the recurrence function G and its derivatives

classdef PolynomialG < handle
    properties
        degree
        cross_indices 
        num_w_x % =degree, #of coeffs for x^i, i>0
        num_w_h % =degree, #of coeffs for h^i, i>0
        num_w_c % =degree*(degree-1)/2, #of coeffs for h^ix^j, i,j>0 
        num_params %#of parameters, sum of previous three
        params
    end
    
    methods
        function obj = PolynomialG(degree)
            obj.degree = degree;
            
            % Build interaction pairs: i >= 1, j >= 1, i + j <= degree
            pairs = [];
            for i = 1:degree
                for j = 1:degree
                    if (i + j) <= degree
                        pairs = [pairs; i, j]; 
                    end
                end
            end
            obj.cross_indices = pairs;
            
            obj.num_w_x = degree;
            obj.num_w_h = degree;
            obj.num_w_c = size(pairs, 1);
            obj.num_params = obj.num_w_x + obj.num_w_h + obj.num_w_c;
            
            init_vals = zeros(obj.num_params, 1, 'single');
            init_vals(1) = 0.2;
            if obj.num_w_x + 1 <= obj.num_params
                init_vals(obj.num_w_x + 1) = 0.2;
            end
            obj.params = init_vals;
        end
        
        % Computes only G
        function Z = forward_trajectory_only(obj, X, params)
            if nargin < 3, params = obj.params; end
            D = obj.degree;
            % Parameters are tanh squashed to prevent explosion
            w = tanh(params);
            
            w_x = w(1:obj.num_w_x);
            w_h = w(obj.num_w_x+1 : obj.num_w_x+obj.num_w_h);
            w_c = w(obj.num_w_x+obj.num_w_h+1 : end);
            
            T = length(X);
            
            % Initialize (Z_list will have the h values)
            Z_list = zeros(T, 1, 'like', X);
            Z_list(1) = 0; 
            h = 0;
            
            % Run the recurrence to compute Z_list
            for t = 1:(T-1)
                x = X(t);
                
                x_pow = zeros(D+1, 1, 'like', X);
                h_pow = zeros(D+1, 1, 'like', X);
                
                x_pow(1) = 1; 
                h_pow(1) = 1;
                % Compute powers with Taylor-like/factorial normalization
                for d = 1:D
                    x_pow(d+1) = (x^d) / factorial(d);
                    h_pow(d+1) = (h^d) / factorial(d);
                end
                
                sum_x = 0; sum_h = 0; sum_c = 0;
                for d = 1:D
                    sum_x = sum_x + w_x(d) * x_pow(d+1);
                    sum_h = sum_h + w_h(d) * h_pow(d+1);
                end
                
                if obj.num_w_c > 0
                    for k = 1:obj.num_w_c
                        idx_i = obj.cross_indices(k, 1);
                        idx_j = obj.cross_indices(k, 2);
                        sum_c = sum_c + w_c(k) * x_pow(idx_i+1) * h_pow(idx_j+1);
                    end
                end
                
                h_new = sum_x + sum_h + sum_c;
                h = h_new;
                Z_list(t+1) = h;
            end
            Z = Z_list;
        end
        
        % Computes G, its Jacobian and Hessian
        function [Z, J_list, H_list] = forward_second_order(obj, X, params)
            if nargin < 3, params = obj.params; end
            D = obj.degree;
            N = obj.num_params;
            
            w = tanh(params);
            dtanh = 1.0 - w.^2;
            d2tanh = -2.0 * w .* dtanh;
            
            w_x = w(1:D); 
            w_h = w(D+1 : 2*D); 
            w_c = w(2*D+1 : end);
            
            dtanh_x = dtanh(1:D);
            dtanh_h = dtanh(D+1 : 2*D);
            dtanh_c = dtanh(2*D+1 : end);
            
            d2tanh_x = d2tanh(1:D);
            d2tanh_h = d2tanh(D+1 : 2*D);
            d2tanh_c = d2tanh(2*D+1 : end);
            
            T = length(X);
            h = 0;
            Jh = zeros(N, 1, 'like', X);
            Hh = zeros(N, N, 'like', X);
            
            Z_list = zeros(T, 1, 'like', X);
            J_list = zeros(T, N, 'like', X);
            H_list = zeros(T, N, N, 'like', X);
            
            Z_list(1) = h;
            J_list(1, :) = Jh;
            H_list(1, :, :) = Hh;
            
            for t = 1:(T-1)
                x = X(t);
                h_prev = h;
                Jh_prev = Jh;
                Hh_prev = Hh;
               
                x_pow = zeros(D+1, 1, 'like', X);
                h_pow = zeros(D+1, 1, 'like', X);
                x_pow(1) = 1; h_pow(1) = 1;
                for d = 1:D
                    x_pow(d+1) = (x^d) / factorial(d);
                    h_pow(d+1) = (h_prev^d) / factorial(d);
                end
                
                % Compute the Z/h values
                sum_x = 0; sum_h = 0; sum_c = 0;
                for d = 1:D
                    sum_x = sum_x + w_x(d) * x_pow(d+1);
                    sum_h = sum_h + w_h(d) * h_pow(d+1);
                end
                if obj.num_w_c > 0
                    for k = 1:obj.num_w_c
                        i = obj.cross_indices(k, 1);
                        j = obj.cross_indices(k, 2);
                        sum_c = sum_c + w_c(k) * x_pow(i+1) * h_pow(j+1);
                    end
                end
                h_val = sum_x + sum_h + sum_c;
                h = h_val;
                Z_list(t+1) = h;
                
                % Compute Jacobian
                dg_dw = zeros(N, 1, 'like', X);
                
                % w_x part
                for d = 1:D
                    dg_dw(d) = x_pow(d+1) * dtanh_x(d);
                end
                % w_h part
                for d = 1:D
                    dg_dw(D + d) = h_pow(d+1) * dtanh_h(d);
                end
                % w_c part
                if obj.num_w_c > 0
                    for k = 1:obj.num_w_c
                        i = obj.cross_indices(k, 1);
                        j = obj.cross_indices(k, 2);
                        dg_dw(2*D + k) = x_pow(i+1) * h_pow(j+1) * dtanh_c(k);
                    end
                end
                
                % dg_dh part
                dg_dh = 0;
                for d = 1:D
                    dg_dh = dg_dh + w_h(d) * h_pow(d); 
                end
                if obj.num_w_c > 0
                    for k = 1:obj.num_w_c
                        i = obj.cross_indices(k, 1);
                        j = obj.cross_indices(k, 2);
                        % h^(j-1) is at index j in h_pow
                        dg_dh = dg_dh + w_c(k) * x_pow(i+1) * h_pow(j);
                    end
                end
                
                Jh = dg_dw + dg_dh * Jh_prev;
                J_list(t+1, :) = Jh;
                
                % Compute Hessian
                % g_wh=V part
                g_wh = zeros(N, 1, 'like', X);
                for d = 1:D
                    g_wh(D + d) = h_pow(d) * dtanh_h(d); 
                end
                if obj.num_w_c > 0
                    for k = 1:obj.num_w_c
                        i = obj.cross_indices(k, 1);
                        j = obj.cross_indices(k, 2);
                        g_wh(2*D + k) = x_pow(i+1) * h_pow(j) * dtanh_c(k);
                    end
                end
                % g_ww part
                g_ww = zeros(N, N, 'like', X);
                for d = 1:D
                    g_ww(d, d) = x_pow(d+1) * d2tanh_x(d);
                end
                for d = 1:D
                    idx = D + d;
                    g_ww(idx, idx) = h_pow(d+1) * d2tanh_h(d);
                end
                if obj.num_w_c > 0
                    for k = 1:obj.num_w_c
                        i = obj.cross_indices(k, 1);
                        j = obj.cross_indices(k, 2);
                        idx = 2*D + k;
                        g_ww(idx, idx) = x_pow(i+1) * h_pow(j+1) * d2tanh_c(k);
                    end
                end
                
                % g_hh part
                g_hh = 0;
                if D >= 2
                    for d = 2:D
                        g_hh = g_hh + w_h(d) * h_pow(d-1);
                    end
                    if obj.num_w_c > 0
                        for k = 1:obj.num_w_c
                            i = obj.cross_indices(k, 1);
                            j = obj.cross_indices(k, 2);
                            if j >= 2
                                g_hh = g_hh + w_c(k) * x_pow(i+1) * h_pow(j-1);
                            end
                        end
                    end
                end
                
                out_gwh_J = g_wh * Jh_prev';
                out_JJ = Jh_prev * Jh_prev';
                
                Hh = g_ww + out_gwh_J + out_gwh_J' + g_hh*out_JJ + dg_dh*Hh_prev;
                H_list(t+1, :, :) = Hh;
            end
            
            Z = Z_list;
        end
    end
end