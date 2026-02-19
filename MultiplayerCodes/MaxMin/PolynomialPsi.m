classdef PolynomialPsi < handle
    properties
        degree
        num_params
        params
    end
    
    methods
        function obj = PolynomialPsi(degree)
            obj.degree = degree;
            obj.num_params = degree;
            init_vals = zeros(degree, 1, 'single');
            init_vals(1) = 1.47;
            obj.params = init_vals;
        end
        
        % Computes only Psi
        function res = forward(obj, Y, params)
            if nargin < 3, params = obj.params; end
            w = tanh(params);
            D = obj.degree;
            
            res = zeros(size(Y), 'like', Y);
            for d = 1:D
                term = (Y.^d) / factorial(d);
                res = res + w(d) * term;
            end
        end
        
        %Computes Psi and its derivatives wrt y and w
        function [psi_vals, d_psi_dy, d2_psi_dy2, J_Psi_w, J_PsiPrime_w, H_Psi_ww] = ...
                get_derivatives(obj, Y, params)
            if nargin < 3, params = obj.params; end
            D = obj.degree;
            w = tanh(params);
            dtanh = 1.0 - w.^2;
            d2tanh = -2.0 * w .* dtanh;
            
            T = length(Y);
            
            % Compute Psi
            psi_vals = zeros(T, 1, 'like', Y);
            for d = 1:D
                psi_vals = psi_vals + w(d) * (Y.^d)/factorial(d);
            end
            
            % Compute d_psi_dy
            d_psi_dy = zeros(T, 1, 'like', Y);
            for d = 1:D
                if d == 1
                    term = ones(T, 1, 'like', Y);
                else
                    term = (Y.^(d-1))/factorial(d-1);
                end
                d_psi_dy = d_psi_dy + w(d) * term;
            end
            
            % Compute d2_psi_dy2
            d2_psi_dy2 = zeros(T, 1, 'like', Y);
            for d = 2:D
                if d == 2
                    term = ones(T, 1, 'like', Y);
                else
                    term = (Y.^(d-2))/factorial(d-2);
                end
                d2_psi_dy2 = d2_psi_dy2 + w(d) * term;
            end
            
            % Compute J_Psi_w 
            J_Psi_w = zeros(T, D, 'like', Y);
            for d = 1:D
                J_Psi_w(:, d) = ((Y.^d)/factorial(d)) * dtanh(d);
            end
            
            % Compute H_Psi_ww (T x D because Psi_ww is diagonal)
            H_Psi_ww = zeros(T, D, 'like', Y);
            for d = 1:D
                H_Psi_ww(:, d) = ((Y.^d)/factorial(d)) * d2tanh(d);
            end
            
            % J_PsiPrime_w (cross-derivative, T x D)
            J_PsiPrime_w = zeros(T, D, 'like', Y);
            for d = 1:D
                if d == 1
                     term = ones(T, 1, 'like', Y);
                else
                     term = (Y.^(d-1))/factorial(d-1);
                end
                J_PsiPrime_w(:, d) = term * dtanh(d);
            end
        end
    end
end