classdef HessianSolver < handle
    properties
        X
        T
        G
        Psi
        Y
        lmbda
        eta_z_min
        eta_y_min
        eta_z
        eta_y
    end
    
    methods
        function obj = HessianSolver(X, G, Psi, lmbda, eta_z_m, eta_y_m, eta_z_i, eta_y_i)
            obj.X = X;
            obj.T = length(X);
            obj.G = G;
            obj.Psi = Psi;
            
            % Random initialization for Y as N(0,1)*0.2 + 0.3
            obj.Y = randn(size(X), 'like', X) * 0.2 + 0.3;
            
            obj.lmbda = lmbda;
            obj.eta_z_min = eta_z_m;
            obj.eta_y_min = eta_y_m;
            obj.eta_z = eta_z_i;
            obj.eta_y = eta_y_i;
        end
        
        % Returns parameters vector
        function w_vec = get_w_vector(obj)
            w_vec = [obj.G.params; obj.Psi.params];
        end
        
        % Splits parameters vector
        function [w_g, w_psi] = split_w(obj, w_vector)
            len_g = obj.G.num_params;
            w_g = w_vector(1:len_g);
            w_psi = w_vector(len_g+1 : end);
        end
        
        % Computes L without derivatives (for comparison to update eta)
        function val = compute_L_at(obj, w_vector, y_val)
            [w_g, w_psi] = obj.split_w(w_vector);
            Z = obj.G.forward_trajectory_only(obj.X, w_g);
            psi_val = obj.Psi.forward(y_val, w_psi);
            
            mse_val = mean((obj.X - y_val).^2);
            Z_norm_val = Z - mean(Z);
            corr_val = mean(Z_norm_val .* psi_val);
            
            val = mse_val + obj.lmbda * corr_val;
        end
        
        % The update step
        function [obj, L, mse, corr, nGw, nGy, ndw, ndy, accept] = ...
                update_step(obj, grad_min)
            
            w_current = obj.get_w_vector();
            [w_g, w_psi] = obj.split_w(w_current);
            
            % Forward G and center Z, J_Z and H_Z
            [Z, J_Z, H_Z] = obj.G.forward_second_order(obj.X, w_g);
            Z_centered = Z - mean(Z);
            J_Z_centered = J_Z - mean(J_Z, 1);
            H_Z_mean = mean(H_Z, 1); %(tensor dims: T x N x N)
            H_Z_centered = bsxfun(@minus, H_Z, H_Z_mean);
            
            % Compute Psi
            Y_curr = obj.Y;
            [psi_vals, d_psi_dy, d2_psi_dy2, J_Psi_w, J_PsiPrime_w, H_Psi_ww] = ...
                obj.Psi.get_derivatives(Y_curr, w_psi);
            
            % Gradients
            % grad_w_G: mean over T of (psi * J_Z)
            grad_w_G = obj.lmbda * mean(bsxfun(@times, psi_vals, J_Z_centered), 1)';
            
            % grad_w_Psi: mean over T of (Z * J_Psi)
            grad_w_Psi = obj.lmbda * mean(bsxfun(@times, Z_centered, J_Psi_w), 1)';
            
            gw = [grad_w_G; grad_w_Psi];            
            gy = -(2.0/obj.T)*(obj.X - obj.Y) + (obj.lmbda/obj.T) * (Z_centered .* d_psi_dy);
            
            % Hessian blocks
            % L_GG (N_G x N_G)
            psi_H = bsxfun(@times, reshape(psi_vals, [obj.T, 1, 1]), H_Z_centered);
            L_GG = squeeze(mean(psi_H, 1)) * obj.lmbda;
            
            % L_GPsi (N_G x N_Psi)
            L_GPsi = (obj.lmbda / obj.T) * (J_Z_centered' * J_Psi_w);
            
            % L_PsiPsi (Diagonal)
            diag_vals = obj.lmbda * mean(bsxfun(@times, Z_centered, H_Psi_ww), 1);
            L_PsiPsi = diag(diag_vals);
            
            % Assemble L_ww
            row1 = [L_GG, L_GPsi];
            row2 = [L_GPsi', L_PsiPsi];
            L_ww = [row1; row2];
            
            % L_yy components
            diag_correction = (obj.lmbda / obj.T) * (Z_centered .* d2_psi_dy2);
            alpha_vec = (2.0 / obj.T) + diag_correction;
            
            % L_yw construction
            vec_G = (obj.lmbda / obj.T) * d_psi_dy;
            L_yw_G = bsxfun(@times, J_Z_centered, vec_G); % T x N_G            
            vec_Z = (obj.lmbda / obj.T) * Z_centered;
            L_yw_Psi = bsxfun(@times, J_PsiPrime_w, vec_Z); % T x N_Psi            
            L_yw = [L_yw_G, L_yw_Psi]; % T x N_total
            
            % Solver loop
            
            accept = false;
            dw = zeros(size(gw), 'like', gw);
            dy = zeros(size(gy), 'like', gy);
            
            mse = mean((obj.X - obj.Y).^2);
            corr = mean(Z_centered .* psi_vals);
            L_curr = mse + obj.lmbda * corr;
            
            norm_dw = 0;
            norm_dy = 0;
            
            while (~accept) && (obj.eta_y + obj.eta_z) > (obj.eta_y_min + obj.eta_z_min)
                obj.eta_z = max(obj.eta_z, obj.eta_z_min);
                obj.eta_y = max(obj.eta_y, obj.eta_y_min);
                
                inv_Lyy_diag = 1.0 ./ alpha_vec;
                
                scaled_gy = gy .* inv_Lyy_diag;
                term_correction = L_yw' * scaled_gy;
                
                D_w = -obj.eta_z * (gw - term_correction);
                D_y = obj.eta_y * gy;
                
                scaled_Lyw = L_yw' .* (inv_Lyy_diag'); 
                term_cov = scaled_Lyw * L_yw;
                
                block_inner = L_ww - term_cov;
                
                eye_w = eye(length(w_current));
                H_11 = eye_w - obj.eta_z * block_inner;
                damping = 1e-4 * eye_w;
                A_sys = H_11 + damping;

                % --- ROBUSTNESS CHECK ---
                % If the matrix is singular, eta_z is hitting a resonance. 
                % We must back off (reduce step size) rather than forcing a solve.
                if rcond(A_sys) < 1e-12
                    accept = false;
                    obj.eta_z = obj.eta_z * 0.5;
                    obj.eta_y = obj.eta_y * 0.5;
                    continue; % Skip to next iteration of while loop
                end
                
                % Safe to solve
                dw = A_sys \ (-D_w);
                
                rhs_y = -D_y - (obj.eta_y * (L_yw * dw));
                lhs_diag = 1.0 + obj.eta_y * alpha_vec;
                dy = rhs_y ./ lhs_diag;
                
                % Clipping
                max_norm = 0.4;
                norm_dw = norm(dw);
                if norm_dw > max_norm
                    dw = dw * (max_norm / (norm_dw + 1e-8));
                end
                
                norm_dy = norm(dy);
                if norm_dy > max_norm
                    dy = dy * (max_norm / (norm_dy + 1e-8));
                end
                
                w_new = w_current + dw;
                y_new = obj.Y + dy;
                
                L_new_w_old = obj.compute_L_at(w_new, obj.Y);
                L_new_w_new = obj.compute_L_at(w_new, y_new);
                
                accept_y = (L_new_w_new <= L_new_w_old);
                accept_z = (L_new_w_new >= L_curr) || (norm(gy) > grad_min);
                
                accept = accept_z && accept_y;
                
                if ~accept
                    if ~accept_z, obj.eta_z = obj.eta_z * 0.5;
                    else, obj.eta_z = obj.eta_z * 0.9; end
                    
                    if ~accept_y, obj.eta_y = obj.eta_y * 0.5;
                    else, obj.eta_y = obj.eta_y * 0.9; end
                end
            end
            
            if accept
                obj.eta_z = obj.eta_z * 1.1;
                obj.eta_y = obj.eta_y * 1.1;
                
                % Update parameters
                [wg_new, wpsi_new] = obj.split_w(w_current + dw);
                obj.G.params = wg_new;
                obj.Psi.params = wpsi_new;
                obj.Y = obj.Y + dy;
                
                L = L_new_w_new; 
            else
                L = L_curr;
            end
            
            nGw = norm(gw);
            nGy = norm(gy);
            ndw = norm_dw;
            ndy = norm_dy;
        end
        
        % Train loop
        function obj = train(obj, steps)
            fprintf('%-5s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s\n', ...
                'Step', 'Loss', 'MSE', 'Corr', '|GradW|', '|GradY|', '|NormDW|', '|NormDY|');
            fprintf('%s\n', repmat('-', 1, 105));
            
            grad_min = 0.0001;
            
            for i = 1:steps
                [obj, L, mse, corr, nGw, nGy, ndw, ndy, accept] = obj.update_step(grad_min);
                
                if ~accept
                    fprintf('Optimization stopped (step rejected).\n');
                    break; 
                end
                
                if mod(i, 5) == 0 || i == 1
                    fprintf('%-5d | %-10.6f | %-10.6f | %-10.6f | %-10.4f | %-10.4f | %-10.4f | %-10.4f\n', ...
                        i, L, mse, corr, nGw, nGy, ndw, ndy);
                end
            end
        end
    end
end