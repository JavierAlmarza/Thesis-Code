% 1. Main

function run_hessian_solver()
    clear; clc;
    
    %Degrees of polynomials
    degree_G = 4;
    degree_Psi = 2;

    % Data parameters
    T = 3000;
    alpha_kal = 0.8;
    beta_kal = 1.0;
    tau_x = 0.4;
    tau_z = 0.4;
    omega = 0.05;
    alpha_garch = 0.20;
    beta_garch = 0.79;
    
    % Solver parameters
    lmbda = 1.0;
    eta_z_m = 0.05;
    eta_y_m = 0.05;
    eta_z_i = T * 0.05;
    eta_y_i = T * 0.1;

    % Choose mode (options: "Kalman", "GARCH") 
    mode_data = "Kalman";    
    fprintf('\nInput data: %s\n', mode_data);
    
    % Generate data
    if mode_data == "Kalman"
        [Xsim, Ztrue] = generate_kalman_data(T, alpha_kal, beta_kal, tau_x, tau_z);
    elseif mode_data == "GARCH"
        [Xsim, Ztrue, ~] = generate_garch_data(T, omega, alpha_garch, beta_garch);
    else
        error('Unknown data mode');
    end
    
    X = single(reshape(Xsim, [T, 1])); % Ensure column vector
    
    % Calculate true solutions/predictors
    if mode_data == "Kalman"
        P = kalman_predictors(double(X), alpha_kal, beta_kal, tau_x, tau_z);
        Ystar = double(X) - P;
    elseif mode_data == "GARCH"
        P = sqrt(Ztrue) - mean(sqrt(Ztrue));
        Ystar = (double(X) .* mean(sqrt(Ztrue))) ./ sqrt(Ztrue');
    end
   
    
    fprintf('Data metrics\n');
    fprintf('Var(X) = %.4f\n', mean(std(double(X))^2));
    fprintf('Equilibrium V = |Pred|^2 = %.4f\n', mean(P.^2));
    %fprintf('|Y*|^2 = %.4f \n\n', mean(Ystar.^2));
    
    % Initialize G and Psi
    G = PolynomialG(degree_G);
    Psi = PolynomialPsi(degree_Psi);
    
    solver = HessianSolver(X, G, Psi, lmbda, eta_z_m, eta_y_m, eta_z_i, eta_y_i);
    
    fprintf('\nStarting Training (deg_G=%d, deg_Psi=%d)...\n\n', degree_G, degree_Psi);
    
    solver = solver.train(150);
    
    % Print final results
    Y_final = solver.Y;
    err = mean((Y_final - Ystar).^2);
    true_var = mean(Ystar.^2);
    
    fprintf('\n--- Final Results ---\n');
    fprintf('Equilibrium value |Pred|^2: %.6f\n', mean(P.^2));
    fprintf('Squared Relative Error (|Y - Y*|^2 / |Y*|^2): %.6f\n', err/true_var);
    %fprintf('|Y|^2: %.6f\n', mean(Y_final.^2));
    %fprintf('Var(X): %.6f\n', var(double(X)));
    
    % --- Print Learned Parameters ---
    w_x = solver.G.params(1:degree_G);
    w_h = solver.G.params(degree_G+1 : 2*degree_G);
    w_c = solver.G.params(2*degree_G+1 : end);
    
    w_x_tanh = tanh(w_x);
    w_h_tanh = tanh(w_h);
    
    fprintf('\nLearned G Params (w_x): ');
    fprintf('%.4f ', w_x_tanh);
    fprintf('\nLearned G Params (w_h): ');
    fprintf('%.4f ', w_h_tanh);
    
    if solver.G.num_w_c > 0
        w_c_tanh = tanh(w_c);
        fprintf('\nLearned G Params (w_c): ');
        fprintf('%.4f ', w_c_tanh);
        fprintf(' (Interaction coefficients)');
    end
    
    psi_params_tanh = tanh(solver.Psi.params);
    fprintf('\nLearned Psi Params: ');
    fprintf('%.4f ', psi_params_tanh);
    fprintf('\n');

end

% 2. Helper functions (Kalman and GARCH data generators)


function [X, Z] = generate_kalman_data(T, alpha, beta, tau_eta, tau_epsilon, burn_in)
    if nargin < 6, burn_in = 500; end
    
    if abs(alpha) < 1
        var_Z = tau_epsilon^2 / (1 - alpha^2);
        Z0 = randn() * sqrt(var_Z);
    else
        Z0 = 0.0;
    end
    
    total = T + burn_in;
    Z = zeros(1, total);
    X = zeros(1, total);
    
    Z(1) = Z0;
    for t = 1:total-1
        eps = tau_epsilon * randn();
        Z(t+1) = alpha * Z(t) + eps;
        X(t) = beta * Z(t) + tau_eta * randn();
    end
    X(total) = beta * Z(total) + tau_eta * randn();
    
    Z = Z(burn_in+1:end);
    X = X(burn_in+1:end);
end

function [X, h, epsi] = generate_garch_data(T, omega, alpha, beta, burn_in)
    if nargin < 5, burn_in = 500; end
    total_T = T + burn_in;
    
    X = zeros(1, total_T);
    h = zeros(1, total_T);
    epsi = zeros(1, total_T);
    
    h(1) = omega / (1 - alpha - beta);
    
    for t = 2:total_T
        h(t) = omega + alpha * X(t-1)^2 + beta * h(t-1);
        eps_val = randn();
        X(t) = sqrt(h(t)) * eps_val;
        epsi(t) = eps_val;
    end
    
    X = X(burn_in+1:end);
    h = h(burn_in+1:end);
    epsi = epsi(burn_in+1:end);
end

function preds = kalman_predictors(X, alpha, beta, tau_eta, tau_eps)
    X = X(:);
    T = length(X);
    R = float_s(tau_eta)^2;
    Q = float_s(tau_eps)^2;
    
    Z_hat = 0.0;
    P = Q / (1 - alpha^2);
    
    preds = zeros(T, 1);
    
    for t = 1:T
        preds(t) = beta * Z_hat;
        y = X(t) - preds(t);
        
        S = beta^2 * P + R;
        K = (P * beta) / S;
        
        Z_hat = Z_hat + K * y;
        P = (1 - K * beta) * P;
        
        Z_hat = alpha * Z_hat;
        P = alpha^2 * P + Q;
    end
end

function v = float_s(x)
    v = double(x);
end