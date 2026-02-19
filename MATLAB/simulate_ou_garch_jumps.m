function X = simulate_ou_garch_jumps(n, phi, mu, alpha0, alpha1, beta1, ...
                                     sigma0, lambda, muJ, sigmaJ, X0)
% Simulates a discrete-time OU process with GARCH volatility and Poisson jumps
% Inputs:
%   n      - number of time steps
%   phi    - mean reversion strength
%   mu     - long-run mean
%   alpha0, alpha1, beta1 - GARCH(1,1) parameters
%   sigma0 - initial standard deviation
%   lambda - jump probability (per time step)
%   muJ    - mean of jump size
%   sigmaJ - std deviation of jump size
%   X0     - initial value of the process
% Output:
%   X      - simulated process (1 x n vector)

    % Preallocate vectors
    X = zeros(1, n);
    sigma2 = zeros(1, n);
    epsilon = zeros(1, n);
    
    % Initial values
    X(1) = X0;
    sigma2(1) = sigma0^2;
    epsilon(1) = 0;

    % Main loop
    for t = 2:n
        % Update GARCH volatility
        sigma2(t) = alpha0 + alpha1 * epsilon(t-1)^2 + beta1 * sigma2(t-1);
        sigma_t = sqrt(sigma2(t));

        % Random draw for innovation
        e_t = randn;

        % Random jump
        if rand < lambda
            jump = muJ + sigmaJ * randn;
        else
            jump = 0;
        end

        % Update process
        drift = phi * (mu - X(t-1));
        noise = sigma_t * e_t ;
        X(t) = X(t-1) + drift + noise  * X(t-1) + jump;

        % Save residual
        epsilon(t) = noise;
    end
end
