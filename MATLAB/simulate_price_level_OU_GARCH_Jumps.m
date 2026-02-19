function X = simulate_price_level_OU_GARCH_Jumps(T, phi, mu0,  alpha0, alpha1, beta1, sigma_0, lambda, mu_J, sigma_J,X0)
% Simulates a discrete-time Ornstein-Uhlenbeck process in price space
% with GARCH(1,1) volatility and Poisson jumps.

% Inputs:
% T         - number of time steps
% mu0        - initial mean of price
% phi       - mean-reversion strength
% alpha0    - GARCH intercept
% alpha1    - GARCH ARCH parameter
% beta1     - GARCH GARCH parameter
% lambda - probability of jump per time step
% mu_J      - mean of jump size (in return units)
% sigma_J   - std. dev. of jump size (in return units)
% X0        - initial price
% sigma_0  - initial conditional sd

    % Preallocate
    X = zeros(T, 1);
    sigma2 = zeros(T, 1);
    jump_indicator = zeros(T, 1);

    % Initialize
    X(1) = X0;
    sigma2(1) = sigma_0^2;
    epsilon = randn;
    mu = mu0;

    % Simulate
    for t = 2:T

        % Update variance
        sigma2(t) = alpha0 + alpha1 * (epsilon^2) * sigma2(t-1) + beta1 * sigma2(t-1);
        sigma_t = sqrt(sigma2(t));

        % Mean-reversion term
        
        mr_term = phi * (mu - X(t-1));

        epsilon = randn;

        % GARCH shock (scaled by price)
        garch_noise = sigma_t * X(t-1) * epsilon;

        % Jump
        if rand < lambda
            jump_indicator(t) = 1;
            jump_size = (mu_J + sigma_J * randn)*X(t-1);  % price-level jump
        else
            jump_size = 0;
        end

        % Price update (additive)
        X(t) = X(t-1) + mr_term + garch_noise + jump_size;
        if t< 101
            mu = ((100-t)/100) * mu0 + (t/100) * mean(X(1:t));
        else
            mu =  mean(X(t-100:t));
        end

    end
end
