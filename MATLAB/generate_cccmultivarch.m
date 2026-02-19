function paths = generate_cccmultivarch(N, T, seed)
% Generate N paths of length T of a bivariate CCC-ARCH(1) model
% using the principal square root of the conditional covariance matrix.
%
%   sigma_{i,t}^2 = omega_i + alpha_{i,1} eps_{1,t-1}^2 + alpha_{i,2} eps_{2,t-1}^2
%   Sigma_t = [sigma1^2, rho*sigma1*sigma2;
%              rho*sigma1*sigma2, sigma2^2]
%   eps_t   = Sigma_t^(1/2) * z_t,  z_t ~ N(0,I)
%
%
% Output:
%   paths: [N x T x 2] matrix of simulated returns

    if nargin > 2 && ~isempty(seed)
        rng(seed);
    end

    % Parameters
 
    mu1 = 0.15 / 252;
    mu2 = 0.06 / 252;
    omega = [0.0004, 0.00008];
    A = [0.15, 0.06;
         0.04, 0.1];
    rho = 0.5;

    % Allocate output
    paths = zeros(N, T, 2);

    for n = 1:N
        sigma2 = zeros(T,2);
        eps = zeros(T,2);
        sigma2(1,:) = omega;
        z = randn(T,2);

        for t = 2:T
            eps_prev = eps(t-1,:);

            % ARCH dynamics
            sigma2(t,1) = omega(1) + A(1,1)*eps_prev(1)^2 + A(1,2)*eps_prev(2)^2;
            sigma2(t,2) = omega(2) + A(2,1)*eps_prev(1)^2 + A(2,2)*eps_prev(2)^2;

            % Build conditional covariance
            a = sigma2(t,1);
            b = sigma2(t,2);
            c = rho*sqrt(sigma2(t,1))*sqrt(sigma2(t,2));

            % Principal symmetric square root of 2x2 covariance
            delta = sqrt(a*b - c^2);
            s = sqrt(a + b + 2*delta);
            Sigma_sqrt = [a + delta, c;
                          c, b + delta] / s;

            % Shocks
            eps(t,:) = (Sigma_sqrt * z(t,:)')';
        end

        % Add mean
        paths(n,:,1) = mu1 + eps(:,1)';
        paths(n,:,2) = mu2 + eps(:,2)';
    end
end
