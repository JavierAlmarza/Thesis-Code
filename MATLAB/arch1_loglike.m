function logL = arch1_loglike(omega,alpha, residuals)
    T = length(residuals);
    sigma2 = zeros(T,1);
    logL = 0;

    % Initial variance estimate (can also set to var(residuals) or omega/(1-alpha))
    sigma2(1) = var(residuals);  

    for t = 2:T
        sigma2(t) = omega + alpha * residuals(t-1)^2;

        if sigma2(t) <= 0  % infeasible parameters
            logL = -Inf;
            return
        end

        logL = logL - 0.5 * (log(2*pi) + log(sigma2(t)) + residuals(t)^2 / sigma2(t));
    end
end