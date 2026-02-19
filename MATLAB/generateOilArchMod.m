function Oil = generateOilArchMod(T)


    m = T;
    T = T+100;
    start_idx = T - m + 1;
    q=1;
    
    dx = 1;
    dz = 2;
    
    z      = cell(dz,1);
    type_v = z;
    ndv    = zeros(dz,1);
    for i = 1:dz
        type_v{i} = 'Real'; ndv(i) = 0;
    end
    
    % parameters
    alpha0 = 0.0001;
    alpha = 0.25 * ones(q,1);
    sigma0 = sqrt(alpha0 / (1 - sum(alpha)))
    
    % simulate the process
    r     = zeros(T,1);     % r_1 … r_T
    sigma = zeros(T,1);
    Fz  = zeros(T,1);
    epsi = randn(T,1);     % σ_1 … σ_T
    
    % Commodity simulation
    
    % Parameters
    n = T;          % number of steps (e.g., daily for 1 year)
    phi = 0.1;       % mean reversion strength
    mu = 25;          % long-term mean
    alpha0oil = 0.0004;    % GARCH params
    alpha1oil = 0.1;
    beta1oil = 0.45;
    sigma_0 = 0.02;     % initial volatility
    lambda = 0.005;    % jump probability
    mu_J = 0;        % mean jump size
    sigma_J = 0.45;     % jump std deviation
    X0 = 25;          % initial price
    
    
    
    % Simulate
    %xoil = simulate_ou_garch_jumps(n, phi, mu, alpha0oil, alpha1oil, beta1oil, sigma0oil, lambda, muJ, sigmaJ, X0);
    
    xoil = simulate_price_level_OU_GARCH_Jumps(n, phi,mu, alpha0oil, alpha1oil, beta1oil, sigma_0, lambda, mu_J, sigma_J, X0);
    
    
    % Sensitivity
    FSens = 0.2;
    
    retoil = diff(xoil)./xoil(1:end-1);
    rxoil = [0;retoil(:)];
    
    sigma(1:q) = sigma0;
    r(1:q) = FSens * rxoil(1:q)+ sigma(1,q) .* epsi(1:q);
    Fz(1:q) = alpha0 +  sum(alpha .* r(1:q).^2);
    
    
    for t = q+1:T
        Fz(t)    = alpha0+sum(alpha .* (r(t-1)-FSens * rxoil(t)).^2);
        sigma(t) = sqrt( Fz(t) );
        r(t)     = FSens * rxoil(t)+sigma(t) * epsi(t);
    end
    
    for i = q+1:dz
        z{i} = rxoil(start_idx-i+q+1:T-i+q+1);
    end
    
    for i = 1:q
        z{i} = r(start_idx - i : T - i);  % r_{t-i}
    end
    R = r(start_idx:T);
    Oil = rxoil(start_idx:T);

end
