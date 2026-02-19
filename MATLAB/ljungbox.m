function [h, pValue, Qstat, crit] = ljungbox1(x, hLag, alpha)
  % ljungbox  Ljung–Box Q test for autocorrelation in Octave
  %   [h, pValue, Q, crit] = ljungbox(x, hLag, alpha)
  %
  % INPUTS:
  %   x     - Time series (residuals or squared residuals)
  %   hLag  - Number of lags to include (e.g., 20)
  %   alpha - Significance level (default 0.05)
  %
  % OUTPUTS:
  %   h      - Test decision: 1 if reject H0 (autocorrelation exists), 0 otherwise
  %   pValue - p-value for test
  %   Qstat  - Ljung–Box Q statistic
  %   crit   - Chi-squared critical value

  if nargin < 2
    hLag = 20;
  end
  if nargin < 3
    alpha = 0.05;
  end

  x = x(:);               % Ensure column vector
  T = length(x);          % Sample size
  x = x - mean(x);        % Demean

  % Sample autocorrelations
  acf = autocorr_manual(x, hLag);

  % Compute Q-statistic (Ljung–Box formula)
  Qstat = 0;
  for k = 1:hLag
    Qstat = Qstat + (acf(k)^2) / (T - k);
  end
  Qstat = T * (T + 2) * Qstat;

  % Chi-square critical value
  crit = chi2inv(1 - alpha, hLag);
  pValue = 1 - chi2cdf(Qstat, hLag);
  h = pValue < alpha;
end

function acf = autocorr_manual(x, maxLag)
  % Computes sample autocorrelations up to lag maxLag
  T = length(x);
  acf = zeros(maxLag, 1);
  var_x = var(x, 1); % population variance (normalizing by N)

  for k = 1:maxLag
    acf(k) = sum(x(1:T-k) .* x(k+1:T)) / (T * var_x);
  end
end

