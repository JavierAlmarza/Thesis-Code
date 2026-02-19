function [h, pValue, stat, crit] = archtest(x, varargin)
  % ARCHTEST Engle's ARCH test (Octave-compatible)
  %
  % Usage:
  %   [h, p, stat, crit] = archtest(x, 'Lags', [1 3 5], 'Alpha', 0.05)
  %
  % INPUT:
  %   x        - Residual series (column or row vector)
  %   'Lags'   - Vector of lags to test (default: 1)
  %   'Alpha'  - Significance level (default: 0.05)
  %
  % OUTPUT:
  %   h        - Vector of 0/1 decisions (1: reject H0 of no ARCH)
  %   pValue   - Vector of p-values
  %   stat     - Vector of LM statistics
  %   crit     - Vector of chi-squared critical values

  % Defaults
  lags = 1;
  alpha = 0.05;

  % Parse varargin
  for i = 1:2:length(varargin)
    switch lower(varargin{i})
      case 'lags'
        lags = varargin{i+1};
      case 'alpha'
        alpha = varargin{i+1};
      otherwise
        error(['Unknown parameter: ', varargin{i}])
    end
  end

  x = x(:);  % Ensure column vector
  x = x(~isnan(x));  % Remove NaNs
  T = length(x);
  e2 = x.^2;

  % Preallocate results
  k = length(lags);
  h = false(k,1);
  pValue = zeros(k,1);
  stat = zeros(k,1);
  crit = zeros(k,1);

  for j = 1:k
    q = lags(j);
    if T <= q
      error(['Not enough data for lag q = ', num2str(q)]);
    end

    % Construct regression: e2_t = alpha_0 + sum alpha_i e2_{t-i}
    Y = e2(q+1:end);
    X = ones(length(Y), q+1);  % intercept + q lags
    for l = 1:q
      X(:, l+1) = e2(q+1 - l:end - l);
    end

    % Run regression and get R^2
    [~, ~, ~, ~, stats] = regress(Y, X);
    R2 = stats(1);
    LM = length(Y) * R2;

    % Test result
    p = 1 - chi2cdf(LM, q);
    c = chi2inv(1 - alpha, q);
    h(j) = p < alpha;
    pValue(j) = p;
    stat(j) = LM;
    crit(j) = c;
  end
end


