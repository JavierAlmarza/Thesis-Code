import numpy as np
import gymnasium as gym
from gymnasium import spaces

def simulate_price_level_OU_GARCH_Jumps(T, phi, mu0, alpha0, alpha1, beta1,
                                        sigma_0, lambda_, mu_J, sigma_J, X0):
    """
    Simulates a discrete-time Ornstein-Uhlenbeck process in price space
    with GARCH(1,1) volatility and Poisson jumps.
    
    Parameters
    ----------
    T : int
        Number of time steps.
    phi : float
        Mean-reversion strength.
    mu0 : float
        Initial mean of price.
    alpha0 : float
        GARCH intercept.
    alpha1 : float
        GARCH ARCH parameter.
    beta1 : float
        GARCH GARCH parameter.
    sigma_0 : float
        Initial conditional sd.
    lambda_ : float
        Probability of jump per time step.
    mu_J : float
        Mean of jump size (in return units).
    sigma_J : float
        Std. dev. of jump size (in return units).
    X0 : float
        Initial price.

    Returns
    -------
    X : ndarray
        Simulated price path of length T.
    """
    # Preallocate
    X = np.zeros(T)
    sigma2 = np.zeros(T)
    sigmat = np.zeros(T)
    eps_arr = np.zeros(T)
    J_arr = np.zeros(T)
    jump_indicator = np.zeros(T)

    # Initialize
    X[0] = X0
    sigma2[0] = sigma_0**2
    sigmat[0] = sigma_0
    epsilon = np.random.randn()
    eps_arr[0] = epsilon
    mu = mu0

    for t in range(1, T):
        # Update variance
        sigma2[t] = alpha0 + alpha1 * (epsilon**2) * sigma2[t-1] + beta1 * sigma2[t-1]
        sigma_t = np.sqrt(sigma2[t])
        sigmat[t] = sigma_t

        # Mean reversion term
        mr_term = phi * (mu - X[t-1])

        epsilon = np.random.randn()
        eps_arr[t] = epsilon

        # GARCH noise (scaled by price)
        garch_noise = sigma_t * X[t-1] * epsilon

        # Jump
        if np.random.rand() < lambda_:
            jump_indicator[t] = 1
            jump_size = (mu_J + sigma_J * np.random.randn()) * X[t-1]
        else:
            jump_size = 0
        J_arr[t] = jump_size

        # Price update (note: jump_size^3 as in MATLAB code)
        X[t] = X[t-1] + mr_term + garch_noise + jump_size

        # Adaptive mean update
        if t < 100:
            mu = ((100 - t) / 100) * mu0 + (t / 100) * np.mean(X[:t+1])
        else:
            mu = np.mean(X[t-99:t+1])

    return X, sigmat, eps_arr, jump_indicator, J_arr



def generateOilArchData(T, N=1, alpha0=0.0001, alpha1=0.25, FSens=0.2):
    """
    Generate N paths of length T for the ARCH(1) + OU-GARCH-Jump commodity model.
    
    Parameters
    ----------
    T : int
        Number of time steps.
    N : int
        Number of independent paths.
    alpha0 : float
        ARCH intercept.
    alpha1 : float
        ARCH coefficient.
    FSens : float
        Sensitivity scaling for the commodity returns.

    Returns
    -------
    r : ndarray, shape (N, T)
        Simulated return paths.
    """
    q = 1
    alpha = alpha1 * np.ones(q)
    sigma0 = np.sqrt(alpha0 / (1 - np.sum(alpha)))

    # Allocate arrays
    r = np.zeros((N, T))
    sigma = np.zeros((N, T))
    Fz = np.zeros((N, T))
    epsi = np.random.randn(N, T)

    # Commodity simulation parameters
    phi = 0.1
    mu = 25
    alpha0oil = 0.0001
    alpha1oil = 0.1
    beta1oil = 0.45
    sigma0oil = 0.02
    lambda_ = 0.01
    muJ = 0
    sigmaJ = 0.15
    X0 = 25

    # Simulate commodity prices for each path
    xoil_all = np.zeros((N, T))
    for i in range(N):
        xoil_all[i] = simulate_price_level_OU_GARCH_Jumps(
            T, phi, mu, alpha0oil, alpha1oil,
            beta1oil, sigma0oil, lambda_,
            muJ, sigmaJ, X0
        )

    # Compute commodity returns
    retoil = np.diff(xoil_all, axis=1) / xoil_all[:, :-1]
    rxoil = np.concatenate((np.zeros((N, 1)), retoil), axis=1)

    # Initialize ARCH process
    sigma[:, :q] = sigma0
    r[:, :q] = sigma[:, :q] * epsi[:, :q]
    Fz[:, :q] = alpha0 + alpha[0] * r[:, :q]**2

    # Simulate ARCH process for each path
    for t in range(q, T):
        Fz[:, t] = alpha0 + alpha[0] * r[:, t-1]**2  # since q=1
        sigma[:, t] = np.sqrt(Fz[:, t])
        r[:, t] = sigma[:, t] * epsi[:, t]

    # Combine ARCH + commodity returns
    r = FSens * rxoil + r
    return r , rxoil

def generateOilArchDataMod(T, N=1, alpha0=0.0001, alpha1=0.25, FSens=0.2, phi = 0.1, mu = 25, alpha0oil = 0.0001, alpha1oil = 0.1, beta1oil = 0.45, sigma0oil = 0.02, sigmaJ = 0.15, X0 = 25):
    """
    Generate N paths of length T for the ARCH(1) + OU-GARCH-Jump commodity model.
    
    Parameters
    ----------
    T : int
        Number of time steps.
    N : int
        Number of independent paths.
    alpha0 : float
        ARCH intercept.
    alpha1 : float
        ARCH coefficient.
    FSens : float
        Sensitivity scaling for the commodity returns.

    Returns
    -------
    r : ndarray, shape (N, T)
        Simulated return paths.
    """
    q = 1
    alpha = alpha1 * np.ones(q)
    sigma0 = np.sqrt(alpha0 / (1 - np.sum(alpha)))

    # Allocate arrays
    r = np.zeros((N, T))
    sigma = np.zeros((N, T))
    Fz = np.zeros((N, T))
    epsi = np.random.randn(N, T)

    # Commodity simulation parameters
    lambda_ = 0.005
    muJ = 0
    

    # Simulate commodity prices for each path
    xoil_all = np.zeros((N, T))
    for i in range(N):
        xoil_all[i],_,_,_,_ = simulate_price_level_OU_GARCH_Jumps(T, phi, mu, alpha0oil, alpha1oil,beta1oil, sigma0oil, lambda_,muJ, sigmaJ, X0)

    # Compute commodity returns
    retoil = np.diff(xoil_all, axis=1) / xoil_all[:, :-1]
    rxoil = np.concatenate((np.zeros((N, 1)), retoil), axis=1)

    # Initialize ARCH process
    sigma[:, :q] = sigma0
    r[:, :q] = FSens * rxoil[:,:q]+ sigma[:, :q] * epsi[:, :q]
    Fz[:, :q] = alpha0 + alpha[0] * r[:, :q]**2

    # Simulate ARCH process for each path
    for t in range(q, T):
        Fz[:, t] = alpha0 + alpha[0] * (r[:, t-1] - FSens * rxoil[:,t])**2  # since q=1
        sigma[:, t] = np.sqrt(Fz[:, t])
        r[:, t] = FSens * rxoil[:,t]+sigma[:, t] * epsi[:, t]

    return r , rxoil


class PortfolioEnvOil(gym.Env):
    def __init__(self, R, Oil, q=1, r=0.01, k=0.05, include_action=False, include_vol=False, include_full = False):
        super().__init__()
        R = np.atleast_2d(R)
        Oil = np.atleast_2d(Oil)
        self.R = R
        self.Oil = Oil
        self.N, self.T = R.shape
        assert Oil.shape[1] == self.T
        self.q = q
        self.r = r
        self.k = k
        self.include_action = include_action
        self.include_vol = include_vol
        self.include_full = include_full
        self.logged_actions = []
        self.last_action = None
        self.include_oil = True

        # --- Construct observation vector components
        obs_low = []
        obs_high = []

        if self.include_oil:
            obs_low.append(-np.inf)
            obs_high.append(np.inf)
            if self.include_full and q>0:
                obs_low += [-np.inf] * q
                obs_high += [np.inf] * q
                
        if q > 0:
            obs_low += [-np.inf] * q
            obs_high += [np.inf] * q

        if include_action:
            obs_low.append(0.0)
            obs_high.append(1.0)

        if include_vol:
            obs_low.append(0.0)
            obs_high.append(np.inf)




        # Handle empty observation vector: add dummy dimension
        if len(obs_low) == 0:
            self.use_dummy_obs = True
            obs_low = [0.0]
            obs_high = [0.0]
        else:
            self.use_dummy_obs = False

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array(obs_low, dtype=np.float32),
            high=np.array(obs_high, dtype=np.float32),
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.batch_idx = self.np_random.integers(self.N)
        self.t = self.q
        self.prev_w = 0.5
        self.done = False
        self.terminated = False
        self.ret_buffer = []

        R_slice = self.R[self.batch_idx, self.t - self.q:self.t][::-1] if self.q > 0 else []
        #Oil_slice = self.Oil[self.batch_idx, self.t - self.q:self.t+1][::-1] if self.include_oil else []      
        Oil_slice = (
            [] if not self.include_oil
            else self.Oil[self.batch_idx, self.t:self.t+1] if not self.include_full
            else self.Oil[self.batch_idx, self.t - self.q:self.t+1][::-1]
        )

        vol = 0.0

        a = [self.prev_w] if self.include_action else []
        b = [vol] if self.include_vol else []
        state_parts = list(Oil_slice) + list(R_slice) + a + b

        if self.use_dummy_obs:
            self.state = np.zeros(1, dtype=np.float32)
        else:
            self.state = np.array(state_parts, dtype=np.float32)

        return self.state, {}

    def step(self, action):
        print('Passed action is ', action)
        w = float(np.clip(action, 0.0, 1.0))
        R_t = self.R[self.batch_idx, self.t]
        Oil_t = self.Oil[self.batch_idx, self.t+1] if self.t<self.T-1 else 0
        self.logged_actions.append(w)
        self.last_action = w

        reward = w * R_t + self.r * (1 - w) - self.k * (w * R_t) ** 2
        print('Reward is ', reward)

        self.ret_buffer.append(R_t)
        if len(self.ret_buffer) > 30:
            self.ret_buffer.pop(0)
        vol = np.std(self.ret_buffer) if len(self.ret_buffer) >= 2 else 0.0
        if self.include_oil:
            Oil_lags = (
                np.concatenate(([Oil_t],self.state[0:self.q])) if self.include_full
                else np.array([Oil_t])
            )
                

        if self.q > 0:
            # Extract past lags
            lag_start = 0 if not self.include_oil else (1 if not self.include_full else 1 + self.q)
            R_lags = np.concatenate(([R_t], self.state[lag_start:lag_start + self.q - 1]))  # shift lags
            lagged_returns = R_lags
        else:
            lagged_returns = []

        a = [w] if self.include_action else []
        b = [vol] if self.include_vol else []
        state_parts = list(Oil_lags) + list(lagged_returns) + a + b

        if self.use_dummy_obs:
            self.state = np.zeros(1, dtype=np.float32)
        else:
            self.state = np.array(state_parts, dtype=np.float32)

        self.prev_w = w
        self.t += 1
        self.done = self.terminated = (self.t >= self.T)

        return self.state, reward, self.done, False, {}

    def render(self):
        print(f"t={self.t}, state={self.state}")


class PortfolioEnvOilFull(gym.Env):
    def __init__(self, R, Oil, q=1, r=0.01, k=0.05, include_action=True, include_vol=True):
        super().__init__()
        self.R = R
        self.Oil = Oil
        self.N, self.T = R.shape
        assert Oil.shape[1] == self.T
        self.q = q
        self.r = r
        self.k = k
        self.include_action = include_action
        self.include_vol = include_vol
        self.logged_actions = []
        self.last_action = None
        self.include_oil = True

        # --- Construct observation vector components
        obs_low = []
        obs_high = []

        if self.include_oil:
            obs_low.append(-np.inf)
            obs_high.append(np.inf)
        if q > 0:
            obs_low += [-np.inf] * q
            obs_high += [np.inf] * q

        if include_action:
            obs_low.append(0.0)
            obs_high.append(1.0)

        if include_vol:
            obs_low.append(0.0)
            obs_high.append(np.inf)




        # Handle empty observation vector: add dummy dimension
        if len(obs_low) == 0:
            self.use_dummy_obs = True
            obs_low = [0.0]
            obs_high = [0.0]
        else:
            self.use_dummy_obs = False

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array(obs_low, dtype=np.float32),
            high=np.array(obs_high, dtype=np.float32),
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.batch_idx = self.np_random.integers(self.N)
        self.t = self.q
        self.prev_w = 0.5
        self.done = False
        self.terminated = False
        self.ret_buffer = []

        R_slice = self.R[self.batch_idx, self.t - self.q:self.t][::-1] if self.q > 0 else []
        Oil_slice = self.Oil[self.batch_idx, self.t - self.q:self.t+1][::-1] if self.include_oil else []      
        vol = 0.0

        a = [self.prev_w] if self.include_action else []
        b = [vol] if self.include_vol else []
        state_parts = list(Oil_slice) + list(R_slice) + a + b

        if self.use_dummy_obs:
            self.state = np.zeros(1, dtype=np.float32)
        else:
            self.state = np.array(state_parts, dtype=np.float32)

        return self.state, {}

    def step(self, action):
        print('Passed action is ', action)
        w = float(np.clip(action, 0.0, 1.0))
        R_t = self.R[self.batch_idx, self.t]
        Oil_t = self.Oil[self.batch_idx, self.t+1] if self.t<self.T-1 else 0
        self.logged_actions.append(w)
        self.last_action = w

        reward = w * R_t + self.r * (1 - w) - self.k * (w * R_t) ** 2
        print('Reward is ', reward)

        self.ret_buffer.append(R_t)
        if len(self.ret_buffer) > 30:
            self.ret_buffer.pop(0)
        vol = np.std(self.ret_buffer) if len(self.ret_buffer) >= 2 else 0.0
        if self.include_oil:
            Oil_lags = np.concatenate(([Oil_t], self.state[0:self.q]))  # shift lags

        if self.q > 0:
            # Extract past lags
            lag_start = 1+self.q if self.include_oil else 0
            R_lags = np.concatenate(([R_t], self.state[lag_start:lag_start + self.q - 1]))  # shift lags
            lagged_returns = R_lags
        else:
            lagged_returns = []

        a = [w] if self.include_action else []
        b = [vol] if self.include_vol else []
        state_parts = list(Oil_lags) + list(lagged_returns) + a + b

        if self.use_dummy_obs:
            self.state = np.zeros(1, dtype=np.float32)
        else:
            self.state = np.array(state_parts, dtype=np.float32)

        self.prev_w = w
        self.t += 1
        self.done = self.terminated = (self.t >= self.T)

        return self.state, reward, self.done, False, {}

    def render(self):
        print(f"t={self.t}, state={self.state}")