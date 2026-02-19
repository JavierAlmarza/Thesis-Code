import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import GenerateData as gd

class MaxMinMaxSolver:
    def __init__(self, X, burn_in=50, lr_Y=1e-3, lr_ab=1e-3, n_inner_Y = 50, lr_lambda=1e-4, device='cpu'):
        """
        Args:
            X: torch.Tensor of shape [T], the observed process
            burn_in: int, number of steps to ignore for Z_t convergence
            lr_Y, lr_ab, lr_lambda: learning rates for inner min, outer max variables
            device: 'cpu' or 'cuda'
        """
        self.X = X.to(device)
        self.T = X.shape[0]
        self.t0 = burn_in
        self.device = device
        self.n_inner_Y = n_inner_Y
        self.n_inner_Z = 2

        
        # Initialize variables
        self.Y = torch.zeros(self.T, requires_grad=True, device=device)
        self.raw_b = torch.randn((), requires_grad=True, device=device)
        self.raw_a = torch.randn((), requires_grad=True, device=device)
        self.raw_lambda = torch.randn((), requires_grad=True, device=device)

                    
        # Separate optimizers
        self.opt_Y = torch.optim.Adam([self.Y], lr=lr_Y)
        self.opt_ab = torch.optim.Adam([self.raw_a, self.raw_b], lr=lr_ab)
        self.opt_lambda = torch.optim.Adam([self.raw_lambda], lr=lr_lambda)
        P = gd.kalman_predictors(X.detach().cpu().view(-1).numpy(),0.7,0.7,0.7,0.7)
        self.P = P

    def get_a(self):
        return 0.5 * torch.tanh(self.raw_a)

    def get_b(self):
        return torch.tanh(self.raw_b)

    def get_lambda(self):
        return 125.0 * torch.tanh(self.raw_lambda)

        
    def compute_Z(self, a, b):
        """Compute Z_t recursively without in-place operations."""
        Z_list = [torch.tensor(0.0, device=self.device)]  # initial Z_0
        for t in range(self.T - 1):
            Z_next = a * self.X[t] + b * Z_list[-1]
            Z_list.append(Z_next)
        Z = torch.stack(Z_list)  # shape [T]
        return Z

        
    def step(self):

        # ===== INNER LOOP: approximately minimize over Y =====
        for _ in range(self.n_inner_Y):
            self.opt_Y.zero_grad()

            a = self.get_a().detach()
            b = self.get_b().detach()
            lambda_ = self.get_lambda().detach()

            Z = self.compute_Z(a, b)
            Z_mean = Z[self.t0:].mean()
            lambda_ = 50

            loss_Y = ((self.X[self.t0:] - self.Y[self.t0:])**2).mean() \
                     + lambda_ * ((Z[self.t0:] - Z_mean) * self.Y[self.t0:]).mean()

            loss_Y.backward()
            self.opt_Y.step()

        # ===== OUTER STEP: maximize over (a,b,lambda) =====
        for _ in range(self.n_inner_Z):
            # --- (a,b) update ---
            self.opt_ab.zero_grad()

            a = self.get_a()
            b = self.get_b()
            lambda_ = self.get_lambda().detach()   # IMPORTANT

            Z = self.compute_Z(a, b)
            Z_mean = Z[self.t0:].mean()
            
            lambda_ = 50

            loss_ab = -lambda_ * ((Z[self.t0:] - Z_mean) * self.Y[self.t0:]).mean()
            loss_ab.backward()
            self.opt_ab.step()

            # --- lambda update ---
            self.opt_lambda.zero_grad()

            a = self.get_a().detach()
            b = self.get_b().detach()
            lambda_ = self.get_lambda()

            Z = self.compute_Z(a, b)
            Z_mean = Z[self.t0:].mean()
            lambda_ = 50

            loss_lambda = -lambda_ * ((Z[self.t0:] - Z_mean) * self.Y[self.t0:]).mean()
            loss_lambda.backward()
            self.opt_lambda.step()



    def train(self, num_steps=1000, verbose=False):
        """Run optimization for a number of steps."""
        for step in range(num_steps):
            self.step()
            if verbose and (step % 50 == 0):
                a = self.get_a()
                b = self.get_b()
                lambda_ = self.get_lambda()
                lambda_ = 50
                Z = self.compute_Z(a, b)
                Z_mean = Z[self.t0:].mean()
                current_loss = ((self.X[self.t0:] - self.Y[self.t0:]) ** 2).mean() + \
                               lambda_ * ((Z[self.t0:] - Z_mean) * self.Y[self.t0:]).mean()
                Diff =  ((self.X[self.t0:].detach().numpy().reshape(-1) - self.Y[self.t0:].detach().numpy().reshape(-1) - self.P[self.t0:])**2).mean()
                print(f"Step {step:4d}: Loss={current_loss.item():.6f}, "
                      f"a={a.item():.4f}, b={b.item():.4f}, "
                      #f"lambda={lambda_.item():.4f}, "
                      f"|Y-Pred|^2 ={Diff.item():.4f}, "
                      f"coupling={lambda_ * ((Z[self.t0:] - Z_mean) * self.Y[self.t0:]).mean().item():.4f}")
                print("corr(Z,X) =", torch.dot((Z[self.t0:] - Z_mean), self.X[self.t0:]).item() / ((Z[self.t0:] - Z_mean).norm() * self.X[self.t0:].norm()).item())

    
    def get_solution(self):
        """Return current solution."""
        return {
            'Y': self.Y.detach().clone(),
            'a': self.get_a().item(),
            'b': self.get_b().item(),
            'lambda': self. get_lambda().item(),
            'mse loss': ((self.X[self.t0:] - self.Y[self.t0:]) ** 2).mean().item()
        }

# ===== Example usage =====
if __name__ == "__main__":
    T = 2000
    burn_in = 500
    # Simulate a stationary Gaussian process
    #X = torch.randn(T)
    Xsim, Ztrue = gd.generate_kalman_data(T,0.7,0.7,0.7,0.7)
    X = torch.tensor(Xsim.reshape(-1), dtype=torch.float32)
    
    solver = MaxMinMaxSolver(X, burn_in=burn_in, lr_Y=1e-2, lr_ab=1e-2, lr_lambda=1e-3,n_inner_Y = 50)
    solver.train(num_steps=2000, verbose=True)
    
    solution = solver.get_solution()
    print(solution)
