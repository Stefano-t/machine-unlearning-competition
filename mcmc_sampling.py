"""Minimum code to run the MCMC unlearning algorithm.

This code implements a slightly different version from the reference paper
"Markov Chain Monte Carlo-Based Machine Unlearning: Unlearning What Needs to be
Forgotten" (https://arxiv.org/pdf/2202.13585.pdf). The paper suggests to
precompute all the model weights. This is okay if your model is very
small. Since the competition requires the unlearning of a ResNet (~1.5M
parameters), storing them is unfeasible. Thus, this code doesn't save the best
model but tries to find it while scanning and running the chain.
"""

import numpy as np
import torch
from scipy import stats
from contextlib import contextmanager
import time



def load_params_in_model(model, params):
    """Helper function to load `params` inside `model`

    **Note**: this method modifies `model` **in place**.
    """
    total_params = len(params)
    params_set = 0
    with torch.no_grad():
        for w_name, w in model.named_parameters():
             # skip un-learnable params
            if not w.requires_grad:
                continue
            to_set = model
            names = w_name.split(".")
            for idx in range(len(names)-1):
                to_set = getattr(to_set, names[idx])
            setattr(to_set, names[-1], torch.nn.Parameter(
                torch.tensor(params[params_set: params_set + w.numel()],
                             dtype=w.dtype, device=DEVICE).reshape(w.shape))
            )
            params_set += w.numel()


def get_numpy_model_params(model, device="cpu", get_only_grads=True):
    """Extract all the weights from `model` as numpy flatten array."""
    params = torch.concat([p.flatten() for p in model.parameters() if get_only_grads and p.requires_grad])
    return params.to(device).detach().numpy()


@contextmanager
def timeit(msg):
    """Time the function inside the context manager"""
    elapsed = time.perf_counter_ns()
    try:
        yield
    finally:
        elapsed = time.perf_counter_ns() - elapsed
        debug("  [TIME] {}, msg = {}".format(elapsed / 10 ** 9, msg))


DEBUG = True
def debug(msg, *args):
    if DEBUG: print("  [DEBUG]", msg, *args)


class Metropolis_Model_Unlearner:
    """Unlearn a model through the Metropolis algorithm."""
    def __init__(self, model, params_cdf, proposal_fn="normal", delta=0.3):
        # Model info
        self.model = model
        # self._backup_weights = get_numpy_model_params(model, get_only_grads=True)
        self.total_param = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Metropolis info
        self.params_cdf = params_cdf
        if proposal_fn == "normal":
            self.proposal = stats.norm(0, delta)
        elif proposal_fn == "uniform":
            self.proposal = stats.uniform(-delta, 2*delta)
        else:
            raise ValueError("invalid proposal fn '{}'".format(proposal_fn))
        assert self.proposal is not None
        self.steps = 0


    def generate_best_model(self, train_set, forget_set, early_stopping=100, scale=1.0):
        best_score = -1 * 10 ** 10

        # Caching the last valid result.
        x0 = get_numpy_model_params(self.model)
        f_x0 = self.h_function(x0, train_set)
        best_weight = x0

        stop = 0
        while stop <= early_stopping:
            stop += 1
            self.steps += 1

            x_n = x0 + self.proposal.rvs(self.total_param)
            f_xn = self.h_function(x_n, train_set)

            alpha = (f_xn * scale - f_x0 * scale).exp()
            u = np.random.uniform(0, 1)
            if u <= alpha: # Accept
                debug("ACCEPTING with alpha {} and u {}".format(alpha.item(), u))
                x0 = x_n
                f_x0 = f_xn
                # Find if this is a suitable value for the erase set.
                with torch.no_grad():
                    e_prob = 0.0
                    for inputs, labels in forget_set:
                        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                        e_prob += self.model(inputs).softmax(1).gather(
                            1, labels.unsqueeze(1)
                        ).log().sum()
                    g = f_xn - e_prob
                if g > best_score:
                    debug("new best score", g.item())
                    best_score = g
                    best_weight = x_n
                    stop = 0
            else:
                debug("REJECTING with alpha {} and u {}".format(alpha.item(), u))

        return best_weight, best_score


    def h_function(self, theta, dataloader):
        """Computes the formula 'log(P(D | Theta)) + log(P(Theta))'

        Logs are used to avoid numeric cancellation.
        """
        load_params_in_model(self.model, theta)
        with torch.no_grad():
            p_dataset = 0.0
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                p_dataset += self.model(inputs).softmax(1).gather(1, labels.unsqueeze(1)).log().sum()
            p_weights = np.log(self.params_cdf(theta)).sum()
            return p_dataset + p_weights
