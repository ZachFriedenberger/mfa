import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import cross_val_score

import torch
from torch import optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import copy


########################### Two-step FA model functions ###########################

def fit_fa_model(transformed_data, max_nb_latents, cv=5, force_nb_latents=None, refit_model=True):
    """
    Fit a factor analysis (FA) model to the transformed data, optionally using cross-validation to select the number of latent dimensions.
    
    :param transformed_data: Numpy array of shape (nb_samples, nb_neurons) containing the transformed data.
    :param max_nb_latents: Maximum number of latent dimensions to consider during cross-validation.
    :param cv: Number of cross-validation folds to use when selecting the number of latent dimensions.
    :param force_nb_latents: If specified, forces the FA model to use this number of latent dimensions, bypassing cross-validation.
    :param refit_model: Boolean indicating whether to refit the FA model with the selected number of latent dimensions.
    :return: Tuple containing the estimated factor loading matrix (W_est), the estimated latent variables (z_est), and the best number of latent dimensions (best_nb_latent). If refit_model is False, W_est and z_est will be None.
    """

    nb_latents = np.arange(1, max_nb_latents)

    # Estimate the dimensionality of the latent space using cross-validation
    mean_scores = []
    if force_nb_latents is None:
        for nb_latent in nb_latents:
            fa_model = FactorAnalysis(n_components=nb_latent)
            scores = cross_val_score(fa_model, transformed_data, cv=cv)
            mean_scores.append(np.mean(scores))

    if force_nb_latents is not None:
        best_nb_latent = force_nb_latents
    else:
        best_nb_latent = nb_latents[np.argmax(mean_scores)]

    # Refit the FA model with the best number of latent dimensions
    if refit_model:
        fa_model = FactorAnalysis(n_components=best_nb_latent)
        fa_model.fit(transformed_data)
        W_est = fa_model.components_.T
        z_est = fa_model.transform(transformed_data)
    else:
        W_est = None
        z_est = None

    return W_est, z_est, best_nb_latent

def fit_fa_model_events_and_bursts(data, nb_latents):

    """
    Fit factor analysis (FA) models separately to event counts and burst fractions.
    
    :param data: Dictionary containing the data, with keys "events" and "bursts".
    :param nb_latents: Dictionary specifying the number of latent variables for each component, e.g., {"events": nb_latents_events, "bursts": nb_latents_bursts}.
    :return: Tuple containing the estimated factor loading matrices for events (W_s_est) and bursts (W_d_est).
    """

    events = data["events"].cpu().numpy()
    bursts = data["bursts"].cpu().numpy()

    nb_latents_events = nb_latents["events"]
    nb_latents_bursts = nb_latents["bursts"]

    # FA on event counts with sqrt transform
    transformed_events = np.sqrt(events)

    fa_events = FactorAnalysis(n_components=nb_latents_events, max_iter=10000, tol=1e-3)
    fa_events.fit(transformed_events)
    W_s_est = fa_events.components_.T

    # FA on burst fraction with arcsin-sqrt transform
    burst_fraction = np.nan_to_num(bursts/events, nan=0.0)
    transformed_burst_fraction = np.arcsin(np.sqrt(burst_fraction))

    fa_bursts = FactorAnalysis(n_components=nb_latents_bursts, max_iter=10000, tol=1e-3)
    fa_bursts.fit(transformed_burst_fraction)
    W_d_est = fa_bursts.components_.T

    
    return W_s_est, W_d_est

################################# Variational model classes and optimization functions ###########################

class NullDecoder(nn.Module):
    """
    Decoder for the null model. Linear function of the centered log rates.
    """

    def __init__(self, nb_neurons: int, mu_init: torch.Tensor, w_init: torch.Tensor = None, w_init_scale: float = 1e-3):
        super().__init__()

        # Low variance random weight initialization if not provided
        if w_init is None:
            w_init = torch.randn(nb_neurons) * w_init_scale

        self.w = nn.Parameter(w_init.clone(), requires_grad=True)

        # Verify that self.w has shape (nb_neurons,)
        assert self.w.shape == (nb_neurons,), f"Expected self.w to have shape ({nb_neurons},), but got {self.w.shape}"

        # Baseline initialization should be given my the mean of the data
        self.mu = nn.Parameter(mu_init.clone(), requires_grad=True)

    def forward(self, log_rate: torch.Tensor):
        log_rate_mean = log_rate.mean(dim=1, keepdim=True)  # average over data samples only
        log_rate_centered = log_rate - log_rate_mean
        return torch.einsum('n,nts->nts', self.w, log_rate_centered) + self.mu[:, None, None]

class LinearDecoder(nn.Module):
    """
    Linear decoder of the latent variables.
    """

    def __init__(self, nb_neurons: int, nb_latents: int, mu_init: torch.Tensor, W_init: torch.Tensor = None, W_init_scale: float = 1e-3):
        super().__init__()

        # Low variance random weight initialization if not provided
        if W_init is None:
            W_init = torch.randn(nb_neurons, nb_latents) * W_init_scale

        self.W = nn.Parameter(W_init.clone(), requires_grad=True)

        # Baseline initialization should be given my the mean of the data
        self.mu = nn.Parameter(mu_init.clone(), requires_grad=True)

    def forward(self, z: torch.Tensor):
        return torch.einsum('nk,stk->nts', self.W, z) + self.mu[:, None, None]

class HybridDecoder(nn.Module):
    """
    Hybrid decoder combining linear latent variable decoding and null model decoding.
    """

    def __init__(self, nb_neurons: int, nb_latents: int, mu_init: torch.Tensor, W_init: torch.Tensor = None,
                 w_init: torch.Tensor = None, W_init_scale: float = 1e-3, w_init_scale: float = 1e-3):
        super().__init__()

        # Low variance random weight initialization if not provided
        if W_init is None:
            W_init = torch.randn(nb_neurons, nb_latents) * W_init_scale
        if w_init is None:
            w_init = torch.randn(nb_neurons) * w_init_scale

        self.W = nn.Parameter(W_init.clone(), requires_grad=True)
        self.w = nn.Parameter(w_init.clone(), requires_grad=True)

        # Baseline initialization should be given my the mean of the data
        self.mu = nn.Parameter(mu_init.clone(), requires_grad=True)

    def forward(self, z: torch.Tensor, log_rate: torch.Tensor):
        log_rate_mean = log_rate.mean(dim=1, keepdim=True)  # average over data samples only
        log_rate_centered = log_rate - log_rate_mean
        return torch.einsum('nk,stk->nts', self.W, z) + torch.einsum('n,nts->nts', self.w, log_rate_centered) + self.mu[:, None, None]
    
class EventLikelihood(nn.Module):

    def __init__(self, max_log_rate: float = 20.0):
        super().__init__()

        self.max_log_rate = max_log_rate
    
    def log_prob(self, e: torch.Tensor, log_rate:torch.Tensor):

        log_rate = log_rate.clamp(max=self.max_log_rate, min=-self.max_log_rate)

        return torch.distributions.Poisson(torch.exp(log_rate)).log_prob(e.T[..., None]) # (nb_neurons, nb_samples, nb_posterior_samples)
    
class BurstLikelihood(nn.Module):

    def __init__(self):
        super().__init__()

    def log_prob(self, b: torch.Tensor, e: torch.Tensor, logit_prob: torch.Tensor):
        return torch.distributions.Binomial(e.T[..., None], torch.sigmoid(logit_prob)).log_prob(b.T[..., None]) # (nb_neurons, nb_samples, nb_posterior_samples)
    

def kl_diag_standard_normal(mu: torch.Tensor, log_var: torch.Tensor):
    return 0.5 * torch.sum(torch.exp(log_var) + mu**2 - 1.0 - log_var)

class VariationalTable(nn.Module):
    def __init__(self, nb_samples: int, nb_latents: int, init_logvar: float = -2.0):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(nb_samples, nb_latents))
        self.log_var = nn.Parameter(torch.full((nb_samples, nb_latents), float(init_logvar)))

    def forward(self, idx: torch.Tensor):
        return self.mu[idx], self.log_var[idx]
    
    def dist(self, idx: torch.Tensor):
        mu, log_var = self.mu[idx], self.log_var[idx]
        base = torch.distributions.Normal(loc=mu, scale=torch.exp(0.5 * log_var))
        return torch.distributions.Independent(base, 1)
    
def exponential_decay_lambda(step, *, total_steps, final_ratio=0.05):
    gamma = final_ratio ** (1.0 / total_steps)
    return gamma ** step

def update_stats_hist(stats_hist, stats):
    stats_hist["elbo_full"].append(stats["elbo_full"])
    stats_hist["loss"].append(stats["loss"])
    stats_hist["log_likelihood_scaled"].append(stats["log_likelihood_scaled"])
    stats_hist["kl_divergence_scaled"].append(stats["kl_divergence_scaled"])
    return stats_hist

def optimize(model, loss_function, data, device, nb_iter=10_000, nb_posterior_samples=1, lr=1e-4, batch_size=None, lr_lambda=None, print_loss=False):
    """
    Optimize the model using the specified loss function and data.
    Supports full batch optimization and learning rate scheduling.
    
    :param model: The model to be optimized.
    :param loss_function: The loss function used for optimization.
    :param data: A dictionary containing the training data with keys "events" and "bursts".
    :param device: The device (CPU or GPU) on which to perform the optimization.
    :param nb_iter: The number of optimization iterations.
    :param nb_posterior_samples: The number of posterior samples to use in the loss computation.
    :param lr: The initial learning rate for the optimizer.
    :param batch_size: The batch size for mini-batch optimization. If None, full batch optimization is used.
    :param lr_lambda: A function that computes a multiplicative factor for the learning rate at each epoch.
    :param print_loss: If True, prints the loss every 100 iterations.
    """


    events = data["events"]
    bursts = data["bursts"]
    nb_samples = events.shape[0]

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.95, 0.999))

    # If no lr decay function, keep lr constant.
    if lr_lambda is None:
        lr_lambda = lambda epoch : 1

    # Define the learning rate scheduler
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if batch_size is None:
        batch_size = nb_samples

    if batch_size == nb_samples:

        events_dev, bursts_dev = events.to(device), bursts.to(device)
        idx = torch.arange(nb_samples).to(device)

        for i in range(nb_iter):

            optimizer.zero_grad()

            loss, stats = loss_function(model, events_dev, bursts_dev, idx, nb_posterior_samples=nb_posterior_samples)

            loss.backward()
            optimizer.step()
            scheduler.step()

            # Print loss 
            if i % 100 == 0 and print_loss:
                print(f"Iteration: {i} Loss: {loss.item():.3f}, Current LR: {scheduler.get_last_lr()}")

            # Initialize dictionary of training stats history
            if i == 0: 
                stats_hist = {}
                for key in stats.keys():
                    stats_hist[key] = []

            # Update taining stats history
            stats_hist = update_stats_hist(stats_hist, stats)

    else:
        raise NotImplementedError("Mini-batch optimization was removed. Use batch_size=None (full batch).")

    return model, stats_hist

### Loss functions helpers for different model configurations
def event_only_loss_function(model, events, bursts, idx, nb_posterior_samples):
    return model.loss(events, idx, nb_posterior_samples=nb_posterior_samples)

def full_loss_function(model, events, bursts, idx, nb_posterior_samples):
    return model.loss(events, bursts, idx, nb_posterior_samples=nb_posterior_samples)

def estimate_init_params(data, nb_latents, eps=1e-6, model_type='base'):
    """
    Estimate initial parameters for the multiplexed factor analysis model using method of moments and factor analysis.
    
    :param data: Dictionary containing the data, with keys "events" and "bursts".
    :param nb_latents: Dictionary specifying the number of latent variables for each component, e.g., {"events": nb_latents_events, "bursts": nb_latents_bursts}.
    :param eps: Small constant to avoid numerical issues when computing log and logit.
    :param model_type: String indicating the type of model for which to estimate initial parameters. Can be 'base', 'null', or 'hybrid'.
    :return: Dictionary containing the initialized parameters for the model. Keys include "W_event", "W_burst", "mu_event", and "mu_burst". If null_model is True, "W_burst" is set to None.
    """

    events = data["events"]
    bursts = data["bursts"]

    # Use method of moments to get initial estimates for baselines
    mean_event_counts = events.mean(dim=0)
    mean_burst_counts = bursts.mean(dim=0)
    burst_fraction = torch.nan_to_num(mean_burst_counts/mean_event_counts, nan=0.0)

    mu_init_event = torch.log(mean_event_counts + eps)
    mu_init_burst = torch.logit(burst_fraction, eps)

    # Fit using FA for intialization
    W_event_learned_FA, W_burst_learned_FA = fit_fa_model_events_and_bursts(data, nb_latents)

    # Initialize the multiplexed model with FA weights and baselines from the training data
    init_params = {}
    init_params["W_event"] = torch.tensor(W_event_learned_FA, dtype=torch.float32)
    init_params["mu_event"] = mu_init_event
    init_params["mu_burst"] = mu_init_burst

    if model_type == 'null':
        init_params["w_burst"] = None
    if model_type == 'hybrid':
        init_params["w_burst"] = None
        init_params["W_burst"] = torch.tensor(W_burst_learned_FA, dtype=torch.float32)
    if model_type == 'base':
        init_params["W_burst"] = torch.tensor(W_burst_learned_FA, dtype=torch.float32)


    return init_params

# Model definitions for event and burst models
class EventModel(nn.Module):
    """
    Event model for count data.
    """

    def __init__(self, nb_samples: int, nb_neurons: int, nb_latents_events: int,
                 init_params: dict):
        super().__init__()

        self.nb_neurons = nb_neurons
        self.nb_latents_events = nb_latents_events
        self.nb_samples = nb_samples
     
        # Define the encoder
        self.event_encoder = VariationalTable(nb_samples=nb_samples, nb_latents=nb_latents_events)

        # Define the decoder 
        self.event_decoder = LinearDecoder(nb_neurons=nb_neurons, nb_latents=nb_latents_events, 
                                           mu_init=init_params["mu_event"], W_init=init_params["W_event"])

        # Likelihood
        self.event_likelihood = EventLikelihood()


    def load_decoder_parameters(self, fitted_model: nn.Module, freeze: bool = True):
        """
        Copy decoder parameters from a fitted model and optionally freeze them.
        Assumes fitted_model has:
            - event_decoder
            - burst_decoder
        """

        self.event_decoder.load_state_dict(copy.deepcopy(fitted_model.event_decoder.state_dict()))

        if freeze:
           self.freeze_decoders()

    def freeze_decoders(self):
        for p in self.event_decoder.parameters():
            p.requires_grad = False
    
    def unfreeze_decoders(self):
        for p in self.event_decoder.parameters():
            p.requires_grad = True

    def posterior_dist(self, idx: torch.Tensor):
        q_event = self.event_encoder.dist(idx)
        return q_event
       
    def sample_posterior(self, idx: torch.Tensor, nb_posterior_samples: int = 1):
        q_event = self.posterior_dist(idx)
        z_event_samples = q_event.rsample((nb_posterior_samples,))  # (nb_posterior_samples, nb_samples, nb_latents)
        return z_event_samples
    
    def sample_posterior_predictive(self, idx: torch.Tensor, nb_posterior_samples: int = 1):
        z_event_samples = self.sample_posterior(idx, nb_posterior_samples)
        log_event_rate = self.event_decoder(z_event_samples)
        new_events = torch.distributions.Poisson(torch.exp(log_event_rate)).sample()  # (nb_posterior_samples, nb_samples, nb_neurons)
        return new_events
    
    def analytic_kl_divergence(self, idx: torch.Tensor):
        mu_event, log_var_event = self.event_encoder(idx)
        kl_event = kl_diag_standard_normal(mu_event, log_var_event)
        return kl_event

    def elbo_terms(self, events: torch.Tensor,  idx: torch.Tensor, nb_posterior_samples: int = 1):
        """
        Returns dict with:
          - elbo
          - loglike (scaled)
          - kl_z (scaled, with beta applied)
          - theta_log_prior (unscaled)
        """

        # Scale factor for minibatching over datapoints 
        scale = float(self.nb_samples) / float(events.shape[0])

        # Sample from the approximate posteriors
        z_event_samples = self.sample_posterior(idx, nb_posterior_samples)

        # Expected log-likelihood via MC
        log_event_rate = self.event_decoder(z_event_samples)
        log_likelihood = self.event_likelihood.log_prob(events, log_event_rate).sum(dim=(0, 1)).mean()

        # KL divergences
        kl_divergence = self.analytic_kl_divergence(idx)

        # Scaled ELBO
        elbo = scale * (log_likelihood - kl_divergence)
        elbo_full = scale * (log_likelihood - kl_divergence) 

        return {
            "elbo": elbo,
            "elbo_full": elbo_full,
            "log_likelihood_scaled": scale * log_likelihood,
            "kl_divergence_scaled": scale * kl_divergence,
        }

    def loss(self, events: torch.Tensor, idx: torch.Tensor, nb_posterior_samples: int = 1):
        """
        Returns (loss_to_minimize, stats_dict)
        """
        terms = self.elbo_terms(events=events, idx=idx, nb_posterior_samples=nb_posterior_samples)
        loss = -1*terms["elbo"]

        stats = {"loss": loss.detach().item(), "log_likelihood_scaled": terms["log_likelihood_scaled"].detach().item(),
            "kl_divergence_scaled": terms["kl_divergence_scaled"].detach().item(), "elbo_full": terms["elbo_full"].detach().item()}
        
        return loss, stats
    
    def log_evidence(self, events: torch.Tensor, nb_posterior_samples: int = 1000):
        """
        Estimate log evidence via importance sampling with the approximate posterior as the proposal distribution.
        """

        idx = torch.arange(events.shape[0])
        nb_latents_events = self.nb_latents_events
        device = events.device

        with torch.no_grad():

            # Sample from the approximate posteriors
            z_event_samples = self.sample_posterior(idx, nb_posterior_samples)

            # Evaluate log-likelihood for each sample
            log_event_rate = self.event_decoder(z_event_samples)
            log_likelihood = self.event_likelihood.log_prob(events, log_event_rate).sum(dim=0) # (nb_samples, nb_posterior_samples)

            # Evaluate the log prior for each sample
            event_prior_dist_base = torch.distributions.Normal(loc=torch.zeros(nb_latents_events, device=device), scale=torch.ones(nb_latents_events, device=device))
            event_prior_dist = torch.distributions.Independent(event_prior_dist_base, 1)
            log_prior = event_prior_dist.log_prob(z_event_samples) # (nb_posterior_samples, nb_samples)

            # Evaluate the log posterior for each sample
            q_event = self.posterior_dist(idx)
            log_posterior = q_event.log_prob(z_event_samples) # (nb_posterior_samples, nb_samples)

            # Compute importance weights and estimate log evidence
            log_importance_weights = log_likelihood.T + log_prior - log_posterior  # (nb_posterior_samples, nb_samples)
            log_evidence_score = (torch.logsumexp(log_importance_weights, dim=0) - torch.log(torch.tensor(nb_posterior_samples)))  # per sample log evidence

            return log_evidence_score
        
class BurstMultiplexedModel(nn.Module):
    """
    Base MFA model.
    """
    def __init__(self, nb_samples: int, nb_neurons: int, nb_latents_events: int, nb_latents_bursts: int, 
                 init_params: dict):
        super().__init__()

        self.nb_neurons = nb_neurons
        self.nb_latents_events = nb_latents_events
        self.nb_latents_bursts = nb_latents_bursts
        self.nb_samples = nb_samples
     
        # Define the encoders
        self.event_encoder = VariationalTable(nb_samples=nb_samples, nb_latents=nb_latents_events)
        self.burst_encoder = VariationalTable(nb_samples=nb_samples, nb_latents=nb_latents_bursts)

        # Decoders produce logits/etas
        self.event_decoder = LinearDecoder(nb_neurons=nb_neurons, nb_latents=nb_latents_events, mu_init=init_params["mu_event"], W_init=init_params["W_event"])
        self.burst_decoder = LinearDecoder(nb_neurons=nb_neurons, nb_latents=nb_latents_bursts, mu_init=init_params["mu_burst"], W_init=init_params["W_burst"])

        # Likelihoods
        self.event_likelihood = EventLikelihood()
        self.burst_likelihood = BurstLikelihood()

    def load_decoder_parameters(self, fitted_model: nn.Module, freeze: bool = True):
        """
        Copy decoder parameters from a fitted model and optionally freeze them.
        Assumes fitted_model has:
            - event_decoder
            - burst_decoder
        """

        self.event_decoder.load_state_dict(copy.deepcopy(fitted_model.event_decoder.state_dict()))
        self.burst_decoder.load_state_dict(copy.deepcopy(fitted_model.burst_decoder.state_dict()))

        if freeze:
           self.freeze_decoders()

    def freeze_decoders(self):
        for p in self.event_decoder.parameters():
            p.requires_grad = False
        for p in self.burst_decoder.parameters():
            p.requires_grad = False
    
    def unfreeze_decoders(self):
        for p in self.event_decoder.parameters():
            p.requires_grad = True
        for p in self.burst_decoder.parameters():
            p.requires_grad = True

    def load_event_parameters(self, fitted_event_model: nn.Module, freeze: bool = True):
        """
        Copy event-side parameters from a fitted model and optionally freeze them.
        Assumes fitted_event_model has:
            - event_encoder
            - event_decoder
        """

        self.event_encoder.load_state_dict(copy.deepcopy(fitted_event_model.event_encoder.state_dict()))
        self.event_decoder.load_state_dict(copy.deepcopy(fitted_event_model.event_decoder.state_dict()))

        if freeze:
            self.freeze_event_parameters()

    def freeze_event_parameters(self):
        for p in self.event_encoder.parameters():
            p.requires_grad = False
        for p in self.event_decoder.parameters():
            p.requires_grad = False

    def unfreeze_event_parameters(self):
        for p in self.event_encoder.parameters():
            p.requires_grad = True
        for p in self.event_decoder.parameters():
            p.requires_grad = True

    def posterior_dist(self, idx: torch.Tensor):
        
        q_event = self.event_encoder.dist(idx)
        q_burst = self.burst_encoder.dist(idx)

        return q_event, q_burst
       
    def sample_posterior(self, idx: torch.Tensor, nb_posterior_samples: int = 1):

        q_event, q_burst = self.posterior_dist(idx)

        z_event_samples = q_event.rsample((nb_posterior_samples,))  # (nb_posterior_samples, nb_samples, nb_latents)
        z_burst_samples = q_burst.rsample((nb_posterior_samples,))  # (nb_posterior_samples, nb_samples, nb_latents)

        return z_event_samples, z_burst_samples
    
    def sample_posterior_predictive(self, idx: torch.Tensor, nb_posterior_samples: int = 1):

        z_event_samples, z_burst_samples = self.sample_posterior(idx, nb_posterior_samples)

        log_event_rate = self.event_decoder(z_event_samples)
        logit_burst_prob = self.burst_decoder(z_burst_samples)

        new_events = torch.distributions.Poisson(torch.exp(log_event_rate)).sample()  # (nb_posterior_samples, nb_samples, nb_neurons)
        new_bursts = torch.distributions.Binomial(new_events, torch.sigmoid(logit_burst_prob)).sample()  # (nb_posterior_samples, nb_samples, nb_neurons)

        return new_events, new_bursts
    
    def analytic_kl_divergence(self, idx: torch.Tensor):

        mu_event, log_var_event = self.event_encoder(idx)
        mu_burst, log_var_burst = self.burst_encoder(idx)

        kl_event = kl_diag_standard_normal(mu_event, log_var_event)
        kl_burst = kl_diag_standard_normal(mu_burst, log_var_burst)
        kl_divergence = kl_event + kl_burst

        return kl_divergence


    def elbo_terms(self, events: torch.Tensor, bursts: torch.Tensor, idx: torch.Tensor, nb_posterior_samples: int = 1):
        """
        Returns dict with:
          - elbo
          - loglike (scaled)
          - kl_z (scaled, with beta applied)
          - theta_log_prior (unscaled)
        """

        # Scale factor for minibatching over datapoints 
        scale = float(self.nb_samples) / float(events.shape[0])

        # Sample from the approximate posteriors
        z_event_samples, z_burst_samples = self.sample_posterior(idx, nb_posterior_samples)

        # Expected log-likelihood via MC
        log_event_rate = self.event_decoder(z_event_samples)
        logit_burst_prob = self.burst_decoder(z_burst_samples)

        event_log_likelihood = self.event_likelihood.log_prob(events, log_event_rate).sum(dim=(0, 1)).mean()
        burst_log_likelihood = self.burst_likelihood.log_prob(bursts, events, logit_burst_prob).sum(dim=(0, 1)).mean()
        log_likelihood = event_log_likelihood + burst_log_likelihood

        # KL divergences
        kl_divergence = self.analytic_kl_divergence(idx)

        # Scaled ELBO
        elbo = scale * (log_likelihood - kl_divergence)
        elbo_full = scale * (log_likelihood - kl_divergence) 

        return {
            "elbo": elbo,
            "elbo_full": elbo_full,
            "log_likelihood_scaled": scale * log_likelihood,
            "kl_divergence_scaled": scale * kl_divergence,
        }

    def loss(self, events: torch.Tensor, bursts: torch.Tensor, idx: torch.Tensor, nb_posterior_samples: int = 1):
        """
        Returns (loss_to_minimize, stats_dict)
        """
        terms = self.elbo_terms(events=events, bursts=bursts, idx=idx, nb_posterior_samples=nb_posterior_samples)
        loss = -1*terms["elbo"]

        stats = {"loss": loss.detach().item(), "log_likelihood_scaled": terms["log_likelihood_scaled"].detach().item(),
            "kl_divergence_scaled": terms["kl_divergence_scaled"].detach().item(), "elbo_full": terms["elbo_full"].detach().item()}
        
        return loss, stats
    
    def log_evidence(self, events: torch.Tensor, bursts: torch.Tensor, nb_posterior_samples: int = 1000):
        """
        Estimate log evidence via importance sampling with the approximate posterior as the proposal distribution.
        """

        idx = torch.arange(events.shape[0])
        nb_latents_events = self.nb_latents_events
        nb_latents_bursts = self.nb_latents_bursts
        device = events.device

        with torch.no_grad():

            # Sample from the approximate posteriors
            z_event_samples, z_burst_samples = self.sample_posterior(idx, nb_posterior_samples)

            # Evaluate log-likelihood for each sample
            log_event_rate = self.event_decoder(z_event_samples)
            logit_burst_prob = self.burst_decoder(z_burst_samples)

            event_log_likelihood = self.event_likelihood.log_prob(events, log_event_rate).sum(dim=0)
            burst_log_likelihood = self.burst_likelihood.log_prob(bursts, events, logit_burst_prob).sum(dim=0)

            # Combine log-likelihoods
            log_likelihood = event_log_likelihood + burst_log_likelihood # (nb_samples, nb_posterior_samples)

            # Evaluate the log prior for each sample
            event_prior_dist_base = torch.distributions.Normal(loc=torch.zeros(nb_latents_events, device=device), scale=torch.ones(nb_latents_events, device=device))
            event_prior_dist = torch.distributions.Independent(event_prior_dist_base, 1)
            event_log_prior = event_prior_dist.log_prob(z_event_samples) # (nb_posterior_samples, nb_samples)

            burst_prior_dist_base = torch.distributions.Normal(loc=torch.zeros(nb_latents_bursts, device=device), scale=torch.ones(nb_latents_bursts, device=device))
            burst_prior_dist = torch.distributions.Independent(burst_prior_dist_base, 1)
            burst_log_prior = burst_prior_dist.log_prob(z_burst_samples) # (nb_posterior_samples, nb_samples)

            # Combine log priors
            log_prior = event_log_prior + burst_log_prior

            # Evaluate the log posterior for each sample
            q_event, q_burst = self.posterior_dist(idx)
            q_event_log_prob = q_event.log_prob(z_event_samples) # (nb_posterior_samples, nb_samples)
            q_burst_log_prob = q_burst.log_prob(z_burst_samples) # (nb_posterior_samples, nb_samples)

            # Combine log posteriors
            log_posterior = q_event_log_prob + q_burst_log_prob

            # Compute importance weights and estimate log evidence
            log_importance_weights = log_likelihood.T + log_prior - log_posterior  # (nb_posterior_samples, nb_samples)
            log_evidence_score = (torch.logsumexp(log_importance_weights, dim=0) - torch.log(torch.tensor(nb_posterior_samples)))  # per sample log evidence

            return log_evidence_score
        
class BurstNullModel(nn.Module):
    """
    Null model for burst activity, with event activity modeled separately.
    """
    
    def __init__(self, nb_samples: int, nb_neurons: int, nb_latents_events: int,  init_params: dict):
        super().__init__()

        self.nb_neurons = nb_neurons
        self.nb_latents_events = nb_latents_events
        self.nb_samples = nb_samples

        # Define the encoders
        self.event_encoder = VariationalTable(nb_samples=nb_samples, nb_latents=nb_latents_events)

        # Decoders produce logits/etas
        self.event_decoder = LinearDecoder(nb_neurons=nb_neurons, nb_latents=nb_latents_events, mu_init=init_params["mu_event"], W_init=init_params["W_event"])
        self.burst_decoder = NullDecoder(nb_neurons=nb_neurons, mu_init=init_params["mu_burst"], w_init=init_params["w_burst"])

        # Likelihoods
        self.event_likelihood = EventLikelihood()
        self.burst_likelihood = BurstLikelihood()


    def load_decoder_parameters(self, fitted_model: nn.Module, freeze: bool = True):
        """
        Copy decoder parameters from a fitted model and optionally freeze them.
        Assumes fitted_model has:
            - event_decoder
            - burst_decoder
        """

        self.event_decoder.load_state_dict(copy.deepcopy(fitted_model.event_decoder.state_dict()))
        self.burst_decoder.load_state_dict(copy.deepcopy(fitted_model.burst_decoder.state_dict()))

        if freeze:
           self.freeze_decoders()

    def freeze_decoders(self):
        for p in self.event_decoder.parameters():
            p.requires_grad = False
        for p in self.burst_decoder.parameters():
            p.requires_grad = False
    
    def unfreeze_decoders(self):
        for p in self.event_decoder.parameters():
            p.requires_grad = True
        for p in self.burst_decoder.parameters():
            p.requires_grad = True

    def load_event_parameters(self, fitted_event_model: nn.Module, freeze: bool = True):
        """
        Copy event-side parameters from a fitted model and optionally freeze them.
        Assumes fitted_event_model has:
            - event_encoder
            - event_decoder
        """

        self.event_encoder.load_state_dict(copy.deepcopy(fitted_event_model.event_encoder.state_dict()))
        self.event_decoder.load_state_dict(copy.deepcopy(fitted_event_model.event_decoder.state_dict()))

        if freeze:
            self.freeze_event_parameters()

    def freeze_event_parameters(self):
        for p in self.event_encoder.parameters():
            p.requires_grad = False
        for p in self.event_decoder.parameters():
            p.requires_grad = False

    def unfreeze_event_parameters(self):
        for p in self.event_encoder.parameters():
            p.requires_grad = True
        for p in self.event_decoder.parameters():
            p.requires_grad = True

    def posterior_dist(self, idx: torch.Tensor):
        q_event = self.event_encoder.dist(idx)
        return q_event
       
    def sample_posterior(self, idx: torch.Tensor, nb_posterior_samples: int = 1):
        q_event = self.posterior_dist(idx)
        z_event_samples = q_event.rsample((nb_posterior_samples,))  # (nb_posterior_samples, nb_samples, nb_latents)
        return z_event_samples
    
    def sample_posterior_predictive(self, idx: torch.Tensor, nb_posterior_samples: int = 1):

        z_event_samples = self.sample_posterior(idx, nb_posterior_samples)

        log_event_rate = self.event_decoder(z_event_samples)
        event_rate = torch.exp(log_event_rate)
        logit_burst_prob = self.burst_decoder(log_event_rate)

        new_events = torch.distributions.Poisson(event_rate).sample()  # (nb_posterior_samples, nb_samples, nb_neurons)
        new_bursts = torch.distributions.Binomial(new_events, torch.sigmoid(logit_burst_prob)).sample()  # (nb_posterior_samples, nb_samples, nb_neurons)

        return new_events, new_bursts
    
    def analytic_kl_divergence(self, idx: torch.Tensor):

        mu_event, log_var_event = self.event_encoder(idx)
    
        kl_event = kl_diag_standard_normal(mu_event, log_var_event)
        kl_divergence = kl_event 

        return kl_divergence


    def elbo_terms(self, events: torch.Tensor, bursts: torch.Tensor, idx: torch.Tensor, beta: float = 1.0, nb_posterior_samples: int = 1):
        """
        Returns dict with:
          - elbo
          - loglike (scaled)
          - kl_z (scaled, with beta applied)
          - theta_log_prior (unscaled)
        """

        # Scale factor for minibatching over datapoints 
        scale = float(self.nb_samples) / float(events.shape[0])

        # Sample from the approximate posteriors
        z_event_samples = self.sample_posterior(idx, nb_posterior_samples)

        # Expected log-likelihood via MC
        log_event_rate = self.event_decoder(z_event_samples)
        #event_rate = torch.exp(log_event_rate)
        logit_burst_prob = self.burst_decoder(log_event_rate.detach()) # detach to prevent gradients flowing into burst decoder from event latents

        event_log_likelihood = self.event_likelihood.log_prob(events, log_event_rate).sum(dim=(0, 1)).mean()
        burst_log_likelihood = self.burst_likelihood.log_prob(bursts, events, logit_burst_prob).sum(dim=(0, 1)).mean()
        log_likelihood = event_log_likelihood + burst_log_likelihood

        # KL divergences
        kl_divergence = self.analytic_kl_divergence(idx)

        # Scaled ELBO
        elbo = scale * (log_likelihood - beta*kl_divergence)

        elbo_full = scale * (log_likelihood - kl_divergence) 

        return {
            "elbo": elbo,
            "elbo_full": elbo_full,
            "log_likelihood_scaled": scale * log_likelihood,
            "kl_divergence_scaled": scale * (float(beta) * kl_divergence),
        }

    def loss(self, events: torch.Tensor, bursts: torch.Tensor, idx: torch.Tensor, beta: float = 1.0, nb_posterior_samples: int = 1):
        """
        Returns (loss_to_minimize, stats_dict)
        """
        terms = self.elbo_terms(events=events, bursts=bursts, idx=idx, beta=beta, nb_posterior_samples=nb_posterior_samples)
        loss = -1*terms["elbo"]

        stats = {"loss": loss.detach().item(), "log_likelihood_scaled": terms["log_likelihood_scaled"].detach().item(),
            "kl_divergence_scaled": terms["kl_divergence_scaled"].detach().item(), "elbo_full": terms["elbo_full"].detach().item()}
        
        return loss, stats
    
    def log_evidence(self, events: torch.Tensor, bursts: torch.Tensor, nb_posterior_samples: int = 1000):
        """
        Estimate log evidence via importance sampling with the approximate posterior as the proposal distribution.
        """

        idx = torch.arange(events.shape[0])
        nb_latents_events = self.nb_latents_events

        device = events.device

        with torch.no_grad():

            # Sample from the approximate posteriors
            z_event_samples = self.sample_posterior(idx, nb_posterior_samples)

            # Evaluate log-likelihood for each sample
            log_event_rate = self.event_decoder(z_event_samples)
            #event_rate = torch.exp(log_event_rate)
            logit_burst_prob = self.burst_decoder(log_event_rate)

            event_log_likelihood = self.event_likelihood.log_prob(events, log_event_rate).sum(dim=0)
            burst_log_likelihood = self.burst_likelihood.log_prob(bursts, events, logit_burst_prob).sum(dim=0)

            # Combine log-likelihoods
            log_likelihood = event_log_likelihood + burst_log_likelihood # (nb_samples, nb_posterior_samples)

            # Evaluate the log prior for each sample
            event_prior_dist_base = torch.distributions.Normal(loc=torch.zeros(self.nb_latents_events, device=device), scale=torch.ones(self.nb_latents_events, device=device))
            event_prior_dist = torch.distributions.Independent(event_prior_dist_base, 1)
            event_log_prior = event_prior_dist.log_prob(z_event_samples) # (nb_posterior_samples, nb_samples)

            # Combine log priors
            log_prior = event_log_prior 

            # Evaluate the log posterior for each sample
            q_event = self.posterior_dist(idx)
            q_event_log_prob = q_event.log_prob(z_event_samples) # (nb_posterior_samples, nb_samples)
      
            # Combine log posteriors
            log_posterior = q_event_log_prob

            # Compute importance weights and estimate log evidence
            log_importance_weights = log_likelihood.T + log_prior - log_posterior  # (nb_posterior_samples, nb_samples)
            log_evidence_score = (torch.logsumexp(log_importance_weights, dim=0) - torch.log(torch.tensor(nb_posterior_samples)))  # per sample log evidence

            return log_evidence_score

class BaselineModel(nn.Module):
    def __init__(self, nb_neurons: int):
        super().__init__()

        self.event_rate = torch.zeros(nb_neurons)
        self.burst_prob = torch.zeros(nb_neurons)

    def maximum_likelihood_estimation(self, events: torch.Tensor, bursts: torch.Tensor):

        self.event_rate = events.mean(dim=0) # Mean along the sample dimension
        self.burst_prob = bursts.mean(dim=0) / (events.mean(dim=0))
        self.burst_prob = torch.nan_to_num(self.burst_prob, nan=0.0, posinf=0.0, neginf=0.0)

    def log_evidence(self, events: torch.Tensor, bursts: torch.Tensor):

        event_log_likelihood = torch.distributions.Poisson(self.event_rate).log_prob(events) # (nb_samples, nb_neurons)
        burst_log_likelihood = torch.distributions.Binomial(events, self.burst_prob).log_prob(bursts) # (nb_samples, nb_neurons)

        log_likelihood = event_log_likelihood + burst_log_likelihood # (nb_samples, nb_neurons)
        return log_likelihood.sum(dim=1) # Sum over all neurons to get total log-likelihood per sample (nb_samples,)

class BurstIndepModel(nn.Module):
    def __init__(self, nb_samples: int, nb_neurons: int, nb_latents_events: int, init_params: dict):
        super().__init__()

        self.nb_neurons = nb_neurons
        self.nb_latents_events = nb_latents_events
        self.nb_samples = nb_samples
     
        # Define the encoders
        self.event_encoder = VariationalTable(nb_samples=nb_samples, nb_latents=nb_latents_events)

        # Decoders produce logits/etas
        self.event_decoder = LinearDecoder(nb_neurons=nb_neurons, nb_latents=nb_latents_events, mu_init=init_params["mu_event"], W_init=init_params["W_event"])
        self.burst_prob = torch.zeros(nb_neurons)

        # Likelihoods
        self.event_likelihood = EventLikelihood()
        self.burst_likelihood = BurstLikelihood()

    def load_decoder_parameters(self, fitted_model: nn.Module, freeze: bool = True):
        """
        Copy decoder parameters from a fitted model and optionally freeze them.
        Assumes fitted_model has:
            - event_decoder
        """

        self.event_decoder.load_state_dict(copy.deepcopy(fitted_model.event_decoder.state_dict()))

        if freeze:
           self.freeze_decoders()

    def freeze_decoders(self):
        for p in self.event_decoder.parameters():
            p.requires_grad = False
    
    def unfreeze_decoders(self):
        for p in self.event_decoder.parameters():
            p.requires_grad = True

    def load_event_parameters(self, fitted_event_model: nn.Module, freeze: bool = True):
        """
        Copy event-side parameters from a fitted model and optionally freeze them.
        Assumes fitted_event_model has:
            - event_encoder
            - event_decoder
        """

        self.event_encoder.load_state_dict(copy.deepcopy(fitted_event_model.event_encoder.state_dict()))
        self.event_decoder.load_state_dict(copy.deepcopy(fitted_event_model.event_decoder.state_dict()))

        if freeze:
            self.freeze_event_parameters()

    def freeze_event_parameters(self):
        for p in self.event_encoder.parameters():
            p.requires_grad = False
        for p in self.event_decoder.parameters():
            p.requires_grad = False

    def unfreeze_event_parameters(self):
        for p in self.event_encoder.parameters():
            p.requires_grad = True
        for p in self.event_decoder.parameters():
            p.requires_grad = True

    def posterior_dist(self, idx: torch.Tensor):
        q_event = self.event_encoder.dist(idx)
        return q_event
    
    def maximum_likelihood_estimation(self, events: torch.Tensor, bursts: torch.Tensor):

        self.burst_prob = bursts.mean(dim=0) / (events.mean(dim=0))
        self.burst_prob = torch.nan_to_num(self.burst_prob, nan=0.0, posinf=0.0, neginf=0.0)
       
    def sample_posterior(self, idx: torch.Tensor, nb_posterior_samples: int = 1):

        q_event = self.posterior_dist(idx)
        z_event_samples = q_event.rsample((nb_posterior_samples,))  # (nb_posterior_samples, nb_samples, nb_latents)

        return z_event_samples
    
    def log_evidence(self, events: torch.Tensor, bursts: torch.Tensor, nb_posterior_samples: int = 1000):
        """
        Estimate log evidence via importance sampling with the approximate posterior as the proposal distribution.
        """

        idx = torch.arange(events.shape[0])
        nb_latents_events = self.nb_latents_events
        device = events.device

        with torch.no_grad():

            # Sample from the approximate posteriors
            z_event_samples = self.sample_posterior(idx, nb_posterior_samples)

            # Evaluate log-likelihood for each sample
            log_event_rate = self.event_decoder(z_event_samples)
            event_log_likelihood = self.event_likelihood.log_prob(events, log_event_rate).sum(dim=0)
  
            # Combine log-likelihoods
            log_likelihood = event_log_likelihood # (nb_samples, nb_posterior_samples)

            # Evaluate the log prior for each sample
            event_prior_dist_base = torch.distributions.Normal(loc=torch.zeros(nb_latents_events, device=device), scale=torch.ones(nb_latents_events, device=device))
            event_prior_dist = torch.distributions.Independent(event_prior_dist_base, 1)
            event_log_prior = event_prior_dist.log_prob(z_event_samples) # (nb_posterior_samples, nb_samples)

            # Combine log priors
            log_prior = event_log_prior

            # Evaluate the log posterior for each sample
            q_event = self.posterior_dist(idx)
            q_event_log_prob = q_event.log_prob(z_event_samples) # (nb_posterior_samples, nb_samples)

            # Combine log posteriors
            log_posterior = q_event_log_prob

            # Compute importance weights and estimate log evidence
            log_importance_weights = log_likelihood.T + log_prior - log_posterior  # (nb_posterior_samples, nb_samples)
            log_evidence_score_event = (torch.logsumexp(log_importance_weights, dim=0) - torch.log(torch.tensor(nb_posterior_samples)))  # per sample log evidence

            # Log evidence from the burst model (which is independent of the latents in this simplified model)
            log_evidence_score_burst = torch.distributions.Binomial(events, self.burst_prob).log_prob(bursts).sum(dim=1) # (nb_samples,)
            log_evidence_score = log_evidence_score_event + log_evidence_score_burst

            return log_evidence_score

class BurstHybridModel(nn.Module):
    """
    Hybrid model for burst activity. Merges the base model and the null model. 
    """
    def __init__(self, nb_samples: int, nb_neurons: int, nb_latents_events: int, nb_latents_bursts: int, 
                 init_params: dict):
        super().__init__()

        self.nb_neurons = nb_neurons
        self.nb_latents_events = nb_latents_events
        self.nb_latents_bursts = nb_latents_bursts
        self.nb_samples = nb_samples
     
        # Define the encoders
        self.event_encoder = VariationalTable(nb_samples=nb_samples, nb_latents=nb_latents_events)
        self.burst_encoder = VariationalTable(nb_samples=nb_samples, nb_latents=nb_latents_bursts)

        # Decoders produce logits/etas
        self.event_decoder = LinearDecoder(nb_neurons=nb_neurons, nb_latents=nb_latents_events, mu_init=init_params["mu_event"], W_init=init_params["W_event"])
        self.burst_decoder = HybridDecoder(nb_neurons=nb_neurons, nb_latents=nb_latents_bursts, mu_init=init_params["mu_burst"], W_init=init_params["W_burst"], w_init=init_params["w_burst"])

        # Likelihoods
        self.event_likelihood = EventLikelihood()
        self.burst_likelihood = BurstLikelihood()

    def load_decoder_parameters(self, fitted_model: nn.Module, freeze: bool = True):
        """
        Copy decoder parameters from a fitted model and optionally freeze them.
        Assumes fitted_model has:
            - event_decoder
            - burst_decoder
        """

        self.event_decoder.load_state_dict(copy.deepcopy(fitted_model.event_decoder.state_dict()))
        self.burst_decoder.load_state_dict(copy.deepcopy(fitted_model.burst_decoder.state_dict()))

        if freeze:
           self.freeze_decoders()

    def freeze_decoders(self):
        for p in self.event_decoder.parameters():
            p.requires_grad = False
        for p in self.burst_decoder.parameters():
            p.requires_grad = False
    
    def unfreeze_decoders(self):
        for p in self.event_decoder.parameters():
            p.requires_grad = True
        for p in self.burst_decoder.parameters():
            p.requires_grad = True

    def load_event_parameters(self, fitted_event_model: nn.Module, freeze: bool = True):
        """
        Copy event-side parameters from a fitted model and optionally freeze them.
        Assumes fitted_event_model has:
            - event_encoder
            - event_decoder
        """

        self.event_encoder.load_state_dict(copy.deepcopy(fitted_event_model.event_encoder.state_dict()))
        self.event_decoder.load_state_dict(copy.deepcopy(fitted_event_model.event_decoder.state_dict()))

        if freeze:
            self.freeze_event_parameters()

    def freeze_event_parameters(self):
        for p in self.event_encoder.parameters():
            p.requires_grad = False
        for p in self.event_decoder.parameters():
            p.requires_grad = False

    def unfreeze_event_parameters(self):
        for p in self.event_encoder.parameters():
            p.requires_grad = True
        for p in self.event_decoder.parameters():
            p.requires_grad = True

    def posterior_dist(self, idx: torch.Tensor):
        
        q_event = self.event_encoder.dist(idx)
        q_burst = self.burst_encoder.dist(idx)

        return q_event, q_burst
       
    def sample_posterior(self, idx: torch.Tensor, nb_posterior_samples: int = 1):

        q_event, q_burst = self.posterior_dist(idx)

        z_event_samples = q_event.rsample((nb_posterior_samples,))  # (nb_posterior_samples, nb_samples, nb_latents)
        z_burst_samples = q_burst.rsample((nb_posterior_samples,))  # (nb_posterior_samples, nb_samples, nb_latents)

        return z_event_samples, z_burst_samples
    
    def sample_posterior_predictive(self, idx: torch.Tensor, nb_posterior_samples: int = 1):

        z_event_samples, z_burst_samples = self.sample_posterior(idx, nb_posterior_samples)

        log_event_rate = self.event_decoder(z_event_samples)
        logit_burst_prob = self.burst_decoder(z_burst_samples, log_event_rate)

        new_events = torch.distributions.Poisson(torch.exp(log_event_rate)).sample()  # (nb_posterior_samples, nb_samples, nb_neurons)
        new_bursts = torch.distributions.Binomial(new_events, torch.sigmoid(logit_burst_prob)).sample()  # (nb_posterior_samples, nb_samples, nb_neurons)

        return new_events, new_bursts
    
    def analytic_kl_divergence(self, idx: torch.Tensor):

        mu_event, log_var_event = self.event_encoder(idx)
        mu_burst, log_var_burst = self.burst_encoder(idx)

        kl_event = kl_diag_standard_normal(mu_event, log_var_event)
        kl_burst = kl_diag_standard_normal(mu_burst, log_var_burst)
        kl_divergence = kl_event + kl_burst

        return kl_divergence


    def elbo_terms(self, events: torch.Tensor, bursts: torch.Tensor, idx: torch.Tensor, nb_posterior_samples: int = 1):
        """
        Returns dict with:
          - elbo
          - loglike (scaled)
          - kl_z (scaled, with beta applied)
          - theta_log_prior (unscaled)
        """

        # Scale factor for minibatching over datapoints 
        scale = float(self.nb_samples) / float(events.shape[0])

        # Sample from the approximate posteriors
        z_event_samples, z_burst_samples = self.sample_posterior(idx, nb_posterior_samples)

        # Expected log-likelihood via MC
        log_event_rate = self.event_decoder(z_event_samples)
        logit_burst_prob = self.burst_decoder(z_burst_samples, log_event_rate.detach())

        event_log_likelihood = self.event_likelihood.log_prob(events, log_event_rate).sum(dim=(0, 1)).mean()
        burst_log_likelihood = self.burst_likelihood.log_prob(bursts, events, logit_burst_prob).sum(dim=(0, 1)).mean()
        log_likelihood = event_log_likelihood + burst_log_likelihood

        # KL divergences
        kl_divergence = self.analytic_kl_divergence(idx)

        # Scaled ELBO
        elbo = scale * (log_likelihood - kl_divergence)
        elbo_full = scale * (log_likelihood - kl_divergence) 

        return {
            "elbo": elbo,
            "elbo_full": elbo_full,
            "log_likelihood_scaled": scale * log_likelihood,
            "kl_divergence_scaled": scale * kl_divergence,
        }

    def loss(self, events: torch.Tensor, bursts: torch.Tensor, idx: torch.Tensor, nb_posterior_samples: int = 1):
        """
        Returns (loss_to_minimize, stats_dict)
        """
        terms = self.elbo_terms(events=events, bursts=bursts, idx=idx, nb_posterior_samples=nb_posterior_samples)
        loss = -1*terms["elbo"]

        stats = {"loss": loss.detach().item(), "log_likelihood_scaled": terms["log_likelihood_scaled"].detach().item(),
            "kl_divergence_scaled": terms["kl_divergence_scaled"].detach().item(), "elbo_full": terms["elbo_full"].detach().item()}
        
        return loss, stats
    
    def log_evidence(self, events: torch.Tensor, bursts: torch.Tensor, nb_posterior_samples: int = 1000):
        """
        Estimate log evidence via importance sampling with the approximate posterior as the proposal distribution.
        """

        idx = torch.arange(events.shape[0])
        nb_latents_events = self.nb_latents_events
        nb_latents_bursts = self.nb_latents_bursts
        device = events.device

        with torch.no_grad():

            # Sample from the approximate posteriors
            z_event_samples, z_burst_samples = self.sample_posterior(idx, nb_posterior_samples)

            # Evaluate log-likelihood for each sample
            log_event_rate = self.event_decoder(z_event_samples)
            logit_burst_prob = self.burst_decoder(z_burst_samples, log_event_rate.detach())

            event_log_likelihood = self.event_likelihood.log_prob(events, log_event_rate).sum(dim=0)
            burst_log_likelihood = self.burst_likelihood.log_prob(bursts, events, logit_burst_prob).sum(dim=0)

            # Combine log-likelihoods
            log_likelihood = event_log_likelihood + burst_log_likelihood # (nb_samples, nb_posterior_samples)

            # Evaluate the log prior for each sample
            event_prior_dist_base = torch.distributions.Normal(loc=torch.zeros(nb_latents_events, device=device), scale=torch.ones(nb_latents_events, device=device))
            event_prior_dist = torch.distributions.Independent(event_prior_dist_base, 1)
            event_log_prior = event_prior_dist.log_prob(z_event_samples) # (nb_posterior_samples, nb_samples)

            burst_prior_dist_base = torch.distributions.Normal(loc=torch.zeros(nb_latents_bursts, device=device), scale=torch.ones(nb_latents_bursts, device=device))
            burst_prior_dist = torch.distributions.Independent(burst_prior_dist_base, 1)
            burst_log_prior = burst_prior_dist.log_prob(z_burst_samples) # (nb_posterior_samples, nb_samples)

            # Combine log priors
            log_prior = event_log_prior + burst_log_prior

            # Evaluate the log posterior for each sample
            q_event, q_burst = self.posterior_dist(idx)
            q_event_log_prob = q_event.log_prob(z_event_samples) # (nb_posterior_samples, nb_samples)
            q_burst_log_prob = q_burst.log_prob(z_burst_samples) # (nb_posterior_samples, nb_samples)

            # Combine log posteriors
            log_posterior = q_event_log_prob + q_burst_log_prob

            # Compute importance weights and estimate log evidence
            log_importance_weights = log_likelihood.T + log_prior - log_posterior  # (nb_posterior_samples, nb_samples)
            log_evidence_score = (torch.logsumexp(log_importance_weights, dim=0) - torch.log(torch.tensor(nb_posterior_samples)))  # per sample log evidence

            return log_evidence_score
 
### Functions for fitting each of the models
def fit_event_model(data_train, nb_latents, training_params, device, print_loss=False, data_test=None):
    """
    Function to fit an event model to the training data, optionally evaluating on test data.
    
    :param data_train: Dictionary containing the training data, with keys "events" and optionally "bursts".
    :param nb_latents: Dictionary specifying the number of latent variables for each component, e.g., {"events": nb_latents_events}.
    :param training_params: Dictionary containing training parameters such as "nb_iter", "batch_size", "lr", and "final_ratio".
    :param device: Torch device to run the model on (e.g., "cpu" or "cuda").
    :param print_loss: Boolean indicating whether to print the loss during training.
    :param data_test: Dictionary containing the test data, with keys "events" and optionally "bursts". If None, no test evaluation is performed.
    :return: The trained event model and training statistics history. If test data is provided, the test model and evaluation results are also included.
    """

    # Unpack parameters
    nb_samples_train, nb_neurons = data_train["events"].shape
    
    if data_test is not None:
        nb_samples_test, _ = data_test["events"].shape

    nb_latents_events = nb_latents["events"]

    # Intit parameters using FA on the event data
    init_params = estimate_init_params(data_train, nb_latents, eps=1e-6, model_type='base')

    # Define the models
    event_model_train = EventModel(nb_samples=nb_samples_train, nb_neurons=nb_neurons, nb_latents_events=nb_latents_events, init_params=init_params).to(device)

    # Setup learning rate schedule
    lr_lambda= lambda step: exponential_decay_lambda(step, total_steps=training_params["nb_iter"], final_ratio=training_params["final_ratio"])

    # Fit the model
    event_model_train, stats_hist_train = optimize(event_model_train, event_only_loss_function, data_train, device, 
                                nb_iter=training_params["nb_iter"], batch_size=training_params["batch_size"], lr=training_params["lr"], 
                                lr_lambda=lr_lambda, print_loss=print_loss)
    

    if data_test is not None:
        
        # Define the test model
        event_model_test = EventModel(nb_samples=nb_samples_test, nb_neurons=nb_neurons, nb_latents_events=nb_latents_events, init_params=init_params).to(device)

        # Load the trained parameters into the test model and freeze them
        event_model_test.load_decoder_parameters(event_model_train, freeze=True)

        # Fit the test model
        event_model_test, stats_hist_test = optimize(event_model_test, event_only_loss_function, data_test, device,
                                    nb_iter=training_params["nb_iter"], batch_size=training_params["batch_size"], lr=training_params["lr"], 
                                    lr_lambda=lr_lambda, print_loss=print_loss)
        
        return event_model_train, stats_hist_train, event_model_test, stats_hist_test

    return event_model_train, stats_hist_train

def fit_burst_multiplexed_model(data_train, event_model, nb_latents, training_params, device, print_loss=False, data_test=None):
    """
    Function to fit the burst multiplexed model.
    :param data_train: Dictionary containing the training data, with keys "events" and optionally "bursts".
    :param nb_latents: Dictionary specifying the number of latent variables for each component, e.g., {"events": nb_latents_events, "bursts": nb_latents_bursts}.
    :param training_params: Dictionary containing training parameters such as "nb_iter", "batch_size", "lr", and "final_ratio".
    :param device: Torch device to run the model on (e.g., "cpu" or "cuda").
    :param print_loss: Boolean indicating whether to print the loss during training.
    :param data_test: Dictionary containing the test data, with keys "events" and optionally "bursts". If None, no test evaluation is performed.
    :return: The trained burst multiplexed model and training statistics history. If test data is provided, the test model and evaluation results are also included.
    """

    # Unpack parameters
    nb_samples, nb_neurons = data_train["events"].shape

    if data_test is not None:
        nb_samples_test, _ = data_test["events"].shape
    
    nb_latents_events = nb_latents["events"]
    nb_latents_bursts = nb_latents["bursts"]

    # Intit parameters using FA on the event data
    init_params = estimate_init_params(data_train, nb_latents, eps=1e-6, model_type='base')

    # Define the model
    multiplexed_model_train = BurstMultiplexedModel(nb_samples=nb_samples, nb_neurons=nb_neurons, nb_latents_events=nb_latents_events, nb_latents_bursts=nb_latents_bursts,
                            init_params=init_params).to(device)

    # Load the event parameters into the multiplexed model and freeze them
    multiplexed_model_train.load_event_parameters(event_model, freeze=True)

    # Setup learning rate schedule
    lr_lambda= lambda step: exponential_decay_lambda(step, total_steps=training_params["nb_iter"], final_ratio=training_params["final_ratio"])

    # Fit the model
    multiplexed_model_train, stats_hist_train = optimize(multiplexed_model_train, full_loss_function, data_train, device, 
                                nb_iter=training_params["nb_iter"], batch_size=training_params["batch_size"], lr=training_params["lr"], 
                                lr_lambda=lr_lambda, print_loss=print_loss)
    

    if data_test is not None:
        
        # Define the test model
        multiplexed_model_test = BurstMultiplexedModel(nb_samples=nb_samples_test, nb_neurons=nb_neurons, nb_latents_events=nb_latents_events, nb_latents_bursts=nb_latents_bursts,
                            init_params=init_params).to(device)

        # Load the trained parameters into the test model and freeze them
        multiplexed_model_test.load_decoder_parameters(multiplexed_model_train, freeze=True)

        # Fit the test model
        multiplexed_model_test, stats_hist_test = optimize(multiplexed_model_test, full_loss_function, data_test, device,
                                    nb_iter=training_params["nb_iter"], batch_size=training_params["batch_size"], lr=training_params["lr"], 
                                    lr_lambda=lr_lambda, print_loss=print_loss)
        
        return multiplexed_model_train, stats_hist_train, multiplexed_model_test, stats_hist_test

    return multiplexed_model_train, stats_hist_train

def fit_baseline_burst_model(data_train, event_model, nb_latents, device):

    # Unpack parameters
    nb_samples, nb_neurons = data_train["events"].shape
    nb_latents_events = nb_latents["events"]

    # Intit parameters using FA on the event data
    init_params = estimate_init_params(data_train, nb_latents, eps=1e-6, model_type='base')

    # Define the model
    baseline_burst_model_train = BurstIndepModel(nb_samples=nb_samples, nb_neurons=nb_neurons, nb_latents_events=nb_latents_events,
                            init_params=init_params).to(device)

    # Load the event parameters into the baseline model and freeze them
    baseline_burst_model_train.load_event_parameters(event_model, freeze=True)

    # Learn the burst probabilities
    baseline_burst_model_train.maximum_likelihood_estimation(data_train["events"], data_train["bursts"])

    return baseline_burst_model_train

def fit_burst_null_model(data_train, event_model, nb_latents, training_params, device, print_loss=False, data_test=None):
    
    # Unpack parameters
    nb_samples, nb_neurons = data_train["events"].shape

    if data_test is not None:
        nb_samples_test, _ = data_test["events"].shape

    nb_latents_events = nb_latents["events"]

    # Intit parameters using FA on the event data
    init_params = estimate_init_params(data_train, nb_latents, eps=1e-6, model_type='null')

    # Define the model
    null_model = BurstNullModel(nb_samples=nb_samples, nb_neurons=nb_neurons, nb_latents_events=nb_latents_events,init_params=init_params).to(device)

    # Load the event parameters into the null model and freeze them
    null_model.load_event_parameters(event_model, freeze=True)

    # Setup learning rate schedule
    lr_lambda= lambda step: exponential_decay_lambda(step, total_steps=training_params["nb_iter"], final_ratio=training_params["final_ratio"])

    # Fit the model
    null_model_train, stats_hist_train = optimize(null_model, full_loss_function, data_train, device, 
                                nb_iter=training_params["nb_iter"], batch_size=training_params["batch_size"], lr=training_params["lr"], 
                                lr_lambda=lr_lambda, print_loss=print_loss)
    
    if data_test is not None:
        
        # Define the test model
        null_model_test = BurstNullModel(nb_samples=nb_samples_test, nb_neurons=nb_neurons, nb_latents_events=nb_latents_events, init_params=init_params).to(device)

        # Load the trained parameters into the test model and freeze them
        null_model_test.load_decoder_parameters(null_model_train, freeze=True)

        # Fit the test model
        null_model_test, stats_hist_test = optimize(null_model_test, full_loss_function, data_test, device,
                                    nb_iter=training_params["nb_iter"], batch_size=training_params["batch_size"], lr=training_params["lr"], 
                                    lr_lambda=lr_lambda, print_loss=print_loss)
        
        return null_model_train, stats_hist_train, null_model_test, stats_hist_test

    return null_model_train, stats_hist_train

def fit_burst_hybrid_model(data_train, event_model, nb_latents, training_params, device, print_loss=False, data_test=None):
    """
    Function to fit the burst hybrid model.
    :param data_train: Dictionary containing the training data, with keys "events" and optionally "bursts".
    :param nb_latents: Dictionary specifying the number of latent variables for each component, e.g., {"events": nb_latents_events, "bursts": nb_latents_bursts}.
    :param training_params: Dictionary containing training parameters such as "nb_iter", "batch_size", "lr", and "final_ratio".
    :param device: Torch device to run the model on (e.g., "cpu" or "cuda").
    :param print_loss: Boolean indicating whether to print the loss during training.
    :param data_test: Dictionary containing the test data, with keys "events" and optionally "bursts". If None, no test evaluation is performed.
    :return: The trained burst hybrid model and training statistics history. If test data is provided, the test model and evaluation results are also included.
    """

    # Unpack parameters
    nb_samples, nb_neurons = data_train["events"].shape

    if data_test is not None:
        nb_samples_test, _ = data_test["events"].shape
    
    nb_latents_events = nb_latents["events"]
    nb_latents_bursts = nb_latents["bursts"]

    # Intit parameters using FA on the event data
    init_params = estimate_init_params(data_train, nb_latents, eps=1e-6, model_type='hybrid')

    # Define the model
    hybrid_model_train = BurstHybridModel(nb_samples=nb_samples, nb_neurons=nb_neurons, nb_latents_events=nb_latents_events, nb_latents_bursts=nb_latents_bursts,
                            init_params=init_params).to(device)

    # Load the event parameters into the multiplexed model and freeze them
    hybrid_model_train.load_event_parameters(event_model, freeze=True)

    # Setup learning rate schedule
    lr_lambda= lambda step: exponential_decay_lambda(step, total_steps=training_params["nb_iter"], final_ratio=training_params["final_ratio"])

    # Fit the model
    hybrid_model_train, stats_hist_train = optimize(hybrid_model_train, full_loss_function, data_train, device, 
                                nb_iter=training_params["nb_iter"], batch_size=training_params["batch_size"], lr=training_params["lr"], 
                                lr_lambda=lr_lambda, print_loss=print_loss)
    

    if data_test is not None:
        
        # Define the test model
        hybrid_model_test = BurstHybridModel(nb_samples=nb_samples_test, nb_neurons=nb_neurons, nb_latents_events=nb_latents_events, nb_latents_bursts=nb_latents_bursts,
                            init_params=init_params).to(device)

        # Load the trained parameters into the test model and freeze them
        hybrid_model_test.load_decoder_parameters(hybrid_model_train, freeze=True)

        # Fit the test model
        hybrid_model_test, stats_hist_test = optimize(hybrid_model_test, full_loss_function, data_test, device,
                                    nb_iter=training_params["nb_iter"], batch_size=training_params["batch_size"], lr=training_params["lr"], 
                                    lr_lambda=lr_lambda, print_loss=print_loss)
        
        return hybrid_model_train, stats_hist_train, hybrid_model_test, stats_hist_test

    return hybrid_model_train, stats_hist_train

### Scoring and evaluation functions for the models
def log_evidence_scores(model, data, device, nb_posterior_samples=1000, event_model=False):
    """
    Computes the log evidence scores for the given model and data.

    :param model: The model to evaluate.
    :param data: A dictionary containing the evaluation data with keys "events" and "bursts".
    :param device: The device (CPU or GPU) on which to perform the evaluation.
    :param nb_posterior_samples: The number of posterior samples to use in the computation.
    :param event_model: If True, indicates that the model is an event-only model.
    :return: The log evidence scores.
    """

    with torch.inference_mode():

        # Events and bursts of test data
        events_dev = data["events"].to(device) # Move to the correct device
        bursts_dev = data["bursts"].to(device)

        if event_model:
            scores = model.log_evidence(events_dev, nb_posterior_samples=nb_posterior_samples)
        else:
            scores = model.log_evidence(events_dev, bursts_dev, nb_posterior_samples=nb_posterior_samples)

    return scores

def sample_posterior_predictive(model, nb_posterior_samples=1000, event_model=False):
    """
    Samples from the posterior predictive distribution of the given model.
    
    :param model: The model from which to sample.
    :param nb_posterior_samples: The number of posterior samples to generate.
    :param event_model: If True, indicates that the model is an event-only model.
    :return: A dictionary containing the sampled events and, if applicable, bursts.
    """
        
    with torch.inference_mode():

        if event_model:
            new_events = model.sample_posterior_predictive(idx=torch.arange(model.nb_samples), nb_posterior_samples=nb_posterior_samples)
            data_posterior_predictive = {"events": new_events.cpu()}
        else:
            new_events, new_bursts = model.sample_posterior_predictive(idx=torch.arange(model.nb_samples), nb_posterior_samples=nb_posterior_samples)
            data_posterior_predictive = {"events": new_events.cpu(), "bursts": new_bursts.cpu()}

    return data_posterior_predictive