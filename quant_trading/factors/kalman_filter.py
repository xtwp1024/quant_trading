"""
quant_trading.factors.kalman_filter — Kalman Filter and KalmanNet implementations.

Based on KalmanBOT_ICASSP23 (ICASSP 2023) - KalmanNet: GRU架构神经网络增强卡尔曼滤波

Classes
-------
KalmanFilter : Traditional linear Kalman filter for state estimation.
LinearSystemModel : Linear state-space model (motion + observation).
KalmanNetNN : GRU-based neural network that learns adaptive Kalman gain.
KalmanNetSystemModel : System model for KalmanNet with learnable dynamics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from typing import Optional, Tuple, Callable


# =============================================================================
# Device management
# =============================================================================
if torch.cuda.is_available():
    _DEV = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    _DEV = torch.device("cpu")


# =============================================================================
# Linear Kalman Filter
# =============================================================================
class KalmanFilter:
    """
    Linear Kalman Filter for 1D observation with 2D state [beta; intercept].

    State evolution: x_{t+1} = F x_t + w_t,  w_t ~ N(0, Q)
    Observation:     y_t     = H_t x_t + v_t,  v_t ~ N(0, R)

    For pairs trading: x = [hedge_ratio, intercept]^T,
                       y = price_spread (e.g., EUR - beta * CHF)
    """

    def __init__(self, ss_model: "LinearSystemModel", ratio: int = 1):
        """
        Parameters
        ----------
        ss_model : LinearSystemModel
            State-space model with F, Q, H, R matrices.
        ratio : int
            Prediction-to-observation ratio (1 = every step).
        """
        self.F = ss_model.F
        self.F_T = self.F.t()
        self.m = ss_model.m  # state dim

        self.Q = ss_model.Q

        self.H = ss_model.H
        self.n = ss_model.n  # observation dim

        self.R = ss_model.R

        self.T = ss_model.T
        self.T_test = getattr(ss_model, "T_test", ss_model.T)
        self.ratio = ratio

    def predict(self, H: torch.Tensor, H_T: torch.Tensor) -> None:
        """Compute prior moments m1x_prior, m2x_prior, m1y, m2y."""
        for i in range(self.ratio):
            if i == self.ratio - 1:
                self.m1x_prior = torch.matmul(self.F, self.m1x_posterior)
                self.m2x_prior = torch.matmul(self.F, self.m2x_posterior)
                self.m2x_prior = torch.matmul(self.m2x_prior, self.F_T) + self.Q
                self.m1y = torch.matmul(H, self.m1x_prior)
                self.m2y = torch.matmul(H, self.m2x_prior)
                self.m2y = torch.matmul(self.m2y, H_T) + self.R
            else:
                m1x_temp = torch.matmul(self.F, self.m1x_posterior)
                m2x_temp = torch.matmul(self.F, self.m2x_posterior)
                m2x_temp = torch.matmul(m2x_temp, self.F_T) + self.Q

    def k_gain(self, H_T: torch.Tensor) -> None:
        """Compute Kalman gain."""
        self.KG = torch.matmul(self.m2x_prior, H_T)
        self.KG = torch.matmul(self.KG, torch.inverse(self.m2y))

    def innovate(self, y: torch.Tensor) -> None:
        """Innovation (measurement residual)."""
        self.dy = y - self.m1y

    def correct(self) -> None:
        """Compute posterior moments."""
        self.m1x_posterior = self.m1x_prior + torch.matmul(self.KG, self.dy)
        self.m2x_posterior = torch.matmul(self.m2y, self.KG.t())
        self.m2x_posterior = self.m2x_prior - torch.matmul(self.KG, self.m2x_posterior)

    def update(
        self, y: torch.Tensor, H: torch.Tensor, H_T: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full predict-correct step."""
        self.predict(H, H_T)
        self.k_gain(H_T)
        self.innovate(y)
        self.correct()
        return self.m1x_posterior, self.m2x_posterior

    def init_sequence(self, m1x_0: torch.Tensor, m2x_0: torch.Tensor) -> None:
        """Initialize filter with prior moments."""
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

    def generate_sequence(
        self, y: torch.Tensor, T: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run filter over observation sequence.

        Returns
        -------
        x : (m, T) estimated states
        sigma : (m, m, T) state covariances
        innovations : (T,) innovation sequence
        y_vars : (T,) predicted observation variances
        """
        x = torch.empty(self.m, T, device=_DEV)
        sigma = torch.empty(self.m, self.m, T, device=_DEV)
        innovations = torch.empty(T, device=_DEV)
        y_vars = torch.empty(T, device=_DEV)

        self.m1x_posterior = self.m1x_0
        self.m2x_posterior = self.m2x_0

        for t in range(T):
            yt = torch.unsqueeze(y[:, t], 1)
            H_t = torch.tensor(self.H[t] if self.H.dim() > 1 else self.H).unsqueeze(0)
            H_T_t = H_t.t()

            xt, sigmat = self.update(yt, H_t, H_T_t)
            x[:, t] = torch.squeeze(xt)
            sigma[:, :, t] = torch.squeeze(sigmat)
            innovations[t] = torch.squeeze(self.dy)
            y_vars[t] = torch.squeeze(self.m2y)

        return x, sigma, innovations, y_vars


# =============================================================================
# Linear System Model
# =============================================================================
class LinearSystemModel:
    """
    Linear Gaussian state-space model for pairs trading.

    State x = [beta, intercept]^T  (hedge ratio + intercept)
    Observation y = spread = y_asset - H @ x

    Dynamics: x_{t+1} = F x_t + w_t,  w_t ~ N(0, Q)
    Observation: y_t = H_t @ x_t + v_t,  v_t ~ N(0, R)
    """

    def __init__(
        self,
        F: torch.Tensor,
        q: float,
        H: torch.Tensor,
        r: float,
        T: int,
        T_test: Optional[int] = None,
        hedge: int = 1,
        prior_Q: Optional[torch.Tensor] = None,
        prior_Sigma: Optional[torch.Tensor] = None,
        prior_S: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        F : (m, m) state transition matrix
        q : float, process noise std
        H : (n, m) observation matrix (or (T, n, m) for time-varying)
        r : float, observation noise std
        T : int, training sequence length
        T_test : int, test sequence length
        hedge : int, observation dimension (1 for pairs trading)
        """
        self.F = F
        self.m = F.size(0)

        self.q = q
        self.Q = q * q * torch.eye(self.m)

        self.H = H
        if hedge == 0:
            self.n = H.size(0)
        else:
            self.n = 1  # pairs trading: 1D spread observation

        self.r = r
        self.R = r * r * torch.eye(self.n)

        self.T = T
        self.T_test = T_test if T_test is not None else T

        # Covariance priors for KalmanNet
        self.prior_Q = prior_Q if prior_Q is not None else torch.eye(self.m)
        self.prior_Sigma = prior_Sigma if prior_Sigma is not None else torch.eye(self.m)
        self.prior_S = prior_S if prior_S is not None else torch.eye(self.n)

    def init_sequence(self, m1x_0: torch.Tensor, m2x_0: torch.Tensor) -> None:
        """Set initial state distribution."""
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

    def update_covariance_gain(self, q: float, r: float) -> None:
        """Update process and observation noise std."""
        self.q = q
        self.Q = q * q * torch.eye(self.m)
        self.r = r
        self.R = r * r * torch.eye(self.n)

    def update_covariance_matrix(self, Q: torch.Tensor, R: torch.Tensor) -> None:
        """Update covariance matrices directly."""
        self.Q = Q
        self.R = R


# =============================================================================
# KalmanNet: GRU-based Adaptive Kalman Filter
# =============================================================================
class KalmanNetNN(nn.Module):
    """
    KalmanNet: GRU-enhanced Kalman Filter that learns adaptive Kalman gain.

    Reference: KalmanBOT_ICASSP23 (ICASSP 2023)
    Architecture: 3 GRUs (Q, Sigma, S tracking) + 7 FC layers

    The network learns to estimate:
    - Q: process noise covariance
    - Sigma: prior state covariance
    - S: innovation covariance

    And computes an adaptive Kalman gain without requiring explicit matrix
    inversions.
    """

    def __init__(self):
        super().__init__()
        self.device = _DEV
        self.to(self.device)

    def build(self, sys_model: "KalmanNetSystemModel") -> None:
        """Build KalmanNet from system model."""
        self._init_system_dynamics(sys_model.f, sys_model.h, sys_model.m, sys_model.n)
        self._init_sequence(sys_model.m1x_0, sys_model.T)
        self._init_k_gain_net(
            sys_model.prior_Q, sys_model.prior_Sigma, sys_model.prior_S
        )

    def _init_system_dynamics(
        self, f: Callable, h: Callable, m: int, n: int, infoString: str = "fullInfo"
    ) -> None:
        """Set state evolution and observation functions."""
        if infoString == "partialInfo":
            self.f_string = "ModInacc"
            self.h_string = "ObsInacc"
        else:
            self.f_string = "ModAcc"
            self.h_string = "ObsAcc"

        self.f = f
        self.m = m
        self.h = h
        self.H = torch.tensor([[1.0, 1.0]], device=self.device)
        self.n = n

    def _init_sequence(self, m1x_0: torch.Tensor, T: int) -> None:
        """Initialize filter sequence state."""
        self.T = T
        self.x_out = torch.empty(self.m, T, device=self.device)

        self.m1x_posterior = m1x_0.to(self.device)
        self.m1x_posterior_previous = self.m1x_posterior.clone()
        self.m1x_prior_previous = self.m1x_posterior.clone()
        self.y_previous = torch.matmul(self.H, self.m1x_posterior).to(self.device)

        self.i = 0

    def _init_k_gain_net(
        self, prior_Q: torch.Tensor, prior_Sigma: torch.Tensor, prior_S: torch.Tensor
    ) -> None:
        """Build the Kalman gain estimation network (GRU + FC layers)."""
        self.seq_len_input = 1
        self.batch_size = 1

        self.prior_Q = prior_Q
        self.prior_Sigma = prior_Sigma
        self.prior_S = prior_S

        in_mult = 5
        out_mult = 40

        # GRU to track Q
        self.d_input_Q = self.m * in_mult
        self.d_hidden_Q = self.m**2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q)
        self.h_Q = torch.randn(
            self.seq_len_input, self.batch_size, self.d_hidden_Q, device=self.device
        )

        # GRU to track Sigma
        self.d_input_Sigma = self.d_hidden_Q + self.m * in_mult
        self.d_hidden_Sigma = self.m**2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma)
        self.h_Sigma = torch.randn(
            self.seq_len_input, self.batch_size, self.d_hidden_Sigma, device=self.device
        )

        # GRU to track S
        self.d_input_S = self.n**2 + 2 * self.n * in_mult
        self.d_hidden_S = self.n**2
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S)
        self.h_S = torch.randn(
            self.seq_len_input, self.batch_size, self.d_hidden_S, device=self.device
        )

        # FC 1: Sigma hidden -> S observation covariance
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.n**2
        self.FC1 = nn.Sequential(
            nn.Linear(self.d_input_FC1, self.d_output_FC1), nn.ReLU()
        )

        # FC 2: [Sigma_hidden, S_hidden] -> Kalman gain
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.n * self.m
        self.d_hidden_FC2 = self.d_input_FC2 * out_mult
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC2, self.d_output_FC2),
        )

        # FC 3: [S_hidden, KG] -> state covariance correction
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.m**2
        self.FC3 = nn.Sequential(
            nn.Linear(self.d_input_FC3, self.d_output_FC3), nn.ReLU()
        )

        # FC 4: [Sigma_hidden, FC3_out] -> Sigma hidden update
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
            nn.Linear(self.d_input_FC4, self.d_output_FC4), nn.ReLU()
        )

        # FC 5: forward evolution diff -> Q-GRU input
        self.d_input_FC5 = self.m
        self.d_output_FC5 = self.m * in_mult
        self.FC5 = nn.Sequential(
            nn.Linear(self.d_input_FC5, self.d_output_FC5), nn.ReLU()
        )

        # FC 6: forward update diff -> Sigma-GRU input
        self.d_input_FC6 = self.m
        self.d_output_FC6 = self.m * in_mult
        self.FC6 = nn.Sequential(
            nn.Linear(self.d_input_FC6, self.d_output_FC6), nn.ReLU()
        )

        # FC 7: observation diffs -> S-GRU input
        self.d_input_FC7 = 2 * self.n
        self.d_output_FC7 = 2 * self.n * in_mult
        self.FC7 = nn.Sequential(
            nn.Linear(self.d_input_FC7, self.d_output_FC7), nn.ReLU()
        )

    def _step_prior(self) -> None:
        """Compute prior moments."""
        self.m1x_prior = self.f(self.m1x_posterior)
        self.m1y = torch.matmul(self.H, self.m1x_prior)

    def _step_k_gain_est(
        self, y: torch.Tensor
    ) -> None:
        """Estimate Kalman gain using the neural network."""
        obs_diff = y - torch.squeeze(self.y_previous)
        obs_innov_diff = y - torch.squeeze(self.m1y)
        fw_evol_diff = torch.squeeze(self.m1x_posterior) - torch.squeeze(
            self.m1x_posterior_previous
        )
        fw_update_diff = torch.squeeze(self.m1x_posterior) - torch.squeeze(
            self.m1x_prior_previous
        )

        # L2 normalize
        obs_diff = func.normalize(obs_diff, p=2, dim=0, eps=1e-12)
        obs_innov_diff = func.normalize(obs_innov_diff, p=2, dim=0, eps=1e-12)
        fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=0, eps=1e-12)
        fw_update_diff = func.normalize(fw_update_diff, p=2, dim=0, eps=1e-12)

        KG = self._k_gain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)
        self.KGain = torch.reshape(KG, (self.m, self.n))

    def _k_gain_step(
        self,
        obs_diff: torch.Tensor,
        obs_innov_diff: torch.Tensor,
        fw_evol_diff: torch.Tensor,
        fw_update_diff: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through Kalman gain estimation network."""

        def expand_dim(x: torch.Tensor) -> torch.Tensor:
            try:
                expanded = torch.empty(
                    self.seq_len_input, self.batch_size, x.shape[-1], device=self.device
                )
            except IndexError:
                expanded = torch.empty(
                    self.seq_len_input, self.batch_size, 1, device=self.device
                )
            expanded[0, 0, :] = x
            return expanded

        obs_diff = expand_dim(obs_diff)
        obs_innov_diff = expand_dim(obs_innov_diff)
        fw_evol_diff = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)

        # Forward flow
        out_FC5 = self.FC5(fw_evol_diff)
        out_Q, self.h_Q = self.GRU_Q(out_FC5, self.h_Q)

        out_FC6 = self.FC6(fw_update_diff)
        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)

        out_FC1 = self.FC1(out_Sigma)

        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        out_FC7 = self.FC7(in_FC7)

        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)
        self.S_t = out_S

        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2)

        # Backward flow
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)

        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)

        self.h_Sigma = out_FC4

        return out_FC2

    def knet_step(self, y: torch.Tensor) -> torch.Tensor:
        """Single step of KalmanNet forward pass."""
        if y.dim() > 1:
            self.H = y[:, 1:]
            y = y[:, 0:1]

        # Compute priors
        self._step_prior()

        # Estimate Kalman gain
        self._step_k_gain_est(y)

        self.i += 1

        # Innovation
        y_obs = torch.unsqueeze(y, 1)
        dy = y_obs - self.m1y
        self.dy = dy

        # Posterior update
        INOV = torch.matmul(self.KGain, dy)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_posterior = self.m1x_prior + INOV
        self.m1x_prior_previous = self.m1x_prior

        self.y_previous = y

        return torch.squeeze(self.m1x_posterior)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Full forward pass through KalmanNet."""
        y = y.to(self.device)
        self.x_out = self.knet_step(y)
        return self.x_out

    def init_hidden(self) -> None:
        """Initialize GRU hidden states from prior covariances."""
        weight = next(self.parameters()).data
        hidden = weight.new(1, self.batch_size, self.d_hidden_S).zero_()
        self.h_S = hidden.data
        self.h_S[0, 0, :] = self.prior_S.flatten()

        hidden = weight.new(1, self.batch_size, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden.data
        self.h_Sigma[0, 0, :] = self.prior_Sigma.flatten()

        hidden = weight.new(1, self.batch_size, self.d_hidden_Q).zero_()
        self.h_Q = hidden.data
        self.h_Q[0, 0, :] = self.prior_Q.flatten()


# =============================================================================
# KalmanNet System Model
# =============================================================================
class KalmanNetSystemModel:
    """
    State-space model for KalmanNet with learnable (non-linear) dynamics.

    Used for pairs trading with non-linear state evolution or
    time-varying observation matrices.
    """

    def __init__(
        self,
        F: torch.Tensor,
        f: Callable,
        q: float,
        H: torch.Tensor,
        h: Callable,
        r: float,
        T: int,
        hedge: int = 0,
        prior_Q: Optional[torch.Tensor] = None,
        prior_Sigma: Optional[torch.Tensor] = None,
        prior_S: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        F : (m, m) base transition matrix
        f : callable, state evolution function
        q : float, process noise std
        H : (n, m) or (T, n, m) observation matrix
        h : callable, observation function
        r : float, observation noise std
        T : int, sequence length
        """
        self.F = F
        self.f = f
        self.m = F.size(0)

        self.q = q
        self.Q = q * q * torch.eye(self.m)

        self.H = H
        self.h = h
        if hedge == 0:
            self.n = H.size(0)
        else:
            self.n = 1

        self.r = r
        self.R = r * r * torch.eye(self.n)

        self.T = T

        self.prior_Q = prior_Q if prior_Q is not None else torch.eye(self.m)
        self.prior_Sigma = prior_Sigma if prior_Sigma is not None else torch.eye(self.m)
        self.prior_S = prior_S if prior_S is not None else torch.eye(self.n)

    def init_sequence(self, m1x_0: torch.Tensor, m2x_0: torch.Tensor) -> None:
        """Initialize prior moments."""
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

    def update_covariance_gain(self, q: float, r: float) -> None:
        """Update noise std."""
        self.q = q
        self.Q = q * q * torch.eye(self.m)
        self.r = r
        self.R = r * r * torch.eye(self.n)

    def update_covariance_matrix(self, Q: torch.Tensor, R: torch.Tensor) -> None:
        """Update covariance matrices directly."""
        self.Q = Q
        self.R = R


# =============================================================================
# KalmanNet with Delta (Simplified version for online trading)
# =============================================================================
class KNetDelta(nn.Module):
    """
    Simplified KalmanNet variant returning innovation and covariance estimates.

    Returns (dy, x_out, S_t) for use with position management models.
    Suitable for online trading scenarios.
    """

    def __init__(self):
        super().__init__()
        self.device = _DEV
        self.to(self.device)

    def build(self, sys_model: KalmanNetSystemModel) -> None:
        """Build from system model."""
        self._init_system_dynamics(
            sys_model.f, sys_model.h, sys_model.m, sys_model.n
        )
        self._init_sequence(sys_model.m1x_0, sys_model.T)
        self._init_k_gain_net(
            sys_model.prior_Q, sys_model.prior_Sigma, sys_model.prior_S
        )

    def _init_system_dynamics(
        self, f: Callable, h: Callable, m: int, n: int, infoString: str = "fullInfo"
    ) -> None:
        if infoString == "partialInfo":
            self.f_string = "ModInacc"
            self.h_string = "ObsInacc"
        else:
            self.f_string = "ModAcc"
            self.h_string = "ObsAcc"
        self.f = f
        self.m = m
        self.h = h
        self.H = torch.tensor([[1.0, 1.0]], device=self.device)
        self.n = n

    def _init_sequence(self, m1x_0: torch.Tensor, T: int) -> None:
        self.T = T
        self.x_out = torch.empty(self.m, T, device=self.device)
        self.m1x_posterior = m1x_0.to(self.device)
        self.m1x_posterior_previous = self.m1x_posterior.clone()
        self.m1x_prior_previous = self.m1x_posterior.clone()
        self.y_previous = torch.matmul(self.H, self.m1x_posterior).to(self.device)
        self.i = 0

    def _init_k_gain_net(
        self, prior_Q: torch.Tensor, prior_Sigma: torch.Tensor, prior_S: torch.Tensor
    ) -> None:
        self.seq_len_input = 1
        self.batch_size = 1
        self.prior_Q = prior_Q
        self.prior_Sigma = prior_Sigma
        self.prior_S = prior_S

        in_mult, out_mult = 5, 40

        self.d_input_Q = self.m * in_mult
        self.d_hidden_Q = self.m**2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q)
        self.h_Q = torch.randn(
            self.seq_len_input, self.batch_size, self.d_hidden_Q, device=self.device
        )

        self.d_input_Sigma = self.d_hidden_Q + self.m * in_mult
        self.d_hidden_Sigma = self.m**2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma)
        self.h_Sigma = torch.randn(
            self.seq_len_input, self.batch_size, self.d_hidden_Sigma, device=self.device
        )

        self.d_input_S = self.n**2 + 2 * self.n * in_mult
        self.d_hidden_S = self.n**2
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S)
        self.h_S = torch.randn(
            self.seq_len_input, self.batch_size, self.d_hidden_S, device=self.device
        )

        self.FC1 = nn.Sequential(
            nn.Linear(self.d_hidden_Sigma, self.n**2), nn.ReLU()
        )
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_hidden_S + self.d_hidden_Sigma, self.n * self.m * out_mult),
            nn.ReLU(),
            nn.Linear(self.n * self.m * out_mult, self.n * self.m),
        )
        self.FC3 = nn.Sequential(
            nn.Linear(self.d_hidden_S + self.d_output_FC2 if hasattr(self, 'd_output_FC2') else self.n * self.m + self.d_hidden_S, self.m**2),
            nn.ReLU(),
        )
        self.FC4 = nn.Sequential(
            nn.Linear(self.d_hidden_Sigma + self.m**2, self.d_hidden_Sigma), nn.ReLU()
        )
        self.FC5 = nn.Sequential(
            nn.Linear(self.m, self.m * in_mult), nn.ReLU()
        )
        self.FC6 = nn.Sequential(
            nn.Linear(self.m, self.m * in_mult), nn.ReLU()
        )
        self.FC7 = nn.Sequential(
            nn.Linear(2 * self.n, 2 * self.n * in_mult), nn.ReLU()
        )

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        dy : innovation (scalar for pairs trading)
        x_out : estimated state
        S_t : innovation covariance estimate
        """
        if y.dim() > 1:
            self.H = y[:, 1:]
            y = y[:, 0:1]

        self.m1x_prior = self.f(self.m1x_posterior)
        self.m1y = torch.matmul(self.H, self.m1x_prior)

        # Kalman gain estimation
        obs_diff = func.normalize(
            y - torch.squeeze(self.y_previous), p=2, dim=0, eps=1e-12
        )
        obs_innov_diff = func.normalize(
            y - torch.squeeze(self.m1y), p=2, dim=0, eps=1e-12
        )
        fw_evol_diff = func.normalize(
            torch.squeeze(self.m1x_posterior) - torch.squeeze(self.m1x_posterior_previous),
            p=2, dim=0, eps=1e-12,
        )
        fw_update_diff = func.normalize(
            torch.squeeze(self.m1x_posterior) - torch.squeeze(self.m1x_prior_previous),
            p=2, dim=0, eps=1e-12,
        )

        # Forward pass
        out_FC5 = self.FC5(fw_evol_diff.unsqueeze(0).unsqueeze(0))
        out_Q, self.h_Q = self.GRU_Q(out_FC5, self.h_Q)

        out_FC6 = self.FC6(fw_update_diff.unsqueeze(0).unsqueeze(0))
        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)

        out_FC1 = self.FC1(out_Sigma)

        in_FC7 = torch.cat(
            (
                obs_diff.unsqueeze(0).unsqueeze(0),
                obs_innov_diff.unsqueeze(0).unsqueeze(0),
            ),
            2,
        )
        out_FC7 = self.FC7(in_FC7)

        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)
        S_t = out_S

        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2)

        KG = torch.reshape(out_FC2.squeeze(), (self.m, self.n))

        # Update
        y_obs = torch.unsqueeze(y, 1)
        dy = y_obs - self.m1y
        self.dy = dy

        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_posterior = self.m1x_prior + torch.matmul(KG, dy)
        self.m1x_prior_previous = self.m1x_prior
        self.y_previous = y
        self.x_out = self.m1x_posterior

        return dy.squeeze(), torch.squeeze(self.m1x_posterior), S_t.squeeze()

    def init_hidden(self) -> None:
        """Initialize hidden states from priors."""
        weight = next(self.parameters()).data
        self.h_S = weight.new(1, self.batch_size, self.d_hidden_S).zero_()
        self.h_S[0, 0, :] = self.prior_S.flatten()

        self.h_Sigma = weight.new(1, self.batch_size, self.d_hidden_Sigma).zero_()
        self.h_Sigma[0, 0, :] = self.prior_Sigma.flatten()

        self.h_Q = weight.new(1, self.batch_size, self.d_hidden_Q).zero_()
        self.h_Q[0, 0, :] = self.prior_Q.flatten()
