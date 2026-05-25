"""channel_model.py — Explicit wireless channel model (Extension 4).

Bridges the gap between abstract attack injection and physical RF parameters.

The original attack_injection.py treated communication failure as binary
(messages arrive or don't) with jam_power as a direct drop probability.

This module adds a physical layer beneath the attack injection:

  1.  Path loss model (log-distance)
  2.  Noise floor + SNR computation
  3.  Packet error rate (PER) vs SNR curve (BPSK/AWGN approximation)
  4.  Jamming: jammer raises noise floor → SNR drops → PER rises
  5.  Link quality metric qcomm_physical: derived from PER rather than
      a raw probability parameter

The output qcomm_physical[i][j] (range [0, 1]) replaces or supplements
the qcomm dict passed to the IDS — same interface, physical grounding.

Usage
-----
    ch = ChannelModel(channel_cfg)

    # At each timestep:
    qcomm = ch.link_quality(
        positions,          # (N, 3) true UAV positions
        t,                  # current time
        jam_active,         # bool — is jammer active this step?
        jam_power_w,        # jammer EIRP in Watts
        jam_pos,            # (3,) jammer position (or None → centroid)
    )
    # qcomm[i] = mean link quality for agent i (average over neighbors)

Default parameters are representative of 2.4 GHz WiFi/ZigBee at
UAV inter-agent distances of 0.5–3 m.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class ChannelModel:
    """Physical link quality model for UAV inter-agent communication.

    Parameters (from channel_cfg dict or defaults)
    -----------------------------------------------
    freq_ghz : float        Carrier frequency in GHz (default 2.4)
    tx_power_dbm : float    Transmit power in dBm (default 20 = 100 mW)
    tx_gain_dbi : float     Transmit antenna gain in dBi (default 2)
    rx_gain_dbi : float     Receive antenna gain in dBi (default 2)
    noise_figure_db : float Receiver noise figure in dB (default 6)
    bandwidth_mhz : float   Channel bandwidth in MHz (default 1)
    path_loss_exp : float   Path loss exponent (default 2.5, urban)
    d0_m : float            Reference distance in m (default 1.0)
    shadowing_std_db : float Log-normal shadowing std dev (default 2.0)
    per_snr_threshold_db : float SNR below which PER > 0.5 (default 5)
    per_snr_slope : float   Steepness of PER curve (default 2.0)
    rng_seed : int          Seed for shadowing realisation (default 42)
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        c = cfg or {}
        self.freq_ghz = float(c.get("freq_ghz", 2.4))
        self.tx_power_dbm = float(c.get("tx_power_dbm", 20.0))
        self.tx_gain_dbi = float(c.get("tx_gain_dbi", 2.0))
        self.rx_gain_dbi = float(c.get("rx_gain_dbi", 2.0))
        self.noise_figure_db = float(c.get("noise_figure_db", 6.0))
        self.bandwidth_mhz = float(c.get("bandwidth_mhz", 1.0))
        self.path_loss_exp = float(c.get("path_loss_exp", 2.5))
        self.d0_m = float(c.get("d0_m", 1.0))
        self.shadowing_std_db = float(c.get("shadowing_std_db", 2.0))
        self.per_snr_threshold_db = float(c.get("per_snr_threshold_db", 5.0))
        self.per_snr_slope = float(c.get("per_snr_slope", 2.0))
        self.rng = np.random.default_rng(int(c.get("rng_seed", 42)))

        # Derived constants
        # Thermal noise power: kTB  (k=1.38e-23, T=290 K, B in Hz)
        B_hz = self.bandwidth_mhz * 1e6
        self._noise_floor_dbm = float(
            10 * np.log10(1.38e-23 * 290.0 * B_hz * 1e3)  # in mW → dBm
            + self.noise_figure_db
        )
        # Free-space path loss at reference distance d0
        lambda_m = 3e8 / (self.freq_ghz * 1e9)
        self._fspl_d0_db = float(
            20 * np.log10(4 * np.pi * self.d0_m / lambda_m)
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def link_quality(
        self,
        positions: np.ndarray,
        t: float,
        jam_active: bool = False,
        jam_power_w: float = 0.0,
        jam_pos: np.ndarray | None = None,
    ) -> dict[int, float]:
        """Compute per-agent mean link quality metric.

        Returns
        -------
        qcomm : dict[int, float]
            qcomm[i] = mean(1 - PER) over all neighbors j≠i for agent i.
            Range [0, 1].  Higher = better communication.
        """
        n = positions.shape[0]
        per_matrix = self._per_matrix(positions, t, jam_active, jam_power_w, jam_pos)
        qcomm: dict[int, float] = {}
        for i in range(n):
            neighbor_q = []
            for j in range(n):
                if i == j:
                    continue
                neighbor_q.append(1.0 - per_matrix[i, j])
            qcomm[i] = float(np.mean(neighbor_q)) if neighbor_q else 1.0
        return qcomm

    def snr_matrix(
        self,
        positions: np.ndarray,
        t: float,
        jam_active: bool = False,
        jam_power_w: float = 0.0,
        jam_pos: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return (N, N) SNR matrix in dB (diagonal = 0)."""
        n = positions.shape[0]
        snr = np.zeros((n, n), dtype=float)
        noise_floor_dbm = self._effective_noise_floor_dbm(
            positions, jam_active, jam_power_w, jam_pos
        )
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                rx_power_dbm = self._rx_power_dbm(positions[i], positions[j], t)
                snr[i, j] = rx_power_dbm - noise_floor_dbm
        return snr

    def per_matrix(
        self,
        positions: np.ndarray,
        t: float,
        jam_active: bool = False,
        jam_power_w: float = 0.0,
        jam_pos: np.ndarray | None = None,
    ) -> np.ndarray:
        """Public wrapper returning (N, N) PER matrix."""
        return self._per_matrix(positions, t, jam_active, jam_power_w, jam_pos)

    def jam_power_to_drop_prob(
        self,
        positions: np.ndarray,
        t: float,
        jam_power_w: float,
        jam_pos: np.ndarray | None = None,
    ) -> float:
        """Mean packet drop probability induced by a jammer of given power.

        Convenience function for reporting / paper tables.
        Returns the swarm-average PER under jamming.
        """
        per = self._per_matrix(positions, t, jam_active=True,
                                jam_power_w=jam_power_w, jam_pos=jam_pos)
        n = positions.shape[0]
        off_diag = [(per[i, j]) for i in range(n) for j in range(n) if i != j]
        return float(np.mean(off_diag)) if off_diag else 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rx_power_dbm(
        self,
        tx_pos: np.ndarray,
        rx_pos: np.ndarray,
        t: float,
    ) -> float:
        """Received power in dBm using log-distance path loss + shadowing."""
        d = float(np.linalg.norm(tx_pos - rx_pos))
        d = max(d, self.d0_m * 0.01)  # avoid log(0)
        # Log-distance path loss
        pl_db = self._fspl_d0_db + 10.0 * self.path_loss_exp * np.log10(d / self.d0_m)
        # Log-normal shadowing (slow-fading; resample infrequently in real use)
        shadowing = float(self.rng.normal(0.0, self.shadowing_std_db))
        rx_dbm = (
            self.tx_power_dbm
            + self.tx_gain_dbi
            + self.rx_gain_dbi
            - pl_db
            + shadowing
        )
        return float(rx_dbm)

    def _effective_noise_floor_dbm(
        self,
        positions: np.ndarray,
        jam_active: bool,
        jam_power_w: float,
        jam_pos: np.ndarray | None,
    ) -> float:
        """Noise floor raised by jammer interference."""
        if not jam_active or jam_power_w <= 0.0:
            return self._noise_floor_dbm
        # Jammer at centroid if no explicit position given
        centroid = np.mean(positions, axis=0) if jam_pos is None else np.asarray(jam_pos)
        # Jammer distance from swarm centroid
        d_jam = float(np.linalg.norm(centroid - np.mean(positions, axis=0)))
        d_jam = max(d_jam, 0.5)
        # Jammer received power at swarm (simplified isotropic, d_jam path loss)
        jam_dbm = 10.0 * np.log10(jam_power_w * 1e3)  # W → dBm
        jam_pl = self._fspl_d0_db + 10.0 * self.path_loss_exp * np.log10(
            max(d_jam, self.d0_m) / self.d0_m
        )
        jam_rx_dbm = jam_dbm - jam_pl
        # Total noise = thermal + jammer (add in linear scale)
        noise_mw = 10 ** (self._noise_floor_dbm / 10.0)
        jam_mw = 10 ** (jam_rx_dbm / 10.0)
        total_mw = noise_mw + jam_mw
        return float(10.0 * np.log10(total_mw))

    def _per_matrix(
        self,
        positions: np.ndarray,
        t: float,
        jam_active: bool,
        jam_power_w: float,
        jam_pos: np.ndarray | None,
    ) -> np.ndarray:
        """(N, N) packet error rate matrix.

        PER model: sigmoid-like mapping from SNR.
        PER ≈ 0 for high SNR, PER ≈ 1 for SNR << threshold.
        Concretely:
            PER(snr_db) = 1 / (1 + exp(slope * (snr_db - threshold_db)))
        This is the logistic function centred at the threshold SNR.
        """
        n = positions.shape[0]
        snr = self.snr_matrix(positions, t, jam_active, jam_power_w, jam_pos)
        per = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                snr_ij = snr[i, j]
                per[i, j] = 1.0 / (
                    1.0 + np.exp(
                        self.per_snr_slope * (snr_ij - self.per_snr_threshold_db)
                    )
                )
        return per
