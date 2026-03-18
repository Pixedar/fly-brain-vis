"""
Spike data playback engine.

Loads simulation spike data from parquet files and provides time-windowed
firing rate / brightness arrays for driving the 3D renderer animation.
"""

import numpy as np
import pandas as pd
from pathlib import Path


class SpikePlayer:
    """Manages spike data playback with time-windowed brightness computation.

    Spikes are pre-binned into time slots for O(1) per-frame lookup.
    Brightness uses exponential decay from last spike time for smooth visuals.
    """

    def __init__(self, spike_path, neuron_index, trial=0,
                 time_window_ms=20.0, decay_ms=0.0, playback_speed=1.0,
                 dt_ms=16.0):
        """
        Args:
            spike_path: path to spike parquet file
            neuron_index: dict mapping flywire_id -> array index (0..N-1)
            trial: which trial to visualize (default: 0)
            time_window_ms: width of firing rate window
            decay_ms: exponential decay time constant for brightness
            playback_speed: multiplier for playback speed
            dt_ms: animation timestep in ms (default ~60fps)
        """
        self.neuron_index = neuron_index
        self.n_neurons = len(neuron_index)
        self.decay_ms = decay_ms
        self.playback_speed = playback_speed
        self.dt_ms = dt_ms
        self.time_window_ms = time_window_ms

        # Load spike data
        print(f"[visualizer] Loading spikes from {spike_path}")
        df = pd.read_parquet(spike_path)

        # Filter to selected trial
        if 'trial' in df.columns:
            df = df[df['trial'] == trial]

        # Map flywire IDs to indices
        self.spike_times = df['t'].values.astype(np.float64)  # ms
        fids = df['flywire_id'].values
        self.spike_neurons = np.array([
            neuron_index.get(int(fid), -1) for fid in fids
        ], dtype=np.int32)

        # Filter out unmapped neurons
        valid = self.spike_neurons >= 0
        self.spike_times = self.spike_times[valid]
        self.spike_neurons = self.spike_neurons[valid]

        # Sort by time
        order = np.argsort(self.spike_times)
        self.spike_times = self.spike_times[order]
        self.spike_neurons = self.spike_neurons[order]

        self.duration_ms = float(self.spike_times[-1]) if len(self.spike_times) > 0 else 0.0
        self.current_time_ms = 0.0
        self.is_playing = True

        # Pre-compute per-neuron last spike time for fast brightness
        self._last_spike = np.full(self.n_neurons, -1e6, dtype=np.float64)

        # Pre-bin spikes into time bins for fast window queries
        self._bin_width_ms = max(1.0, time_window_ms / 2.0)
        n_bins = int(np.ceil(self.duration_ms / self._bin_width_ms)) + 1
        self._bins = [[] for _ in range(n_bins)]
        for i, (t, n) in enumerate(zip(self.spike_times, self.spike_neurons)):
            b = int(t / self._bin_width_ms)
            if b < n_bins:
                self._bins[b].append(i)

        # Track which spikes we've already processed for last_spike updates
        self._spike_cursor = 0

        n_spikes = len(self.spike_times)
        n_active = len(set(self.spike_neurons))
        print(f"[visualizer] Loaded {n_spikes} spikes from {n_active} neurons, "
              f"duration={self.duration_ms:.1f} ms")

    def step(self):
        """Advance playback by one frame. Returns brightness array (0..1 per neuron)."""
        if not self.is_playing:
            return self.get_brightness()

        self.current_time_ms += self.dt_ms * self.playback_speed

        if self.current_time_ms >= self.duration_ms:
            self.current_time_ms = 0.0
            self._spike_cursor = 0
            self._last_spike[:] = -1e6

        # Update last spike times for all spikes up to current time
        while (self._spike_cursor < len(self.spike_times) and
               self.spike_times[self._spike_cursor] <= self.current_time_ms):
            neuron_idx = self.spike_neurons[self._spike_cursor]
            self._last_spike[neuron_idx] = self.spike_times[self._spike_cursor]
            self._spike_cursor += 1

        return self.get_brightness()

    def get_brightness(self, decay_ms=None):
        """Get per-neuron brightness (0..1) based on exponential decay from last spike.

        Args:
            decay_ms: optional override for decay time constant.
                      If None, uses the default self.decay_ms.
                      If 0, decay is disabled: brightness is 1.0 for any
                      neuron that spiked within the current time window,
                      0.0 otherwise (instant on/off).
        """
        d = decay_ms if decay_ms is not None else self.decay_ms
        dt = self.current_time_ms - self._last_spike
        if d <= 0:
            # No decay: binary on/off within time window
            brightness = np.where(dt <= self.time_window_ms, 1.0, 0.0)
        else:
            brightness = np.exp(-dt / d)
        brightness = np.clip(brightness, 0.0, 1.0)
        return brightness

    def get_firing_rates(self):
        """Get per-neuron firing rate (Hz) in the current time window."""
        rates = np.zeros(self.n_neurons, dtype=np.float64)
        t_start = max(0.0, self.current_time_ms - self.time_window_ms)
        t_end = self.current_time_ms

        bin_start = max(0, int(t_start / self._bin_width_ms))
        bin_end = min(len(self._bins) - 1, int(t_end / self._bin_width_ms))

        for b in range(bin_start, bin_end + 1):
            for si in self._bins[b]:
                t = self.spike_times[si]
                if t_start <= t <= t_end:
                    rates[self.spike_neurons[si]] += 1

        # Convert count to Hz
        window_sec = self.time_window_ms / 1000.0
        if window_sec > 0:
            rates /= window_sec

        return rates

    def reset(self):
        """Reset playback to start."""
        self.current_time_ms = 0.0
        self._spike_cursor = 0
        self._last_spike[:] = -1e6

    def set_speed(self, speed):
        """Set playback speed multiplier."""
        self.playback_speed = speed

    def toggle_pause(self):
        """Toggle play/pause."""
        self.is_playing = not self.is_playing
