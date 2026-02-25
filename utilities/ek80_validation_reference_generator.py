#!/usr/bin/env python3
"""
ek80_validation_reference_generator.py

Generates fill-outable CSV reference files for cross-validating the EK80
processing pipeline (EK80_pipeline_01.ipynb) against Simrad EK80 desktop
software.

Outputs (in ../Data/ relative to this script's location):
  1. validation_ping_ts120_timeseries.csv
     - ~15 pings sampled across the recording (moving + stationary)
     - Our TS(f) at 120 kHz + echopype broadband TS
     - Blank column for EK80 software values

  2. validation_tsf_detailed.csv
     - TS(f) at 10 key frequencies for 3 selected pings
     - Frequencies: 90, 100, 110, 115, 120, 130, 134, 136, 140, 150 kHz
     - Blank column for EK80 software values

Usage:
    python ek80_validation_reference_generator.py
"""

import csv
import os
import numpy as np
from scipy.signal import convolve as sig_convolve
import echopype as ep
from echopype.calibrate.ek80_complex import get_filter_coeff

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_FILE = os.path.join(PROJECT_ROOT, 'Data', 'prod-D20250904-T165452.raw')
UTC_OFFSET_HOURS = -4
TARGET_BAND = [1.293, 1.793]       # target range gate (m)
NFFT = 4096
ALPHA_DBM = 0.04                   # absorption, dB/m
Z_ET = 75.0                        # transducer impedance (echopype default)

# Key frequencies for TS(f) comparison
TSF_FREQS_KHZ = [90, 100, 110, 115, 120, 130, 134, 136, 140, 150]

OUT_DIR = os.path.join(PROJECT_ROOT, 'Data')

# ═══════════════════════════════════════════════════════════════════
# Load raw file
# ═══════════════════════════════════════════════════════════════════
print("Loading raw file...")
ed = ep.open_raw(RAW_FILE, sonar_model='EK80')

beam_ds = ed['Sonar/Beam_group1']
env = ed['Environment']
vs = ed['Vendor_specific']

ping_times = beam_ds['ping_time'].values
n_pings = len(ping_times)
n_samples = beam_ds.dims['range_sample']
n_beams = beam_ds.dims['beam']

f_start = float(beam_ds['transmit_frequency_start'].values[0, 0])
f_stop = float(beam_ds['transmit_frequency_stop'].values[0, 0])
t_dur = float(beam_ds['transmit_duration_nominal'].values[0, 0])
slope_val = float(beam_ds['slope'].values[0, 0])
fs_rx = float(vs['receiver_sampling_frequency'].values[0])
sample_int = float(beam_ds['sample_interval'].values[0, 0])
fs_eff = 1.0 / sample_int
c_sw = float(env['sound_speed_indicative'].values[0])
Ptx = float(beam_ds['transmit_power'].values[0, 0])
z_er = float(vs['impedance_transceiver'].values[0].item())
gain = float(vs['gain_correction'].values[0, 0])

r_raw = np.arange(n_samples) * sample_int * c_sw / 2

# Local times
local_times = ping_times + np.timedelta64(int(UTC_OFFSET_HOURS * 3600), 's')

print(f"  {n_pings} pings, {n_samples} samples/ping, {n_beams} beams")
print(f"  Local time: {str(local_times[0])[:19]} to {str(local_times[-1])[:19]}")
print(f"  f_start={f_start/1e3:.0f} kHz, f_stop={f_stop/1e3:.0f} kHz, "
      f"tau={t_dur*1e3:.3f} ms, Ptx={Ptx:.0f} W, gain={gain:.1f} dB")

# ═══════════════════════════════════════════════════════════════════
# Compute echopype broadband TS (reference)
# ═══════════════════════════════════════════════════════════════════
print("\nComputing echopype broadband TS...")
ds_TS = ep.calibrate.compute_TS(ed, waveform_mode='BB', encode_mode='complex')
TS = ds_TS['TS'].values[0]

j0t = np.searchsorted(r_raw, TARGET_BAND[0])
j1t = np.searchsorted(r_raw, TARGET_BAND[1], side='right')
ts_peak_broadband = np.nanmax(TS[:, j0t:j1t], axis=1)

# ═══════════════════════════════════════════════════════════════════
# Build chirp replica (same as EK80_pipeline_01 cell 19)
# ═══════════════════════════════════════════════════════════════════
print("Building chirp replica...")

coeff = get_filter_coeff(vs)
ch_id = list(coeff.keys())[0]
wbt_fil = coeff[ch_id]['wbt_fil']
wbt_decifac = coeff[ch_id]['wbt_decifac']
pc_fil = coeff[ch_id]['pc_fil']
pc_decifac = coeff[ch_id]['pc_decifac']

# RF chirp
n_rf = int(np.floor(t_dur * fs_rx))
t_rf = np.arange(n_rf) / fs_rx
y_rf = np.cos(np.pi * (f_stop - f_start) / t_dur * t_rf**2
              + 2 * np.pi * f_start * t_rf)

# Hanning edge taper
L_taper = round(t_dur * fs_rx * slope_val * 2.0)
w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(L_taper) / (L_taper - 1)))
w1 = w[:L_taper // 2]
w2 = w[L_taper // 2:-1]
y_rf[:len(w1)] *= w1
y_rf[-len(w2):] *= w2
y_rf /= np.max(np.abs(y_rf))

# WBT filter + decimate
ytx_wbt = sig_convolve(y_rf, wbt_fil)
ytx_wbt_dec = ytx_wbt[::wbt_decifac]

# PC filter + decimate
ytx_pc = sig_convolve(ytx_wbt_dec, pc_fil)
tx_filtered = ytx_pc[::pc_decifac]

norm_fac = np.linalg.norm(tx_filtered) ** 2
mf_replica = np.flipud(np.conj(tx_filtered))

print(f"  Chirp: {len(tx_filtered)} samples, norm_fac={norm_fac:.4f}")

# ═══════════════════════════════════════════════════════════════════
# Aliasing-aware frequency mapping
# ═══════════════════════════════════════════════════════════════════
print("Building frequency mapping...")

f_bb_fft = np.fft.fftfreq(NFFT, d=1.0 / fs_eff)
f_actual_map = np.full(NFFT, np.nan)

for i, fb in enumerate(f_bb_fft):
    for k in range(-10, 11):
        f_try = fb + k * fs_eff
        if f_start * 0.90 <= f_try <= f_stop * 1.10:
            f_actual_map[i] = f_try
            break

band_idx = np.where(~np.isnan(f_actual_map))[0]
sort_order = np.argsort(f_actual_map[band_idx])
band_idx_sorted = band_idx[sort_order]
f_band_sorted = f_actual_map[band_idx_sorted]

# TX power spectrum for deconvolution
TX_fft = np.fft.fft(tx_filtered, n=NFFT)
TX_power_band = np.abs(TX_fft[band_idx_sorted]) ** 2
TX_power_band = np.maximum(TX_power_band, np.max(TX_power_band) * 1e-10)

# Impedance/beam calibration factor
impedance_factor_dB = (10 * np.log10(n_beams / 8)
                       + 20 * np.log10(np.abs(z_er + Z_ET) / z_er)
                       - 10 * np.log10(Z_ET))

print(f"  In-band bins: {len(band_idx_sorted)} of {NFFT}")
print(f"  Freq range: {f_band_sorted[0]/1e3:.1f} - {f_band_sorted[-1]/1e3:.1f} kHz")

# ═══════════════════════════════════════════════════════════════════
# Processing functions
# ═══════════════════════════════════════════════════════════════════

def process_ping_pc(ping_idx):
    """Pulse-compress a single ping (mean-beam)."""
    raw = (beam_ds['backscatter_r'].values[0, ping_idx, :, :]
           + 1j * beam_ds['backscatter_i'].values[0, ping_idx, :, :])
    raw_sig = raw.mean(axis=1)
    pc = sig_convolve(raw_sig, mf_replica, mode='full')[len(mf_replica)-1:][:n_samples]
    return pc / norm_fac


def compute_absolute_tsf(ping_idx):
    """Compute absolute TS(f) for one ping. Returns (freq_hz, ts_dB)."""
    pc = process_ping_pc(ping_idx)

    j0 = np.searchsorted(r_raw, TARGET_BAND[0])
    j1 = np.searchsorted(r_raw, TARGET_BAND[1], side='right')
    gated = pc[j0:j1]
    if len(gated) > 1:
        gated = gated * np.hanning(len(gated))
    spec = np.fft.fft(gated, n=NFFT)

    R_center = np.mean(TARGET_BAND)
    H = spec[band_idx_sorted] / TX_power_band
    H_dB = 10 * np.log10(np.abs(H)**2 + 1e-30)
    H_dB += 20 * np.log10(norm_fac)
    H_dB += impedance_factor_dB
    H_dB += 40 * np.log10(max(R_center, 0.01))
    H_dB += 2 * ALPHA_DBM * R_center
    lam_f = c_sw / f_band_sorted
    H_dB -= 10 * np.log10(lam_f**2 * Ptx / (16 * np.pi**2))
    H_dB -= 2 * gain

    return f_band_sorted, H_dB


def interp_tsf(f_hz, ts_dB, target_freqs_hz):
    """Interpolate TS(f) curve to specific frequencies."""
    return np.interp(target_freqs_hz, f_hz, ts_dB)


# ═══════════════════════════════════════════════════════════════════
# Sampling plan
# ═══════════════════════════════════════════════════════════════════
print("\nDesigning sampling plan...")

# Stationary start: ~13:33 local = 17:33 UTC
stat_start_utc = np.datetime64('2025-09-04T17:33:00')
stat_mask = ping_times >= stat_start_utc
stat_start_idx = int(np.where(stat_mask)[0][0]) if stat_mask.any() else n_pings - 100

print(f"  Stationary from ping {stat_start_idx} "
      f"({str(local_times[stat_start_idx])[:19]} local)")
print(f"  Moving pings: 0 - {stat_start_idx - 1}")
print(f"  Stationary pings: {stat_start_idx} - {n_pings - 1}")

# Moving period: 10 evenly spaced
moving_indices = np.linspace(0, stat_start_idx - 1, 10, dtype=int)

# Stationary period: 5 evenly spaced
stat_indices = np.linspace(stat_start_idx, n_pings - 1, 5, dtype=int)

# Combined time series
ts_series_indices = np.unique(np.concatenate([moving_indices, stat_indices]))

# Detailed TS(f): 3 pings (early, mid-recording, mid-stationary)
tsf_detail_indices = [
    int(moving_indices[1]),      # early in recording
    int(moving_indices[5]),      # mid-recording
    int(stat_indices[2]),        # mid-stationary
]

print(f"  Time series pings: {len(ts_series_indices)}")
print(f"  TS(f) detail pings: {len(tsf_detail_indices)}")

# ═══════════════════════════════════════════════════════════════════
# Compute: ping time series (TS at 120 kHz)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("COMPUTING TS AT 120 kHz FOR TIME SERIES PINGS")
print("="*70)

ts120_results = []
for idx in ts_series_indices:
    idx = int(idx)
    f_hz, ts_dB = compute_absolute_tsf(idx)
    ts_at_120 = float(interp_tsf(f_hz, ts_dB, [120e3])[0])
    bb_ts = float(ts_peak_broadband[idx])
    local_str = str(local_times[idx])[:23]
    utc_str = str(ping_times[idx])[:23]
    is_stat = "Y" if idx >= stat_start_idx else "N"
    ts120_results.append((idx, utc_str, local_str, is_stat, ts_at_120, bb_ts))
    print(f"  Ping {idx:>5d}  {local_str}  "
          f"TS(120kHz)={ts_at_120:+7.1f} dB  "
          f"BB_peak={bb_ts:+7.1f} dB  "
          f"stat={is_stat}")

# ═══════════════════════════════════════════════════════════════════
# Compute: detailed TS(f) at key frequencies
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("COMPUTING DETAILED TS(f) AT KEY FREQUENCIES")
print("="*70)

target_freqs_hz = np.array([f * 1e3 for f in TSF_FREQS_KHZ])

tsf_results = []
for idx in tsf_detail_indices:
    idx = int(idx)
    f_hz, ts_dB = compute_absolute_tsf(idx)
    ts_at_freqs = interp_tsf(f_hz, ts_dB, target_freqs_hz)
    local_str = str(local_times[idx])[:23]
    utc_str = str(ping_times[idx])[:23]
    is_stat = "Y" if idx >= stat_start_idx else "N"

    print(f"\n  Ping {idx} — {local_str} (stat={is_stat})")
    for fk, tv in zip(TSF_FREQS_KHZ, ts_at_freqs):
        tsf_results.append((idx, utc_str, local_str, is_stat, fk, float(tv)))
        print(f"    {fk:>6d} kHz:  {tv:+7.1f} dB")

# ═══════════════════════════════════════════════════════════════════
# Write CSV files
# ═══════════════════════════════════════════════════════════════════
os.makedirs(OUT_DIR, exist_ok=True)

# File 1: Time series — TS at 120 kHz
ts_file = os.path.join(OUT_DIR, 'validation_ping_ts120_timeseries.csv')
with open(ts_file, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow([
        'ping_index',
        'utc_timestamp',
        'local_timestamp',
        'stationary',
        'pipeline_TS_120kHz_dB',
        'echopype_BB_TS_peak_dB',
        'ek80_software_TS_120kHz_dB'
    ])
    for row in ts120_results:
        idx, utc, local, stat, ts120, bb = row
        w.writerow([idx, utc, local, stat, f'{ts120:.2f}', f'{bb:.2f}', ''])

# File 2: Detailed TS(f) at key frequencies
tsf_file = os.path.join(OUT_DIR, 'validation_tsf_detailed.csv')
with open(tsf_file, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow([
        'ping_index',
        'utc_timestamp',
        'local_timestamp',
        'stationary',
        'freq_kHz',
        'pipeline_TS_dB',
        'ek80_software_TS_dB'
    ])
    for row in tsf_results:
        idx, utc, local, stat, freq, ts = row
        w.writerow([idx, utc, local, stat, freq, f'{ts:.2f}', ''])

# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("VALIDATION FILES READY")
print("="*70)
print(f"\n  1. {ts_file}")
print(f"     {len(ts120_results)} pings — fill in 'ek80_software_TS_120kHz_dB' column")
print(f"\n  2. {tsf_file}")
print(f"     {len(tsf_results)} rows ({len(tsf_detail_indices)} pings x "
      f"{len(TSF_FREQS_KHZ)} freqs) — fill in 'ek80_software_TS_dB' column")
print(f"\nOpen these CSVs in a spreadsheet and fill in the blank columns")
print(f"by reading values from the EK80 desktop software.")
print(f"\nTip: Use ping_index + utc_timestamp to locate pings in the EK80 software.")
