#!/usr/bin/env python3
"""
generate_report_figures.py

Generates all PNG figures for the status report.

Run from echobot_v2/ project root:
    conda activate echobot
    python utilities/generate_report_figures.py

Output: status_reports/assets/*.png
"""

import os
import numpy as np
import scipy.io as sio
from scipy.signal import firwin, lfilter, correlate, convolve as sig_convolve
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA = os.path.join(ROOT, 'Data')
ASSETS = os.path.join(ROOT, 'status_reports', 'assets')
os.makedirs(ASSETS, exist_ok=True)

Nfft = 4096


# ═══════════════════════════════════════════════════════════════════
# Shared: Echobot SNR computation
# ═══════════════════════════════════════════════════════════════════
def compute_echobot_snr(mat_path):
    """Load echobot .mat, bandpass + matched filter, return per-ping floor SNR."""
    S = sio.loadmat(mat_path, squeeze_me=False)
    data = S['data']
    hdr = S['header'][0, 0]
    fs = float(hdr['fs'].flat[0])
    c = float(hdr['c'].flat[0]) if 'c' in hdr.dtype.names else 1486.0
    s_chirp = hdr['s_chirp'].flatten()
    T_pre = float(hdr['T_pre'].flat[0])
    T_post = float(hdr['T_post'].flat[0])

    Ns = data.shape[0]
    n_ch = min(data.shape[1], 3)
    n_pings = data.shape[2]
    R = 0.5 * c * np.arange(Ns) / fs

    n_pre = int(round(T_pre * fs))
    n_post = int(round(T_post * fs))
    tx_ref = np.concatenate([np.zeros(n_pre), s_chirp, np.zeros(n_post)])
    n_ref = len(tx_ref)

    lp = firwin(101, 175e3, fs=fs)
    hp = firwin(101, 80e3, fs=fs, pass_zero=False)

    j0n = np.searchsorted(R, 2.0)
    j1n = np.searchsorted(R, 2.5, side='right')
    j0f = np.searchsorted(R, 2.5)
    j1f = np.searchsorted(R, 5.0, side='right')

    snr = np.zeros(n_pings)
    for p in range(n_pings):
        secs = []
        for ch in range(n_ch):
            df = lfilter(hp, 1.0, lfilter(lp, 1.0, data[:, ch, p]))
            cc = correlate(df, tx_ref, mode='full')
            secs.append(cc[n_ref - 1: n_ref - 1 + Ns])
        env = np.abs(sum(secs))
        snr[p] = 20 * np.log10(
            np.max(env[j0f:j1f]) / (np.mean(env[j0n:j1n]) + 1e-30))
    return snr


def make_summary_figure(results, test_configs, title, save_path):
    """Bar chart + summary table for a set of SNR test results."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                             gridspec_kw={'height_ratios': [2, 1]})
    labels = list(results.keys())
    means = [results[k].mean() for k in labels]
    stds = [results[k].std() for k in labels]
    x = np.arange(len(labels))
    colors = ['C0' if m > 20 else 'C3' for m in means]

    ax = axes[0]
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor='k',
           linewidth=0.5, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i+1}' for i in range(len(labels))], fontsize=9)
    ax.set_ylabel('Mean Floor SNR (dB)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 1.5, f'{m:.1f}', ha='center', va='bottom', fontsize=8)

    ax = axes[1]
    ax.axis('off')
    tbl = [['Test', 'Configuration', 'Pings', 'Mean SNR', 'Std', 'Min', 'Max']]
    for i, k in enumerate(labels):
        snr = results[k]
        tbl.append([
            f'{i+1}', test_configs[i][1], f'{test_configs[i][2]}',
            f'{snr.mean():.1f} dB', f'{snr.std():.2f} dB',
            f'{snr.min():.1f} dB', f'{snr.max():.1f} dB'
        ])
    table = ax.table(cellText=tbl, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    for j in range(len(tbl[0])):
        table[0, j].set_facecolor('#d4e6f1')
        table[0, j].set_text_props(fontweight='bold')
    ax.set_title('Summary Table', fontsize=11, pad=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# PART A: Echobot CRL Test SNR Figures
# ═══════════════════════════════════════════════════════════════════

# --- Figure 1: Sept 4 Test 1 (downsweep, center, stationary) ---
print('[1/9] 0904 Test 1 SNR example...')
snr_t1 = compute_echobot_snr(
    os.path.join(DATA, '0904-CRL-tests',
                 'backcyl_bis_rgh0.01271_T143330_100.mat'))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(snr_t1, 'o-', ms=3, color='C0', alpha=0.7)
axes[0].axhline(snr_t1.mean(), color='k', ls='--', lw=0.8,
                label=f'mean = {snr_t1.mean():.1f} dB')
axes[0].set_xlabel('Ping index')
axes[0].set_ylabel('SNR (dB)')
axes[0].set_title('Test 1: Downsweep, center, stationary (100 pings)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(snr_t1, bins=15, color='C0', edgecolor='k', alpha=0.7)
axes[1].axvline(snr_t1.mean(), color='k', ls='--', lw=0.8)
axes[1].set_xlabel('SNR (dB)')
axes[1].set_ylabel('Count')
axes[1].set_title(f'Distribution (mean={snr_t1.mean():.1f}, '
                  f'std={snr_t1.std():.2f} dB)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(ASSETS, 'fig_1b_snr_example.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f'  Done. mean={snr_t1.mean():.1f} dB')

# --- Figure 2: Nov 5 Summary ---
print('[2/9] Nov 5 summary (10 tests)...')
tests_1105 = [
    ('backcyl_bis_rgh0.01271_T120308_020.mat', 'No target, upsweep', 20),
    ('backcyl_bis_rgh0.01271_T120550_020.mat', 'No target, downsweep', 20),
    ('backcyl_bis_rgh0.01271_T121721_020.mat', '38mm WC stat., downsweep', 20),
    ('backcyl_bis_rgh0.01271_T122733_020.mat', 'Dummy load (65\u03A9), downsweep', 20),
    ('backcyl_bis_rgh0.01271_T123400_020.mat', 'Amp unpowered, downsweep', 20),
    ('backcyl_bis_rgh0.01271_T124029_100.mat', '38mm WC mov., downsweep', 100),
    ('backcyl_bis_rgh0.01271_T125313_100.mat', '38mm WC mov., upsweep', 100),
    ('backcyl_bis_rgh0.01271_T130710_100.mat', '22mm WC mov., upsweep', 100),
    ('backcyl_bis_rgh0.01271_T131215_100.mat', '22mm WC mov., downsweep', 100),
    ('backcyl_bis_rgh0.01271_T131647_100.mat', '22mm WC mov., \u221280dB TX', 100),
]
results_1105 = {}
for i, (fname, desc, _) in enumerate(tests_1105):
    print(f'  Test {i+1}/10: {desc}')
    results_1105[f'Test {i+1}'] = compute_echobot_snr(
        os.path.join(DATA, '1105-CRL-tests', fname))

make_summary_figure(results_1105, tests_1105,
                    'Nov 5, 2025 \u2014 Tank Floor SNR Across All Tests',
                    os.path.join(ASSETS, 'fig_1b_summary_1105.png'))
print('  Done.')

# --- Figure 3: Oct 15 Summary ---
print('[3/9] Oct 15 summary (5 tests)...')
tests_1015 = [
    ('backcyl_bis_rgh0.01271_T105726_100.mat', '0.0 dB TX, target moving', 100),
    ('backcyl_bis_rgh0.01271_T110424_100.mat', '0.0 dB TX, target still', 100),
    ('backcyl_bis_rgh0.01271_T110817_100.mat', '+3.0 dB TX, target still', 100),
    ('backcyl_bis_rgh0.01271_T111929_100.mat', '+3.0 dB TX, no target', 100),
    ('backcyl_bis_rgh0.01271_T112219_100.mat', '\u221240.0 dB TX, no target', 100),
]
results_1015 = {}
for i, (fname, desc, _) in enumerate(tests_1015):
    print(f'  Test {i+1}/5: {desc}')
    results_1015[f'Test {i+1}'] = compute_echobot_snr(
        os.path.join(DATA, '1015-CRL-tests', fname))

make_summary_figure(results_1015, tests_1015,
                    'Oct 15, 2025 \u2014 Tank Floor SNR Across All Tests',
                    os.path.join(ASSETS, 'fig_1b_summary_1015.png'))
print('  Done.')


# ═══════════════════════════════════════════════════════════════════
# PART B: EK80 Processing (shared for figures 4, 8, 9)
# ═══════════════════════════════════════════════════════════════════
print('[4/9] EK80 processing (this may take a few minutes)...')
import echopype as ep
from echopype.calibrate.ek80_complex import get_filter_coeff

raw_file = os.path.join(DATA, 'prod-D20250904-T165452.raw')
ed = ep.open_raw(raw_file, sonar_model='EK80')

beam_ds = ed['Sonar/Beam_group1']
vs = ed['Vendor_specific']

ping_times = beam_ds.ping_time.values
n_pings_ek = len(ping_times)
n_samples = beam_ds.dims['range_sample']
n_beams = beam_ds.dims['beam']

sample_int = float(beam_ds['sample_interval'].values[0, 0])
fs_eff = 1.0 / sample_int
f_start = float(beam_ds['transmit_frequency_start'].values[0, 0])
f_stop = float(beam_ds['transmit_frequency_stop'].values[0, 0])
t_dur = float(beam_ds['transmit_duration_nominal'].values[0, 0])
slope_val = float(beam_ds['slope'].values[0, 0])
fs_rx = float(vs['receiver_sampling_frequency'].values[0])
c_sw = float(ed['Environment']['sound_speed_indicative'].values[0])
gain_ek = float(vs['gain_correction'].values[0, 0])
Ptx = float(beam_ds['transmit_power'].values[0, 0])
z_er = float(vs['impedance_transceiver'].values[0].item())
z_et = 75.0
alpha_dBm = 0.04

# Build chirp replica: RF -> WBT filter -> PC filter
n_rf = int(np.floor(t_dur * fs_rx))
t_rf = np.arange(n_rf) / fs_rx
y_rf = np.cos(np.pi * (f_stop - f_start) / t_dur * t_rf**2
              + 2 * np.pi * f_start * t_rf)
L_taper = int(np.round(t_dur * fs_rx * slope_val * 2.0))
if L_taper > 1:
    w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(L_taper) / (L_taper - 1)))
    y_rf[:len(w[:L_taper // 2])] *= w[:L_taper // 2]
    w2 = w[L_taper // 2:-1]
    if len(w2):
        y_rf[-len(w2):] *= w2
y_rf /= np.max(np.abs(y_rf))

coeff = get_filter_coeff(vs)
ch_id = list(coeff.keys())[0]
wbt_fil, pc_fil = coeff[ch_id]['wbt_fil'], coeff[ch_id]['pc_fil']
wbt_dec, pc_dec = int(coeff[ch_id]['wbt_decifac']), int(coeff[ch_id]['pc_decifac'])

tx_filtered = sig_convolve(sig_convolve(y_rf, wbt_fil)[::wbt_dec], pc_fil)[::pc_dec]
norm_fac = np.linalg.norm(tx_filtered) ** 2
mf_replica = np.flipud(np.conj(tx_filtered))
r_raw = np.arange(n_samples) * sample_int * c_sw / 2

# Aliasing-aware frequency mapping
f_bb = np.fft.fftfreq(Nfft, d=1.0 / fs_eff)
f_actual = np.full(Nfft, np.nan)
for i, fb in enumerate(f_bb):
    for k in range(-10, 11):
        ft = fb + k * fs_eff
        if f_start * 0.90 <= ft <= f_stop * 1.10:
            f_actual[i] = ft
            break

band_idx = np.where(~np.isnan(f_actual))[0]
order = np.argsort(f_actual[band_idx])
band_idx_s = band_idx[order]
f_ek_hz = f_actual[band_idx_s]

TX_fft = np.fft.fft(tx_filtered, n=Nfft)
TX_pow_ek = np.maximum(np.abs(TX_fft[band_idx_s]) ** 2,
                       np.max(np.abs(TX_fft[band_idx_s]) ** 2) * 1e-10)
plot_mask_ek = (f_ek_hz >= f_start * 0.97) & (f_ek_hz <= f_stop * 1.03)

# Calibration
imp_dB = (10 * np.log10(n_beams / 8)
          + 20 * np.log10(np.abs(z_er + z_et) / z_er)
          - 10 * np.log10(z_et))


def calibrate_tsf(spec_avg, R_ctr):
    H = spec_avg[band_idx_s] / TX_pow_ek
    H_dB = 10 * np.log10(np.abs(H) ** 2 + 1e-30)
    H_dB += 20 * np.log10(norm_fac) + imp_dB
    H_dB += 40 * np.log10(max(R_ctr, 0.01)) + 2 * alpha_dBm * R_ctr
    H_dB -= 10 * np.log10((c_sw / f_ek_hz) ** 2 * Ptx / (16 * np.pi ** 2))
    H_dB -= 2 * gain_ek
    return H_dB


# Range gates
target_band_tsf = [1.32, 1.82]
noise_band_tsf = [0.40, 0.70]
snr_noise_gate = [2.0, 2.5]
snr_floor_gate = [2.5, 5.0]
R_target_center = np.mean(target_band_tsf)
R_noise_center = np.mean(noise_band_tsf)

# SNR gate indices
j0n_ek = np.searchsorted(r_raw, snr_noise_gate[0])
j1n_ek = np.searchsorted(r_raw, snr_noise_gate[1], side='right')
j0f_ek = np.searchsorted(r_raw, snr_floor_gate[0])
j1f_ek = np.searchsorted(r_raw, snr_floor_gate[1], side='right')


def gate_fft(pc, gate, nfft=4096):
    j0 = np.searchsorted(r_raw, gate[0])
    j1 = np.searchsorted(r_raw, gate[1], side='right')
    g = pc[j0:j1] * np.hanning(j1 - j0)
    return np.fft.fft(g, n=nfft)


# Stationary mask
stat_mask_ek = ping_times >= np.datetime64('2025-09-04T17:40:00')
stat_idx_ek = np.where(stat_mask_ek)[0]

# --- Single pass through all EK80 pings ---
spec_tgt_all = np.zeros(Nfft, dtype=complex)
spec_noi_all = np.zeros(Nfft, dtype=complex)
spec_tgt_stat = np.zeros(Nfft, dtype=complex)
spec_noi_stat = np.zeros(Nfft, dtype=complex)
snr_ek_all = np.zeros(n_pings_ek)
n_stat_ek = 0

for i in range(n_pings_ek):
    if i % 200 == 0:
        print(f'  EK80 ping {i+1}/{n_pings_ek}...', end='\r')
    raw = (beam_ds['backscatter_r'].values[0, i, :, :]
           + 1j * beam_ds['backscatter_i'].values[0, i, :, :]).mean(axis=1)
    pc = sig_convolve(raw, mf_replica, mode='full')[len(mf_replica)-1:][:n_samples]
    pc = pc / norm_fac

    # SNR
    env = np.abs(pc)
    snr_ek_all[i] = 20 * np.log10(
        np.max(env[j0f_ek:j1f_ek]) / (np.mean(env[j0n_ek:j1n_ek]) + 1e-30))

    # TS(f) accumulation
    spec_tgt_all += gate_fft(pc, target_band_tsf)
    spec_noi_all += gate_fft(pc, noise_band_tsf)
    if stat_mask_ek[i]:
        spec_tgt_stat += gate_fft(pc, target_band_tsf)
        spec_noi_stat += gate_fft(pc, noise_band_tsf)
        n_stat_ek += 1

print(f'  EK80: {n_pings_ek} pings processed ({n_stat_ek} stationary)    ')
snr_ek_stat = snr_ek_all[stat_mask_ek]

# Calibrated TS(f)
TSf_tgt_all = calibrate_tsf(spec_tgt_all / n_pings_ek, R_target_center)
TSf_noi_all = calibrate_tsf(spec_noi_all / n_pings_ek, R_noise_center)
TSf_tgt_stat = calibrate_tsf(spec_tgt_stat / n_stat_ek, R_target_center)
TSf_noi_stat = calibrate_tsf(spec_noi_stat / n_stat_ek, R_noise_center)

# --- Figure 4: EK80 TS(f) ---
CLR_TARGET = '#1f77b4'
CLR_NOISE = '#2ca02c'

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, tgt, noi, ttl in [
    (axes[0], TSf_tgt_all, TSf_noi_all, 'TS(f) \u2014 All Pings'),
    (axes[1], TSf_tgt_stat, TSf_noi_stat, 'TS(f) \u2014 Stationary Only'),
]:
    ax.plot(f_ek_hz[plot_mask_ek] / 1e3, noi[plot_mask_ek],
            color=CLR_NOISE, lw=1, label='noise gate')
    ax.plot(f_ek_hz[plot_mask_ek] / 1e3, tgt[plot_mask_ek],
            color=CLR_TARGET, lw=1, label='target')
    ax.set_xlim([f_start / 1e3 - 5, f_stop / 1e3 + 5])
    ax.set_ylim([-80, -20])
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('TS (dB re 1 m\u00b2)')
    ax.set_title(ttl)
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(ASSETS, 'fig_2a_ek80_tsf.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print('  Figure 4 saved.')


# ═══════════════════════════════════════════════════════════════════
# PART C: Validation Figures (CSV-based)
# ═══════════════════════════════════════════════════════════════════
print('[5-7/9] Validation figures...')
ts_df = pd.read_csv(os.path.join(DATA, 'validation_ping_ts120_timeseries.csv'))
tsf_df = pd.read_csv(os.path.join(DATA, 'validation_tsf_detailed.csv'))

ts_df['residual_pipeline'] = ts_df['pipeline_TS_120kHz_dB'] - ts_df['ek80_software_TS_120kHz_dB']
ts_df['residual_echopype'] = ts_df['echopype_BB_TS_peak_dB'] - ts_df['ek80_software_TS_120kHz_dB']
tsf_df['residual'] = tsf_df['pipeline_TS_dB'] - tsf_df['ek80_software_TS_dB']

stat_v = ts_df[ts_df['stationary'] == 'Y']

# --- Figure 5: Ping-by-ping TS at 120 kHz ---
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                         gridspec_kw={'height_ratios': [3, 1]})
ax = axes[0]
ax.plot(ts_df['ping_index'], ts_df['ek80_software_TS_120kHz_dB'],
        'ko-', ms=6, label='EK80 Software (reference)')
ax.plot(ts_df['ping_index'], ts_df['pipeline_TS_120kHz_dB'],
        's-', color='C0', ms=5, label='Our Pipeline TS(f) @ 120 kHz')
ax.plot(ts_df['ping_index'], ts_df['echopype_BB_TS_peak_dB'],
        '^-', color='C1', ms=5, alpha=0.7, label='echopype BB TS (peak in gate)')
if len(stat_v) > 0:
    ax.axvspan(stat_v['ping_index'].min(), stat_v['ping_index'].max(),
               alpha=0.08, color='green', label='Stationary period')
ax.set_ylabel('TS (dB re 1 m\u00b2)')
ax.legend(loc='lower right', fontsize=9)
ax.set_title('Ping-by-Ping TS Comparison at 120 kHz')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.bar(ts_df['ping_index'] - 20, ts_df['residual_pipeline'], width=40,
       color='C0', alpha=0.7, label='Pipeline \u2212 EK80')
ax.bar(ts_df['ping_index'] + 20, ts_df['residual_echopype'], width=40,
       color='C1', alpha=0.7, label='echopype \u2212 EK80')
ax.axhline(0, color='k', lw=0.8)
ax.set_ylabel('Residual (dB)')
ax.set_xlabel('Ping Index')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(ASSETS, 'fig_2c_validation_s1.png'),
            dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 6: TS(f) spectral comparison ---
pings_v = tsf_df['ping_index'].unique()
fig, axes_v = plt.subplots(1, len(pings_v), figsize=(5 * len(pings_v), 5),
                           sharey=True)
if len(pings_v) == 1:
    axes_v = [axes_v]
for ax, pid in zip(axes_v, pings_v):
    sub = tsf_df[tsf_df['ping_index'] == pid]
    sflag = sub['stationary'].iloc[0] if 'stationary' in sub.columns else '?'
    ax.plot(sub['freq_kHz'], sub['ek80_software_TS_dB'],
            'ko-', ms=6, label='EK80 Software')
    ax.plot(sub['freq_kHz'], sub['pipeline_TS_dB'],
            's-', color='C0', ms=5, label='Our Pipeline')
    ax.set_title(f'Ping {pid} ({"stationary" if sflag == "Y" else "moving"})',
                 fontsize=10)
    ax.set_xlabel('Frequency (kHz)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
axes_v[0].set_ylabel('TS (dB re 1 m\u00b2)')
fig.suptitle('TS(f) Spectral Shape Comparison', fontsize=12, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(ASSETS, 'fig_2c_validation_s2.png'),
            dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 7: Spectral residuals ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
markers = ['o', 's', 'D']
ax = axes[0]
for i, pid in enumerate(pings_v):
    sub = tsf_df[tsf_df['ping_index'] == pid]
    sflag = 'stat' if sub.get('stationary', pd.Series(['?'])).iloc[0] == 'Y' else 'mov'
    ax.plot(sub['freq_kHz'], sub['residual'], f'{markers[i % 3]}-',
            ms=6, label=f'Ping {pid} ({sflag})')
ax.axhline(0, color='k', lw=1, ls='--')
ax.set_xlabel('Frequency (kHz)')
ax.set_ylabel('Residual: Pipeline \u2212 EK80 (dB)')
ax.set_title('Frequency-Dependent Offset')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
# Use last 2 pings (stationary, cleaner) for mean residual
clean_pings = pings_v[-2:] if len(pings_v) >= 2 else pings_v
sub_clean = tsf_df[tsf_df['ping_index'].isin(clean_pings)]
mean_resid = sub_clean.groupby('freq_kHz')['residual'].mean()
colors_bar = ['C3' if r > 0 else 'C0' for r in mean_resid.values]
ax.bar(mean_resid.index, mean_resid.values, width=3, color=colors_bar,
       alpha=0.7, edgecolor='k', linewidth=0.5)
ax.axhline(0, color='k', lw=1, ls='--')
ax.set_xlabel('Frequency (kHz)')
ax.set_ylabel('Mean Residual (dB)')
ax.set_title('Mean Offset (stationary pings)\nBlue = lower, Red = higher')
ax.grid(True, alpha=0.3)

if len(mean_resid) > 0:
    lo_mean = mean_resid.loc[mean_resid.index <= 115].mean()
    hi_mean = mean_resid.loc[mean_resid.index >= 130].mean()
    ax.text(0.02, 0.95,
            f'Mean 90\u2013115 kHz: {lo_mean:+.1f} dB\n'
            f'Mean 130\u2013150 kHz: {hi_mean:+.1f} dB\n'
            f'Tilt: {hi_mean - lo_mean:+.1f} dB',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
fig.savefig(os.path.join(ASSETS, 'fig_2c_validation_s3.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print('  Validation figures saved.')


# ═══════════════════════════════════════════════════════════════════
# PART D: Comparison Figures
# ═══════════════════════════════════════════════════════════════════

# --- Echobot deconvolution (ping 0) ---
print('[8/9] Echobot deconvolution + TS(f) comparison...')
mat_file = os.path.join(DATA, 'backcyl_bis_rgh0.01271_T143330_100.mat')
S = sio.loadmat(mat_file, squeeze_me=False)
data_eb = S['data']
hdr_eb = S['header'][0, 0]
fs_eb = float(hdr_eb['fs'].flat[0])
c_eb = float(hdr_eb['c'].flat[0]) if 'c' in hdr_eb.dtype.names else 1486.0
s_chirp_eb = hdr_eb['s_chirp'].flatten()
T_pre_eb = float(hdr_eb['T_pre'].flat[0])
T_post_eb = float(hdr_eb['T_post'].flat[0])

Ns_eb = data_eb.shape[0]
Rmf_eb = 0.5 * c_eb * np.arange(Ns_eb) / fs_eb

n_pre_eb = int(round(T_pre_eb * fs_eb))
n_post_eb = int(round(T_post_eb * fs_eb))
tx_ref_eb = np.concatenate([np.zeros(n_pre_eb), s_chirp_eb, np.zeros(n_post_eb)])
n_ref_eb = len(tx_ref_eb)

lp_eb = firwin(101, 175e3, fs=fs_eb)
hp_eb = firwin(101, 80e3, fs=fs_eb, pass_zero=False)

# Process ping 0: BP -> MF -> sum 3 sectors
C_sectors = []
for ch in range(3):
    df = lfilter(hp_eb, 1.0, lfilter(lp_eb, 1.0, data_eb[:, ch, 0]))
    cc = correlate(df, tx_ref_eb, mode='full')
    C_sectors.append(cc[n_ref_eb - 1: n_ref_eb - 1 + Ns_eb])
C_eb_sum = sum(C_sectors)

# Gate target, window, FFT, deconvolve by chirp
target_band_eb = [1.32, 1.82]
j0_eb = np.searchsorted(Rmf_eb, target_band_eb[0])
j1_eb = np.searchsorted(Rmf_eb, target_band_eb[1], side='right')
gated_eb = C_eb_sum[j0_eb:j1_eb] * np.hanning(j1_eb - j0_eb)

f_eb_hz = np.arange(Nfft) * (fs_eb / Nfft)
FFT_eb = np.fft.fft(gated_eb, n=Nfft)
TX_chirp_fft = np.fft.fft(s_chirp_eb, n=Nfft)
TX_pow_eb = np.maximum(np.abs(TX_chirp_fft) ** 2,
                       np.max(np.abs(TX_chirp_fft) ** 2) * 1e-10)
H_eb = FFT_eb / TX_pow_eb
H_eb_dB = 20 * np.log10(np.abs(H_eb) + 1e-30)

eb_band = (f_eb_hz >= 90e3) & (f_eb_hz <= 150e3)
f_eb_band = f_eb_hz[eb_band]
H_eb_band = H_eb_dB[eb_band]

# --- Figure 8: Spectral shape comparison (normalized to 120 kHz) ---
f_common = np.arange(93e3, 147.5e3, 500)
interp_eb_fn = interp1d(f_eb_band, H_eb_band, kind='linear', bounds_error=False)
H_eb_common = interp_eb_fn(f_common)
interp_ek_fn = interp1d(f_ek_hz[plot_mask_ek], TSf_tgt_stat[plot_mask_ek],
                        kind='linear', bounds_error=False)
TSf_ek_common = interp_ek_fn(f_common)

idx_120 = np.argmin(np.abs(f_common - 120e3))
eb_norm = H_eb_common - H_eb_common[idx_120]
ek_norm = TSf_ek_common - TSf_ek_common[idx_120]

fig, axes = plt.subplots(2, 1, figsize=(13, 8),
                         gridspec_kw={'height_ratios': [3, 1]})
ax = axes[0]
ax.plot(f_common / 1e3, ek_norm, 'C1', lw=1.5, label='EK80 (stationary avg)')
ax.plot(f_common / 1e3, eb_norm, 'C0', lw=1.5, alpha=0.8,
        label='Echobot (ping 0, deconvolved)')
ax.set_ylabel('TS relative to 120 kHz (dB)')
ax.set_title('Spectral Shape Comparison: Echobot vs EK80')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([93, 147])

ax = axes[1]
resid = eb_norm - ek_norm
ax.plot(f_common / 1e3, resid, 'k', lw=0.8)
ax.axhline(0, color='r', ls='--', lw=0.8)
ax.fill_between(f_common / 1e3, resid, 0, alpha=0.15, color='gray')
ax.set_xlabel('Frequency (kHz)')
ax.set_ylabel('Echobot \u2212 EK80 (dB)')
ax.set_title('Shape Residual (normalized to 120 kHz)')
ax.grid(True, alpha=0.3)
ax.set_xlim([93, 147])

plt.tight_layout()
fig.savefig(os.path.join(ASSETS, 'fig_3a_tsf_comparison.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print('  Figure 8 saved.')

# --- Figure 9: SNR comparison ---
print('[9/9] SNR comparison...')
Npings_eb = data_eb.shape[2]

j0n_eb2 = np.searchsorted(Rmf_eb, snr_noise_gate[0])
j1n_eb2 = np.searchsorted(Rmf_eb, snr_noise_gate[1], side='right')
j0f_eb2 = np.searchsorted(Rmf_eb, snr_floor_gate[0])
j1f_eb2 = np.searchsorted(Rmf_eb, snr_floor_gate[1], side='right')

snr_eb_comp = np.zeros(Npings_eb)
for p in range(Npings_eb):
    secs = []
    for ch in range(3):
        df = lfilter(hp_eb, 1.0, lfilter(lp_eb, 1.0, data_eb[:, ch, p]))
        cc = correlate(df, tx_ref_eb, mode='full')
        secs.append(cc[n_ref_eb - 1: n_ref_eb - 1 + Ns_eb])
    env_p = np.abs(sum(secs))
    snr_eb_comp[p] = 20 * np.log10(
        np.max(env_p[j0f_eb2:j1f_eb2]) / (np.mean(env_p[j0n_eb2:j1n_eb2]) + 1e-30))

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

ax = axes[0, 0]
ax.plot(np.arange(Npings_eb), snr_eb_comp, 'o-', ms=3, color='C0', alpha=0.7)
ax.axhline(snr_eb_comp.mean(), color='k', ls='--', lw=0.8,
           label=f'mean = {snr_eb_comp.mean():.1f} dB')
ax.set_xlabel('Ping index')
ax.set_ylabel('SNR (dB)')
ax.set_title(f'Echobot: Tank Floor SNR ({Npings_eb} pings)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(np.arange(n_pings_ek), snr_ek_all, '.', ms=2, color='C1', alpha=0.5)
stat_start = stat_idx_ek[0] if len(stat_idx_ek) > 0 else n_pings_ek
ax.axvline(stat_start, color='r', ls='--', lw=0.8, label='stationary start')
ax.axhline(snr_ek_stat.mean(), color='k', ls='--', lw=0.8,
           label=f'stat. mean = {snr_ek_stat.mean():.1f} dB')
ax.set_xlabel('Ping index')
ax.set_ylabel('SNR (dB)')
ax.set_title(f'EK80: Tank Floor SNR ({n_pings_ek} pings)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
lo = min(snr_eb_comp.min(), snr_ek_stat.min()) - 1
hi = max(snr_eb_comp.max(), snr_ek_stat.max()) + 1
bins = np.linspace(lo, hi, 30)
ax.hist(snr_eb_comp, bins=bins, alpha=0.6, color='C0', edgecolor='k',
        linewidth=0.5, label=f'Echobot (n={Npings_eb})', density=True)
ax.hist(snr_ek_stat, bins=bins, alpha=0.6, color='C1', edgecolor='k',
        linewidth=0.5, label=f'EK80 stat. (n={len(snr_ek_stat)})', density=True)
ax.set_xlabel('SNR (dB)')
ax.set_ylabel('Density')
ax.set_title('SNR Distributions (stationary pings)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.axis('off')
tbl = [
    ['', 'Echobot', 'EK80 (stationary)'],
    ['Pings', f'{Npings_eb}', f'{len(snr_ek_stat)}'],
    ['Mean SNR', f'{snr_eb_comp.mean():.1f} dB', f'{snr_ek_stat.mean():.1f} dB'],
    ['Std SNR', f'{snr_eb_comp.std():.2f} dB', f'{snr_ek_stat.std():.2f} dB'],
    ['Min SNR', f'{snr_eb_comp.min():.1f} dB', f'{snr_ek_stat.min():.1f} dB'],
    ['Max SNR', f'{snr_eb_comp.max():.1f} dB', f'{snr_ek_stat.max():.1f} dB'],
]
table = ax.table(cellText=tbl, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.6)
for j in range(3):
    table[0, j].set_facecolor('#d4e6f1')
    table[0, j].set_text_props(fontweight='bold')
ax.set_title('SNR Summary', fontsize=12, pad=20)

plt.tight_layout()
fig.savefig(os.path.join(ASSETS, 'fig_3b_snr_comparison.png'),
            dpi=150, bbox_inches='tight')
plt.close()

print(f'\nAll 9 figures saved to: {ASSETS}')
print('Done!')
