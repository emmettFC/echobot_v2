# EchoBot Processing Pipeline

**Notebook:** `echobot_pipeline_01.ipynb`
**Translated from:** `Matlab/echobot_pipeline_SCRIPTED.m`
**Instrument:** EchoBot custom sonar system (3-sector receive array)
**Purpose:** Process a wideband FM chirp recording of calibration spheres (38 mm and 22 mm tungsten carbide) in the CRL tank, producing frequency-dependent target strength approximations TS(f) and system SNR estimates. Companion to the EK80 reference pipeline (`EK80_pipeline_01.ipynb`).

---

## Table of Contents

1. [Parameters](#1-parameters)
2. [Assumptions](#2-assumptions)
3. [Workflow — Data Flow Diagram](#3-workflow--data-flow-diagram)
4. [Step-by-Step Process](#4-step-by-step-process)
5. [Key Results](#5-key-results)
6. [Differences from the EK80 Pipeline](#6-differences-from-the-ek80-pipeline)

---

## 1. Parameters

### 1.1 Acquisition Parameters

| Parameter | Value | Source | Why This Value |
|---|---|---|---|
| Data file | `Data/backcyl_bis_rgh0.01271_T143330_100.mat` | CRL tank recording | 100-ping stationary calibration dataset |
| Date | 2025-09-04 | File metadata | Same day as EK80 reference recording |
| Sample rate (`fs`) | 2,000 kHz (2 MHz) | `header['fs']` | EchoBot ADC sample rate; provides Nyquist of 1 MHz, well above the 150 kHz chirp band |
| Sound speed (`c`) | 1,486 m/s | `header['c']` | Measured/configured for CRL tank water |
| Pre-trigger time (`T_pre`) | From header | `header['T_pre']` | Duration of silence before chirp transmission; defines range = 0 |
| Post-trigger time (`T_post`) | From header | `header['T_post']` | Duration of recording after chirp ends |
| Chirp waveform (`s_chirp`) | 1,000 samples | `header['s_chirp']` | Stored transmit waveform; duration = 1000/2 MHz = 0.50 ms |
| Chirp duration (`T_chirp`) | 0.50 ms | `len(s_chirp) / fs` | Bandwidth-time product ≈ 30 for 60 kHz bandwidth |
| Chirp type | Downsweep | Verified by spectrogram | Frequency decreasing with time (~150→90 kHz) |
| Power level | 3 dB | File metadata | Transmit power setting |
| Receive channels used | 0, 1, 2 | Hardcoded (`rx_chans`) | Sectors 1–3 of the 5-channel system; channels 3–4 are unused |
| Number of pings | 100 | `data.shape[2]` | Full recording |
| Samples per ping | 101,003 | `data.shape[0]` | Determined by `T_pre + T_chirp + T_post` at 2 MHz |
| Total channels | 5 | `data.shape[1]` | Only 3 used as receive sectors |

### 1.2 Bandpass Filter Parameters

| Parameter | Value | Why This Value |
|---|---|---|
| Low-pass passband | 150 kHz | Upper edge of the chirp band |
| Low-pass stopband | 200 kHz | 50 kHz transition band provides steep rolloff while keeping the filter stable at 101 taps |
| Low-pass cutoff (`firwin`) | 175 kHz | Midpoint of passband/stopband edges; `firwin` places the −6 dB point here |
| High-pass passband | 90 kHz | Lower edge of the chirp band |
| High-pass stopband | 70 kHz | 20 kHz transition band |
| High-pass cutoff (`firwin`) | 80 kHz | Midpoint of stopband/passband edges |
| Filter order | 101 taps (both) | Provides sufficient stopband attenuation (~40 dB) while keeping group delay reasonable |
| Filter type | FIR (`scipy.signal.firwin`) | Linear phase, no stability concerns |

### 1.3 Detection / Gating Parameters

| Parameter | Value | Why This Value |
|---|---|---|
| 38 mm target center | 1.57 m | Known deployment position of the calibration sphere |
| Target gate half-width | 0.25 m | Captures the full pulse-compressed echo extent |
| Target gate (`target_38mm_band`) | [1.32, 1.82] m | `1.57 ± 0.25` m |
| Envelope smoothing kernel | 0.05 m (~67 samples) | Smooths fast oscillations while preserving echo envelope shape |
| Signal threshold | 3.0 × median | Flags regions with significant acoustic returns above the background; 3x median provides robust separation |
| Signal exclusion dilation | 0.15 m each side | Safety margin around detected signals to prevent edge contamination of noise estimates |
| Empty-region window size | 0.25 m (~337 samples) | Matches the target gate width for consistent spectral resolution |

### 1.4 FFT / Spectral Parameters

| Parameter | Value | Why This Value |
|---|---|---|
| FFT size (`Nfft`) | 4,096 | Fine frequency resolution (~488 Hz/bin at 2 MHz) while being a power of 2 for FFT efficiency |
| Frequency axis | `k × (fs / Nfft)` for k = 0..4095 | Direct FFT frequency mapping (no aliasing — `fs` ≫ signal bandwidth) |
| Display range | 90–150 kHz | Chirp bandwidth |
| dB range | −60 to +3 dB | Relative scale normalized to combined peak |
| Windowing | None (rectangular) | Gated segment passed directly to FFT without tapering |

### 1.5 SNR Parameters

| Parameter | Value | Why This Value |
|---|---|---|
| Signal reference | Tank floor peak | Strongest fixed reflector; provides a stable, high-SNR reference |
| Floor search region | [target_38mm_band[1]+0.3, 5.0] m = [~2.12, 5.0] m | Beyond the target with 0.3 m clearance |
| Noise region (`empty_deep`) | [2.08, 2.33] m | Auto-detected: quietest 0.25 m window between sphere and floor |
| SNR formula | `20·log10(V_peak / V_noise)` | Voltage-domain ratio; `20·log10` because these are amplitude quantities |

### 1.6 Derived Constants

| Constant | Value | Derivation |
|---|---|---|
| Range resolution (`dr`) | ~0.000743 m | `c / (2 × fs)` = 1486 / (2 × 2,000,000) |
| Transmit reference length | 101,003 samples | `n_pre + len(s_chirp) + n_post` (zero-padded to match record length) |
| MF output length (positive lags) | 101,003 samples | Only positive lags retained from the full cross-correlation |
| Tank floor range | 3.164 m | Auto-detected from envelope peak |
| Noise region (shallow) | [0.706, 0.956] m | Auto-detected: quietest region above the target |
| Noise region (deep) | [2.080, 2.330] m | Auto-detected: quietest region between target and floor |
| Signal-occupied fraction | 32.9% | Of 101,003 samples flagged by the exclusion mask |
| Strongest ping | Index 58 | `argmax` of per-ping RMS (excluding first/last) |

---

## 2. Assumptions

1. **Sound speed is constant at 1,486 m/s** — No sound speed profile is applied. This is reasonable for the small, well-mixed CRL tank where temperature gradients are minimal.

2. **Only channels 0, 1, 2 are used** — The EchoBot has 5 receive channels but only the first 3 are treated as active sectors. Channels 3 and 4 are ignored (assumed unused or diagnostic).

3. **Range origin at chirp transmission** — `Range = 0.5 × c × (t − T_pre)` places range = 0 at the moment the chirp begins. This assumes `T_pre` exactly marks the chirp start time.

4. **The `firwin` bandpass filter approximates the MATLAB design** — The original MATLAB pipeline used `designfilt` with explicit passband ripple (1 dB) and stopband attenuation (40 dB) specifications. The Python translation uses `scipy.signal.firwin` with a simple cutoff at the midpoint between passband and stopband. The filter shapes are similar but not identical.

5. **No absolute calibration is applied** — The EchoBot pipeline produces **relative** TS(f) values (normalized to the combined peak of noise + target spectra). No TVG, impedance correction, gain correction, absorption correction, or sonar equation terms are applied. This is because the EchoBot lacks embedded calibration metadata (unlike the EK80 raw file format).

6. **The matched filter uses the full zero-padded chirp** — `tx_ref` is the chirp embedded in a full-length zero-padded vector (101,003 samples). The cross-correlation is therefore equivalent to a standard matched filter but with a very long reference. The matched filter output is not normalized by chirp energy.

7. **Target location is hardcoded at 1.57 m** — Unlike the EK80 pipeline which auto-detects the target, the EchoBot pipeline uses a known deployment position. This is appropriate since the sphere position was measured during the experiment.

8. **Coherent sector summation is valid** — `C_sum = C[:,0] + C[:,1] + C[:,2]` assumes the 3 sectors receive in-phase (coherent) signals from the target. This is valid for a target near the acoustic axis of the array.

9. **Rectangular FFT window is acceptable** — No tapering function (Hanning, etc.) is applied before the FFT. This maximizes frequency resolution but introduces spectral leakage. The chirp's inherent bandwidth and the gating window width mitigate this somewhat.

10. **Tank floor is the appropriate signal reference for SNR** — The SNR uses the tank floor echo (not the calibration sphere) as the "signal." This measures the system's overall signal-to-noise capability against the strongest fixed reflector, rather than target-specific detectability.

11. **The noise region is signal-free** — Both `empty_shallow` and `empty_deep` regions are identified by finding the lowest-RMS 0.25 m window while excluding detected signal zones. Any low-level reverberation or sidelobes in these windows would bias noise estimates upward.

12. **Single-ping spectral analysis** — TS(f) is computed from one ping only (ping 0). No multi-ping spectral averaging is performed, unlike the EK80 pipeline which averages across hundreds of pings.

---

## 3. Workflow — Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│  INPUT: backcyl_bis_rgh0.01271_T143330_100.mat               │
│  State: MATLAB .mat file with raw voltage data               │
│    • data: (101003 samples, 5 channels, 100 pings)           │
│    • header: {fs, c, T_pre, T_post, s_chirp}                 │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 1: Load & Parse                                        │
│  scipy.io.loadmat()                                          │
│  State: NumPy arrays + header dict                           │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 2: Build Axes & Transmit Reference                     │
│  t = arange(Ns) / fs                                         │
│  Range = 0.5 · c · (t − T_pre)                               │
│  tx_ref = [zeros | s_chirp | zeros]  (zero-padded to Ns)     │
│  State: Time axis, range axis, full-length chirp reference   │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 3: Select Ping & Extract Sectors                       │
│  raw = data[:, 0:3, ping_idx]                                │
│  State: Raw real-valued voltages (101003 × 3 sectors)        │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 4: Bandpass Filter (per sector)                        │
│  LP filter: 101-tap FIR, cutoff 175 kHz                      │
│  HP filter: 101-tap FIR, cutoff 80 kHz                       │
│  df_i = HP(LP(raw[:, i]))                                    │
│  State: Bandpass-filtered voltages (90–150 kHz), 3 sectors   │
└───────────────┬──────────────────────────┬───────────────────┘
                │                          │
    ┌───────────┘                          └──────────────┐
    ▼                                                     ▼
┌─────────────────────────────────┐    ┌──────────────────────────────┐
│  PATH A: Voltage-Domain         │    │  PATH B: Matched Filter      │
│  Analysis                       │    │                              │
│                                 │    │  c_i = correlate(df_i,       │
│  Combined envelope:             │    │         tx_ref, mode='full') │
│  env = |df1| + |df2| + |df3|   │    │                              │
│                                 │    │  Retain positive lags only   │
│  State: Voltage envelope        │    │  Rmf = 0.5·c·(lags/fs)      │
│  vs range                       │    │  C = [c1, c2, c3]            │
│                                 │    │                              │
│  ┌── Detection ──┐              │    │  State: MF output            │
│  │ • Smooth (5cm) │              │    │  (101003 × 3 sectors)        │
│  │ • Threshold    │              │    │  vs MF range axis            │
│  │   (3× median)  │              │    └─────────────┬────────────────┘
│  │ • Dilate ±15cm │              │                  │
│  │ • Floor detect │              │       ┌──────────┴──────────┐
│  │ • Empty region │              │       │                     │
│  │   finder       │              │       ▼                     ▼
│  └────────────────┘              │  ┌──────────────┐  ┌────────────────┐
│                                 │  │ Per-Sector   │  │ Summed-Sector  │
│  Results:                       │  │ FFT          │  │ FFT            │
│  • Floor: 3.164 m               │  │              │  │                │
│  • Empty shallow:               │  │ For ch 0,1,2:│  │ C_sum = Σ C_ch │
│    [0.706, 0.956] m             │  │  Gate noise  │  │ Gate noise     │
│  • Empty deep:                  │  │  Gate target │  │ Gate target    │
│    [2.080, 2.330] m             │  │  FFT(4096)   │  │ FFT(4096)      │
│                                 │  │  Rel. dB     │  │ Rel. dB        │
│  ┌── Multi-Ping (100) ──┐       │  │              │  │                │
│  │ For each ping:        │       │  │ State:       │  │ State:         │
│  │  BP filter → envelope │       │  │ Relative     │  │ Relative       │
│  │  floor_peak = max(env)│       │  │ TS(f) per    │  │ TS(f) for      │
│  │  empty_mean = mean    │       │  │ sector       │  │ coherent sum   │
│  │  SNR = 20·log10(      │       │  │ 90–150 kHz   │  │ 90–150 kHz     │
│  │   peak / mean)        │       │  └──────────────┘  └────────────────┘
│  └───────────────────────┘       │
│                                 │
│  State: Per-ping SNR             │
│  Mean: 33.7 ± 0.17 dB           │
└─────────────────────────────────┘
```

---

## 4. Step-by-Step Process

### Step 1 — Load Data

**What:** Load the MATLAB `.mat` file containing raw voltage recordings and header metadata.

**Python:**
```python
import scipy.io as sio
mat = sio.loadmat(mat_file, squeeze_me=False)
data = mat['data']       # shape (101003, 5, 100)
header = mat['header']   # structured array with fs, c, T_pre, T_post, s_chirp
```

**Result:** `data` array of raw voltages (samples × channels × pings) and header parameters.

---

### Step 2 — Build Axes and Transmit Reference

**What:** Construct time and range axes, and build the zero-padded chirp reference for matched filtering.

**Equations:**

$$t[n] = \frac{n}{f_s}, \quad n = 0, 1, \ldots, N_s - 1$$

$$R[n] = \frac{c}{2} \cdot (t[n] - T_{pre})$$

The factor of 1/2 accounts for two-way travel (sound travels to target and back). Subtracting `T_pre` places range = 0 at the transmit instant.

**Python:**
```python
t = np.arange(Ns) / fs
Range = 0.5 * c * (t - T_pre)

n_pre = round(T_pre * fs)
n_post = round(T_post * fs)
tx_ref = np.zeros(n_pre + len(s_chirp) + n_post)
tx_ref[n_pre : n_pre + len(s_chirp)] = s_chirp
```

**Result:** `tx_ref` — the chirp embedded in a full-length zero vector (101,003 samples), positioned at the correct time offset.

---

### Step 3 — Chirp Verification (Spectrogram)

**What:** Compute and display the spectrogram of the transmit reference to visually confirm the chirp characteristics (downsweep, bandwidth, duration).

**Python:**
```python
from scipy.signal import spectrogram
f_spec, t_spec, Sxx = spectrogram(tx_ref, fs=fs, window='hamming',
                                   nperseg=256, noverlap=200, nfft=1024)
```

**Result:** Spectrogram confirming a downsweep from ~150 kHz to ~90 kHz over 0.50 ms.

---

### Step 4 — Find Key Pings

**What:** Identify the strongest, first, and last pings for subsequent analysis.

**Equation:**

$$s(p) = \text{median}\!\left(\sqrt{\frac{1}{N_s}\sum_{n=0}^{N_s-1} |x_n^{(p)}|^2}\right)$$

where the median is taken across channels (RMS per channel, then median across channels).

**Python:**
```python
for p in range(Npings):
    rms_per_ch = np.sqrt(np.mean(data[:, :, p]**2, axis=0))
    s[p] = np.median(rms_per_ch)
max_idx = np.argmax(s[1:-1]) + 1   # strongest ping (excluding endpoints)
```

---

### Step 5 — Bandpass Filtering

**What:** Filter each receive sector to isolate the chirp band (roughly 80–175 kHz), removing out-of-band noise and DC offsets.

**Equations:**

The cascade of low-pass then high-pass FIR filters implements a bandpass response:

$$y[n] = h_{HP} * (h_{LP} * x[n])$$

where $h_{LP}$ and $h_{HP}$ are the 101-tap FIR impulse responses.

**Python:**
```python
from scipy.signal import firwin, lfilter

lp_coeff = firwin(101, 175e3, fs=fs)                      # LP at 175 kHz
hp_coeff = firwin(101, 80e3, fs=fs, pass_zero=False)       # HP at 80 kHz

df1 = lfilter(hp_coeff, 1.0, lfilter(lp_coeff, 1.0, data[:, 0, ping_idx]))
df2 = lfilter(hp_coeff, 1.0, lfilter(lp_coeff, 1.0, data[:, 1, ping_idx]))
df3 = lfilter(hp_coeff, 1.0, lfilter(lp_coeff, 1.0, data[:, 2, ping_idx]))
```

**Why cascaded LP+HP instead of a single bandpass?** Matching the original MATLAB implementation structure. The two-stage cascade provides equivalent bandpass behavior with independent control of each edge.

---

### Step 6 — Matched Filtering (Cross-Correlation)

**What:** Cross-correlate each bandpass-filtered sector with the transmit chirp reference. This is the pulse compression step — it compresses the extended chirp echo into a narrow peak, improving temporal resolution and SNR.

**Equation:**

$$c_i[k] = \sum_{n} df_i[n] \cdot tx_{ref}[n - k]$$

This is mathematically equivalent to convolving with the time-reversed chirp: $c_i = df_i * \text{flip}(tx_{ref})$. For a real-valued chirp, this equals the matched filter output.

**Python:**
```python
from scipy.signal import correlate

c1 = correlate(df1, tx_ref, mode='full')
c2 = correlate(df2, tx_ref, mode='full')
c3 = correlate(df3, tx_ref, mode='full')

lags = np.arange(-(len(tx_ref) - 1), len(df1))
idx_pos = lags >= 0
C = np.column_stack([c1[idx_pos], c2[idx_pos], c3[idx_pos]])
Rmf = 0.5 * c * (lags[idx_pos] / fs)
```

**Notes:**
- `mode='full'` returns the complete cross-correlation (length = `len(df) + len(tx_ref) − 1`)
- Only positive lags are retained (causal output)
- The MF range axis `Rmf` maps lag to one-way distance
- The output is **not normalized** by chirp energy (unlike the EK80 pipeline)

---

### Step 7 — Envelope Detection

**What:** Extract the instantaneous amplitude envelope of the matched filter output using the Hilbert transform.

**Equation:**

$$\text{env}[n] = |x[n] + j\cdot\mathcal{H}\{x\}[n]| = |\text{hilbert}(x)[n]|$$

$$\text{cc}_{dB}[n] = 20\log_{10}(\text{env}[n] + 10^{-30})$$

The Hilbert transform produces the analytic signal; its magnitude is the envelope.

**Python:**
```python
from scipy.signal import hilbert
cc_dB = 20 * np.log10(np.abs(hilbert(C, axis=0)) + 1e-30)
```

---

### Step 8 — Automated Feature Detection

Three detection algorithms identify key features in the voltage-domain data.

#### 8a — Signal Exclusion Mask

**What:** Identify regions containing acoustic returns (sphere echoes, floor reflection) to exclude from noise estimation.

**Python:**
```python
from scipy.ndimage import uniform_filter1d, binary_dilation

voltage_envelope = np.abs(df1) + np.abs(df2) + np.abs(df3)
smooth_samples = round(0.05 / dr)
env_smooth = uniform_filter1d(voltage_envelope, smooth_samples)
signal_threshold = 3.0 * np.median(env_smooth[Range > 0.05])
signal_present = env_smooth > signal_threshold
signal_zones = binary_dilation(signal_present, iterations=round(0.15 / dr))
```

**Result:** A boolean mask where `True` = signal-occupied, covering 32.9% of samples.

#### 8b — Tank Floor Detection

**What:** Find the range of the tank floor by locating the peak envelope beyond the target region.

**Python:**
```python
def find_tank_floor(envelope, range_axis, min_range):
    search_mask = range_axis > min_range
    idx = np.argmax(envelope[search_mask])
    return range_axis[search_mask][idx]

floor_range = find_tank_floor(voltage_envelope, Range, target_38mm_band[1] + 0.3)
```

**Result:** Floor at 3.164 m.

#### 8c — Empty Region Finder

**What:** Locate the quietest 0.25 m window in a specified range interval, avoiding signal-occupied zones.

**Algorithm:**
1. Compute cumulative sum of `envelope²`
2. Sliding-window energy: `win_energy[k] = cs[k + W] − cs[k]`
3. Set energy to `inf` for windows overlapping `signal_zones`
4. Return the window with minimum energy

**Python:**
```python
def find_empty_region(envelope, range_axis, search_range, win_m, signal_mask):
    win_samples = round(win_m / dr)
    cs = np.cumsum(envelope**2)
    win_energy = cs[win_samples:] - cs[:-win_samples]
    # Penalize windows overlapping signal zones
    for k where signal_mask overlaps window k:
        win_energy[k] = np.inf
    best_k = np.argmin(win_energy[valid_range])
    return center, [lo, hi]
```

**Results:**
- `empty_shallow`: [0.706, 0.956] m (center 0.831 m) — above the sphere
- `empty_deep`: [2.080, 2.330] m (center 2.205 m) — between sphere and floor

---

### Step 9 — Define Range Windows

**What:** Assign the detected features as the noise and target gates for spectral analysis.

| Window | Range | Source |
|---|---|---|
| Target gate | [1.32, 1.82] m | Hardcoded: 38 mm sphere at 1.57 ± 0.25 m |
| Noise gate | [0.706, 0.956] m | Auto-detected `empty_shallow` |

---

### Step 10 — Per-Sector TS(f) (FFT of Gated MF Output)

**What:** Compute the frequency spectrum of the matched filter output within the target and noise range gates, independently for each sector.

**Equations:**

For each sector $ch \in \{0, 1, 2\}$:

$$S_{target,ch}(f) = \text{FFT}\!\left(C[j_{0f}:j_{1f},\, ch],\ N_{fft}=4096\right)$$

$$S_{noise,ch}(f) = \text{FFT}\!\left(C[j_{0n}:j_{1n},\, ch],\ N_{fft}=4096\right)$$

$$S_{dB}(f) = 20\log_{10}\!\left(\frac{|S(f)|}{|S|_{max}} + 10^{-30}\right)$$

The normalization to the combined maximum of noise + target makes the dB values **relative**, not absolute.

**Python:**
```python
for ch in range(3):
    m = C[:, ch]
    j0n = np.searchsorted(Rmf, noise_band[0])
    j1n = np.searchsorted(Rmf, noise_band[1]) - 1
    j0f = np.searchsorted(Rmf, target_38mm_band[0])
    j1f = np.searchsorted(Rmf, target_38mm_band[1]) - 1

    FFTn = np.fft.fft(m[j0n:j1n+1], n=4096)
    FFTf = np.fft.fft(m[j0f:j1f+1], n=4096)

    all_max = np.max(np.abs(np.concatenate([FFTn, FFTf]))) + 1e-30
    SdB_noise = 20 * np.log10(np.abs(FFTn) / all_max)
    SdB_target = 20 * np.log10(np.abs(FFTf) / all_max)

f_Hz = np.arange(4096) * (fs / 4096)    # frequency axis: 0 to 2 MHz
```

**Notes:**
- No windowing function is applied (rectangular window)
- The number of input samples depends on gate width: ~337 samples (noise, 0.25 m) and ~674 samples (target, 0.50 m)
- Both are zero-padded to 4096 by `np.fft.fft`
- Display is limited to 90–150 kHz

---

### Step 11 — Summed-Sector TS(f)

**What:** Coherently sum the MF outputs from all 3 sectors, then compute the spectrum.

**Equation:**

$$C_{sum}[n] = C[n, 0] + C[n, 1] + C[n, 2]$$

The coherent sum constructively reinforces signals that are in-phase across sectors while partially canceling incoherent noise, improving SNR.

**Python:**
```python
C_sum = C[:, 0] + C[:, 1] + C[:, 2]
FFTn_sum = np.fft.fft(C_sum[j0n:j1n+1], n=4096)
FFTf_sum = np.fft.fft(C_sum[j0f:j1f+1], n=4096)
# Same relative normalization as per-sector
```

---

### Step 12 — Multi-Ping Noise Characterization

**What:** Assess system stability by processing all 100 pings and tracking the tank floor return strength and background noise level.

**Equations:**

For each ping $p$:

1. Bandpass filter all 3 sectors: $d_1^{(p)}, d_2^{(p)}, d_3^{(p)}$

2. Combined voltage envelope: $\text{env}^{(p)} = |d_1^{(p)}| + |d_2^{(p)}| + |d_3^{(p)}|$

3. Floor peak: $V_{floor}^{(p)} = \max_{r \in [\text{floor region}]} \text{env}^{(p)}(r)$

4. Empty mean: $V_{empty}^{(p)} = \text{mean}_{r \in [\text{empty\_deep}]} \text{env}^{(p)}(r)$

5. Ratio: $\text{ratio}^{(p)} = V_{floor}^{(p)} / V_{empty}^{(p)}$

**Python:**
```python
for p in range(Npings):
    d1 = lfilter(hp_coeff, 1.0, lfilter(lp_coeff, 1.0, data[:, 0, p]))
    d2 = lfilter(hp_coeff, 1.0, lfilter(lp_coeff, 1.0, data[:, 1, p]))
    d3 = lfilter(hp_coeff, 1.0, lfilter(lp_coeff, 1.0, data[:, 2, p]))
    env = np.abs(d1) + np.abs(d2) + np.abs(d3)

    floor_peak[p] = np.max(env[i_floor_lo:i_floor_hi])
    empty_mean[p] = np.mean(env[i_empty_lo:i_empty_hi])
```

**Results across 100 pings:**
| Metric | Min | Max | Spread |
|---|---|---|---|
| Floor peak (V) | 17.85 | 18.54 | ±2% |
| Empty mean (V) | 0.361 | 0.389 | ±4% |
| Ratio | 46.2 | 51.4 | narrow |

These tight ranges indicate excellent ping-to-ping stability.

---

### Step 13 — SNR Computation

**What:** Compute per-ping signal-to-noise ratio in dB.

**Equation:**

$$SNR_{dB}(p) = 20\log_{10}\!\left(\frac{V_{floor}^{(p)}}{V_{empty}^{(p)}}\right)$$

The `20·log10` form is used because `V_floor` and `V_empty` are voltage (amplitude) quantities, not power.

**Python:**
```python
snr_dB = 20 * np.log10(floor_peak / (empty_mean + 1e-30))
```

**Result:** Mean SNR = 33.7 dB, std = 0.17 dB, range [33.3, 34.2] dB.

---

## 5. Key Results

| Metric | Value |
|---|---|
| TS(f) type | Relative (not calibrated) |
| TS(f) bandwidth | 90–150 kHz |
| SNR (floor/empty, all pings) | 33.7 ± 0.17 dB |
| Floor peak stability | ±2% across 100 pings |
| Noise level stability | ±4% across 100 pings |
| Tank floor range | 3.164 m |
| Target range | 1.57 m (hardcoded) |
| Pings processed | 100 (all stationary) |

---

## 6. Differences from the EK80 Pipeline

| Aspect | EchoBot (this pipeline) | EK80 (`EK80_pipeline_01`) |
|---|---|---|
| **Data format** | Real voltages in `.mat` | Complex I/Q baseband in `.raw` |
| **Sample rate** | 2,000 kHz (RF) | 93.75 kHz (baseband after 16x decimation) |
| **Samples/ping** | 101,003 | 1,257 |
| **Chirp source** | Stored in header (`s_chirp`) | Reconstructed from RF chirp + WBT/PC filter chain |
| **Pre-filtering** | Explicit bandpass FIR (LP + HP cascade) | Implicit via WBT/PC decimation filters |
| **Matched filter** | `correlate(filtered, tx_ref)` (real, unnormalized) | `convolve(raw, flipud(conj(chirp))) / norm_fac` (complex, normalized) |
| **Frequency mapping** | Direct (`f = k × fs/Nfft`) — no aliasing | Aliasing-aware mapping (`f = f_bb + k·fs_eff`) |
| **FFT windowing** | None (rectangular) | Hanning window |
| **TS(f) calibration** | Relative dB (normalized to peak) | Absolute dB re 1 m² (full sonar equation) |
| **TVG correction** | Not applied | `40·log10(R) + 2α·R` |
| **Impedance correction** | Not applied | `(|z_er+z_et|/z_er)² / z_et` |
| **Gain correction** | Not applied | `−2 × 18.0 dB` |
| **Multi-ping TS(f)** | Single ping only | Coherent average across all/stationary pings |
| **SNR signal** | Tank floor peak voltage | Target peak calibrated TS |
| **SNR noise** | Mean voltage in empty_deep | Mean calibrated TS in [2.0, 2.5] m |
| **SNR formula** | `20·log10(V_sig/V_noise)` | `TS_peak − TS_noise` (dB difference) |
| **Target detection** | Hardcoded position | Auto-detected from mean TS profile |
| **Noise region** | Auto-detected (sliding window) | Manually specified |

### Why the differences?

The EK80 is a commercial echosounder with:
- Embedded calibration metadata (gain, impedance, transmit power, filter coefficients)
- Standardized SONAR-netCDF4 data format via echopype
- Complex baseband I/Q storage (after onboard decimation)

The EchoBot is a custom prototype with:
- Raw RF voltage storage (no onboard signal processing)
- Minimal metadata (only chirp waveform, timing, and sound speed)
- No calibration constants embedded in the data file

This means the EchoBot pipeline focuses on **relative** spectral analysis and **system-level** SNR characterization, while the EK80 pipeline can produce **absolute** calibrated TS values suitable for direct comparison with theoretical target strength predictions.

---

## Dependencies

- [scipy](https://scipy.org/) — `scipy.io.loadmat` (data loading), `scipy.signal.firwin` / `lfilter` (filtering), `scipy.signal.correlate` (matched filtering), `scipy.signal.hilbert` (envelope), `scipy.signal.spectrogram` (chirp verification)
- [scipy.ndimage](https://docs.scipy.org/doc/scipy/reference/ndimage.html) — `uniform_filter1d` (smoothing), `binary_dilation` (signal mask expansion)
- [numpy](https://numpy.org/) — Array operations, FFT
- [matplotlib](https://matplotlib.org/) — Visualization
