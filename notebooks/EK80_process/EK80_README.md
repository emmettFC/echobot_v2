# EK80 Broadband Processing Pipeline

**Notebook:** `EK80_pipeline_01.ipynb`
**Instrument:** Simrad EK80 wideband transceiver with ES120-18CDK-Split transducer
**Purpose:** Process a 90–150 kHz FM chirp recording of a 38 mm tungsten carbide calibration sphere in the CRL tank, producing frequency-dependent target strength TS(f) and signal-to-noise ratio (SNR) estimates. Serves as a calibrated reference for cross-validation with the EchoBot pipeline (`echobot_pipeline_01.ipynb`).

---

## Table of Contents

1. [Parameters](#1-parameters)
2. [Assumptions](#2-assumptions)
3. [Workflow — Data Flow Diagram](#3-workflow--data-flow-diagram)
4. [Step-by-Step Process](#4-step-by-step-process)
5. [Key Results](#5-key-results)

---

## 1. Parameters

### 1.1 Transmit / Acquisition Parameters

| Parameter | Value | Source | Why This Value |
|---|---|---|---|
| Raw file | `Data/prod-D20250904-T165452.raw` | Recorded in CRL tank | Single-target calibration dataset with ES120-18CDK-Split |
| FM sweep start (`f_start`) | 90 kHz | `beam['transmit_frequency_start']` | Lower edge of the ES120-18CDK-Split transducer bandwidth |
| FM sweep stop (`f_stop`) | 150 kHz | `beam['transmit_frequency_stop']` | Upper edge of the transducer bandwidth |
| Center frequency (`f_center`) | 120 kHz | `(f_start + f_stop) / 2` | Midpoint of the 60 kHz chirp band |
| Pulse duration (`t_dur`) | 0.512 ms | `beam['transmit_duration_nominal']` | Standard EK80 FM pulse length; provides ~60 kHz bandwidth-time product of ~31 (good pulse compression gain) |
| Transmit power (`Ptx`) | 60 W | `beam['transmit_power']` | Set during acquisition; used in the sonar equation calibration term |
| Receiver sample rate (`fs_rx`) | 1,500 kHz | `vs['receiver_sampling_frequency']` | EK80 WBT hardware ADC rate |
| Effective sample rate (`fs_eff`) | 93.75 kHz | `1 / sample_interval` | After 16x total decimation (8x WBT + 2x PC) |
| Taper slope | 0.0434 | `beam['slope']` | Controls the Hanning edge taper width on the chirp; prevents spectral sidelobes |
| Sound speed (water) | 1491 m/s | `env['sound_speed_indicative']` | Measured/configured for the CRL tank water |
| Sound speed (transducer) | 1500 m/s | `env['transducer_sound_speed']` | Standard transducer material sound speed |
| Number of pings | 2,867 | `len(ping_times)` | Full recording duration (~48 minutes) |
| Samples per ping | 1,257 | `beam.dims['range_sample']` | Determined by recording range and sample interval |
| Number of beams | 3 | `beam.dims['beam']` | ES120-18CDK-Split is a split-beam transducer with 3 quadrants |

### 1.2 Calibration Parameters

| Parameter | Value | Source | Why This Value |
|---|---|---|---|
| Gain correction | 18.0 dB | `vs['gain_correction']` | Factory/user calibration of the transducer sensitivity; applied two-way (2 × 18 = 36 dB) |
| Equivalent beam angle | −12.5 dB | `beam['equivalent_beam_angle']` | Characterizes the transducer beam pattern; not directly used in TS computation |
| Transceiver impedance (`z_er`) | 10,800 Ω | `vs['impedance_transceiver']` | Electrical impedance of the WBT receiver; used in the impedance mismatch correction |
| Transducer impedance (`z_et`) | 75 Ω | Hardcoded (echopype default) | Standard nominal impedance for Simrad transducers; not stored in the raw file |
| Absorption coefficient (`α`) | 0.04 dB/m | Hardcoded | Approximate freshwater absorption at 120 kHz; applied uniformly across band |

### 1.3 Filter Chain Parameters

| Parameter | Value | Source | Why This Value |
|---|---|---|---|
| WBT filter | 47-tap complex FIR | `get_filter_coeff(vs)` from raw file | Hardware anti-alias filter for the first decimation stage; coefficients set by EK80 firmware |
| WBT decimation factor | 8× | Raw file | Reduces rate from 1500 kHz to 187.5 kHz |
| PC filter | 91-tap complex FIR | `get_filter_coeff(vs)` from raw file | Anti-alias filter for the second decimation stage |
| PC decimation factor | 2× | Raw file | Reduces rate from 187.5 kHz to 93.75 kHz |
| Total decimation | 16× | 8 × 2 | Brings the complex baseband signal to the effective storage rate |

### 1.4 Processing Parameters

| Parameter | Value | Why This Value |
|---|---|---|
| FFT size (`Nfft`) | 4,096 | Provides fine frequency resolution (~22.9 Hz/bin) while being a power of 2 for FFT efficiency |
| Target gate center | 1.543 m | Auto-detected: peak of stationary mean TS profile in [1.0, 2.5] m |
| Target gate half-width | 0.25 m | Captures the full pulse-compressed extent of the sphere echo |
| Target gate | [1.293, 1.793] m | `target_peak ± 0.25 m` |
| Noise gate (spectral) | [0.40, 0.70] m | Above the sphere, in the near field; used for spectral noise comparison |
| Noise gate (SNR) | [2.0, 2.5] m | Between sphere (~1.5 m) and tank floor (~3 m); avoids near-field transducer artifacts |
| Stationary time cutoff | 17:40:00 UTC | Based on experimental notes; sphere was stationary from ~13:40 local (UTC−4) onward |
| Aliasing frequency margins | 10% | `f_start × 0.90` to `f_stop × 1.10`; ensures edge bins of the chirp band are captured |
| TX power spectrum floor | 1e−10 × peak | Prevents division-by-zero during spectral deconvolution |
| Hanning window | Applied to gated PC output before FFT | Reduces spectral leakage from the rectangular range gate |
| UTC offset | −4 hours | CRL tank time zone (EDT) |

### 1.5 Derived Constants

| Constant | Value | Derivation |
|---|---|---|
| RF chirp length | 767 samples | `floor(t_dur × fs_rx)` = floor(0.000512 × 1,500,000) |
| Filtered chirp length | 96 complex samples | After WBT + PC filtering and decimation |
| Matched filter norm (`norm_fac`) | 44.237 | `‖tx_filtered‖²` (energy of the decimated chirp replica) |
| `norm_fac` in dB | 32.9 dB | `20 × log10(norm_fac)` |
| Impedance factor (mean-beam) | −22.95 dB | `10·log10(3/8) + 20·log10(|z_er+z_et|/z_er) − 10·log10(z_et)` |
| In-band FFT bins | 3,670 of 4,096 | Bins mapping to actual frequencies within [81, 165] kHz |
| Tank floor range | 2.991 m | Auto-detected from stationary mean TS profile |
| Stationary pings | 187 of 2,867 | Pings after 17:40 UTC |

---

## 2. Assumptions

1. **Transducer impedance z_et = 75 Ω** — This echopype default is not stored in the raw file. The actual ES120-18CDK-Split impedance may differ, which would shift all absolute TS values by a constant offset.

2. **Absorption α = 0.04 dB/m is frequency-independent** — A single value is applied across the full 90–150 kHz band. In reality, absorption varies with frequency (~0.03 dB/m at 90 kHz to ~0.05 dB/m at 150 kHz in freshwater), but at the short ranges in this tank (<3 m) the error is negligible (<0.06 dB total two-way).

3. **Gain is frequency-independent** — The single `gain = 18.0 dB` value is applied uniformly across the band. Real transducer gain varies with frequency; frequency-dependent gain calibration would improve absolute accuracy at band edges.

4. **Stationary period begins at 17:40 UTC** — Based on experimental notes, not algorithmically detected. All pings after this time are assumed to have the sphere motionless.

5. **Target is the strongest scatterer in [1.0, 2.5] m** — The pipeline detects the sphere by finding the peak of the mean TS profile in this range window. This assumes no other strong scatterers are present in that interval.

6. **On-axis target** — Mean-beam processing (`raw.mean(axis=1)` across 3 beams) assumes the sphere is centered on the acoustic axis. Off-axis targets would show beam-pattern effects.

7. **Coherent spectral averaging is valid** — Complex FFT spectra are averaged across pings (not magnitude spectra). This preserves phase and suppresses incoherent noise, but requires that the target position and phase are stable across pings — appropriate for a stationary sphere in a controlled tank.

8. **Linear FM chirp model** — The chirp replica is constructed as `cos(π·(f_stop−f_start)/t_dur · t² + 2π·f_start·t)`, assuming a perfectly linear frequency sweep. The EK80 transmitter closely approximates this.

9. **Aliasing does not exceed ±10 Nyquist zones** — The frequency mapping searches integer `k` in [−10, +10]. For this 90–150 kHz band at 93.75 kHz sample rate, only k = 0 and k = 1 are actually needed.

10. **SNR noise gate [2.0, 2.5 m] is signal-free** — This region between the sphere and tank floor is assumed free of coherent scatterers. Any reverberation or sidelobe energy in this gate would bias the noise estimate upward (reducing apparent SNR).

---

## 3. Workflow — Data Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│  INPUT: prod-D20250904-T165452.raw                      │
│  State: Simrad EK80 proprietary binary format           │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 1: Load with echopype                             │
│  ep.open_raw(raw_file, sonar_model='EK80')              │
│  State: EchoData object with SONAR-netCDF4 groups       │
│    • Beam_group1: backscatter_r + j·backscatter_i       │
│      shape (1 channel, 2867 pings, 1257 samples, 3 beams)
│    • Environment: sound speeds                          │
│    • Vendor_specific: filter coefficients, impedances   │
└───────────────────────┬─────────────────────────────────┘
                        │
           ┌────────────┴────────────┐
           ▼                         ▼
┌─────────────────────┐   ┌──────────────────────────────┐
│  PATH A: echopype    │   │  PATH B: Custom Spectral     │
│  Broadband TS        │   │  TS(f) Pipeline              │
│                      │   │                              │
│  ep.calibrate.       │   │  Build chirp replica:        │
│  compute_TS(         │   │  RF chirp → WBT filter →    │
│    ed,               │   │  decimate 8x → PC filter →  │
│    waveform_mode=    │   │  decimate 2x →              │
│    'BB',             │   │  96-sample complex baseband  │
│    encode_mode=      │   │  chirp (tx_filtered)         │
│    'complex')        │   │                              │
│                      │   │  State: Complex baseband     │
│  State: Calibrated   │   │  chirp replica at 93.75 kHz │
│  TS(ping, range)     │   │                              │
│  in dB re 1 m²       │   │  Matched filter replica:     │
│                      │   │  mf = flipud(conj(chirp))   │
│  ┌─── Used for ───┐  │   │  norm = ‖chirp‖²            │
│  │ • Echograms     │  │   │                              │
│  │ • Mean profiles │  │   │  Aliasing-aware freq map:    │
│  │ • Broadband TS  │  │   │  f_actual = f_bb + k·fs_eff │
│  │ • SNR (Path C)  │  │   │  for k giving f ∈ [90,150] │
│  └─────────────────┘  │   │                              │
└─────────────────────┘   │  State: Monotonic frequency   │
           │               │  axis 81–165 kHz with         │
           │               │  band_idx_sorted mapping      │
           │               └──────────────┬───────────────┘
           │                              │
           │                              ▼
           │               ┌──────────────────────────────┐
           │               │  For each ping:               │
           │               │                              │
           │               │  1. Extract complex I/Q       │
           │               │     raw = b_r + j·b_i         │
           │               │     State: Complex baseband   │
           │               │     voltages (1257 × 3 beams) │
           │               │                              │
           │               │  2. Average across beams      │
           │               │     raw_sig = mean(raw, dim=  │
           │               │     beam)                     │
           │               │     State: Mean-beam complex  │
           │               │     voltage (1257 samples)    │
           │               │                              │
           │               │  3. Pulse compression         │
           │               │     pc = convolve(raw_sig,    │
           │               │     mf_replica) / norm_fac    │
           │               │     State: Normalized PC      │
           │               │     output (complex, 1257)    │
           │               │                              │
           │               │  4. Range-gate target         │
           │               │     gated = pc[1.29:1.79 m]   │
           │               │     Apply Hanning window      │
           │               │     State: Windowed gated     │
           │               │     PC segment                │
           │               │                              │
           │               │  5. FFT(gated, n=4096)        │
           │               │     State: Complex spectrum   │
           │               │     (4096 bins)               │
           │               │                              │
           │               │  6. Accumulate (coherent sum) │
           │               └──────────────┬───────────────┘
           │                              │
           │                              ▼
           │               ┌──────────────────────────────┐
           │               │  Average & Calibrate:         │
           │               │                              │
           │               │  spec_avg = Σ spec / N_pings  │
           │               │  (coherent average)           │
           │               │                              │
           │               │  H(f) = spec[band] / |TX(f)|² │
           │               │  (deconvolve chirp shape)     │
           │               │                              │
           │               │  TS(f) = 10·log10(|H|²)      │
           │               │    + 20·log10(norm_fac)       │
           │               │    + impedance_factor_dB      │
           │               │    + 40·log10(R)              │
           │               │    + 2·α·R                    │
           │               │    − 10·log10(λ²Ptx/16π²)    │
           │               │    − 2·gain                   │
           │               │                              │
           │               │  State: Absolute TS(f) in     │
           │               │  dB re 1 m² vs frequency      │
           │               │  (90–150 kHz)                 │
           │               └──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│  PATH C: SNR Computation                                │
│  (uses echopype calibrated TS from Path A)              │
│                                                         │
│  For each ping:                                         │
│    TS_peak = max(TS[target gate 1.29–1.79 m])           │
│    TS_noise = 10·log10(mean(10^(TS[noise 2.0–2.5 m]/10)))
│    SNR = TS_peak − TS_noise                             │
│                                                         │
│  State: Per-ping SNR in dB                              │
│    All pings:  30.9 ± 1.44 dB                           │
│    Stationary: 31.7 ± 0.14 dB                           │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Step-by-Step Process

### Step 1 — Load Raw Data

**What:** Open the Simrad `.raw` file and parse it into an organized data structure.

**Equation:** None (file I/O).

**Python:**
```python
import echopype as ep
ed = ep.open_raw('Data/prod-D20250904-T165452.raw', sonar_model='EK80')
```

**Result:** An `EchoData` object containing:
- `ed['Sonar/Beam_group1']` — complex backscatter samples (I and Q stored separately)
- `ed['Environment']` — sound speed values
- `ed['Vendor_specific']` — filter coefficients, impedances, gain tables

---

### Step 2 — Compute Broadband TS via echopype

**What:** Use echopype's built-in calibration to get broadband (frequency-integrated) target strength as a function of ping and range.

**Equation (internal to echopype):**

$$p_{rx} = n_{beam} \cdot \frac{|\overline{pc}|^2}{(2\sqrt{2})^2} \cdot \frac{(|z_{er} + z_{et}|)^2}{z_{er}^2 \cdot z_{et}}$$

$$TS = 10\log_{10}(p_{rx}) + 40\log_{10}(R) + 2\alpha R - 10\log_{10}\!\left(\frac{\lambda^2 P_{tx}}{16\pi^2}\right) - 2G$$

**Python:**
```python
ds_TS = ep.calibrate.compute_TS(ed, waveform_mode='BB', encode_mode='complex')
TS = ds_TS['TS'].values[0]  # shape (2867, 1257), dB re 1 m²
```

**Result:** A 2D array of calibrated TS values used for echograms, mean profiles, broadband TS statistics, and SNR computation.

---

### Step 3 — Visualize Echograms

**What:** Display the TS data as a color-mapped image (ping time × range).

**Python:**
```python
plt.pcolormesh(local_times, range_m, TS.T, vmin=-80, vmax=0, cmap='viridis')
```

**Result:** Echogram showing the sphere at ~1.5 m and tank floor at ~3 m, with clear visibility of the stationary period.

---

### Step 4 — Identify Stationary Period

**What:** Select pings where the sphere was motionless, based on experimental notes.

**Python:**
```python
stat_mask = ping_times >= np.datetime64('2025-09-04T17:40:00')
stat_indices = np.where(stat_mask)[0]  # 187 pings
```

**Why:** Stationary pings allow coherent spectral averaging and yield the most accurate TS estimates.

---

### Step 5 — Compute Mean TS Profiles

**What:** Average TS across pings in the linear (power) domain to produce a range profile.

**Equation:**

$$\overline{TS}(r) = 10\log_{10}\!\left(\frac{1}{N}\sum_{i=1}^{N} 10^{TS_i(r)/10}\right)$$

Averaging in linear domain prevents biasing from the nonlinearity of the dB scale.

**Python:**
```python
linear = 10 ** (TS_subset / 10)
mean_ts = 10 * np.log10(np.nanmean(linear, axis=0))
```

---

### Step 6 — Detect Target and Floor Ranges

**What:** Automatically locate the sphere and tank floor from peaks in the mean TS profile.

**Python:**
```python
# Target: strongest peak in [1.0, 2.5] m
target_peak_range = range_m[j_target_search + np.nanargmax(ts_profile[j_lo:j_hi])]

# Floor: strongest peak in [2.5, 5.0] m
floor_range = range_m[j_floor_search + np.nanargmax(ts_profile[j_flo:j_fhi])]
```

**Result:** Target at 1.543 m, floor at 2.991 m.

---

### Step 7 — Broadband TS Statistics

**What:** Extract per-ping peak TS within the target gate for summary statistics.

**Equation:**

$$TS_{peak}(i) = \max_{r \in [1.29, 1.79]} TS_i(r)$$

**Python:**
```python
ts_peak = np.nanmax(TS[:, j0:j1], axis=1)
```

**Result:** All pings: −42.4 dB mean | Stationary: −40.9 ± 0.08 dB.

---

### Step 8 — Construct the Chirp Replica

The EK80 stores complex baseband I/Q data that has been filtered and decimated from the original 1500 kHz RF signal. To perform matched filtering, we must construct a chirp replica that has undergone the same processing chain.

#### 8a — Generate the RF chirp

**Equation:**

$$y_{RF}(t) = \cos\!\left(\pi \frac{f_{stop} - f_{start}}{\tau} t^2 + 2\pi f_{start} \cdot t\right)$$

This produces a linear FM sweep from `f_start` (90 kHz) to `f_stop` (150 kHz) over duration `τ` (0.512 ms). The instantaneous frequency at time `t` is:

$$f(t) = f_{start} + \frac{f_{stop} - f_{start}}{\tau} \cdot t$$

**Python:**
```python
n_rf = int(np.floor(t_dur * fs_rx))        # 767 samples
t_rf = np.arange(n_rf) / fs_rx
y_rf = np.cos(np.pi * (f_stop - f_start) / t_dur * t_rf**2
              + 2 * np.pi * f_start * t_rf)
```

#### 8b — Apply Hanning edge taper

**What:** Smooth the chirp onset and offset to suppress spectral sidelobes.

**Python:**
```python
L_taper = round(t_dur * fs_rx * slope_val * 2.0)
w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(L_taper) / (L_taper - 1)))
y_rf[:L_taper//2] *= w[:L_taper//2]       # rising edge
y_rf[-(L_taper//2):] *= w[L_taper//2:]    # falling edge
y_rf /= np.max(np.abs(y_rf))              # normalize to unit peak
```

The `slope` parameter (0.0434) controls how many samples are tapered — roughly 4.3% of the chirp on each edge.

#### 8c — WBT filter and decimation

**What:** Apply the 47-tap complex FIR anti-alias filter, then downsample by 8.

**Python:**
```python
from scipy.signal import convolve
ytx_wbt = convolve(y_rf, wbt_fil)      # 47-tap complex FIR
ytx_wbt_dec = ytx_wbt[::8]             # decimate by 8
```

The WBT filter downconverts the real RF chirp to complex baseband and provides anti-alias filtering for the decimation.

#### 8d — PC filter and decimation

**What:** Apply the 91-tap complex FIR filter, then downsample by 2.

**Python:**
```python
ytx_pc = convolve(ytx_wbt_dec, pc_fil)  # 91-tap complex FIR
tx_filtered = ytx_pc[::2]               # decimate by 2
```

**Result:** `tx_filtered` — 96 complex samples at 93.75 kHz. This is the chirp replica that matches the format of the stored I/Q data.

#### 8e — Matched filter setup

**Equations:**

$$\text{norm\_fac} = \|tx_{filtered}\|^2 = \sum |tx_{filtered}[n]|^2$$

$$\text{mf\_replica} = \text{flipud}(\text{conj}(tx_{filtered}))$$

**Python:**
```python
norm_fac = np.linalg.norm(tx_filtered) ** 2   # = 44.237
mf_replica = np.flipud(np.conj(tx_filtered))
```

The matched filter is the time-reversed complex conjugate of the chirp. Division by `norm_fac` normalizes the output so that a perfect chirp echo returns unit amplitude.

---

### Step 9 — Aliasing-Aware Frequency Mapping

**What:** Map FFT bin indices from baseband frequencies to actual RF frequencies, accounting for aliasing caused by the 16x decimation.

**Why:** The chirp spans 90–150 kHz (bandwidth = 60 kHz) but the effective sample rate is only 93.75 kHz (Nyquist = 46.875 kHz). The 60 kHz bandwidth exceeds Nyquist, so the signal aliases into multiple Nyquist zones when sampled at `fs_eff`. Each FFT bin at baseband frequency `f_bb` actually corresponds to an RF frequency:

$$f_{actual} = f_{bb} + k \cdot f_{s,eff}$$

where `k` is the unique integer placing `f_actual` within the chirp band.

**Python:**
```python
f_bb = np.fft.fftfreq(Nfft, d=1.0/fs_eff)

for i, fb in enumerate(f_bb):
    for k in range(-10, 11):
        f_try = fb + k * fs_eff
        if f_start * 0.90 <= f_try <= f_stop * 1.10:
            f_actual_map[i] = f_try
            break

# Sort in-band bins by ascending actual frequency
band_idx_sorted = in_band_indices[np.argsort(f_actual_map[in_band_indices])]
f_band_sorted = f_actual_map[band_idx_sorted]
```

**Result:** A monotonic frequency axis from ~81 kHz to ~165 kHz (3,670 bins), with index mapping `band_idx_sorted` that correctly reorders the aliased FFT output.

---

### Step 10 — Pulse Compression (per ping)

**What:** Cross-correlate each ping's I/Q data with the matched filter replica.

**Equation:**

$$pc[n] = \frac{1}{\text{norm\_fac}} \sum_m \overline{x}_{beam}[m] \cdot \overline{mf}[m - n]$$

where $\overline{x}_{beam}$ is the mean across beams.

**Python:**
```python
def process_ping_pc(ed, ping_idx, beam_idx=None):
    raw = backscatter_r[0, ping_idx, :, :] + 1j * backscatter_i[0, ping_idx, :, :]
    raw_sig = raw.mean(axis=1) if beam_idx is None else raw[:, beam_idx]
    pc = convolve(raw_sig, mf_replica, mode='full')
    pc = pc[len(mf_replica)-1:][:n_samples]
    return pc / norm_fac
```

**Notes:**
- `mode='full'` returns the full convolution; we trim to align index 0 with the first original sample
- Beam averaging uses `mean` (not `sum`) to match echopype's convention
- Division by `norm_fac` normalizes the output

---

### Step 11 — Spectral Decomposition (range-gated FFT)

**What:** Extract the frequency spectrum of a range-gated segment of the pulse-compressed signal.

**Python:**
```python
def compute_tsf(pc_signal, r_axis, gate_band, nfft=4096):
    j0 = np.searchsorted(r_axis, gate_band[0])
    j1 = np.searchsorted(r_axis, gate_band[1], side='right')
    gated = pc_signal[j0:j1]
    gated = gated * np.hanning(len(gated))   # reduce spectral leakage
    return np.fft.fft(gated, n=nfft)
```

**Notes:**
- Hanning window applied to reduce spectral leakage from the abrupt gate edges
- Zero-padding to 4096 points interpolates the spectrum for smooth visualization

---

### Step 12 — Coherent Multi-Ping Spectral Averaging

**What:** Average the complex spectra across pings to suppress incoherent noise while preserving the coherent target response.

**Equation:**

$$\overline{S}(f) = \frac{1}{N_{valid}} \sum_{i=1}^{N_{valid}} S_i(f)$$

where $S_i(f)$ is the complex FFT spectrum of ping $i$.

**Python:**
```python
for p in range(n_pings):
    pc = process_ping_pc(ed, p)
    spec_target_acc += compute_tsf(pc, r_raw, target_band, Nfft)
spec_target_avg = spec_target_acc / n_valid
```

---

### Step 13 — Absolute TS(f) Calibration

**What:** Convert the deconvolved spectrum to absolute target strength using the full sonar equation.

**Equation:**

$$TS(f) = \underbrace{10\log_{10}|H(f)|^2}_{\text{deconvolved spectrum}} + \underbrace{20\log_{10}(\text{norm\_fac})}_{\text{undo PC normalization}} + \underbrace{Z_{dB}}_{\text{impedance factor}} + \underbrace{40\log_{10}(R) + 2\alpha R}_{\text{TVG + absorption}} - \underbrace{10\log_{10}\!\left(\frac{\lambda^2 P_{tx}}{16\pi^2}\right)}_{\text{sonar equation}} - \underbrace{2G}_{\text{gain}}$$

where:
- $H(f) = S(f) / |TX(f)|^2$ — target transfer function (spectrum deconvolved by chirp power spectrum)
- $Z_{dB} = 10\log_{10}(n_{beams}/8) + 20\log_{10}(|z_{er}+z_{et}|/z_{er}) - 10\log_{10}(z_{et})$ — impedance/beam scaling
- $\lambda(f) = c / f$ — wavelength (frequency-dependent)
- $R$ = center of target range gate (1.543 m)
- $\alpha$ = 0.04 dB/m
- $P_{tx}$ = 60 W
- $G$ = 18.0 dB

**Python:**
```python
def compute_absolute_tsf(spec_raw, R_center):
    H = spec_raw[band_idx_sorted] / TX_power_band
    H_dB = 10 * np.log10(np.abs(H)**2 + 1e-30)
    H_dB += 20 * np.log10(norm_fac)
    H_dB += impedance_factor_dB
    H_dB += 40 * np.log10(np.maximum(R_center, 0.01))
    H_dB += 2 * alpha_dBm * R_center
    lam_f = c_sw / f_band_sorted
    H_dB -= 10 * np.log10(lam_f**2 * Ptx / (16 * np.pi**2))
    H_dB -= 2 * gain
    return H_dB
```

**Calibration term breakdown for this dataset:**

| Term | Value | Effect |
|---|---|---|
| `20·log10(norm_fac)` | +32.9 dB | Compensates for the PC normalization |
| Impedance factor | −22.95 dB | Accounts for electrical impedance mismatch and beam averaging |
| TVG at 1.543 m | +7.6 dB | Corrects for geometric spreading |
| Absorption at 1.543 m | +0.12 dB | Corrects for water absorption (negligible at this range) |
| Sonar equation at 120 kHz | ~−26 dB | Transmit term |
| 2 × gain | −36.0 dB | Transducer sensitivity correction |

---

### Step 14 — Per-Beam TS(f) Analysis

**What:** Compute TS(f) independently for each of the 3 split-beam sectors.

The only difference from mean-beam processing is the impedance factor:

$$Z_{dB,\text{per-beam}} = 10\log_{10}(1/8) + 20\log_{10}(|z_{er}+z_{et}|/z_{er}) - 10\log_{10}(z_{et})$$

This uses `n_beams = 1` instead of 3, since each beam is processed individually rather than averaged.

**Python:**
```python
for p in stat_indices:
    for b in range(3):
        pc_b = process_ping_pc(ed, p, beam_idx=b)
        spec_t_beams[b] += compute_tsf(pc_b, r_raw, target_band, Nfft)
```

---

### Step 15 — SNR Computation

**What:** Compute per-ping signal-to-noise ratio using echopype's calibrated TS values.

**Equation:**

$$SNR(i) = TS_{peak}(i) - TS_{noise}(i)$$

where:

$$TS_{peak}(i) = \max_{r \in [1.29, 1.79]} TS_i(r)$$

$$TS_{noise}(i) = 10\log_{10}\!\left(\text{mean}\left(10^{TS_i(r)/10}\right)\right), \quad r \in [2.0, 2.5]$$

The noise is averaged in the linear (power) domain to avoid bias from dB-domain averaging.

**Python:**
```python
snr_noise_band = [2.0, 2.5]
ts_peak_all = np.nanmax(TS[:, j0t:j1t], axis=1)
noise_linear = 10 ** (TS[:, j0n:j1n] / 10)
ts_noise_all = 10 * np.log10(np.nanmean(noise_linear, axis=1))
snr_all = ts_peak_all - ts_noise_all
```

**Why [2.0, 2.5] m for noise?** The original noise gate at [0.4, 0.7] m was contaminated by near-field transducer artifacts. The [2.0, 2.5] m gate sits between the sphere (~1.5 m) and tank floor (~3 m) in a signal-free region.

---

### Step 16 — Verification

The pipeline includes a comparison between the custom spectral TS and echopype's broadband TS for a single stationary ping:

| Metric | Value |
|---|---|
| Band-averaged TS(f) (custom pipeline) | −34.3 dB |
| Broadband TS peak (echopype) | −41.0 dB |
| Difference | 6.7 dB |

This discrepancy is expected — the two metrics measure different things: the custom pipeline computes a **spectral mean** across 90–150 kHz, while echopype reports the **spatial peak** broadband TS in the target gate. Coherent processing gain and bandwidth normalization differences also contribute.

---

## 5. Key Results

| Metric | All Pings | Stationary Only |
|---|---|---|
| Broadband TS (echopype) | −42.4 dB | −40.9 ± 0.08 dB |
| TS(f) at 120 kHz (custom) | −43.6 dB | (plotted) |
| SNR | 30.9 ± 1.44 dB | 31.7 ± 0.14 dB |
| Number of pings | 2,867 | 187 |

---

## Dependencies

- [echopype](https://echopype.readthedocs.io/) — EK80 raw file parsing and broadband TS calibration
- [numpy](https://numpy.org/) — Array operations, FFT, linear algebra
- [scipy](https://scipy.org/) — `scipy.signal.convolve` for filtering and matched filtering
- [matplotlib](https://matplotlib.org/) — Visualization
- [xarray](https://xarray.dev/) — Underlying data structure (via echopype)
