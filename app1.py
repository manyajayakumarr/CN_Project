import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SignalGuard",
    page_icon="📡",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📡 SignalGuard")
st.subheader("Intelligent Transmission Channel Diagnosis and Stress-Test System")

st.write(
    "SignalGuard simulates transmission impairments and analyzes how "
    "attenuation, delay, distortion, and noise affect a communication signal."
)

st.info(
    "💡 SignalGuard does more than display a degraded signal. "
    "It identifies the dominant impairment, evaluates channel health, "
    "tests transmission limits, and simulates recovery strategies."
)


# ============================================================
# SIGNAL GENERATION
# ============================================================

def generate_signal(signal_type, frequency, sampling_rate, duration):

    time = np.arange(
        0,
        duration,
        1 / sampling_rate
    )

    if signal_type == "Sine Wave":

        signal = np.sin(
            2 * np.pi * frequency * time
        )

    elif signal_type == "Square Wave":

        signal = np.where(
            np.sin(2 * np.pi * frequency * time) >= 0,
            1.0,
            -1.0
        )

    elif signal_type == "Composite Signal":

        signal = (
            0.60 * np.sin(2 * np.pi * frequency * time)
            + 0.30 * np.sin(2 * np.pi * 3 * frequency * time)
            + 0.15 * np.sin(2 * np.pi * 5 * frequency * time)
        )

        maximum = np.max(np.abs(signal))

        if maximum != 0:
            signal = signal / maximum

    return time, signal


# ============================================================
# TRANSMISSION MEDIUM
# ============================================================

MEDIUM_ATTENUATION = {
    "Copper Cable": 0.08,
    "Fiber Optic": 0.02,
    "Wireless": 0.15
}


def calculate_channel_loss(distance, medium):

    rate = MEDIUM_ATTENUATION[medium]

    loss = distance * rate

    return loss


# ============================================================
# ATTENUATION
# ============================================================

def apply_attenuation(signal, loss_db):

    amplitude_factor = 10 ** (-loss_db / 20)

    return signal * amplitude_factor


# ============================================================
# DELAY
# ============================================================

def apply_delay(signal, delay_samples):

    if delay_samples <= 0:
        return signal.copy()

    delayed_signal = np.zeros_like(signal)

    if delay_samples < len(signal):

        delayed_signal[delay_samples:] = signal[
            :-delay_samples
        ]

    return delayed_signal


# ============================================================
# DISTORTION
# ============================================================

def apply_distortion(signal, distortion_percent):

    if distortion_percent <= 0:
        return signal.copy()

    distortion_factor = distortion_percent / 100.0

    distorted_signal = (
        signal
        + distortion_factor * (signal ** 3)
    )

    maximum = np.max(np.abs(distorted_signal))

    if maximum > 0:
        distorted_signal = (
            distorted_signal / maximum
        )

    return distorted_signal


# ============================================================
# NOISE
# ============================================================

def apply_noise(signal, noise_level, seed):

    if noise_level <= 0:

        return signal.copy(), np.zeros_like(signal)

    rng = np.random.default_rng(seed)

    signal_rms = np.sqrt(
        np.mean(signal ** 2)
    )

    noise_std = noise_level * signal_rms

    noise = rng.normal(
        0,
        noise_std,
        len(signal)
    )

    noisy_signal = signal + noise

    return noisy_signal, noise


# ============================================================
# RMS
# ============================================================

def calculate_rms(signal):

    if len(signal) == 0:
        return 0

    return np.sqrt(
        np.mean(signal ** 2)
    )


# ============================================================
# SNR
# ============================================================

def calculate_snr(signal, noise):

    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power <= 1e-15:

        return float("inf")

    if signal_power <= 1e-15:

        return -float("inf")

    return 10 * np.log10(
        signal_power / noise_power
    )


# ============================================================
# NORMALIZED DIFFERENCE
# ============================================================

def normalized_difference(original, modified):

    denominator = calculate_rms(original)

    if denominator <= 1e-15:
        return 0

    difference = calculate_rms(
        original - modified
    )

    return difference / denominator


# ============================================================
# CHANNEL HEALTH SCORE
# ============================================================

def calculate_health_score(
    loss_db,
    distortion_percent,
    noise_level,
    delay_samples,
    sampling_rate
):

    # Educational heuristic score.
    # This is not a standard networking specification.

    attenuation_score = np.exp(
        -loss_db / 12
    )

    distortion_score = np.exp(
        -distortion_percent / 30
    )

    noise_score = np.exp(
        -noise_level / 0.20
    )

    delay_reference = max(
        1,
        sampling_rate * 0.10
    )

    delay_score = np.exp(
        -delay_samples / delay_reference
    )

    combined_score = (
        attenuation_score
        * distortion_score
        * noise_score
        * delay_score
    )

    health = combined_score * 100

    return float(
        np.clip(health, 0, 100)
    )


# ============================================================
# QUALITY CLASSIFICATION
# ============================================================

def classify_quality(health_score):

    if health_score >= 80:

        return "Healthy", "🟢"

    elif health_score >= 60:

        return "Moderate", "🟡"

    else:

        return "Degraded", "🔴"


# ============================================================
# COMPLETE SIMULATION
# ============================================================

def simulate_channel(
    signal_type,
    frequency,
    sampling_rate,
    duration,
    medium,
    distance,
    delay_samples,
    distortion_percent,
    noise_level,
    seed
):

    time, original = generate_signal(
        signal_type,
        frequency,
        sampling_rate,
        duration
    )

    # --------------------------------------------------------
    # 1. ATTENUATION
    # --------------------------------------------------------

    channel_loss = calculate_channel_loss(
        distance,
        medium
    )

    attenuated = apply_attenuation(
        original,
        channel_loss
    )

    # --------------------------------------------------------
    # 2. DELAY
    # --------------------------------------------------------

    delayed = apply_delay(
        attenuated,
        delay_samples
    )

    # --------------------------------------------------------
    # 3. DISTORTION
    # --------------------------------------------------------

    distorted = apply_distortion(
        delayed,
        distortion_percent
    )

    # --------------------------------------------------------
    # 4. NOISE
    # --------------------------------------------------------

    received, noise = apply_noise(
        distorted,
        noise_level,
        seed
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    original_rms = calculate_rms(
        original
    )

    received_rms = calculate_rms(
        received
    )

    snr = calculate_snr(
        distorted,
        noise
    )

    health = calculate_health_score(
        channel_loss,
        distortion_percent,
        noise_level,
        delay_samples,
        sampling_rate
    )

    quality, quality_icon = classify_quality(
        health
    )

    signal_loss_percent = 0

    if original_rms > 0:

        signal_loss_percent = (
            1 - received_rms / original_rms
        ) * 100

    # --------------------------------------------------------
    # IMPAIRMENT FINGERPRINT
    # --------------------------------------------------------

    attenuation_effect = normalized_difference(
        original,
        attenuated
    )

    delay_effect = normalized_difference(
        attenuated,
        delayed
    )

    distortion_effect = normalized_difference(
        delayed,
        distorted
    )

    noise_effect = normalized_difference(
        distorted,
        received
    )

    effects = {
        "Attenuation": attenuation_effect,
        "Delay": delay_effect,
        "Distortion": distortion_effect,
        "Noise": noise_effect
    }

    total_effect = sum(
        effects.values()
    )

    if total_effect > 0:

        contributions = {
            key: (value / total_effect) * 100
            for key, value in effects.items()
        }

    else:

        contributions = {
            key: 0
            for key in effects
        }

    dominant_impairment = max(
        contributions,
        key=contributions.get
    )

    return {
        "time": time,
        "original": original,
        "attenuated": attenuated,
        "delayed": delayed,
        "distorted": distorted,
        "received": received,
        "noise": noise,

        "channel_loss": channel_loss,
        "original_rms": original_rms,
        "received_rms": received_rms,
        "snr": snr,
        "health": health,
        "quality": quality,
        "quality_icon": quality_icon,
        "signal_loss_percent": signal_loss_percent,

        "effects": effects,
        "contributions": contributions,
        "dominant_impairment": dominant_impairment
    }


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Transmission Settings")


# ------------------------------------------------------------
# Signal settings
# ------------------------------------------------------------

st.sidebar.subheader("📡 Signal")

signal_type = st.sidebar.selectbox(
    "Signal Type",
    [
        "Sine Wave",
        "Square Wave",
        "Composite Signal"
    ]
)

sampling_rate = st.sidebar.selectbox(
    "Sampling Rate (Hz)",
    [
        1000,
        2000,
        5000,
        10000
    ]
)

frequency = st.sidebar.slider(
    "Signal Frequency (Hz)",
    min_value=1,
    max_value=max(
        2,
        int(sampling_rate * 0.45)
    ),
    value=min(
        10,
        int(sampling_rate * 0.45)
    )
)

duration = st.sidebar.slider(
    "Duration (seconds)",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
    step=0.1
)


# ------------------------------------------------------------
# Channel settings
# ------------------------------------------------------------

st.sidebar.divider()

st.sidebar.subheader("📡 Transmission Channel")

medium = st.sidebar.selectbox(
    "Transmission Medium",
    [
        "Copper Cable",
        "Fiber Optic",
        "Wireless"
    ]
)

distance = st.sidebar.slider(
    "Transmission Distance (km)",
    min_value=1,
    max_value=100,
    value=34
)

delay_samples = st.sidebar.slider(
    "Delay (samples)",
    min_value=0,
    max_value=100,
    value=5
)


# ------------------------------------------------------------
# Impairment settings
# ------------------------------------------------------------

st.sidebar.divider()

st.sidebar.subheader("⚠️ Impairments")

distortion_percent = st.sidebar.slider(
    "Distortion (%)",
    min_value=0,
    max_value=50,
    value=10
)

noise_level = st.sidebar.slider(
    "Noise Level",
    min_value=0.0,
    max_value=0.50,
    value=0.05,
    step=0.01
)

seed = st.sidebar.number_input(
    "Noise Seed",
    min_value=0,
    max_value=99999,
    value=42
)


# ============================================================
# RUN SIMULATION
# ============================================================

run_simulation = st.sidebar.button(
    "🚀 Run Channel Diagnosis",
    type="primary",
    use_container_width=True
)


# ============================================================
# DEFAULT RUN
# ============================================================

if (
    "simulation" not in st.session_state
    or run_simulation
):

    result = simulate_channel(
        signal_type,
        frequency,
        sampling_rate,
        duration,
        medium,
        distance,
        delay_samples,
        distortion_percent,
        noise_level,
        seed
    )

    st.session_state.simulation = result

else:

    result = st.session_state.simulation


# ============================================================
# VARIABLES FROM RESULT
# ============================================================

channel_loss = result["channel_loss"]
received_rms = result["received_rms"]
snr = result["snr"]
health = result["health"]
quality = result["quality"]
quality_icon = result["quality_icon"]

dominant_impairment = result[
    "dominant_impairment"
]

contributions = result[
    "contributions"
]


# ============================================================
# TRANSMISSION STATUS
# ============================================================

st.header("📊 Transmission Status")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Channel Loss",
        f"{channel_loss:.2f} dB"
    )

with col2:

    st.metric(
        "Received RMS",
        f"{received_rms:.3f}"
    )

with col3:

    if np.isinf(snr):

        st.metric(
            "SNR",
            "∞ dB"
        )

    else:

        st.metric(
            "SNR",
            f"{snr:.2f} dB"
        )

with col4:

    st.metric(
        "Channel Health",
        f"{health:.1f}/100"
    )


st.subheader(
    f"{quality_icon} Channel Quality: {quality}"
)


# ============================================================
# TRANSMISSION FLOW
# ============================================================

st.header("🔄 Signal Transmission Flow")

st.info(
    f"Transmitter → {medium} → "
    f"{distance} km → Attenuation → "
    f"Delay → Distortion → Noise → Receiver"
)


# ============================================================
# CHANNEL DIAGNOSIS
# ============================================================

st.header("🩺 Automatic Channel Diagnosis")

diagnosis_col1, diagnosis_col2 = st.columns(2)

with diagnosis_col1:

    st.subheader("🔍 Dominant Impairment")

    st.success(
        f"**{dominant_impairment}** is the dominant "
        f"simulated impairment."
    )

    dominant_value = contributions[
        dominant_impairment
    ]

    st.write(
        f"Estimated contribution to the simulated "
        f"signal change: **{dominant_value:.1f}%**"
    )

with diagnosis_col2:

    st.subheader("📋 Diagnosis")

    if dominant_impairment == "Attenuation":

        st.write(
            "The main degradation is associated with "
            "signal amplitude reduction caused by channel loss."
        )

    elif dominant_impairment == "Delay":

        st.write(
            "The main degradation is associated with "
            "time displacement introduced by the channel."
        )

    elif dominant_impairment == "Distortion":

        st.write(
            "The main degradation is associated with "
            "changes in the signal waveform."
        )

    else:

        st.write(
            "The main degradation is associated with "
            "random noise added to the transmitted signal."
        )


# ============================================================
# IMPAIRMENT FINGERPRINT
# ============================================================

st.header("🧬 Channel Impairment Fingerprint")

fingerprint_data = contributions

fig_fingerprint, ax_fingerprint = plt.subplots(
    figsize=(9, 4)
)

ax_fingerprint.bar(
    list(fingerprint_data.keys()),
    list(fingerprint_data.values())
)

ax_fingerprint.set_ylabel(
    "Relative Contribution (%)"
)

ax_fingerprint.set_xlabel(
    "Impairment"
)

ax_fingerprint.set_title(
    "SignalGuard Impairment Fingerprint"
)

ax_fingerprint.grid(
    axis="y",
    alpha=0.3
)

st.pyplot(
    fig_fingerprint,
    use_container_width=True
)

plt.close(fig_fingerprint)


# ============================================================
# TRANSMITTED VS RECEIVED
# ============================================================

st.header("📈 Transmitted vs Received Signal")

fig1, ax1 = plt.subplots(
    figsize=(12, 5)
)

ax1.plot(
    result["time"],
    result["original"],
    label="Transmitted Signal",
    linewidth=2
)

ax1.plot(
    result["time"],
    result["received"],
    label="Received Signal",
    linewidth=1.5,
    alpha=0.8
)

ax1.set_xlabel(
    "Time (seconds)"
)

ax1.set_ylabel(
    "Amplitude"
)

ax1.set_title(
    "Effect of Transmission Impairments"
)

ax1.legend()

ax1.grid(
    alpha=0.3
)

st.pyplot(
    fig1,
    use_container_width=True
)

plt.close(fig1)


# ============================================================
# PROCESSING STAGES
# ============================================================

st.header("🔬 Channel Processing Stages")

fig2, ax2 = plt.subplots(
    figsize=(12, 6)
)

ax2.plot(
    result["time"],
    result["original"],
    label="Original",
    linewidth=2
)

ax2.plot(
    result["time"],
    result["attenuated"],
    label="After Attenuation"
)

ax2.plot(
    result["time"],
    result["delayed"],
    label="After Delay"
)

ax2.plot(
    result["time"],
    result["distorted"],
    label="After Distortion"
)

ax2.plot(
    result["time"],
    result["received"],
    label="After Noise",
    alpha=0.8
)

ax2.set_xlabel(
    "Time (seconds)"
)

ax2.set_ylabel(
    "Amplitude"
)

ax2.set_title(
    "Progressive Signal Transformation"
)

ax2.legend()

ax2.grid(
    alpha=0.3
)

st.pyplot(
    fig2,
    use_container_width=True
)

plt.close(fig2)


# ============================================================
# NOISE ANALYSIS
# ============================================================

st.header("📡 Noise Analysis")

fig3, ax3 = plt.subplots(
    figsize=(12, 4)
)

ax3.plot(
    result["time"],
    result["noise"],
    linewidth=1
)

ax3.set_xlabel(
    "Time (seconds)"
)

ax3.set_ylabel(
    "Noise Amplitude"
)

ax3.set_title(
    "Noise Introduced into the Channel"
)

ax3.grid(
    alpha=0.3
)

st.pyplot(
    fig3,
    use_container_width=True
)

plt.close(fig3)


# ============================================================
# SIGNAL LOSS
# ============================================================

st.header("📉 Signal Degradation")

signal_loss = result[
    "signal_loss_percent"
]

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Original RMS",
        f"{result['original_rms']:.3f}"
    )

with col2:

    st.metric(
        "Received RMS",
        f"{result['received_rms']:.3f}"
    )

with col3:

    st.metric(
        "RMS Change",
        f"{signal_loss:.2f}%"
    )


# ============================================================
# DISTANCE STRESS TEST
# ============================================================

st.header("🧪 Transmission Distance Stress Test")

st.write(
    "SignalGuard automatically tests different transmission "
    "distances while keeping the selected signal and other "
    "impairment settings unchanged."
)

stress_distances = np.arange(
    5,
    101,
    5
)

stress_health = []

for test_distance in stress_distances:

    stress_result = simulate_channel(
        signal_type,
        frequency,
        sampling_rate,
        duration,
        medium,
        test_distance,
        delay_samples,
        distortion_percent,
        noise_level,
        seed
    )

    stress_health.append(
        stress_result["health"]
    )


stress_health = np.array(
    stress_health
)

fig4, ax4 = plt.subplots(
    figsize=(12, 5)
)

ax4.plot(
    stress_distances,
    stress_health,
    marker="o"
)

ax4.axhline(
    60,
    linestyle="--",
    label="Minimum acceptable health"
)

ax4.set_xlabel(
    "Transmission Distance (km)"
)

ax4.set_ylabel(
    "Channel Health"
)

ax4.set_title(
    "Channel Health vs Transmission Distance"
)

ax4.legend()

ax4.grid(
    alpha=0.3
)

st.pyplot(
    fig4,
    use_container_width=True
)

plt.close(fig4)


# ============================================================
# SAFE OPERATING DISTANCE
# ============================================================

safe_distances = stress_distances[
    stress_health >= 60
]

if len(safe_distances) > 0:

    maximum_safe_distance = np.max(
        safe_distances
    )

    st.success(
        f"📍 Estimated operating range for the current "
        f"conditions: **up to approximately "
        f"{maximum_safe_distance} km**."
    )

else:

    st.warning(
        "⚠️ Under the current impairment settings, "
        "the simulated channel does not reach the "
        "minimum health threshold even at the shortest "
        "tested distance."
    )


# ============================================================
# MEDIUM COMPARISON
# ============================================================

st.header("⚖️ Transmission Medium Comparison")

comparison_results = []

for test_medium in MEDIUM_ATTENUATION.keys():

    comparison = simulate_channel(
        signal_type,
        frequency,
        sampling_rate,
        duration,
        test_medium,
        distance,
        delay_samples,
        distortion_percent,
        noise_level,
        seed
    )

    comparison_results.append(
        {
            "Medium": test_medium,
            "Channel Loss (dB)": round(
                comparison["channel_loss"],
                2
            ),
            "Received RMS": round(
                comparison["received_rms"],
                3
            ),
            "SNR (dB)": (
                "∞"
                if np.isinf(comparison["snr"])
                else round(
                    comparison["snr"],
                    2
                )
            ),
            "Health": round(
                comparison["health"],
                1
            ),
            "Quality": comparison["quality"]
        }
    )


st.dataframe(
    comparison_results,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# RECOVERY SIMULATION
# ============================================================

st.header("🛠️ Automatic Recovery Simulation")

st.write(
    "SignalGuard now tests a possible corrective action "
    "based on the dominant simulated impairment."
)


recovery_medium = medium
recovery_distance = distance
recovery_delay = delay_samples
recovery_distortion = distortion_percent
recovery_noise = noise_level

recovery_action = ""


if dominant_impairment == "Attenuation":

    if medium != "Fiber Optic":

        recovery_medium = "Fiber Optic"

        recovery_action = (
            "Switching the simulated transmission medium "
            "to Fiber Optic."
        )

    else:

        recovery_distance = max(
            1,
            int(distance * 0.7)
        )

        recovery_action = (
            "Reducing the simulated transmission distance."
        )


elif dominant_impairment == "Delay":

    recovery_delay = 0

    recovery_action = (
        "Removing the simulated channel delay."
    )


elif dominant_impairment == "Distortion":

    recovery_distortion = distortion_percent * 0.5

    recovery_action = (
        "Reducing the simulated distortion by 50%."
    )


else:

    recovery_noise = noise_level * 0.5

    recovery_action = (
        "Reducing the simulated noise level by 50%."
    )


recovery_result = simulate_channel(
    signal_type,
    frequency,
    sampling_rate,
    duration,
    recovery_medium,
    recovery_distance,
    recovery_delay,
    recovery_distortion,
    recovery_noise,
    seed
)


recovery_col1, recovery_col2 = st.columns(2)


with recovery_col1:

    st.subheader("Current Channel")

    st.metric(
        "Health",
        f"{health:.1f}/100"
    )

    st.metric(
        "SNR",
        (
            "∞ dB"
            if np.isinf(snr)
            else f"{snr:.2f} dB"
        )
    )

    st.write(
        f"Quality: **{quality}**"
    )


with recovery_col2:

    st.subheader("Simulated Recovery")

    st.metric(
        "Health",
        f"{recovery_result['health']:.1f}/100",
        delta=f"{recovery_result['health'] - health:.1f}"
    )

    recovery_snr = recovery_result["snr"]

    st.metric(
        "SNR",
        (
            "∞ dB"
            if np.isinf(recovery_snr)
            else f"{recovery_snr:.2f} dB"
        )
    )

    st.write(
        f"Quality: **{recovery_result['quality']}**"
    )


st.info(
    f"🔧 **Simulated recovery action:** {recovery_action}"
)


# ============================================================
# BEFORE / AFTER RECOVERY GRAPH
# ============================================================

st.subheader("📊 Recovery Comparison")

fig5, ax5 = plt.subplots(
    figsize=(12, 5)
)

ax5.plot(
    result["time"],
    result["received"],
    label="Before Recovery"
)

ax5.plot(
    recovery_result["time"],
    recovery_result["received"],
    label="After Simulated Recovery"
)

ax5.set_xlabel(
    "Time (seconds)"
)

ax5.set_ylabel(
    "Amplitude"
)

ax5.set_title(
    "Effect of Simulated Recovery"
)

ax5.legend()

ax5.grid(
    alpha=0.3
)

st.pyplot(
    fig5,
    use_container_width=True
)

plt.close(fig5)


# ============================================================
# AUTOMATIC RECOMMENDATION
# ============================================================

st.header("💡 SignalGuard Recommendation")

if dominant_impairment == "Attenuation":

    recommendation = (
        f"The simulation indicates that attenuation is the "
        f"dominant impairment. Consider reducing transmission "
        f"distance or evaluating a lower-loss medium such as "
        f"fiber optic for the simulated conditions."
    )

elif dominant_impairment == "Delay":

    recommendation = (
        f"The simulation indicates that delay is the dominant "
        f"impairment. Investigate the simulated propagation "
        f"delay and evaluate whether lower-delay channel "
        f"conditions are required."
    )

elif dominant_impairment == "Distortion":

    recommendation = (
        f"The simulation indicates that distortion is the "
        f"dominant impairment. Reducing the distortion level "
        f"in the simulated channel improves waveform fidelity."
    )

else:

    recommendation = (
        f"The simulation indicates that noise is the dominant "
        f"impairment. Reducing the simulated noise level "
        f"improves the signal-to-noise ratio."
    )


st.success(
    recommendation
)


# ============================================================
# FINAL SUMMARY
# ============================================================

st.header("📋 Final Channel Summary")

summary_col1, summary_col2 = st.columns(2)


with summary_col1:

    st.write(
        f"**Signal Type:** {signal_type}"
    )

    st.write(
        f"**Transmission Medium:** {medium}"
    )

    st.write(
        f"**Distance:** {distance} km"
    )

    st.write(
        f"**Channel Loss:** {channel_loss:.2f} dB"
    )

    st.write(
        f"**Delay:** {delay_samples} samples"
    )


with summary_col2:

    st.write(
        f"**Distortion:** {distortion_percent}%"
    )

    st.write(
        f"**Noise Level:** {noise_level:.2f}"
    )

    st.write(
        f"**SNR:** "
        f"{'∞' if np.isinf(snr) else f'{snr:.2f}'} dB"
    )

    st.write(
        f"**Dominant Impairment:** "
        f"{dominant_impairment}"
    )

    st.write(
        f"**Channel Health:** {health:.1f}/100"
    )


# ============================================================
# PROJECT NOTE
# ============================================================

st.divider()

st.caption(
    "SignalGuard is an educational simulation. "
    "The Channel Health Score and impairment contribution "
    "values are simulation metrics designed for comparative "
    "analysis and are not standardized network-quality measurements."
)