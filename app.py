# ----------------------- CORRECT STRUCTURE -----------------------

import streamlit as st
import h5py
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

# Set page config
st.set_page_config(
    page_title="CNC Health Check",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(45deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
    }
    .metric-container {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stImage > img {
        width: 100%;
        height: auto;
    }
</style>
""", unsafe_allow_html=True)

# Helper function
def pad_or_truncate(data, target_length=4096):
    if len(data) > target_length:
        return data[:target_length]
    elif len(data) < target_length:
        return np.pad(data, ((0, target_length - len(data)), (0, 0)), mode='constant')
    else:
        return data

# Load model and scaler
autoencoder = tf.keras.models.load_model(
    "cnc_anomaly_detector.h5",
    custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
)
scaler = joblib.load("scaler.pkl")

# UI Layout
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.image("cnc_machine.jpg", caption="Your CNC Machine")

st.title("CNC Machine Health Check")
st.markdown("""
    Upload vibration data to instantly check your machine's operational health  
    Supported format: `.h5` files from Bosch CISS sensors
""")

# File uploader
uploaded_file = st.file_uploader(
    "**Upload Vibration Data File**",
    type=["h5"],
    help="Need sample data? [Download example files](#)"
)

if uploaded_file:
    with h5py.File(uploaded_file, 'r') as f:
        data = f['vibration_data'][:]
    
    # Preprocess
    if data.ndim == 1:
        data = data.reshape(-1, 3)
    data = pad_or_truncate(data, target_length=4096)
    data_normalized = scaler.transform(data.reshape(-1, 3)).reshape(1, 4096, 3)
    
    # Predict
    reconstruction = autoencoder.predict(data_normalized)
    loss = np.mean(np.square(data_normalized - reconstruction), axis=(1, 2))[0]
    vibration_magnitude = np.sqrt(np.sum(data ** 2, axis=1))

    # Metrics Display
    st.markdown("### Analysis Report")
    if loss > 0.1:
        st.error("## 🔴 Critical Alert: Anomaly Detected")
    else:
        st.success("## 🟢 Normal Operation: All Systems Go")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Confidence Score", f"{(1 - loss)*100:.1f}%", help="System confidence in normal operation")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Error Severity", f"{loss:.4f}", help="0 = Perfect health, >0.1 = Critical")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Recommended Action", "Immediate Maintenance" if loss > 0.1 else "Continue Monitoring")
        st.markdown('</div>', unsafe_allow_html=True)

    # Vibration Statistics
    vibration_std = np.std(data, axis=0)
    vibration_peak = np.ptp(data, axis=0)
    st.markdown("### Vibration Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("X-Axis Std. Dev", f"{vibration_std[0]:.4f}")
        st.metric("X-Axis Peak-to-Peak", f"{vibration_peak[0]:.4f}")
    with col2:
        st.metric("Y-Axis Std. Dev", f"{vibration_std[1]:.4f}")
        st.metric("Y-Axis Peak-to-Peak", f"{vibration_peak[1]:.4f}")
    with col3:
        st.metric("Z-Axis Std. Dev", f"{vibration_std[2]:.4f}")
        st.metric("Z-Axis Peak-to-Peak", f"{vibration_peak[2]:.4f}")

    # Signal Visualization
    st.markdown("### Vibration Pattern Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(["X-Axis", "Y-Axis", "Z-Axis", "Magnitude"])
    with tab1:
        st.line_chart(data[:500, 0])
    with tab2:
        st.line_chart(data[:500, 1])
    with tab3:
        st.line_chart(data[:500, 2])
    with tab4:
        st.line_chart(vibration_magnitude[:500])

    # Maintenance Guidance
    st.markdown("---")
    if loss > 0.1:
        st.markdown("""
            ### 🛠 Recommended Maintenance Actions
            1. **Immediate Checks**:
               - Inspect cutting tools for wear/breakage
               - Verify spindle alignment
               - Check hydraulic pressure levels
            2. **Preventative Measures**:
               - Schedule tool replacement
               - Review last maintenance logs
               - Run full diagnostic cycle
            """)
    else:
        st.markdown("""
            ### ✅ Healthy Operation Indicators
            - Consistent vibration patterns
            - No high-frequency anomalies detected
            - All axes within normal operating ranges
            """)


# Add insights from graphs
st.markdown("### 📊 Insights from Graphs")
st.markdown("""
    The graphs above provide a detailed view of vibration patterns across different axes (X, Y, Z) and the combined vibration magnitude:
    
    - **X-Axis Vibration**: The pattern shows periodic spikes, indicating consistent operational behavior. However, any sharp spikes or irregularities might hint at potential misalignment.
    - **Y-Axis Vibration**: A relatively steady pattern with occasional peaks. Peaks could suggest specific load conditions or mechanical stress.
    - **Z-Axis Vibration**: Noticeable irregularity observed in this axis, which might correlate with the identified anomaly. Further investigation is recommended for this direction.
    - **Combined Magnitude**: The overall vibration magnitude graph reveals how combined forces impact the system. Anomalous peaks here directly correspond to critical points in the X, Y, or Z axes.
    
    These insights are crucial for identifying which components or directions require attention, allowing for targeted maintenance.
""")

# Display numeric summary
st.markdown("### 📋 Summary in Numbers")
st.table({
    "Metric": [
        "Confidence Score", 
        "Error Severity", 
        "Recommended Action", 
        "X-Axis Std. Dev", 
        "Y-Axis Std. Dev", 
        "Z-Axis Std. Dev", 
        "X-Axis Peak-to-Peak", 
        "Y-Axis Peak-to-Peak", 
        "Z-Axis Peak-to-Peak"
    ],
    "Value": [
        f"{(1 - loss)*100:.1f}%", 
        f"{loss:.4f}", 
        "Immediate Maintenance" if loss > 0.1 else "Continue Monitoring", 
        f"{vibration_std[0]:.4f}", 
        f"{vibration_std[1]:.4f}", 
        f"{vibration_std[2]:.4f}", 
        f"{vibration_peak[0]:.4f}", 
        f"{vibration_peak[1]:.4f}", 
        f"{vibration_peak[2]:.4f}"
    ]
})

# Maintenance Guidance (repeat for user clarity)
if loss > 0.1:
    st.markdown("""
        ### 🛠 Maintenance Summary:
        - **Critical Issue** detected in the system.
        - Immediate actions are required to prevent further damage.
        - Recommended checks include tool inspection, spindle alignment, and hydraulic systems review.
    """)
else:
    st.markdown("""
        ### ✅ Maintenance Summary:
        - System is **healthy** with no significant anomalies detected.
        - Routine monitoring is recommended to ensure continued optimal performance.
    """)
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray;">
        Made with ❤️ by <a href="https://github.com/ChashmishCoder" target="_blank">ChashmishCoder</a>
    </div>
    """,
    unsafe_allow_html=True
)