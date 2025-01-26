import streamlit as st
import h5py
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import plotly.graph_objects as go
import pandas as pd

# ----------------------- Streamlit Page Configuration -----------------------
st.set_page_config(
    page_title="CNC Anomaly Detector",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------- Custom CSS for Status Styling -----------------------
def add_custom_css():
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(135deg, #6a82fb, #fc5c7d);
            color: white;
        }
        .stApp {
            background: linear-gradient(135deg, #6a82fb, #fc5c7d);
            color: white;
        }
        header, footer {
            background: transparent;
        }

        /* Custom styles for anomaly status */
        .anomaly-detected {
            background-color: black;
            color: red;
            font-size: 18px;
            padding: 10px;
            border-radius: 5px;
        }

        .normal-status {
            background-color: green;
            color: white;
            font-size: 18px;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_custom_css()




# ----------------------- Helper Function -----------------------
def pad_or_truncate(data, target_length=4096):
    if len(data) > target_length:
        return data[:target_length]
    elif len(data) < target_length:
        return np.pad(data, ((0, target_length - len(data)), (0, 0)), mode='constant')
    else:
        return data

# ----------------------- Load Model & Scaler -----------------------
try:
    autoencoder = tf.keras.models.load_model(
        "cnc_anomaly_detector.h5",
        custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
    )
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error("Error loading model or scaler. Please check the files.")
    st.stop()

# ----------------------- Sidebar: Model Summary -----------------------
st.sidebar.subheader("Model Summary")
st.sidebar.markdown("""
- **Model Type**: Autoencoder
- **Input Shape**: 4096 x 3 (Tri-axial Vibration Data)
- **Training Duration**: 12 Hours
- **Data Used**: 3 Machines, 2 Years of Production Data
- **Validation Set Size**: 408 Samples
- **Testing Accuracy**: 88%
""")

# ----------------------- Main App Title and Description -----------------------
st.image("cnc_machine.jpg", use_column_width=True, caption="CNC Milling Machine")
st.title("🔧 CNC Milling Machine Anomaly Detection")
st.write("""
This application uses an **Autoencoder model** to detect anomalies in CNC milling machine vibration data.
Upload your data to analyze reconstruction errors and identify anomalies such as tool wear, misalignment, or process failures.
""")

# ----------------------- Performance Metrics -----------------------
st.subheader("Model Performance")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("False Positive Rate", "1.2%", help="Normal samples flagged as anomalies")
with col2:
    st.metric("Recall (Anomalies)", "86%", help="True anomaly detection rate")
with col3:
    st.metric("Training Samples", "1,632", help="Healthy process samples used for training")

col4, col5, col6 = st.columns(3)
with col4:
    st.metric("Precision", "84%", help="Proportion of true positives among detected anomalies")
with col5:
    st.metric("F1 Score", "85%", help="Harmonic mean of Precision and Recall")
with col6:
    st.metric("Specificity", "98%", help="Proportion of normal samples correctly identified as normal")

st.caption("""
**Legend**:
- ✅ Normal: Reconstruction Error < Threshold (95th percentile of training data)
- 🚨 Anomaly: Reconstruction Error ≥ Threshold
*Model trained on 3 machines over 2 years of production data*
""")

# ----------------------- File Upload Section -----------------------
uploaded_file = st.file_uploader(
    "Upload a file with tri-axial vibration data (.h5 or .csv)",
    type=["h5", "csv"]
)

if uploaded_file:
    file_extension = uploaded_file.name.split('.')[-1]

    # Load Data
    try:
        if file_extension == "h5":
            with h5py.File(uploaded_file, 'r') as f:
                data = f['vibration_data'][:]
        elif file_extension == "csv":
            df = pd.read_csv(uploaded_file)
            data = df.values  # Convert DataFrame to numpy array
    except Exception as e:
        st.error("Error reading the uploaded file. Please ensure it is correctly formatted.")
        st.stop()
    original_data_length = len(data)
    # Preprocess
    if data.ndim == 1:
        data = data.reshape(-1, 3)
    data = pad_or_truncate(data, target_length=4096)
    data_normalized = scaler.transform(data.reshape(-1, 3)).reshape(1, 4096, 3)

    # Predict
    reconstruction = autoencoder.predict(data_normalized)
    reconstruction_error = np.mean(np.square(data_normalized - reconstruction), axis=(1, 2))[0]

    # Threshold Calculation (Dynamic)
    training_errors = np.load("training_reconstruction_errors.npy")  # Pre-saved reconstruction errors from training
    threshold = np.percentile(training_errors, 95)

    is_anomaly = reconstruction_error > threshold

    # ----------------------- Data Preview -----------------------
    st.write("Below is a preview of the first few rows of the uploaded data:")
    st.dataframe(data[:5])

# ----------------------- Analysis Results -----------------------
    st.subheader("Analysis Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Reconstruction Error", f"{reconstruction_error:.4f}",
                  delta="- Normal" if not is_anomaly else "+ Anomaly",
                  delta_color="off")
    with col2:
        if is_anomaly:
            st.markdown('<div class="anomaly-detected">Status: 🚨 Anomaly Detected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="normal-status">Status: ✅ Normal Operation</div>', unsafe_allow_html=True)


    # ----------------------- Visualization -----------------------
    st.subheader("Reconstruction Errors Visualization")
    st.write("""
    **Understanding Reconstruction Errors**:
    - Reconstruction error represents the difference between the input signal and its reconstruction.
    - Low errors indicate normal operation, while high errors indicate anomalies.
    Below is a visualization of the reconstruction errors:
    """)

    # Plot Reconstruction Errors
    errors = np.mean(np.square(data.reshape(-1, 3) - reconstruction.reshape(-1, 3)), axis=1)
    indices = list(range(len(errors)))
    anomalies_idx = [i for i, err in enumerate(errors) if err > threshold]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=indices,
        y=errors,
        mode='lines',
        name='Reconstruction Error',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=anomalies_idx,
        y=[errors[i] for i in anomalies_idx],
        mode='markers',
        name='Anomalies',
        marker=dict(color='red', size=8)
    ))

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="green",
        annotation_text="Threshold",
        annotation_position="top right"
    )

    fig.update_layout(
        title="Reconstruction Errors with Anomaly Threshold",
        xaxis_title="Index",
        yaxis_title="Reconstruction Error",
        height=500
    )

    st.plotly_chart(fig)

    # ----------------------- Results -----------------------
    st.subheader("Results")
    st.write(f"**Reconstruction Error**: {reconstruction_error:.4f}")
    st.write(f"**Anomaly Threshold**: {threshold:.4f}")
    st.write(f"**Number of Anomalies Detected**: {len(anomalies_idx)}")
    st.write(f"**Number of Samples**: {original_data_length}")
# ----------------------- Footer -----------------------
st.sidebar.markdown("---")
st.sidebar.write("""
    <div style="text-align: center; color: gray;">
        Developed by <a href="https://github.com/ChashmishCoder" target="_blank">ChashmishCoder</a>
    </div>
    """, unsafe_allow_html=True)
