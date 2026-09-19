import streamlit as st
import cv2
import torch
import numpy as np
import time
from pathlib import Path
import sys

# Change layout to wide for premium feel
st.set_page_config(page_title="ISL Translator", page_icon="🤟", layout="wide")

# Custom CSS for glassmorphism and premium design
st.markdown("""
<style>
    .reportview-container {
        background: linear-gradient(135deg, #1f1c2c 0%, #928DAB 100%);
    }
    .main {
        background-color: transparent;
    }
    .stApp {
        background-color: #0E1117;
    }
    .metric-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .title-text {
        color: #00F2FE;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        letter-spacing: -1px;
    }
    .subtitle {
        color: #A0AEC0;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Add project root to path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))
try:
    from src.preprocessing.keypoint_extractor import KeypointExtractor
    from src.models.isl_model import ISLTranslator
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False

@st.cache_resource
def load_model(checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not Path(checkpoint_path).exists():
        return None, None
    model = ISLTranslator.load(checkpoint_path, use_translation=False).to(device).eval()
    extractor = KeypointExtractor()
    return model, extractor, device

def main():
    st.markdown('<h1 class="title-text">⚡ NidhiAI Sign Translator</h1>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Real-time Continuous Indian Sign Language Translation running on ST-GCN + Transformer Engine</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown('### ⚙️ Engine Settings')
        checkpoint_path = st.text_input("Checkpoint Path", value="checkpoints/best.pt")
        window_size = st.slider("Context Window (Frames)", min_value=16, max_value=128, value=64)
        run_demo = st.button("🚀 Start Engine", use_container_width=True, type="primary")
        stop_demo = st.button("⏹️ Stop", use_container_width=True)

        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        translation_placeholder = st.empty()
        translation_placeholder.markdown("### 🗣️ Translation\n*Waiting for video...*")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        gloss_placeholder = st.empty()
        gloss_placeholder.markdown("### 📝 Raw Glosses\n*Waiting for video...*")
        st.markdown('</div>', unsafe_allow_html=True)

    with col1:
        st.markdown('### 📹 Live Feed')
        video_placeholder = st.empty()
        
        if not HAS_MODEL:
            st.warning("Core ML engine not found. Ensure application is initialized from project root.")
            return

        if run_demo:
            model, extractor, device = load_model(checkpoint_path)
            if model is None:
                st.error(f"Model checkpoint not found at {checkpoint_path}.")
                return
            
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Could not connect to WebCam.")
                return

            keypoint_buffer = []
            last_translate_time = time.time()
            translate_interval = 0.5  # Translate every half second
            
            progress = st.progress(0, text="Engine initialization complete...")

            try:
                while True:
                    if stop_demo:
                        break
                        
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Lost webcam feed.")
                        break

                    frame = cv2.flip(frame, 1) # Mirror
                    
                    # Extract keypoints
                    keypoints, metadata = extractor.extract_frame(frame)
                    keypoint_buffer.append(keypoints)
                    
                    if len(keypoint_buffer) > window_size * 2:
                        keypoint_buffer = keypoint_buffer[-window_size:]
                        
                    # Translate periodically
                    if len(keypoint_buffer) >= 16 and (time.time() - last_translate_time > translate_interval):
                        seq = torch.tensor(np.array(keypoint_buffer[-window_size:]), dtype=torch.float32).unsqueeze(0).to(device)
                        lengths = torch.tensor([len(seq[0])]).to(device)
                        
                        with torch.no_grad():
                            glosses, translations = model.translate(seq, lengths, beam_width=5)
                            
                        gloss_placeholder.markdown(f"### 📝 Raw Glosses\n<div style='color:#A0AEC0; font-family:monospace;'>{' • '.join(glosses[0])}</div>", unsafe_allow_html=True)
                        if translations and translations[0]:
                            translation_placeholder.markdown(f"### 🗣️ Translation\n<h2 style='color:#00F2FE;'>{translations[0]}</h2>", unsafe_allow_html=True)
                        else:
                            translation_placeholder.markdown(f"### 🗣️ Translation\n<h2 style='color:#00F2FE;'>{' '.join(glosses[0])}</h2>", unsafe_allow_html=True)
                        
                        last_translate_time = time.time()
                        
                    # Visualization drawing using standard OpenCV
                    h, w = frame.shape[:2]
                    # Pose
                    if metadata["pose_detected"]:
                        for i in range(33):
                            if keypoints[i][2] > 0.5:
                                cv2.circle(frame, (int(keypoints[i][0]*w), int(keypoints[i][1]*h)), 3, (0, 255, 100), -1)
                    # Hands
                    hand_start = 501
                    if metadata["left_hand_detected"]:
                        for i in range(21):
                            cx, cy = int(keypoints[hand_start+i][0]*w), int(keypoints[hand_start+i][1]*h)
                            if cx > 0 and cy > 0: cv2.circle(frame, (cx, cy), 4, (255, 100, 100), -1)
                    if metadata["right_hand_detected"]:
                        for i in range(21):
                            cx, cy = int(keypoints[hand_start+21+i][0]*w), int(keypoints[hand_start+21+i][1]*h)
                            if cx > 0 and cy > 0: cv2.circle(frame, (cx, cy), 4, (100, 100, 255), -1)
                    
                    # Buffer status
                    progress.progress(min(len(keypoint_buffer) / window_size, 1.0), text=f"Buffer: {len(keypoint_buffer)} frames")

                    # Convert for Streamlit
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame, channels="RGB", use_container_width=True)
                    
            finally:
                cap.release()
                extractor.close()

if __name__ == "__main__":
    main()
