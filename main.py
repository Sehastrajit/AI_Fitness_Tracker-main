import av
import os
import streamlit as st
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer
from aiortc.contrib.media import MediaRecorder
import cv2
import mediapipe as mp
from utils import get_mediapipe_pose
from process_frame import ProcessFrame
from thresholds import get_thresholds

def main():
    # Set page config
    st.set_page_config(
        page_title="AI Fitness Trainer - Squats Analysis",
        page_icon="💪",
        layout="wide"
    )

    # Add custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 20px;
        }
        .stTitle {
            font-size: 40px !important;
            color: #0066cc;
            padding-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Title and description
    st.title('AI Fitness Trainer: Squats Analysis')
    st.markdown("""
    This application uses AI to analyze your squats form in real-time. 
    It provides instant feedback on your posture and counts your correct and incorrect repetitions.
    
    **Instructions:**
    1. Allow camera access when prompted
    2. Position yourself sideways to the camera
    3. Ensure your full body is visible
    4. Start performing squats
    """)

    # Initialize components
    thresholds = get_thresholds()
    live_process_frame = ProcessFrame(thresholds=thresholds, flip_frame=True)
    pose = get_mediapipe_pose()

    # Callback to process video frames
    def video_frame_callback(frame: av.VideoFrame):
        frame = frame.to_ndarray(format="rgb24")
        frame, _ = live_process_frame.process(frame, pose)
        return av.VideoFrame.from_ndarray(frame, format="rgb24")

    # WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="squats-analysis",
        video_frame_callback=video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": {"width": {'min':480, 'ideal':720}, "height": {'min':480, 'ideal':720}},
            "audio": False
        },
        video_html_attrs=VideoHTMLAttributes(
            autoPlay=True,
            controls=False,
            muted=True,
            style={"width": "100%", "height": "100%"}
        )
    )

    # Instructions column
    with st.expander("Common Mistakes to Avoid"):
        st.markdown("""
        - **Knees Falling Over Toes**: Keep your knees aligned with your toes
        - **Improper Depth**: Don't squat too deep or too shallow
        - **Back Not Straight**: Maintain a neutral spine throughout the movement
        - **Poor Hip Hinge**: Push your hips back as you descend
        - **Heels Rising**: Keep your heels planted firmly on the ground
        """)

if __name__ == "__main__":
    main()
