import streamlit as st
import cv2 as cv
import numpy as np

# Body Parts & Pose Pairs
BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

# Load model
net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

def pose_estimation(frame):
    frameHeight, frameWidth = frame.shape[:2]
    inWidth = 368
    inHeight = 368

    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight),
                                      (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > 0.2 else None)

    for pair in POSE_PAIRS:
        partFrom, partTo = pair
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 2)
            cv.circle(frame, points[idFrom], 4, (0, 0, 255), -1)
            cv.circle(frame, points[idTo], 4, (0, 0, 255), -1)
    return frame

# ------------- Streamlit App UI -------------
st.set_page_config(page_title="Human Pose Estimation", page_icon="🤖", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🤖 Human Pose Estimation</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Ứng dụng nhận diện dáng người bằng AI - Đồ án Digital Image Processing</p>", unsafe_allow_html=True)

st.markdown("---")

option = st.selectbox("📸 Chọn nguồn ảnh:", ["📂 Upload Ảnh", "🎥 Dùng Webcam"])

# Styling buttons
button_style = """
    display: inline-block;
    padding: 0.5em 1em;
    margin: 0.5em;
    border: none;
    border-radius: 8px;
    background: #4CAF50;
    color: white;
    font-size: 16px;
    cursor: pointer;
"""

# --- Option 1: Upload ---
if option == "📂 Upload Ảnh":
    uploaded_file = st.file_uploader("🖼️ Chọn ảnh từ máy tính", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv.imdecode(file_bytes, 1)
        st.image(image, caption="📷 Ảnh gốc", use_container_width=True)

        if st.button("🚀 Nhận diện Pose", use_container_width=True):
            result = pose_estimation(image.copy())
            st.image(result, caption="✅ Kết quả nhận diện", use_container_width=True)

# --- Option 2: Webcam ---
if option == "🎥 Dùng Webcam":
    picture = st.camera_input("📸 Chụp ảnh bằng Webcam")

    if picture is not None:
        file_bytes = np.asarray(bytearray(picture.read()), dtype=np.uint8)
        image = cv.imdecode(file_bytes, 1)
        st.image(image, caption="📷 Ảnh đã chụp", use_container_width=True)

        if st.button("🚀 Nhận diện Pose từ Webcam", use_container_width=True):
            result = pose_estimation(image.copy())
            st.image(result, caption="✅ Kết quả Pose từ Webcam", use_container_width=True)

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Made with ❤️ by Group 12 - 2025</p>", unsafe_allow_html=True)
