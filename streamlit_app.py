import time
import threading

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import (
	RTCConfiguration,
	VideoProcessorBase,
	WebRtcMode,
	webrtc_streamer,
)

# Lazy import to avoid blocking app boot if dependencies are heavy
YOLO = None

def get_yolo_class():
	global YOLO
	if YOLO is None:
		from ultralytics import YOLO as _YOLO
		YOLO = _YOLO
	return YOLO

st.set_page_config(page_title="AI Item Detection (Streamlit)", layout="wide")

st.title("AI Item Detection")

with st.expander("Instructions", expanded=True):
	st.markdown(
		"""
		1. Click Start Detection to activate your camera.
		2. Point your camera at an object you want to detect.
		3. Watch as the app identifies the object and draws a box around it.
		4. Click Stop Detection when you are finished.
		"""
	)

col_left, col_right = st.columns([3, 2])

with col_right:
	st.subheader("Controls")
	enable_detection = st.toggle("Start Detection", value=True)
	conf_threshold = st.slider("Confidence threshold", min_value=0.1, max_value=0.9, value=0.25, step=0.05)
	st.caption("Toggle to start/stop detection. Confidence controls the minimum score for boxes.")

with col_left:
	st.subheader("Webcam")


class YOLOProcessor(VideoProcessorBase):
	def __init__(self):
		self.model = None
		self.names = None
		self.confidence_threshold = 0.25
		self.enable = True
		self._model_lock = threading.Lock()

	def _load_model_once(self):
		if self.model is None:
			with self._model_lock:
				if self.model is None:
					YOLOCls = get_yolo_class()
					# Using the nano model for speed
					self.model = YOLOCls("yolov8n.pt")
					self.names = self.model.names

	def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
		img = frame.to_ndarray(format="bgr24")

		if not self.enable:
			return av.VideoFrame.from_ndarray(img, format="bgr24")

		self._load_model_once()

		# Run inference
		results = self.model.predict(img, conf=float(self.confidence_threshold), verbose=False)
		res = results[0]

		if res.boxes is not None and len(res.boxes) > 0:
			for box in res.boxes:
				x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
				conf = float(box.conf[0]) if box.conf is not None else 0.0
				cls_id = int(box.cls[0]) if box.cls is not None else -1
				label = self.names.get(cls_id, str(cls_id)) if isinstance(self.names, dict) else (
					self.names[cls_id] if isinstance(self.names, (list, tuple)) and 0 <= cls_id < len(self.names) else str(cls_id)
				)

				# Draw bounding box
				color = (199, 132, 2)  # BGR
				cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
				caption = f"{label} {conf*100:.0f}%"
				cv2.putText(
					img,
					caption,
					(x1, max(0, y1 - 8)),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.6,
					color,
					2,
					cv2.LINE_AA,
				)

		return av.VideoFrame.from_ndarray(img, format="bgr24")


rtc_configuration = RTCConfiguration({
	"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
})

webrtc_ctx = webrtc_streamer(
	key="ai-detect",
	mode=WebRtcMode.SENDRECV,
	rtc_configuration=rtc_configuration,
	media_stream_constraints={"video": True, "audio": False},
	video_processor_factory=YOLOProcessor,
	async_processing=True,
)

# Bind UI controls to video processor
if webrtc_ctx and webrtc_ctx.video_processor:
	webrtc_ctx.video_processor.enable = bool(enable_detection)
	webrtc_ctx.video_processor.confidence_threshold = float(conf_threshold)

status_placeholder = st.empty()

if webrtc_ctx and webrtc_ctx.video_processor:
	status_text = "Running" if enable_detection else "Paused"
	status_placeholder.info(f"Status: {status_text} | Threshold: {conf_threshold:.2f}")
else:
	status_placeholder.warning("Click the 'Start' button in the webcam widget if it is not connected.")