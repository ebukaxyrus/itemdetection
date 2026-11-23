# Real-Time Object Detection

An interactive Streamlit webcam object-detection prototype.

## Technology stack
- Python
- Streamlit
- OpenCV
- Ultralytics YOLOv8
- streamlit-webrtc
- NumPy

## Verified implementation
- Uses WebRTC to access the webcam
- runs YOLOv8n inference
- overlays bounding boxes and class names
- provides a confidence threshold and detection toggle.

## Getting started

```bash
git clone https://github.com/ebukaxyrus/itemdetection.git
cd itemdetection
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Review the source for required API credentials, data files, local environment settings, and device permissions before running. Never commit real API keys or passwords.

## Important limitations

Requires browser camera permission, WebRTC support, network/model availability and adequate compute. Inference speed is not benchmarked.

## Project structure

- `streamlit_app.py`: primary application entry point.
- `requirements.txt`: Python package requirements.

## Suggested next improvements

- [ ] Test installation from scratch and pin versions.
- [ ] Include real screenshots, verified live URL and a demo walkthrough.
- [ ] Add automated tests and document tested configuration.
- [ ] Verify source/data rights and operational security.

## Maintainer

**Dimakunne Chukwuebuka Francis**  
[GitHub](https://github.com/ebukaxyrus) · [Portfolio](https://xyrusportfolio.netlify.app/) · [XY AI Solution](https://xyaisolution.online/)

*Documentation is grounded in source review; the application was not independently deployed or benchmarked for this update.*
