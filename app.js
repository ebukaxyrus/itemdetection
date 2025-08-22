(function () {
	const startButton = document.getElementById('startBtn');
	const stopButton = document.getElementById('stopBtn');
	const statusEl = document.getElementById('status');
	const videoEl = document.getElementById('video');
	const canvasEl = document.getElementById('overlay');
	const ctx = canvasEl.getContext('2d');

	/** @type {cocoSsd.ObjectDetection | null} */
	let model = null;
	/** @type {MediaStream | null} */
	let mediaStream = null;
	/** @type {number | null} */
	let rafId = null;
	let running = false;

	function setStatus(message) {
		statusEl.textContent = message;
	}

	async function loadModelOnce() {
		if (model) return model;
		setStatus('Loading model...');
		model = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
		setStatus('Model loaded. Ready.');
		return model;
	}

	async function startCamera() {
		if (mediaStream) return mediaStream;
		setStatus('Requesting camera...');
		const constraints = {
			video: {
				facingMode: { ideal: 'environment' },
				width: { ideal: 1280 },
				height: { ideal: 720 }
			},
			audio: false
		};
		mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
		videoEl.srcObject = mediaStream;
		await new Promise((resolve) => {
			if (videoEl.readyState >= 2) return resolve();
			videoEl.onloadedmetadata = () => resolve();
		});
		await videoEl.play();
		resizeCanvasToVideo();
		setStatus('Camera started.');
		return mediaStream;
	}

	function stopCamera() {
		if (!mediaStream) return;
		mediaStream.getTracks().forEach((t) => t.stop());
		mediaStream = null;
		videoEl.srcObject = null;
	}

	function resizeCanvasToVideo() {
		const videoWidth = videoEl.videoWidth || canvasEl.clientWidth;
		const videoHeight = videoEl.videoHeight || canvasEl.clientHeight;
		const dpr = window.devicePixelRatio || 1;
		canvasEl.width = Math.floor(videoWidth * dpr);
		canvasEl.height = Math.floor(videoHeight * dpr);
		canvasEl.style.width = videoWidth + 'px';
		canvasEl.style.height = videoHeight + 'px';
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
	}

	function clearCanvas() {
		ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
	}

	function drawDetections(predictions) {
		clearCanvas();
		ctx.lineWidth = 2;
		ctx.strokeStyle = 'rgba(2, 132, 199, 1)';
		ctx.fillStyle = 'rgba(2, 132, 199, 0.85)';

		for (const pred of predictions) {
			const [x, y, w, h] = pred.bbox;
			ctx.strokeRect(x, y, w, h);

			const label = `${pred.class} ${(pred.score * 100).toFixed(0)}%`;
			const paddingX = 6;
			const paddingY = 4;
			const metrics = ctx.measureText(label);
			const labelWidth = metrics.width + paddingX * 2;
			const labelHeight = 16 + paddingY * 2;

			ctx.fillRect(x, y - labelHeight, labelWidth, labelHeight);
			ctx.fillStyle = 'white';
			ctx.font = '12px system-ui, -apple-system, Segoe UI, Roboto, Arial';
			ctx.fillText(label, x + paddingX, y - paddingY);
			ctx.fillStyle = 'rgba(2, 132, 199, 0.85)';
		}
	}

	async function detectLoop() {
		if (!running || !model || !videoEl || videoEl.readyState < 2) return;
		try {
			const predictions = await model.detect(videoEl);
			drawDetections(predictions);
			setStatus(predictions.length ? `Detected ${predictions.length} object(s)` : 'No objects detected');
		} catch (err) {
			console.error(err);
			setStatus('Detection error. See console');
		}
		runNextFrame();
	}

	function runNextFrame() {
		rafId = requestAnimationFrame(detectLoop);
	}

	async function start() {
		if (running) return;
		running = true;
		startButton.disabled = true;
		stopButton.disabled = false;
		try {
			await loadModelOnce();
			await startCamera();
			resizeCanvasToVideo();
			runNextFrame();
		} catch (err) {
			console.error(err);
			setStatus('Unable to start. Check camera permissions.');
			startButton.disabled = false;
			stopButton.disabled = true;
			running = false;
		}
	}

	function stop() {
		if (!running) return;
		running = false;
		if (rafId !== null) {
			cancelAnimationFrame(rafId);
			rafId = null;
		}
		stopCamera();
		clearCanvas();
		setStatus('Stopped.');
		startButton.disabled = false;
		stopButton.disabled = true;
	}

	window.addEventListener('resize', () => {
		if (videoEl.readyState >= 2) resizeCanvasToVideo();
	});

	startButton.addEventListener('click', start);
	stopButton.addEventListener('click', stop);

	// Prefetch model weights silently after page load
	window.addEventListener('load', () => {
		loadModelOnce().catch(() => {});
	});
})();