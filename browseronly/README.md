# OTD In-Browser Duplex Optical Application (100% Client-Side Vision)

A zero-dependency, bidirectional air-gap data transmission and reception suite running in the web browser.

---

## Features

### 1. 📤 Outbound Sending Engine
- **Air-Gap Keyboard Typer**:
  - Automatically translates staged files into base64 packets typed directly into RDP, VNC, or local terminal windows (`xlib` / `pyautogui` backend).
  - **Optical ACK Lock-Step**: Automatically listens for `client.py`'s optical `ACK2` grid (<1ms JS computer vision). Confirms focus and steps chunks reliably.
  - **Speed Presets**: Turbo (0.0ms delay), Fast (0.2ms delay), Balanced (1.0ms delay), and Safe (4.0ms delay).
  - **Terminal Quick-Chat**: Injects commands or text directly into remote sessions.
  - **Emergency Stop**: Instantly halts keystroke transmission.
- **Visual Screen Broadcast (Pure Client-Side)**:
  - Generates optical Q2 frames locally with HTML5 canvas.
  - Full-screen support, 10–60 FPS adjustment, Cauchy Reed-Solomon GF($2^8$) parity frames.

### 2. 📥 In-Browser Optical Receiver
- **Flexible Video Sources**: Decode directly from **🖥️ Screen / Window Sharing** (`getDisplayMedia`) OR **📷 Physical Webcams / USB Cameras** (`getUserMedia`) with live device selection and mirror-flip toggles.
- **Real-Time Computer Vision**: Real-time chromatic fiducial segmentation and closed-form quad homography dewarping.
- **Vectorized Cell Sampling**: Reads R, G, B channels at 60 FPS with calibrated thresholds.
- **Error Correction & Verification**: IEEE 802.3 CRC32, Cauchy RS FEC reconstruction, zlib decompression, and SHA-256 verification.
- **Zero Video Streaming Lag**: Computer vision executes inside the browser process on GPU/Canvas memory instead of streaming video to Python.

### 3. 📁 Received File Library
- Browse, download, delete, or 1-click stage received files back into the outbound keyboard sender.

### 4. ⚡ Enhanced Remote Client (`browseronly/client.py`)
- Standalone, zero-dependency duplex receiver for the remote target.
- **Active Optical Heartbeat**: Periodically broadcasts live optical poll frames even during retries so long transfers never trigger the server's dead-man's switch.
- **Anti-Stall Pulse**: Prevents window compositors from pausing video delivery on static windows.

---

## Quick Start

### Start Duplex Server
```bash
python3 browseronly/run.py
```
Automatically launches `http://localhost:8088/index.html` in your default browser.

### Key Server Endpoints
- `GET /api/status`: Current sender telemetry, typer backend, speed preset, and library count.
- `GET /api/files`: List of received files in `./received`.
- `GET /download/<filename>`: Download a file from `./received`.
- `POST /api/upload`: Stage a file for outbound keyboard typing.
- `POST /api/ack`: Forward client optical ACK frame decoded by in-browser JS vision to backend typer.
- `POST /api/save_file`: Save reconstructed received file to `./received`.
- `POST /api/stage_received`: Stage a file directly from `./received` to transmit back.
- `POST /api/speed`: Change speed preset (`turbo`, `fast`, `balanced`, `safe`).
- `POST /api/abort`: Immediate emergency stop.

---

## Automated Tests

Run the test suite verifying all duplex endpoints, optical ACK routing, and file handling:
```bash
python3 tests/test_browseronly_duplex.py
```
