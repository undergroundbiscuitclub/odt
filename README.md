# otd: High-Speed Optical Data Transfer (Air-Gap & RDP)

**otd** (Optical Data Transfer) is an optical data transfer suite designed to transmit files across isolated networks, locked-down RDP/VDI sessions, and air-gapped environments. It transforms arbitrary binary files into high-density animated visual 2D color data grids on a screen, and reconstructs the original files from video recordings or real-time screen captures.

<img width="1510" height="823" alt="Screenshot From 2026-09-18 09-33-56" src="https://github.com/user-attachments/assets/852cfcac-b2d8-4f0c-92e5-cff194fe426e" />


---

## Features

- **High Throughput & 1-Pass FEC:** Uses 8-color RGB multiplexing (3 bits per cell), binary framing, level-9 `zlib` compression, and **systematic Cauchy Reed-Solomon $GF(2^8)$ Forward Error Correction (`--fec 15`)**. Reconstructs missing/dropped frames in single-digit milliseconds so transfers complete on Pass 1 without waiting for a loop!
- **Persistent WebSocket Streaming (45–60 FPS):** Web capture (`--http`) connects via a persistent binary WebSocket (`/ws`) with real-time automatic ROI bounding-box cropping (`auto_roi`), boosting capture framerates from 15 FPS to 60 FPS.
- **Subpixel Fiducial Corner Refinement:** Corner fiducials are refined to subpixel precision using `cv2.cornerSubPix`, ensuring razor-sharp cell sampling across arbitrary resolutions and high-DPI scaling.
- **Real-Time Fiducial Color Calibration:** Corner chromatic fiducials (Red, Green, Blue, Magenta) and black/white reference borders are sampled dynamically per frame to calculate empirical BGR thresholds, making the receiver immune to screen gamma, night-light, and auto-exposure shifts.
- **Scale & Perspective Invariance:** 4-corner concentric chromatic fiducials allow the decoder to mathematically dewarp the grid using `cv2.warpPerspective`. Works across RDP smart sizing, DPI scaling, stretched window aspect ratios, and slight camera or monitor tilt.
- **Pure Headless Shell Mode (`--tty`):** Transmit files from headless Linux servers, SSH sessions, serial consoles, or minimal containers with zero GUI, X11, Wayland, or Tkinter requirements. Features 1-cell outer black isolation margins and symmetric grid centering so transmissions decode reliably across terminal window borders, prompts, and status lines.
- **Multiple Decoding Modes:**
  - **Live Web Capture Tool (`--http`):** **Recommended on Wayland!** Spins up a local web application that uses the browser's native PipeWire / xdg-desktop-portal window and screen sharing with WebSocket binary streaming. Zero extra dependencies.
  - **Video File Mode:** Decodes recorded `.mp4` / `.mkv` / `.avi` / `.webm` video files.
  - **Live Screen Mode (`--live`):** Captures and decodes the transmission in real-time from your monitor using `mss` (requires X11 / Xorg).
  - **Camera Stream Mode (`--camera 0`):** Decodes live transmission through an external camera or webcam.
- **Guaranteed Data Integrity:**
  - **Per-Frame IEEE 802.3 CRC32:** Instantly discards blurry, dropped, or corrupted frames.
  - **End-to-End Cryptographic SHA-256:** Embeds the original file hash in stream metadata and verifies byte-for-byte fidelity upon completion.
- **Paused-on-Launch Safety:** Transmitters display Frame 1 statically and wait for a button click or `[Space]`/`[Enter]` keypress before cycling frames.

---

## Core Applications Matrix

| Script | Primary Role | Direction | Dependencies | Description |
| :--- | :--- | :--- | :--- | :--- |
| **[`server_v2.py`](#unified-bidirectional-server-server_v2py)** | **Unified Server (Both)** | **Bidirectional** (Send & Receive) | `opencv`, `numpy` | **All-in-One Web App:** Combines keyboard sender and visual optical receiver in a single browser interface. |
| **[`server.py`](#full-duplex-bidirectional-transfer-clientpy--serverpy)** | Duplex Sender | Host ➔ Remote Client | `opencv`, `numpy` | Stages files on host and types alphanumeric packets into remote client with lock-step optical ACKs and dead-man switch. |
| **[`client.py`](#receiver-clientpy)** | Duplex Receiver | Host ➔ Remote Client | **Pure Python** (Zero dependencies) | Reads incoming keyboard keystrokes on remote target and displays structured optical `ACK2` grid (GUI or `--tty`). |
| **[`decoder_v2.py`](#receiver-options-decoder_v2py)** | Optical Receiver | Remote ➔ Host | `opencv`, `numpy`, `mss` | Captures optical frames via web screen share (`--http`), camera (`--camera`), or video file; auto-loads ONNX ML tracking. |
| **[`encoder_v2.py`](#transmitter-options-encoder_v2py)** | Optical Transmitter | Remote ➔ Host | Standard (`tkinter`) | Displays animated 8-color RGB data frames with corner fiducials and Reed-Solomon FEC in GUI or `--tty` terminal mode. |
| **[`encoder_min.py`](#minimal-standalone-transmitter-encoder_minpy)** | Minimal Transmitter | Remote ➔ Host | **Pure Python** (Zero dependencies) | Ultra-compact (~250 lines) self-contained script for air-gapped hosts; copy-paste friendly with built-in FEC. |

---

## Performance & Profile Comparison

| Profile | Grid Size | Color Mode | Payload / Frame | Speed @ 15 FPS | Typical 20MB Transfer Time* |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Legacy v1** | 16 x 16 | 1-bit B/W (Base64) | 28 bytes | ~0.22 KB/s | ~25 hours |
| **v2 Standard** | 64 x 64 | 3-bit RGB (8 Colors) | 1,520 bytes | ~23 KB/s | ~175 seconds |
| **v2 High-Speed** | 96 x 96 | 3-bit RGB (8 Colors) | 3,440 bytes | ~51 KB/s | ~78 seconds |
| **v2 Maximum** | 128 x 128 | 3-bit RGB (8 Colors) | 6,128 bytes | ~92 KB/s | ~44 seconds |
| **v2 Shell (`--tty`)** | 48 x 48 | 3-bit RGB (8 Colors) | 848 bytes | ~11 KB/s | ~360 seconds |

*\*Assuming standard ~4 MB compressed payload for logs, binaries, documents, or source code archives.*

---

## Installation

### Receiver / Decoder Setup

Install Python 3.8+ and the decoder dependencies:

```bash
# Clone or navigate to the repository
git clone https://github.com/your-username/otd.git
cd otd

# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install decoder dependencies
pip install -r requirements.txt
```

### Dependencies (`requirements.txt`)

- `opencv-python>=4.8.0` — Image processing, color segmentation, and perspective dewarping.
- `numpy>=1.24.0` — Fast vector upsampling and signal sampling.
- `mss>=9.0.0` — Low-latency desktop screen capture for `--live` decoding mode.

*(Note: The transmitter scripts require only Python 3 and standard libraries; Tkinter is needed only for GUI mode).*

---

## Quickstart Guide

### Step 1: Start the Transmitter (`encoder_v2.py`)

On the machine holding the file (e.g. inside an RDP session, VM, or remote server):

#### Option A: Desktop GUI Window Mode
```bash
python3 encoder_v2.py payload.zip
```

#### Option B: Headless Terminal / SSH Mode
```bash
python3 encoder_v2.py payload.zip --tty
```

1. The transmitter loads the file and displays **Frame 1** in a paused `READY` state.
2. Position the window or focus the screen.
3. Click the green **`▶ START TRANSFER`** button or press **`[Space]`** / **`[Enter]`** to begin broadcasting.

---

### Step 2: Receive and Decode (`decoder_v2.py`)

On the receiving machine:

#### Option A: Live Web Capture (`--http`) — ★ Recommended for Wayland
On Wayland (GNOME, KDE, Sway, etc.), browser screen capture has native access to hardware-accelerated desktop and window streams via PipeWire:

```bash
python3 decoder_v2.py --http
```

1. Starts a local receiver at `http://localhost:8080` and opens your default browser.
2. Click **`📺 Share Window / Screen`**. In the browser dialog, select the transmitter window or your screen.
3. **Transmission Area Masking (ROI):** Click **`Draw Mask`** and drag a rectangle over the transmitter, or click **`Auto-Snap`** (enabled by default via **`Auto-Lock`**). The web app visually shades everything outside the transmission grid and crops outgoing image frames, saving 50–80% bandwidth and CPU encode/decode load.
4. The receiver locks onto the optical fiducials, streams frames, verifies IEEE 802.3 CRC32 per frame, reassembles the file, verifies SHA-256, and writes it directly to disk (with an optional in-browser download button).

#### Option B: From a Video Recording
Record the transmitting window using your screen recorder (e.g., OBS, GNOME Screen Recorder `PrtScn`, QuickTime) and decode the recorded `.mp4`:

```bash
python3 decoder_v2.py recording.mp4
```

#### Option C: Native Screen Capture (`--live`, X11 / Xorg only)
Open the decoder in live desktop capture mode to capture from an X11 display:

```bash
python3 decoder_v2.py --live
```

#### Option D: Live Webcam / Camera
Capture transmission directly through an external webcam:

```bash
python3 decoder_v2.py --camera 0
```

---

## CLI Reference

### Transmitter Options (`encoder_v2.py`)

```text
usage: encoder_v2.py [-h] [--tty] [--grid {32,48,64,80,96,128}]
                     [--cell-size CELL_SIZE] [--delay DELAY] [--mode {rgb,bw}]
                     [--fec FEC]
                     file

positional arguments:
  file                  Path to file to transmit

options:
  -h, --help            Show help message and exit
  --tty                 Run directly in shell/terminal (headless mode, no GUI required)
  --grid, -g {32,48,64,80,96,128}
                        Grid dimension (default: 64 for GUI, 48 for --tty)
  --cell-size, -s CELL_SIZE
                        Pixel size of each cell in GUI mode (default: auto)
  --delay, -d DELAY     Frame delay in milliseconds (default: 70ms for GUI, 80ms for --tty)
  --mode, -m {rgb,bw}   Color mode: rgb (8 colors, 3 bits/cell) or bw (2 colors, 1 bit/cell)
  --fec FEC             Forward Error Correction parity percentage (default: 15, 0 to disable)
```

#### Interactive Controls (GUI & `--tty`):
- **`[Space]` / `[Enter]`**: Start transmission / Pause / Resume
- **`[◀]` / `[▶]`** (Arrow keys): Step single frame backward / forward
- **`[▲]` / `[▼]`** (Arrow keys): Adjust frame speed on the fly (delay ±10ms)
- **`q`** or **`Esc`**: Quit cleanly

---

### Minimal Standalone Transmitter (`encoder_min.py`)

For air-gapped hosts, locked-down RDP/VDI environments, or remote jump boxes where you cannot install packages:

- **Zero Dependencies:** Pure Python 3 standard library only (`sys`, `os`, `time`, `zlib`, `struct`, `hashlib`). Does **not** require NumPy, OpenCV, or external libraries.
- **Built-in Cauchy Reed-Solomon FEC:** Includes pure Python $GF(2^8)$ parity generation (`--fec 15`), recovering dropped or skipped frames instantly on the receiver.
- **Keyboard Emulation & Copy/Paste Friendly:** Compact self-contained script (~250 lines) designed to be typed or pasted onto target hosts via USB Rubber Ducky, KVM keystroke sender, or terminal paste.
- **Dual Display Support:** Automatically displays a scalable/resizable Tkinter GUI window if a display is present; automatically falls back to 24-bit ANSI terminal mode (`--tty`) on headless/SSH consoles.

```bash
# Run minimal transmitter (auto-detects GUI or terminal with 15% FEC):
python3 encoder_min.py payload.zip

# Force headless terminal ANSI mode with custom grid:
python3 encoder_min.py payload.zip --tty --grid 48 --fec 20
```

---

### Receiver Options (`decoder_v2.py`)

```text
usage: decoder_v2.py [-h] [--http [HTTP]] [--port PORT] [--no-browser]
                     [--live] [--camera [CAMERA]] [--no-preview]
                     [--monitor MONITOR] [--model MODEL_PATH] [-o OUTPUT_FILE]
                     [input] [output]

positional arguments:
  input                 Path to input video file (omit if using --http / --live / --camera)
  output                Optional output filepath (default: original filename from metadata)

options:
  -h, --help            Show help message and exit
  -g, --grid GRID_SIZE  Lock to specific grid size (e.g. 32, 48, 64, 80, 96, 128; auto-detected if omitted)
  --http [PORT]         Start web-based screen/window capture receiver (default port: 8080)
  --port PORT           Custom port for --http web capture server (default: 8080)
  --no-browser          Do not automatically launch web browser when using --http
  -o, --output-file OUT Custom output filepath (alternative to positional argument)
  --live                Capture live from screen instead of reading a video file (requires X11 / Xorg)
  --camera, --cam [CAM] Capture live from webcam/camera device index (e.g. --camera 0)
  --no-preview          Disable OpenCV GUI preview window (faster on headless receivers)
  --monitor MONITOR     Monitor index for live capture (default: 1)
  --model, --onnx PATH  Optional ONNX model for ML-accelerated fiducial tracking
```

> [!TIP]
> **Wayland Live Capture Solution (`--http`):**
> On Wayland systems (such as GNOME or KDE on Wayland), the Wayland security model blocks background desktop screen grabbing via X11 (`mss`), resulting in all-black frames when using `--live`.
> - **Best Solution on Wayland:** Run `python3 decoder_v2.py --http`. This opens a web capture app that leverages the browser's native Wayland PipeWire portal (`getDisplayMedia`) to capture any screen or window directly at 30–60 FPS with zero extra dependencies.
> - **Alternative:** Record a video of the transmitting window (`PrtScn` or OBS) and decode with `python3 decoder_v2.py <recording.mp4>`, or point an external camera with `python3 decoder_v2.py --camera 0`.

---

## Technical Protocol Specifications

### 1. Optical Chromatic Fiducials
Four concentric chromatic fiducials anchor the grid:
- **Top-Left (TL):** Red Core (`#FF0000`)
- **Top-Right (TR):** Green Core (`#00FF00`)
- **Bottom-Left (BL):** Blue Core (`#0000FF`)
- **Bottom-Right (BR):** Magenta Core (`#FF00FF`)

Each marker features an outer white boundary, middle black buffer, and saturated color core. The decoder calculates the subpixel centroid of each corner and applies perspective dewarping to a canonical 1000x1000 coordinate plane.

### 2. 8-Color RGB Multiplexing
Each cell represents 3 bits mapped to the vertices of the RGB color cube:

| Bits | Color | Hex / RGB |
| :---: | :--- | :--- |
| `000` | Black | `#000000` (0, 0, 0) |
| `001` | Blue | `#0000FF` (0, 0, 255) |
| `010` | Green | `#00FF00` (0, 255, 0) |
| `011` | Cyan | `#00FFFF` (0, 255, 255) |
| `100` | Red | `#FF0000` (255, 0, 0) |
| `101` | Magenta | `#FF00FF` (255, 0, 255) |
| `110` | Yellow | `#FFFF00` (255, 255, 0) |
| `111` | White | `#FFFFFF` (255, 255, 255) |

Sampling the center 3x3 patch of each cell provides a ±127 pixel noise threshold per channel, ensuring high resistance to lossy H.264 video compression and YUV 4:2:0 chroma subsampling.

### 3. Binary Frame Header (16 Bytes)
```text
[ 2B: Magic "Q2" ] [ 1B: Ver (2) ] [ 1B: Mode ] [ 1B: GridSize ] [ 1B: Flags ]
[ 2B: ChunkIdx ]   [ 2B: TotalChunks ] [ 2B: PayloadLen ] [ 4B: IEEE 802.3 CRC32 ]
```

### 4. Stream Metadata Block (Embedded in Chunk 0)
```text
[ 4B: Magic "QSMD" ] [ 8B: OrigSize ] [ 8B: CompressedSize ]
[ 32B: SHA-256 Hash ] [ 2B: FilenameLength ] [ N Bytes: Filename UTF-8 ]
[ Remainder: Start of Level-9 zlib stream ]
```

---

## Full-Duplex Bidirectional Transfer (`client.py` & `server.py`)

In air-gap and locked-down RDP/VDI environments, clipboard and file copy are disabled, but the local machine can view the remote screen and send keystrokes. **otd Duplex** creates a bidirectional, full-duplex communication channel:

- **Client -> Server (Visual Optical Feed):** `client.py` displays 4 chromatic fiducials and an optical ACK grid (`ACK2`). `server.py` captures the grid via browser screen share (`navigator.mediaDevices.getDisplayMedia`) with transmission masking.
- **Server -> Client (Keyboard Channel):** `server.py` types file data packets directly into the focused client window using standard alphanumeric characters (`0-9`, `a-f`, packet identifiers `h`/`d`/`e`/`m`) and CRC32 verification, completely immune to keyboard layout symbol differences.

### 5-Step Duplex Workflow:

1. **Start the Receiver (`client.py`):**
   ```bash
   python3 client.py
   # Or headless terminal mode:
   python3 client.py --tty
   ```
   The client opens a window displaying the optical grid in a `WAITING FOR FOCUS` state.

2. **Start the Sender (`server.py` or `server_v2.py`):**
   ```bash
   python3 server_v2.py
   # Or python3 server.py
   ```
   Opens the web interface at `http://localhost:8080`.

3. **Lock Optical Stream & Set Mask:**
   In the browser, click **`📺 Share Screen / Window`**, select the client window, and click **`🎯 Auto-Snap Mask`** (or drag with **`✏ Draw Mask`**). The server locks onto the client's optical grid.

4. **Select File to Send:**
   In the browser, select or drag-and-drop any file.

5. **Focus Client Window & Press Key:**
   Switch focus to either the `client.py` terminal window OR its GUI window and press **`[Enter]`** or **`[Space]`**.
   - `client.py` immediately updates its optical grid to broadcast `KEYBOARD_FOCUSED`.
   - The server detects focus confirmation via the optical stream and automatically types the file packets!
   - `client.py` displays real-time ACK bitmasks on its grid, verifying reception of every chunk.
   - Upon completion, `client.py` verifies the end-to-end SHA-256 hash, saves the file to disk (`./received/`), and signals verified completion on the optical grid.

### Duplex Receiver (`client.py`)

Runs on the target remote machine receiving the file:
- **Zero Dependencies:** Pure Python 3 standard library only (`sys`, `os`, `time`, `zlib`, `struct`, `hashlib`, `tkinter`). No OpenCV or NumPy required.
- **Automatic Fallback:** Opens a GUI window if a display is present; automatically falls back to headless 24-bit ANSI terminal mode (`--tty`) over SSH or serial consoles.
- **Packet Integrity:** Validates CRC32 per packet and enforces full SHA-256 cryptographic check before saving.

```bash
# Standard mode (GUI with auto-fallback to TTY):
python3 client.py

# Force headless terminal ANSI mode:
python3 client.py --tty

# Custom output folder and optical grid dimension:
python3 client.py -o ./received -g 32
```

### Keyboard Sender (`server.py`)

Runs on the host machine to transmit files to the client:
- **Keyboard Synthesizer:** Supports Linux `uinput` (kernel-level direct scancodes), X11 `XTest`, `xdotool`, and `pynput`.
- **On-Demand Permission:** Keyboard synthesizer access is requested on-demand via the Web UI or file staging rather than upfront on startup.
- **Optical Watchdog:** Immediately halts keystrokes if client window focus is lost or optical stream is interrupted.

```bash
python3 server.py --port 8080 --preset fast
```

### Safety & Watchdog System (Dead-Man's Switch)

To ensure keystrokes are never typed into the wrong desktop window:
- **Optical Feedback Watchdog:** If the client window is obscured, minimized, closed, or the screen share drops for >2.5s, the server instantly halts all keystrokes and releases modifiers (`Shift`, `Ctrl`, `Alt`, etc.).
- **Lock-Step Optical Verification (10 Retry Attempts):** Every chunk must be acknowledged on the client's optical grid before transmitting the next. If an optical ACK is delayed due to RDP latency, the server retries up to 10 times (with automatic line flushing) before safely halting.
- **Header ACK Guard:** Data chunks are never typed unless the client explicitly acknowledges the header packet on its optical grid (`STATE_RECEIVING`).
- **Emergency Stop:** A prominent **`[🛑 Emergency Stop]`** button in the Web UI allows manual one-click abort at any instant.
- **Speed Presets:** Easily switch throughput in the web header or CLI:
  - ⚡ **Turbo (30–50 KB/s):** 256B chunks, 0ms char delay, 3ms packet delay.
  - 🚀 **Fast (15–25 KB/s - Default):** 256B chunks, 0.2ms char delay, 8ms packet delay.
  - ⚖ **Balanced (5–10 KB/s):** 192B chunks, 1.0ms char delay, 15ms packet delay.
  - 🛡️ **Safe / VDI (1–3 KB/s):** 128B chunks, 2.0ms char delay, 25ms packet delay.

### Unified Bidirectional Server (`server_v2.py`)

`server_v2.py` combines `server.py` (keyboard sender) and `decoder_v2.py` (optical video receiver) into a single, unified server that can **both send and receive**:

```bash
python3 server_v2.py
# Options: --port 8080 --output-dir ./received --preset fast
```

- **📤 Send Tab (Host ➔ Remote):** Stages files and types alphanumeric packets into the remote window with optical lock-step ACKs and safety watchdog.
- **📥 Receive Tab (Remote ➔ Host):** Captures the remote screen running `encoder_min.py` or `encoder_v2.py`, automatically decodes RGB optical data frames, reconstructs missing frames via Cauchy Reed-Solomon FEC, verifies SHA-256, and provides in-browser one-click file downloads.
- **📁 File Library Tab:** Lists all received files in `./received/` with direct download and one-click re-staging to send back to another machine.
- **Shared Screen Share Feed:** A single browser screen share stream dynamically serves both sending and receiving without needing to reconnect.
- **Zero Premature Prompts:** Keyboard synthesizer access is requested on-demand via the Send tab UI button or file staging, making it clean for receive-only workflows.

---

## Best Practices & Tips

1. **RDP Frame Rate:** If transmitting over a low-bandwidth or laggy remote desktop, increase frame delay (e.g. `--delay 100` or `--delay 120`). In LAN or responsive sessions, `--delay 60` or `--delay 70` yields the fastest transfer.
2. **Terminal Sizing (`--tty`):** When using `--tty`, maximize your terminal window or set your terminal font to a slightly smaller size so the character matrix fits comfortably on screen without wrapping.
3. **Looping Transmissions:** Transmitters broadcast indefinitely in a loop. If your recording starts mid-stream or misses a frame, the receiver will automatically capture the missing chunks on the next cycle.
4. **Visual Debug Map:** On the first detected frame, the decoder writes `debug_sample_v2.png` to disk so you can visually verify alignment and dewarping quality.
