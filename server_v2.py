#!/usr/bin/env python3
"""
server_v2.py - Unified Bidirectional Air-Gap Transfer Server (v2)
Incorporates full duplex keyboard sender (from server.py) AND
high-speed visual optical receiver (from decoder_v2.py).

Features:
- 📤 SEND MODE (Host -> Remote):
    Stages files, synthesizes keystrokes (uinput/xlib/xdotool/pynput),
    monitors client focus via ACK2 optical grid, lock-step verification,
    10 retries, and dead-man's safety switch.
- 📥 RECEIVE MODE (Remote -> Host):
    Live optical decoding from screen share (encoder_v2 / encoder_min),
    tracks chromatic fiducials, samples RGB data cells, validates CRC32,
    recovers dropped frames via Cauchy Reed-Solomon FEC, verifies SHA-256,
    and saves files to disk with in-browser one-click download.
- 📁 FILE LIBRARY:
    Manages received files, allows direct download or instant re-staging
    to send to another machine.
- 🔄 UNIFIED WEB WORKER PIPELINE:
    Shared screen share video stream for both sending and receiving,
    unthrottled 25ms timer for background tab streaming, auto-snap mask.
"""

import argparse
import base64
import hashlib
import http.server
import json
import os
import socket
import socketserver
import struct
import sys
import threading
import time
import webbrowser
import zlib

import cv2
import numpy as np

# Import duplex keyboard & packet components from server.py
from server import (
    KeyboardTyper,
    ServerDuplexSession,
    make_header_packet,
    make_data_packet,
    make_chat_packet,
    make_end_packet,
    unpack_optical_ack_payload,
    STATE_WAIT_FOCUS,
    STATE_FOCUSED,
    STATE_RECEIVING,
    STATE_COMPLETE,
    STATE_ERROR,
    STATE_NAMES
)

# Import optical decoder math, tracker, FEC and reassembly from decoder_v2.py
from decoder_v2 import (
    FiducialTracker,
    recover_fec_chunks,
    reassemble_stream,
    print_transfer_summary,
    DEFAULT_CANDIDATE_GRIDS
)


# ==============================================================================
# OPTICAL RECEIVE SESSION (Remote -> Host File Decoder)
# ==============================================================================

def format_bytes(n):
    if n is None:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} TB"


class OpticalReceiveSession:
    """
    Manages visual optical file decoding from remote encoder transmissions.
    Tracks incoming Q2 visual frames, validates CRC32, recovers dropped frames
    using systematic Cauchy Reed-Solomon FEC, and saves verified files to disk.
    """
    def __init__(self, output_dir="./received"):
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.collected_chunks = {}
            self.total_chunks = None
            self.locked_grid_size = None
            self.locked_mode = "rgb"
            self.start_time = None
            self.frames_processed = 0
            self.valid_frames_decoded = 0
            self.is_complete = False
            self.result_info = None
            self.error = None
            self.last_fiducials = None
            self.latest_chunk_idx = None
            self.last_saved_file = None

    def update_with_decoded_frame(self, res, fiducials, frame_ts=None):
        with self.lock:
            self.frames_processed += 1
            if fiducials:
                self.last_fiducials = [[float(pt[0]), float(pt[1])] for pt in fiducials]
            else:
                self.last_fiducials = None

            if self.is_complete or not res:
                return self.get_status_dict()

            idx, total, payload, gs, mode_str = res
            # Guard against optical ACK frames meant for the keyboard sender
            if payload.startswith(b"ACK2"):
                return self.get_status_dict()

            if self.start_time is None:
                self.start_time = frame_ts or time.time()

            self.total_chunks = total
            self.locked_grid_size = gs
            self.locked_mode = mode_str
            self.valid_frames_decoded += 1
            self.latest_chunk_idx = idx

            if idx not in self.collected_chunks:
                self.collected_chunks[idx] = payload
                is_parity = idx >= total
                tag = f"Parity {idx - total + 1}" if is_parity else f"Chunk {idx + 1:3d}/{total:3d}"
                pct = min(100.0, (len(self.collected_chunks) / total) * 100.0)
                elapsed = time.time() - self.start_time
                speed = (len(self.collected_chunks) * len(payload) / 1024.0) / elapsed if elapsed > 0 else 0
                print(f"[Receive] Captured {tag} ({pct:5.1f}%) | Speed: {speed:5.1f} KB/s | {len(self.collected_chunks)}/{total} frames")

            if self.total_chunks and len(self.collected_chunks) >= self.total_chunks:
                missing = [i for i in range(self.total_chunks) if i not in self.collected_chunks]
                if not missing or recover_fec_chunks(self.collected_chunks, self.total_chunks) is not None:
                    # Determine embedded filename
                    full_stream = b"".join(
                        self.collected_chunks[i] for i in range(self.total_chunks) if i in self.collected_chunks
                    )
                    embedded_fname = "received_payload.bin"
                    if len(full_stream) >= 54 and full_stream[:4] == b"QSMD":
                        fname_len = struct.unpack(">H", full_stream[52:54])[0]
                        embedded_fname = full_stream[54:54 + fname_len].decode("utf-8", errors="replace")

                    target_path = os.path.join(self.output_dir, embedded_fname)
                    success, info = reassemble_stream(
                        self.collected_chunks, self.total_chunks, target_path, self.start_time
                    )
                    if success:
                        self.is_complete = True
                        self.result_info = info
                        self.last_saved_file = target_path
                        print_transfer_summary(info)
                    else:
                        self.error = info.get("error", "Reassembly failed")
                        print(f"[Receive Error]: {self.error}")

            return self.get_status_dict()

    def get_status_dict(self, error=None):
        elapsed = (time.time() - self.start_time) if self.start_time else 0
        captured_count = len(self.collected_chunks)
        tot = self.total_chunks or 0
        pct = min(100.0, (captured_count / tot * 100.0)) if tot > 0 else 0.0

        sample_len = len(next(iter(self.collected_chunks.values()))) if self.collected_chunks else 0
        speed = (captured_count * sample_len / 1024.0) / elapsed if elapsed > 0 else 0.0

        state_label = "waiting"
        if self.is_complete:
            state_label = "complete"
        elif self.error or error:
            state_label = "error"
        elif captured_count > 0:
            state_label = "receiving"
        elif self.last_fiducials is not None:
            state_label = "locked"

        d = {
            "status": "ok",
            "state": state_label,
            "is_complete": self.is_complete,
            "captured_chunks": captured_count,
            "total_chunks": self.total_chunks,
            "percent": round(pct, 1),
            "speed_kbps": round(speed, 1),
            "frames_processed": self.frames_processed,
            "valid_frames": self.valid_frames_decoded,
            "grid_size": self.locked_grid_size,
            "mode": self.locked_mode,
            "elapsed_sec": round(elapsed, 1),
            "fiducials": self.last_fiducials,
            "collected_indices": sorted(list(self.collected_chunks.keys())),
            "error": error or self.error
        }
        if self.is_complete and self.result_info:
            d["result"] = self.result_info
        return d


# ==============================================================================
# EXTENDED DUPLEX SEND SESSION (Host -> Remote Keyboard Sender)
# ==============================================================================

class ServerDuplexSessionV2(ServerDuplexSession):
    """
    Subclass of ServerDuplexSession that can accept pre-decoded ACK frames,
    allowing a single unified tracker to process frames for both engines.
    """
    def update_with_decoded_ack(self, res, fiducials):
        with self.lock:
            if res is not None:
                idx, total, payload, gs, mode = res
                ack_info = unpack_optical_ack_payload(payload)
                if ack_info:
                    self.optical_locked = True
                    self.last_optical_frame_time = time.time()
                    self.last_optical_decode_time = time.time()
                    if fiducials:
                        self.last_fiducials = [[float(p[0]), float(p[1])] for p in fiducials]

                    old_state = self.client_state
                    old_seq = self.client_focus_seq
                    self.client_state = ack_info["state"]
                    self.client_focus_seq = ack_info.get("focus_seq", 0)
                    self.client_last_ack = ack_info["last_ack"]
                    self.client_total_chunks = ack_info["total_chunks"]
                    self.client_chunks_count = ack_info["chunks_count"]
                    self.client_bytes_received = ack_info["bytes_received"]
                    self.client_bitmask = ack_info["bitmask"]
                    self.client_sha = ack_info["sha256"]
                    self.client_error = ack_info["error_code"]

                    focus_triggered = (
                        (self.client_state >= STATE_FOCUSED and not self.focus_confirmed_event.is_set()) or
                        (self.client_state >= STATE_FOCUSED and self.client_focus_seq != old_seq)
                    )
                    if focus_triggered:
                        self.focus_confirmed_event.set()
                        st_name = STATE_NAMES.get(self.client_state, str(self.client_state))
                        self.log(f"Client keyboard focus CONFIRMED via optical grid (state={st_name}, seq={self.client_focus_seq})!")

                    if (
                        self.client_state >= STATE_FOCUSED
                        and self.staged_file
                        and not self.transfer_active
                        and not self.transfer_complete
                        and not self.abort_requested
                    ):
                        self.start_transmission_worker()

                    if self.staged_file and self.transfer_active:
                        acked_list = self.staged_file["acked_chunks"]
                        mask = self.client_bitmask
                        count = 0
                        for i in range(len(acked_list)):
                            b_pos = i // 8
                            b_bit = i % 8
                            if b_pos < len(mask) and (mask[b_pos] & (1 << b_bit)):
                                acked_list[i] = True
                                count += 1
                        self.chunks_acked = count

                    if self.client_state == STATE_COMPLETE and self.transfer_active:
                        self.transfer_active = False
                        self.transfer_complete = True
                        self.end_time = time.time()
                        self.log("Transmission 100% COMPLETE & VERIFIED by client optical grid!")
                        self.typer.release_all_keys()
            else:
                if time.time() - self.last_optical_decode_time > 2.5:
                    self.optical_locked = False
                    self.last_fiducials = None
                    if self.transfer_active:
                        self.abort_transmission(reason="Optical signal lost for >2.5s (client window obscured, minimized, or moved)")

        return self.get_telemetry_dict()

    def get_telemetry_dict(self):
        with self.lock:
            progress = 0.0
            tot = 1
            if self.staged_file:
                tot = self.staged_file["total_chunks"]
                progress = (self.chunks_acked / max(1, tot)) * 100.0

            speed = 0.0
            if self.transfer_active and self.start_time > 0:
                dt = time.time() - self.start_time
                if dt > 0.5:
                    speed = (self.chunks_acked * self.chunk_size) / dt

            return {
                "optical_locked": self.optical_locked,
                "fiducials": self.last_fiducials,
                "client_state": STATE_NAMES.get(self.client_state, "UNKNOWN"),
                "client_state_code": self.client_state,
                "focus_confirmed": self.focus_confirmed_event.is_set(),
                "transfer_active": self.transfer_active,
                "transfer_complete": self.transfer_complete,
                "abort_requested": self.abort_requested,
                "chunks_sent": self.chunks_sent,
                "chunks_acked": self.chunks_acked,
                "total_chunks": tot if self.staged_file else 0,
                "progress_pct": round(progress, 1),
                "speed_bps": round(speed, 1),
                "filename": self.staged_file["filename"] if self.staged_file else "",
                "retransmissions": self.retransmissions,
                "typer_backend": self.typer.backend,
                "speed_preset": self.speed_preset
            }


# ==============================================================================
# UNIFIED BIDIRECTIONAL SERVER ENGINE
# ==============================================================================

class ServerV2UnifiedSession:
    """
    Coordinates Send Mode (ServerDuplexSessionV2) and Receive Mode (OpticalReceiveSession)
    over a single shared video capture stream and REST/WebSocket interface.
    """
    def __init__(self, output_dir="./received", chunk_size=256, char_delay=0.0002, speed_preset="fast", model_path=None):
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.active_mode = "auto"   # "auto", "send", "receive"
        self.active_tab = "send"    # "send", "receive", "files"

        self.tracker = FiducialTracker(model_path=model_path)
        self.send_session = ServerDuplexSessionV2(chunk_size=chunk_size, char_delay=char_delay)
        if speed_preset:
            self.send_session.set_speed_preset(speed_preset)

        self.receive_session = OpticalReceiveSession(output_dir=self.output_dir)
        self.lock = threading.RLock()

    def process_image_bytes(self, img_bytes):
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"status": "error", "message": "Failed to decode image frame"}

        # Determine target grid candidate:
        # If sending is active, the client is broadcasting the 32x32 ACK grid.
        # Otherwise, candidate grids are scanned to auto-detect remote optical transmitters.
        target_grid = 32 if (self.send_session.transfer_active or self.active_mode == "send") else None

        res, fiducials, warped = self.tracker.decode_frame(frame, locked_grid_size=target_grid, locked_mode="rgb")

        # Frame routing
        if res is not None:
            idx, total, payload, gs, mode = res
            if payload.startswith(b"ACK2"):
                # Optical ACK grid from client -> update Send session
                send_status = self.send_session.update_with_decoded_ack(res, fiducials)
                receive_status = self.receive_session.get_status_dict()
            else:
                # Optical data frame from remote encoder -> update Receive session
                receive_status = self.receive_session.update_with_decoded_frame(res, fiducials)
                send_status = self.send_session.get_telemetry_dict()
        else:
            send_status = self.send_session.update_with_decoded_ack(None, fiducials)
            receive_status = self.receive_session.update_with_decoded_frame(None, fiducials)

        # Compute Auto-ROI from fiducials for mask auto-snapping
        auto_roi = None
        fids = fiducials or send_status.get("fiducials") or receive_status.get("fiducials")
        if fids and len(fids) == 4:
            tl, tr, br, bl = fids
            min_x = min(tl[0], bl[0])
            max_x = max(tr[0], br[0])
            min_y = min(tl[1], tr[1])
            max_y = max(bl[1], br[1])
            span_w = max_x - min_x
            span_h = max_y - min_y
            if span_w > 50 and span_h > 50:
                pad_w = span_w * 0.06
                pad_h = span_h * 0.06
                auto_roi = [
                    max(0.0, min_x - pad_w),
                    max(0.0, min_y - pad_h),
                    span_w + 2 * pad_w,
                    span_h + 2 * pad_h
                ]

        is_locked = (
            self.send_session.optical_locked
            if (self.send_session.transfer_active or self.active_mode == "send")
            else (receive_status.get("state") in ("locked", "receiving", "complete"))
        )

        return {
            "status": "ok",
            "active_mode": self.active_mode,
            "active_tab": self.active_tab,
            "optical_locked": is_locked,
            "fiducials": [[float(p[0]), float(p[1])] for p in fids] if fids else None,
            "auto_roi": auto_roi,
            "send": send_status,
            "receive": receive_status,
            "received_files_count": len(self.get_received_files_list())
        }

    def get_received_files_list(self):
        files = []
        if not os.path.exists(self.output_dir):
            return files
        for fname in sorted(os.listdir(self.output_dir)):
            fpath = os.path.join(self.output_dir, fname)
            if os.path.isfile(fpath) and not fname.startswith("."):
                try:
                    st = os.stat(fpath)
                    mtime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))
                    files.append({
                        "filename": fname,
                        "size": st.st_size,
                        "size_str": format_bytes(st.st_size),
                        "mtime": st.st_mtime,
                        "mtime_str": mtime_str
                    })
                except Exception:
                    pass
        return sorted(files, key=lambda x: x["mtime"], reverse=True)

    def stage_received_file(self, filename):
        safe_name = os.path.basename(filename)
        fpath = os.path.join(self.output_dir, safe_name)
        if not os.path.isfile(fpath):
            return False, f"File '{safe_name}' not found in received directory"
        with open(fpath, "rb") as f:
            raw_bytes = f.read()
        return self.send_session.stage_file(safe_name, raw_bytes)


# ==============================================================================
# WEB CONTROL INTERFACE (HTML5 / CSS3 / Web Worker)
# ==============================================================================

SERVER_V2_HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OTD v2 - Bidirectional Air-Gap Transfer Server</title>
<style>
  :root {
    --bg-dark: #090d16;
    --card-bg: #111827;
    --border-color: #1f293d;
    --accent-cyan: #06b6d4;
    --accent-green: #10b981;
    --accent-yellow: #f59e0b;
    --accent-red: #ef4444;
    --accent-purple: #a855f7;
    --text-main: #f3f4f6;
    --text-dim: #9ca3af;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  body { background-color: var(--bg-dark); color: var(--text-main); min-height: 100vh; display: flex; flex-direction: column; }

  /* Navbar */
  header {
    background: #0f172a;
    border-bottom: 1px solid var(--border-color);
    padding: 0.75rem 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.75rem;
  }
  .brand { display: flex; align-items: center; gap: 0.6rem; }
  .logo-badge {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
    color: white; font-weight: 800; font-size: 0.9rem; padding: 0.25rem 0.6rem;
    border-radius: 6px; letter-spacing: 0.05em;
  }
  .brand-title { font-size: 1.15rem; font-weight: 700; color: #fff; }
  .brand-sub { font-size: 0.75rem; color: var(--text-dim); }

  .nav-tabs { display: flex; gap: 0.4rem; background: #0b1120; padding: 0.25rem; border-radius: 8px; border: 1px solid var(--border-color); }
  .nav-tab {
    background: transparent; border: none; color: var(--text-dim); padding: 0.45rem 1rem;
    border-radius: 6px; font-size: 0.85rem; font-weight: 600; cursor: pointer; transition: all 0.2s;
    display: flex; align-items: center; gap: 0.4rem;
  }
  .nav-tab:hover { color: var(--text-main); }
  .nav-tab.active { background: #1e293b; color: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.3); }

  .header-actions { display: flex; align-items: center; gap: 0.6rem; }
  .pill { font-size: 0.75rem; padding: 0.3rem 0.65rem; border-radius: 20px; font-weight: 600; display: inline-flex; align-items: center; gap: 0.35rem; }
  .pill-gray { background: #1e293b; color: var(--text-dim); }
  .pill-green { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
  .pill-yellow { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
  .pill-cyan { background: rgba(6, 182, 212, 0.2); color: #38bdf8; border: 1px solid rgba(6, 182, 212, 0.4); }

  .btn-danger {
    background: #dc2626; color: white; border: none; padding: 0.45rem 0.9rem;
    border-radius: 6px; font-weight: 700; font-size: 0.82rem; cursor: pointer;
    transition: background 0.2s; display: flex; align-items: center; gap: 0.35rem;
  }
  .btn-danger:hover { background: #b91c1c; }

  /* Main Layout */
  main {
    flex: 1; padding: 1.25rem 1.5rem; display: grid;
    grid-template-columns: 1.15fr 1fr; gap: 1.25rem;
    max-width: 1700px; width: 100%; margin: 0 auto;
  }
  @media (max-width: 1080px) { main { grid-template-columns: 1fr; } }

  .card {
    background: var(--card-bg); border: 1px solid var(--border-color);
    border-radius: 12px; padding: 1.25rem; display: flex; flex-direction: column; gap: 1rem;
  }

  /* Video Viewport */
  .viewport-header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem; }
  .viewport-title { font-size: 0.95rem; font-weight: 700; display: flex; align-items: center; gap: 0.5rem; }

  .video-stage {
    position: relative; width: 100%; height: 380px; background: #000;
    border-radius: 8px; overflow: hidden; border: 1px solid var(--border-color);
    display: flex; align-items: center; justify-content: center;
  }
  #videoEl { width: 100%; height: 100%; object-fit: contain; display: block; }
  #overlayCanvas {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: auto; cursor: crosshair;
  }
  .stage-placeholder {
    position: absolute; color: var(--text-dim); font-size: 0.9rem;
    text-align: center; pointer-events: none;
  }

  .video-toolbar { display: flex; gap: 0.5rem; flex-wrap: wrap; }
  .btn {
    background: #1e293b; color: #fff; border: 1px solid #334155; padding: 0.45rem 0.85rem;
    border-radius: 6px; font-size: 0.82rem; font-weight: 600; cursor: pointer; transition: all 0.2s;
  }
  .btn:hover { background: #334155; }
  .btn-primary { background: var(--accent-cyan); border-color: var(--accent-cyan); color: #090d16; font-weight: 700; }
  .btn-primary:hover { background: #0891b2; color: #fff; }

  /* Mode Content Tabs */
  .tab-pane { display: none; flex-direction: column; gap: 1rem; }
  .tab-pane.active { display: flex; }

  /* Dropzone */
  .dropzone {
    border: 2px dashed #334155; border-radius: 8px; padding: 1.5rem;
    text-align: center; cursor: pointer; transition: border-color 0.2s;
    background: #0f172a;
  }
  .dropzone:hover, .dropzone.dragover { border-color: var(--accent-cyan); }
  .dropzone-text { font-size: 0.88rem; color: var(--text-dim); }
  .dropzone-browse { color: var(--accent-cyan); font-weight: 600; }

  /* Preset Selector */
  .preset-group { display: flex; gap: 0.4rem; }
  .preset-btn {
    flex: 1; background: #0f172a; border: 1px solid var(--border-color); color: var(--text-dim);
    padding: 0.5rem 0.2rem; border-radius: 6px; font-size: 0.78rem; font-weight: 700; cursor: pointer;
    text-align: center; transition: all 0.2s;
  }
  .preset-btn:hover { color: #fff; border-color: #334155; }
  .preset-btn.active { background: #1e293b; border-color: var(--accent-cyan); color: #38bdf8; }

  /* Progress Banner */
  .status-banner {
    padding: 0.75rem 1rem; border-radius: 8px; font-size: 0.85rem; font-weight: 600;
    display: flex; align-items: center; justify-content: space-between; gap: 0.5rem;
  }
  .banner-wait { background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.3); color: #fbbf24; }
  .banner-active { background: rgba(6, 182, 212, 0.15); border: 1px solid rgba(6, 182, 212, 0.3); color: #38bdf8; }
  .banner-success { background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3); color: #34d399; }
  .banner-error { background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.3); color: #f87171; }

  .progress-wrap { background: #0f172a; border-radius: 6px; height: 10px; overflow: hidden; }
  .progress-fill { background: linear-gradient(90deg, var(--accent-cyan), var(--accent-green)); height: 100%; width: 0%; transition: width 0.15s ease; }

  /* Stats Grid */
  .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.6rem; }
  .stat-box { background: #0f172a; padding: 0.6rem; border-radius: 6px; border: 1px solid #1e293b; text-align: center; }
  .stat-val { font-size: 1.1rem; font-weight: 700; color: #fff; font-family: monospace; }
  .stat-lbl { font-size: 0.7rem; color: var(--text-dim); text-transform: uppercase; margin-top: 0.2rem; }

  /* Chunk Matrix Visualizer */
  .matrix-box { background: #090d16; border: 1px solid var(--border-color); border-radius: 6px; padding: 0.6rem; max-height: 120px; overflow-y: auto; }
  .matrix-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(8px, 1fr)); gap: 2px; }
  .matrix-cell { aspect-ratio: 1; border-radius: 1px; background: #1e293b; }
  .matrix-cell.acked { background: #10b981; }
  .matrix-cell.parity { background: #a855f7; }
  .matrix-cell.active { background: #38bdf8; }

  /* Chat Box */
  .chat-row { display: flex; gap: 0.4rem; }
  .chat-input {
    flex: 1; background: #0f172a; border: 1px solid var(--border-color); color: #fff;
    padding: 0.5rem 0.75rem; border-radius: 6px; font-size: 0.85rem;
  }
  .chat-input:focus { outline: none; border-color: var(--accent-cyan); }

  /* Log Box */
  .log-console {
    background: #090d16; border: 1px solid var(--border-color); border-radius: 6px;
    padding: 0.6rem 0.75rem; font-family: monospace; font-size: 0.75rem; color: #94a3b8;
    height: 120px; overflow-y: auto; display: flex; flex-direction: column; gap: 0.25rem;
  }
  .log-entry { line-height: 1.3; }

  /* Table */
  .file-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  .file-table th { text-align: left; padding: 0.5rem 0.75rem; color: var(--text-dim); border-bottom: 1px solid var(--border-color); }
  .file-table td { padding: 0.6rem 0.75rem; border-bottom: 1px solid #1a2234; }
  .file-table tr:hover { background: #0f172a; }

  /* Modal */
  .modal-backdrop {
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.75);
    backdrop-filter: blur(4px); z-index: 100; align-items: center; justify-content: center;
  }
  .modal-backdrop.active { display: flex; }
  .modal-card {
    background: #111827; border: 1px solid #334155; border-radius: 12px;
    padding: 1.5rem; max-width: 480px; width: 90%; display: flex; flex-direction: column; gap: 1rem;
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
  }
</style>
</head>
<body>

<header>
  <div class="brand">
    <div class="logo-badge">OTD v2</div>
    <div>
      <div class="brand-title">Air-Gap Duplex Server</div>
      <div class="brand-sub">Bidirectional Keyboard Sender & Optical Video Decoder</div>
    </div>
  </div>

  <div class="nav-tabs">
    <button class="nav-tab active" id="tabBtnSend" onclick="switchTab('send')">📤 Send to Remote</button>
    <button class="nav-tab" id="tabBtnReceive" onclick="switchTab('receive')">📥 Receive from Remote</button>
    <button class="nav-tab" id="tabBtnFiles" onclick="switchTab('files')">📁 File Library (<span id="filesCountBadge">0</span>)</button>
  </div>

  <div class="header-actions">
    <div id="wsPill" class="pill pill-gray">● WS OFFLINE</div>
    <div id="opticalPill" class="pill pill-gray">○ OPTICAL SEARCH</div>
    <button class="btn-danger" onclick="emergencyStop()">🛑 Emergency Stop</button>
  </div>
</header>

<main>
  <!-- Left Column: Shared Video Stage -->
  <div class="card">
    <div class="viewport-header">
      <div class="viewport-title">
        <span>📺 Optical Feedback & Camera Stream</span>
        <span id="modeBadge" class="pill pill-cyan">AUTO-DETECT</span>
      </div>
      <div class="video-toolbar">
        <button id="btnShareScreen" class="btn btn-primary" onclick="toggleScreenShare()">📺 Share Screen / Window</button>
        <button id="btnAutoSnap" class="btn" onclick="autoSnapMask()" title="Snap mask tightly to corner chromatic fiducials">🎯 Auto-Snap</button>
        <button id="btnDrawMask" class="btn" onclick="toggleDrawMask()">✏ Draw Mask</button>
        <button id="btnClearMask" class="btn" onclick="clearMask()">🔄 Clear</button>
      </div>
    </div>

    <div class="video-stage">
      <video id="videoEl" autoplay playsinline muted></video>
      <canvas id="overlayCanvas"></canvas>
      <div id="stagePlaceholder" class="stage-placeholder">
        Click <b>Share Screen / Window</b> to lock optical link.<br>
        (Works seamlessly over RDP, VNC, Wayland, and X11)
      </div>
    </div>

    <div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: var(--text-dim);">
      <span id="streamFps">FPS: 0.0</span>
      <span id="cropStatus">Full Video (No Mask)</span>
      <span id="fidsStatus">Fiducials: 0/4</span>
    </div>
  </div>

  <!-- Right Column: Tabbed Workflow Panels -->
  <div class="card">
    <!-- TAB 1: SEND TO REMOTE -->
    <div id="paneSend" class="tab-pane active">
      <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
        <h3 style="font-size: 1rem;">📤 Transmit File to Remote Client</h3>
        <div style="display: flex; align-items: center; gap: 0.5rem;">
          <span style="font-size: 0.75rem; color: var(--text-dim);">Backend: <b id="typerBackendLbl" style="color:#fff;">...</b></span>
          <button id="btnRequestKeyboard" class="btn" style="padding:0.25rem 0.6rem; font-size:0.75rem;" onclick="requestKeyboardAccess()">🔑 Request Keyboard Access</button>
        </div>
      </div>

      <!-- File Dropzone -->
      <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
        <input type="file" id="fileInput" style="display:none;" onchange="handleFileSelect(this.files)">
        <div style="font-size: 1.5rem; margin-bottom: 0.4rem;">📄</div>
        <div class="dropzone-text">Drag & drop any file here, or <span class="dropzone-browse">browse</span></div>
        <div id="stagedFileDesc" style="font-size: 0.78rem; color: #38bdf8; margin-top: 0.4rem; font-weight: 600;">No file staged</div>
      </div>

      <!-- Speed Presets -->
      <div>
        <div style="font-size: 0.75rem; color: var(--text-dim); margin-bottom: 0.3rem;">TRANSMISSION SPEED PRESET:</div>
        <div class="preset-group">
          <button class="preset-btn" onclick="setSpeed('turbo')">⚡ TURBO<br><span style="font-size:0.65rem;font-weight:normal;">256B / 0ms</span></button>
          <button class="preset-btn active" id="btnPresetFast" onclick="setSpeed('fast')">🚀 FAST<br><span style="font-size:0.65rem;font-weight:normal;">256B / 0.2ms</span></button>
          <button class="preset-btn" onclick="setSpeed('balanced')">⚖ BALANCED<br><span style="font-size:0.65rem;font-weight:normal;">192B / 1ms</span></button>
          <button class="preset-btn" onclick="setSpeed('safe')">🛡️ SAFE (RDP)<br><span style="font-size:0.65rem;font-weight:normal;">128B / 2ms</span></button>
        </div>
      </div>

      <!-- Client Focus Banner -->
      <div id="clientStatusBanner" class="status-banner banner-wait">
        <span id="clientStatusText">Waiting for Client Window Focus (Press ENTER or SPACE in client)...</span>
        <span id="clientFocusSeq" class="pill pill-yellow">SEQ 0</span>
      </div>

      <!-- Progress Bar & Stats -->
      <div>
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.3rem;">
          <span id="sendProgressLabel">Progress: 0.0%</span>
          <span id="sendSpeedLabel">0.0 KB/s</span>
        </div>
        <div class="progress-wrap">
          <div id="sendProgressFill" class="progress-fill"></div>
        </div>
      </div>

      <div class="stats-grid">
        <div class="stat-box">
          <div id="statSendChunks" class="stat-val">0 / 0</div>
          <div class="stat-lbl">Chunks Acked</div>
        </div>
        <div class="stat-box">
          <div id="statSendSpeed" class="stat-val">0.0</div>
          <div class="stat-lbl">Speed (KB/s)</div>
        </div>
        <div class="stat-box">
          <div id="statSendRetries" class="stat-val">0</div>
          <div class="stat-lbl">Retries</div>
        </div>
        <div class="stat-box">
          <div id="statSendState" class="stat-val">IDLE</div>
          <div class="stat-lbl">State</div>
        </div>
      </div>

      <!-- Quick Chat Keystroke Injection -->
      <div>
        <div style="font-size: 0.75rem; color: var(--text-dim); margin-bottom: 0.3rem;">QUICK TERMINAL COMMAND / CHAT INJECTION:</div>
        <div class="chat-row">
          <input type="text" id="chatInput" class="chat-input" placeholder="Type text or bash command to type into remote client..." onkeydown="if(event.key==='Enter') sendChat()">
          <button class="btn btn-primary" onclick="sendChat()">Send</button>
        </div>
      </div>

      <!-- Server Log -->
      <div class="log-console" id="serverLog">
        <div class="log-entry">[Server] Ready. Share screen and stage a file to begin.</div>
      </div>
    </div>

    <!-- TAB 2: RECEIVE FROM REMOTE -->
    <div id="paneReceive" class="tab-pane">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <h3 style="font-size: 1rem;">📥 Optical Video Receiver (Remote ➔ Host)</h3>
        <button class="btn" onclick="resetReceiver()">🔄 Reset Receiver</button>
      </div>

      <!-- Optical Signal Banner -->
      <div id="recvSignalBanner" class="status-banner banner-wait">
        <span id="recvSignalText">Waiting for optical signal (run encoder_min.py or encoder_v2.py on remote)...</span>
        <span id="recvGridBadge" class="pill pill-gray">--</span>
      </div>

      <!-- Progress -->
      <div>
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.3rem;">
          <span id="recvProgressLabel">Captured: 0 / 0 chunks (0.0%)</span>
          <span id="recvSpeedLabel">0.0 KB/s</span>
        </div>
        <div class="progress-wrap">
          <div id="recvProgressFill" class="progress-fill"></div>
        </div>
      </div>

      <div class="stats-grid">
        <div class="stat-box">
          <div id="statRecvChunks" class="stat-val">0 / 0</div>
          <div class="stat-lbl">Captured</div>
        </div>
        <div class="stat-box">
          <div id="statRecvSpeed" class="stat-val">0.0</div>
          <div class="stat-lbl">KB/s</div>
        </div>
        <div class="stat-box">
          <div id="statRecvValid" class="stat-val">0</div>
          <div class="stat-lbl">Valid Frames</div>
        </div>
        <div class="stat-box">
          <div id="statRecvTime" class="stat-val">0.0s</div>
          <div class="stat-lbl">Elapsed</div>
        </div>
      </div>

      <!-- Live Chunk Matrix -->
      <div>
        <div style="font-size: 0.75rem; color: var(--text-dim); margin-bottom: 0.3rem;">LIVE CHUNK RECEPTION MATRIX:</div>
        <div class="matrix-box">
          <div id="chunkMatrix" class="matrix-grid"></div>
        </div>
      </div>

      <!-- Verified Result Box (Shown upon completion) -->
      <div id="recvCompleteCard" style="display:none; background:#0f172a; border:1px solid #10b981; border-radius:8px; padding:1rem; flex-direction:column; gap:0.6rem;">
        <div style="color:#34d399; font-weight:700; font-size:0.95rem; display:flex; align-items:center; gap:0.4rem;">
          ✓ FILE VERIFIED & SAVED SUCCESSFULLY!
        </div>
        <div style="font-size:0.82rem; line-height:1.4; color:#cbd5e1;">
          <div><b>Filename:</b> <span id="resFilename">...</span></div>
          <div><b>Size:</b> <span id="resSize">...</span> (<span id="resCompSize">...</span> compressed)</div>
          <div><b>SHA-256:</b> <code id="resSha" style="font-size:0.75rem; color:#38bdf8;">...</code></div>
          <div><b>Elapsed:</b> <span id="resElapsed">...</span> (<span id="resSpeed">...</span>)</div>
        </div>
        <div style="display:flex; gap:0.5rem; margin-top:0.4rem;">
          <a id="btnDownloadReceived" class="btn btn-primary" href="#" download style="text-decoration:none;">⬇ Download File</a>
          <button class="btn" onclick="stageLastReceived()">📤 Stage to Send Back</button>
        </div>
      </div>
    </div>

    <!-- TAB 3: FILE LIBRARY -->
    <div id="paneFiles" class="tab-pane">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <h3 style="font-size: 1rem;">📁 Received Files Directory (<span id="receivedDirLabel">./received</span>)</h3>
        <button class="btn" onclick="refreshFilesList()">🔄 Refresh</button>
      </div>

      <div style="max-height: 440px; overflow-y: auto; border: 1px solid var(--border-color); border-radius: 8px;">
        <table class="file-table">
          <thead>
            <tr>
              <th>Filename</th>
              <th>Size</th>
              <th>Date Modified</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody id="filesTableBody">
            <tr><td colspan="4" style="text-align:center; color:var(--text-dim);">No received files found.</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</main>

<script>
  let ws = null;
  let activeTab = 'send';
  let activeStream = null;
  let timerWorker = null;
  let isStreaming = false;

  // Mask & Drawing state
  let cropRegion = null;
  let isDrawingMask = false;
  let dragStart = null;
  let lastFiducials = null;
  let autoRoi = null;

  // Telemetry caching
  let knownRecvTotal = 0;
  let lastCompletedFile = "";

  const videoEl = document.getElementById('videoEl');
  const overlayCanvas = document.getElementById('overlayCanvas');
  const overlayCtx = overlayCanvas.getContext('2d');
  const offscreenCanvas = document.createElement('canvas');
  const offscreenCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true });

  // 1. WebSocket connection
  function connectWS() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${location.host}/ws`);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      document.getElementById('wsPill').className = 'pill pill-green';
      document.getElementById('wsPill').textContent = '● WS CONNECTED';
      logMsg('[Client] Connected to server WebSocket.');
      refreshFilesList();
    };

    ws.onclose = () => {
      document.getElementById('wsPill').className = 'pill pill-gray';
      document.getElementById('wsPill').textContent = '○ WS OFFLINE';
      setTimeout(connectWS, 1000);
    };

    ws.onmessage = (evt) => {
      if (typeof evt.data === 'string') {
        try {
          const msg = JSON.parse(evt.data);
          handleTelemetry(msg);
        } catch(e) {}
      }
    };
  }
  connectWS();

  // 2. Tab Navigation
  function switchTab(tab) {
    activeTab = tab;
    document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

    if (tab === 'send') {
      document.getElementById('tabBtnSend').classList.add('active');
      document.getElementById('paneSend').classList.add('active');
      fetch('/api/set_mode', { method: 'POST', body: JSON.stringify({ mode: 'send' }) });
    } else if (tab === 'receive') {
      document.getElementById('tabBtnReceive').classList.add('active');
      document.getElementById('paneReceive').classList.add('active');
      fetch('/api/set_mode', { method: 'POST', body: JSON.stringify({ mode: 'receive' }) });
    } else if (tab === 'files') {
      document.getElementById('tabBtnFiles').classList.add('active');
      document.getElementById('paneFiles').classList.add('active');
      refreshFilesList();
    }
  }

  // 3. Screen Sharing & Frame Pipeline (Unthrottled Web Worker)
  async function toggleScreenShare() {
    if (activeStream) {
      stopScreenShare();
      return;
    }
    try {
      activeStream = await navigator.mediaDevices.getDisplayMedia({
        video: { frameRate: { ideal: 60, max: 60 } },
        audio: false
      });
      videoEl.srcObject = activeStream;
      document.getElementById('stagePlaceholder').style.display = 'none';
      document.getElementById('btnShareScreen').textContent = '🛑 Stop Stream';
      document.getElementById('btnShareScreen').className = 'btn btn-danger';

      activeStream.getVideoTracks()[0].onended = () => stopScreenShare();
      startStreamingLoop();
    } catch(err) {
      logMsg(`Screen share error: ${err.message}`);
    }
  }

  function stopScreenShare() {
    if (activeStream) {
      activeStream.getTracks().forEach(t => t.stop());
      activeStream = null;
    }
    if (timerWorker) {
      timerWorker.terminate();
      timerWorker = null;
    }
    videoEl.srcObject = null;
    document.getElementById('stagePlaceholder').style.display = 'block';
    document.getElementById('btnShareScreen').textContent = '📺 Share Screen / Window';
    document.getElementById('btnShareScreen').className = 'btn btn-primary';
    document.getElementById('opticalPill').className = 'pill pill-gray';
    document.getElementById('opticalPill').textContent = '○ OPTICAL SEARCH';
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  }

  function startStreamingLoop() {
    // Web Worker timer (25ms) prevents tab throttling in background
    const workerBlob = new Blob([`
      let timer = null;
      onmessage = (e) => {
        if (e.data === 'start') {
          if (!timer) timer = setInterval(() => postMessage('tick'), 25);
        } else if (e.data === 'stop') {
          clearInterval(timer);
          timer = null;
        }
      };
    `], { type: 'application/javascript' });

    timerWorker = new Worker(URL.createObjectURL(workerBlob));
    timerWorker.onmessage = () => captureAndSendFrame();
    timerWorker.postMessage('start');
  }

  let lastFrameTime = performance.now();
  let frameCount = 0;
  let fpsTimer = performance.now();

  function captureAndSendFrame() {
    if (!ws || ws.readyState !== WebSocket.OPEN || !activeStream || videoEl.videoWidth === 0) return;

    const vw = videoEl.videoWidth;
    const vh = videoEl.videoHeight;

    let sx = 0, sy = 0, sw = vw, sh = vh;
    if (cropRegion && cropRegion.w > 40 && cropRegion.h > 40) {
      sx = Math.max(0, Math.min(vw - 10, cropRegion.x));
      sy = Math.max(0, Math.min(vh - 10, cropRegion.y));
      sw = Math.min(vw - sx, cropRegion.w);
      sh = Math.min(vh - sy, cropRegion.h);
    }

    const targetW = 720;
    const targetH = Math.round(targetW * (sh / sw));
    if (offscreenCanvas.width !== targetW || offscreenCanvas.height !== targetH) {
      offscreenCanvas.width = targetW;
      offscreenCanvas.height = targetH;
    }

    offscreenCtx.drawImage(videoEl, sx, sy, sw, sh, 0, 0, targetW, targetH);
    offscreenCanvas.toBlob((blob) => {
      if (blob && ws && ws.readyState === WebSocket.OPEN) {
        blob.arrayBuffer().then(buf => ws.send(buf));
      }
    }, 'image/jpeg', 0.85);

    // FPS counter
    frameCount++;
    const now = performance.now();
    if (now - fpsTimer >= 1000) {
      const fps = (frameCount * 1000) / (now - fpsTimer);
      document.getElementById('streamFps').textContent = `FPS: ${fps.toFixed(1)}`;
      frameCount = 0;
      fpsTimer = now;
    }
  }

  // 4. Telemetry Handler
  function handleTelemetry(data) {
    lastFiducials = data.fiducials;
    autoRoi = data.auto_roi;

    // Optical pill
    const optPill = document.getElementById('opticalPill');
    if (data.optical_locked) {
      optPill.className = 'pill pill-green';
      optPill.textContent = '● OPTICAL LOCKED';
    } else {
      optPill.className = 'pill pill-gray';
      optPill.textContent = '○ OPTICAL SEARCH';
    }

    document.getElementById('fidsStatus').textContent = `Fiducials: ${data.fiducials ? data.fiducials.length : 0}/4`;
    if (data.received_files_count !== undefined) {
      document.getElementById('filesCountBadge').textContent = data.received_files_count;
    }

    // Redraw Canvas Overlay
    redrawOverlay();

    // Send Tab Telemetry
    if (data.send) {
      updateSendUI(data.send);
    }

    // Receive Tab Telemetry
    if (data.receive) {
      updateReceiveUI(data.receive);
    }
  }

  function updateSendUI(send) {
    document.getElementById('typerBackendLbl').textContent = send.typer_backend || 'uinput';
    document.getElementById('statSendChunks').textContent = `${send.chunks_acked} / ${send.total_chunks}`;
    document.getElementById('statSendSpeed').textContent = (send.speed_bps / 1024).toFixed(1);
    document.getElementById('statSendRetries').textContent = send.retransmissions;
    document.getElementById('statSendState').textContent = send.client_state;

    document.getElementById('sendProgressLabel').textContent = `Progress: ${send.progress_pct}% (${send.chunks_acked}/${send.total_chunks})`;
    document.getElementById('sendSpeedLabel').textContent = `${(send.speed_bps / 1024).toFixed(1)} KB/s`;
    document.getElementById('sendProgressFill').style.width = `${send.progress_pct}%`;

    const banner = document.getElementById('clientStatusBanner');
    const text = document.getElementById('clientStatusText');
    if (send.transfer_complete) {
      banner.className = 'status-banner banner-success';
      text.textContent = '✓ Transmission 100% COMPLETE & VERIFIED by optical grid!';
    } else if (send.transfer_active) {
      banner.className = 'status-banner banner-active';
      text.textContent = `Transmitting chunk ${send.chunks_acked + 1} / ${send.total_chunks}...`;
    } else if (send.focus_confirmed) {
      banner.className = 'status-banner banner-active';
      text.textContent = `Focus CONFIRMED! Staging transfer...`;
    } else {
      banner.className = 'status-banner banner-wait';
      text.textContent = 'Waiting for Client Window Focus (Click client window & press ENTER)...';
    }
  }

  function updateReceiveUI(recv) {
    document.getElementById('statRecvChunks').textContent = `${recv.captured_chunks} / ${recv.total_chunks || 0}`;
    document.getElementById('statRecvSpeed').textContent = recv.speed_kbps ? recv.speed_kbps.toFixed(1) : '0.0';
    document.getElementById('statRecvValid').textContent = recv.valid_frames || 0;
    document.getElementById('statRecvTime').textContent = `${recv.elapsed_sec || 0}s`;

    document.getElementById('recvProgressLabel').textContent = `Captured: ${recv.captured_chunks} / ${recv.total_chunks || 0} (${recv.percent}%)`;
    document.getElementById('recvSpeedLabel').textContent = `${recv.speed_kbps || 0} KB/s`;
    document.getElementById('recvProgressFill').style.width = `${recv.percent}%`;

    const banner = document.getElementById('recvSignalBanner');
    const text = document.getElementById('recvSignalText');
    const gridBadge = document.getElementById('recvGridBadge');

    if (recv.grid_size) {
      gridBadge.textContent = `${recv.grid_size}x${recv.grid_size} ${recv.mode ? recv.mode.toUpperCase() : 'RGB'}`;
      gridBadge.className = 'pill pill-cyan';
    }

    if (recv.state === 'complete') {
      banner.className = 'status-banner banner-success';
      text.textContent = '✓ Transfer 100% Captured and Verified!';
      if (recv.result) {
        showRecvCompleteCard(recv.result);
      }
    } else if (recv.state === 'receiving') {
      banner.className = 'status-banner banner-active';
      text.textContent = `Receiving chunks (${recv.captured_chunks}/${recv.total_chunks})...`;
    } else if (recv.state === 'locked') {
      banner.className = 'status-banner banner-active';
      text.textContent = 'Signal Locked! Waiting for first chunk...';
    } else {
      banner.className = 'status-banner banner-wait';
      text.textContent = 'Waiting for optical signal (run encoder on remote)...';
    }

    // Update Chunk Matrix
    if (recv.total_chunks && knownRecvTotal !== recv.total_chunks) {
      knownRecvTotal = recv.total_chunks;
      initChunkMatrix(recv.total_chunks);
    }
    if (recv.collected_indices) {
      updateChunkMatrixCells(recv.collected_indices, recv.total_chunks);
    }
  }

  function initChunkMatrix(total) {
    const box = document.getElementById('chunkMatrix');
    box.innerHTML = '';
    for (let i = 0; i < total; i++) {
      const cell = document.createElement('div');
      cell.className = 'matrix-cell';
      cell.id = `chunkCell_${i}`;
      box.appendChild(cell);
    }
  }

  function updateChunkMatrixCells(indices, total) {
    indices.forEach(idx => {
      let cell = document.getElementById(`chunkCell_${idx}`);
      if (!cell && idx >= total) {
        // Parity cell
        cell = document.createElement('div');
        cell.className = 'matrix-cell parity';
        cell.id = `chunkCell_${idx}`;
        document.getElementById('chunkMatrix').appendChild(cell);
      } else if (cell) {
        cell.className = idx >= total ? 'matrix-cell parity' : 'matrix-cell acked';
      }
    });
  }

  function showRecvCompleteCard(result) {
    const card = document.getElementById('recvCompleteCard');
    card.style.display = 'flex';
    document.getElementById('resFilename').textContent = result.filename;
    document.getElementById('resSize').textContent = `${(result.orig_size / 1024).toFixed(1)} KB`;
    document.getElementById('resCompSize').textContent = `${(result.comp_size / 1024).toFixed(1)} KB`;
    document.getElementById('resSha').textContent = result.sha256;
    document.getElementById('resElapsed').textContent = `${result.elapsed_sec}s`;
    document.getElementById('resSpeed').textContent = `${result.eff_speed_kbps} KB/s`;

    const btn = document.getElementById('btnDownloadReceived');
    btn.href = `/download/${encodeURIComponent(result.filename)}`;
    lastCompletedFile = result.filename;
  }

  function resetReceiver() {
    fetch('/api/receive/reset', { method: 'POST' }).then(() => {
      document.getElementById('recvCompleteCard').style.display = 'none';
      document.getElementById('chunkMatrix').innerHTML = '';
      knownRecvTotal = 0;
      logMsg('[Receiver] Reset session. Ready for next incoming file.');
    });
  }

  function stageLastReceived() {
    if (!lastCompletedFile) return;
    fetch('/api/stage_received', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename: lastCompletedFile })
    }).then(r => r.json()).then(resp => {
      if (resp.status === 'ok') {
        switchTab('send');
        document.getElementById('stagedFileDesc').textContent = `Staged: ${lastCompletedFile}`;
        logMsg(`[Send] Staged received file '${lastCompletedFile}' for sending.`);
      }
    });
  }

  // 5. Drawing & Canvas Overlay
  function redrawOverlay() {
    const rect = videoEl.getBoundingClientRect();
    if (overlayCanvas.width !== rect.width || overlayCanvas.height !== rect.height) {
      overlayCanvas.width = rect.width;
      overlayCanvas.height = rect.height;
    }
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    const vw = videoEl.videoWidth || 1;
    const vh = videoEl.videoHeight || 1;
    const scaleX = overlayCanvas.width / vw;
    const scaleY = overlayCanvas.height / vh;

    // Dim outside cropRegion
    if (cropRegion && cropRegion.w > 20 && cropRegion.h > 20) {
      const cx = cropRegion.x * scaleX;
      const cy = cropRegion.y * scaleY;
      const cw = cropRegion.w * scaleX;
      const ch = cropRegion.h * scaleY;

      overlayCtx.fillStyle = 'rgba(9, 13, 22, 0.65)';
      overlayCtx.beginPath();
      overlayCtx.rect(0, 0, overlayCanvas.width, overlayCanvas.height);
      overlayCtx.rect(cx, cy, cw, ch);
      overlayCtx.fill('evenodd');

      // Outline
      overlayCtx.strokeStyle = '#06b6d4';
      overlayCtx.lineWidth = 2;
      overlayCtx.strokeRect(cx, cy, cw, ch);
    }

    // Draw Fiducials
    if (lastFiducials && lastFiducials.length === 4) {
      const colors = ['#ef4444', '#10b981', '#a855f7', '#3b82f6']; // TL, TR, BR, BL
      lastFiducials.forEach((pt, idx) => {
        let px = pt[0];
        let py = pt[1];
        if (cropRegion) {
          px += cropRegion.x;
          py += cropRegion.y;
        }
        overlayCtx.fillStyle = colors[idx % 4];
        overlayCtx.beginPath();
        overlayCtx.arc(px * scaleX, py * scaleY, 5, 0, Math.PI * 2);
        overlayCtx.fill();
      });
    }
  }

  function autoSnapMask() {
    if (autoRoi && autoRoi.length === 4) {
      cropRegion = { x: autoRoi[0], y: autoRoi[1], w: autoRoi[2], h: autoRoi[3] };
      document.getElementById('cropStatus').textContent = `Mask: ${Math.round(cropRegion.w)}x${Math.round(cropRegion.h)} px`;
      logMsg(`[Mask] Auto-snapped tightly around fiducials (${Math.round(cropRegion.w)}x${Math.round(cropRegion.h)}).`);
    } else {
      logMsg('[Mask] Fiducials not locked yet. Position the remote window in view first.');
    }
  }

  function clearMask() {
    cropRegion = null;
    document.getElementById('cropStatus').textContent = 'Full Video (No Mask)';
    logMsg('[Mask] Mask cleared. Processing full frame.');
  }

  function toggleDrawMask() {
    isDrawingMask = !isDrawingMask;
    overlayCanvas.style.cursor = isDrawingMask ? 'crosshair' : 'default';
    document.getElementById('btnDrawMask').style.borderColor = isDrawingMask ? 'var(--accent-cyan)' : '#334155';
  }

  overlayCanvas.addEventListener('mousedown', (e) => {
    if (!isDrawingMask) return;
    const r = overlayCanvas.getBoundingClientRect();
    dragStart = { x: e.clientX - r.left, y: e.clientY - r.top };
  });

  overlayCanvas.addEventListener('mouseup', (e) => {
    if (!isDrawingMask || !dragStart) return;
    const r = overlayCanvas.getBoundingClientRect();
    const endX = e.clientX - r.left;
    const endY = e.clientY - r.top;

    const vw = videoEl.videoWidth;
    const vh = videoEl.videoHeight;
    const scaleX = vw / overlayCanvas.width;
    const scaleY = vh / overlayCanvas.height;

    const x = Math.min(dragStart.x, endX) * scaleX;
    const y = Math.min(dragStart.y, endY) * scaleY;
    const w = Math.abs(endX - dragStart.x) * scaleX;
    const h = Math.abs(endY - dragStart.y) * scaleY;

    if (w > 40 && h > 40) {
      cropRegion = { x, y, w, h };
      document.getElementById('cropStatus').textContent = `Mask: ${Math.round(w)}x${Math.round(h)} px`;
      logMsg(`[Mask] Set manual mask: ${Math.round(w)}x${Math.round(h)} px.`);
    }
    dragStart = null;
    toggleDrawMask();
  });

  // 6. Keyboard Access & File Staging
  function requestKeyboardAccess() {
    fetch('/api/request_keyboard', { method: 'POST' })
      .then(r => r.json())
      .then(resp => {
        const btn = document.getElementById('btnRequestKeyboard');
        if (btn) {
          if (resp.status === 'ok') {
            btn.textContent = '✓ Keyboard Access Ready';
            btn.style.borderColor = 'var(--accent-green)';
            btn.style.color = '#34d399';
            logMsg('[Server] Remote control / keyboard access requested. If GNOME displayed a prompt, click Allow.');
          } else {
            btn.textContent = '⚠ Access Denied / Unavailable';
            btn.style.borderColor = 'var(--accent-red)';
            btn.style.color = '#f87171';
            logMsg('[Server Error] Keyboard access request failed.');
          }
        }
      });
  }

  function handleFileSelect(files) {
    if (!files || files.length === 0) return;
    const file = files[0];
    const formData = new FormData();
    formData.append('file', file, file.name);

    document.getElementById('stagedFileDesc').textContent = `Uploading & Staging '${file.name}'...`;

    fetch('/api/upload', { method: 'POST', body: formData })
      .then(r => r.json())
      .then(resp => {
        if (resp.status === 'ok') {
          document.getElementById('stagedFileDesc').textContent = `Staged: ${file.name} (${(file.size/1024).toFixed(1)} KB)`;
          logMsg(`[Server] Staged file '${file.name}'. Focus client terminal and press ENTER.`);
        } else {
          document.getElementById('stagedFileDesc').textContent = `Error: ${resp.message}`;
          logMsg(`[Server Error] ${resp.message}`);
        }
      });
  }

  function setSpeed(preset) {
    fetch('/api/speed', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ preset })
    }).then(r => r.json()).then(resp => {
      document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      event.currentTarget.classList.add('active');
      logMsg(`[Server] Speed preset updated to: ${preset.toUpperCase()}`);
    });
  }

  function emergencyStop() {
    fetch('/api/abort', { method: 'POST' }).then(() => {
      logMsg('[🛑 EMERGENCY STOP] Transmission aborted by user.');
    });
  }

  function sendChat() {
    const input = document.getElementById('chatInput');
    const msg = input.value.trim();
    if (!msg) return;
    fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg })
    }).then(() => {
      logMsg(`[Chat] Sent keystrokes: "${msg}"`);
      input.value = '';
    });
  }

  function refreshFilesList() {
    fetch('/api/files').then(r => r.json()).then(files => {
      document.getElementById('filesCountBadge').textContent = files.length;
      const tbody = document.getElementById('filesTableBody');
      if (files.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; color:var(--text-dim);">No received files found.</td></tr>';
        return;
      }
      tbody.innerHTML = files.map(f => `
        <tr>
          <td><b>${f.filename}</b></td>
          <td>${f.size_str}</td>
          <td>${f.mtime_str}</td>
          <td>
            <a class="btn" style="text-decoration:none; padding:0.25rem 0.6rem; font-size:0.75rem;" href="/download/${encodeURIComponent(f.filename)}" download>⬇ Download</a>
            <button class="btn btn-primary" style="padding:0.25rem 0.6rem; font-size:0.75rem; margin-left:0.3rem;" onclick="stageSpecificFile('${f.filename}')">📤 Stage to Send</button>
          </td>
        </tr>
      `).join('');
    });
  }

  function stageSpecificFile(filename) {
    fetch('/api/stage_received', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename })
    }).then(r => r.json()).then(resp => {
      if (resp.status === 'ok') {
        switchTab('send');
        document.getElementById('stagedFileDesc').textContent = `Staged: ${filename}`;
        logMsg(`[Send] Staged file '${filename}'. Ready to transmit.`);
      }
    });
  }

  function logMsg(msg) {
    const consoleEl = document.getElementById('serverLog');
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    const now = new Date().toTimeString().split(' ')[0];
    entry.textContent = `[${now}] ${msg}`;
    consoleEl.appendChild(entry);
    consoleEl.scrollTop = consoleEl.scrollHeight;
  }
</script>
</body>
</html>
"""


# ==============================================================================
# HTTP & WEBSOCKET DISPATCHER
# ==============================================================================

class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_duplex_v2_http_handler(session):
    class DuplexV2HTTPHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/ws" and self.headers.get("Upgrade", "").lower() == "websocket":
                self.handle_websocket()
            elif self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(SERVER_V2_HTML_PAGE.encode("utf-8"))
            elif self.path == "/api/status":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                status_dict = {
                    "active_mode": session.active_mode,
                    "send": session.send_session.get_telemetry_dict(),
                    "receive": session.receive_session.get_status_dict(),
                    "files": session.get_received_files_list()
                }
                self.wfile.write(json.dumps(status_dict).encode("utf-8"))
            elif self.path == "/api/files":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(session.get_received_files_list()).encode("utf-8"))
            elif self.path.startswith("/download/"):
                raw_name = self.path[len("/download/"):]
                from urllib.parse import unquote
                filename = os.path.basename(unquote(raw_name))
                file_path = os.path.join(session.output_dir, filename)
                if os.path.isfile(file_path):
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                    self.send_header("Content-Length", str(os.path.getsize(file_path)))
                    self.end_headers()
                    with open(file_path, "rb") as f:
                        while True:
                            chunk = f.read(65536)
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                else:
                    self.send_response(404)
                    self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == "/api/upload":
                ctype = self.headers.get("Content-Type", "")
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)

                filename = "transfer_file.bin"
                raw_bytes = body
                if "multipart/form-data" in ctype:
                    boundary = ctype.split("boundary=")[-1].encode("ascii")
                    parts = body.split(b"--" + boundary)
                    for part in parts:
                        if b'Content-Disposition: form-data;' in part and b'filename="' in part:
                            header_part, file_part = part.split(b"\r\n\r\n", 1)
                            for line in header_part.decode("utf-8", errors="replace").split("\r\n"):
                                if "filename=" in line:
                                    filename = line.split('filename="')[1].split('"')[0]
                            raw_bytes = file_part.rstrip(b"\r\n")
                            break

                ok, msg = session.send_session.stage_file(filename, raw_bytes)
                self.send_response(200 if ok else 400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok" if ok else "error", "message": msg}).encode("utf-8"))

            elif self.path == "/api/stage_received":
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    data = json.loads(body.decode("utf-8"))
                    fn = data.get("filename", "")
                    ok, msg = session.stage_received_file(fn)
                    self.send_response(200 if ok else 400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ok" if ok else "error", "message": msg}).encode("utf-8"))
                except Exception as e:
                    self.send_response(400)
                    self.end_headers()

            elif self.path == "/api/abort":
                session.send_session.abort_transmission("User or browser requested abort")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "message": "Transmission halted"}).encode("utf-8"))

            elif self.path == "/api/speed":
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    data = json.loads(body.decode("utf-8"))
                    preset = data.get("preset", "fast")
                    session.send_session.set_speed_preset(preset)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ok", "preset": preset}).encode("utf-8"))
                except Exception:
                    self.send_response(400)
                    self.end_headers()

            elif self.path == "/api/chat":
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    data = json.loads(body.decode("utf-8"))
                    msg = data.get("message", "")
                    session.send_session.send_chat_message(msg)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
                except Exception:
                    self.send_response(400)
                    self.end_headers()

            elif self.path == "/api/reset":
                session.send_session.reset_state()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))

            elif self.path == "/api/receive/reset":
                session.receive_session.reset()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))

            elif self.path == "/api/set_mode":
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    data = json.loads(body.decode("utf-8"))
                    mode = data.get("mode", "auto")
                    session.active_mode = mode
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ok", "mode": mode}).encode("utf-8"))
                except Exception:
                    self.send_response(400)
                    self.end_headers()

            elif self.path == "/api/request_keyboard":
                ok = session.send_session.typer.request_access()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "requested": ok}).encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()

        def handle_websocket(self):
            key = self.headers.get("Sec-WebSocket-Key", "")
            if not key:
                self.send_error(400, "Missing Sec-WebSocket-Key")
                return

            guid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
            accept_val = base64.b64encode(hashlib.sha1((key.strip() + guid).encode("utf-8")).digest()).decode("utf-8")
            handshake = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept_val}\r\n\r\n"
            )
            self.wfile.write(handshake.encode("utf-8"))
            self.wfile.flush()

            sock = self.connection
            sock.settimeout(15.0)

            def recv_exact(n):
                buf = bytearray()
                while len(buf) < n:
                    try:
                        c = sock.recv(n - len(buf))
                        if not c:
                            return None
                        buf.extend(c)
                    except Exception:
                        return None
                return bytes(buf)

            def send_ws_frame(opcode, payload_bytes):
                plen = len(payload_bytes)
                if plen < 126:
                    hdr = struct.pack(">BB", 0x80 | (opcode & 0x0f), plen)
                elif plen < 65536:
                    hdr = struct.pack(">BBH", 0x80 | (opcode & 0x0f), 126, plen)
                else:
                    hdr = struct.pack(">BBQ", 0x80 | (opcode & 0x0f), 127, plen)
                try:
                    self.wfile.write(hdr + payload_bytes)
                    self.wfile.flush()
                except Exception:
                    pass

            try:
                while True:
                    b12 = recv_exact(2)
                    if not b12:
                        break
                    opcode = b12[0] & 0x0f
                    masked = (b12[1] & 0x80) != 0
                    plen = b12[1] & 0x7f

                    if opcode == 0x8:  # Close
                        break
                    if plen == 126:
                        ext = recv_exact(2)
                        if not ext: break
                        plen = struct.unpack(">H", ext)[0]
                    elif plen == 127:
                        ext = recv_exact(8)
                        if not ext: break
                        plen = struct.unpack(">Q", ext)[0]

                    mask_key = recv_exact(4) if masked else None
                    raw = recv_exact(plen)
                    if raw is None or len(raw) < plen:
                        break

                    if masked and mask_key:
                        raw_arr = np.frombuffer(raw, dtype=np.uint8)
                        k_arr = np.frombuffer(mask_key, dtype=np.uint8)
                        payload = (raw_arr ^ np.resize(k_arr, len(raw_arr))).tobytes()
                    else:
                        payload = raw

                    if opcode == 0x2:  # Binary JPEG video frame
                        resp = session.process_image_bytes(payload)
                        send_ws_frame(0x1, json.dumps(resp).encode("utf-8"))
            except Exception:
                pass
            finally:
                session.send_session.abort_transmission("Screen share / WebSocket session ended")

        def log_message(self, format, *args):
            return

    return DuplexV2HTTPHandler


# ==============================================================================
# MAIN RUNNER
# ==============================================================================

def run_server_v2(port=8080, host="0.0.0.0", output_dir="./received",
                  chunk_size=256, delay=0.0002, preset="fast",
                  model_path=None, open_browser=True):

    session = ServerV2UnifiedSession(
        output_dir=output_dir,
        chunk_size=chunk_size,
        char_delay=delay,
        speed_preset=preset,
        model_path=model_path
    )

    server = None
    target_port = port
    for p in range(target_port, target_port + 20):
        try:
            handler_cls = make_duplex_v2_http_handler(session)
            server = ThreadedHTTPServer((host, p), handler_cls)
            target_port = p
            break
        except OSError:
            continue

    if server is None:
        print(f"Error: Unable to bind to port {port} or next 20 ports.")
        return False

    url = f"http://localhost:{target_port}"
    print("=" * 68)
    print("   OTD (OPTICAL DATA TRANSFER) SERVER v2 - UNIFIED SEND & RECEIVE")
    print("=" * 68)
    print(f"Web Control Interface:   {url}")
    print(f"Send Keyboard Synthesizer: {session.send_session.typer.backend} (Alphanumeric only)")
    print(f"Send Speed Preset:        {preset.upper()} ({chunk_size}B chunks, {delay*1000:.2f}ms char delay)")
    print(f"Received Files Location:  {os.path.abspath(output_dir)}")
    print(f"Optical Video Decoder:    ACTIVE (auto-detects 32, 48, 64, 80... RGB grids)")
    print("=" * 68)

    print("[Server v2] Keyboard synthesizer access: On-demand (via Send UI button or file staging)")

    if open_browser:
        def _open():
            time.sleep(0.5)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server v2] Shutting down...")
    finally:
        server.server_close()
        session.send_session.typer.release_all_keys()
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OTD (Optical Data Transfer) Server v2 - Bidirectional Send & Receive"
    )
    parser.add_argument("-p", "--port", type=int, default=8080, help="Web port (default: 8080)")
    parser.add_argument("--host", default="0.0.0.0", help="Binding host (default: 0.0.0.0)")
    parser.add_argument("-o", "--output-dir", default="./received", help="Directory to save received files (default: ./received)")
    parser.add_argument("-c", "--chunk-size", type=int, default=256, help="Send chunk size in bytes (default: 256)")
    parser.add_argument("-d", "--delay", type=float, default=0.0002, help="Send character typing delay in seconds (default: 0.0002)")
    parser.add_argument("--preset", choices=["turbo", "fast", "balanced", "safe"], default="fast", help="Send speed preset (default: fast)")
    parser.add_argument("--model", default=None, help="Path to ONNX neural tracking model (default: auto-load otd_tracker.onnx if present; pass 'none' to disable)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")

    args = parser.parse_args()
    run_server_v2(
        port=args.port,
        host=args.host,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        delay=args.delay,
        preset=args.preset,
        model_path=args.model,
        open_browser=not args.no_browser
    )
