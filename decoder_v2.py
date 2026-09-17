#!/usr/bin/env python3
"""
decoder_v2.py - High-Speed Visual Optical Receiver (v2)
Decodes transmissions from encoder_v2.py.
Features 4-corner perspective dewarping (immune to RDP scaling/rotation),
RGB 8-color sampling, per-frame IEEE 802.3 CRC32 integrity validation,
and end-to-end SHA-256 verification.

Supports:
- Video file input (.mp4, .mkv, .webm, etc.)
- Live Web Capture (--http): spins up a web app for Wayland screen/window sharing
- Native desktop screen capture (--live, X11)
- Live webcam capture (--camera)
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

# Suppress harmless Qt font lookup warnings on Wayland/Linux
os.environ["QT_LOGGING_RULES"] = "*.warning=false"

import cv2
import numpy as np

CANONICAL_DIM = 1000
# In canonical space, the 4 fiducials are at (40, 40), (960, 40), (40, 960), (960, 960)
DST_PTS = np.float32([[40, 40], [960, 40], [960, 960], [40, 960]])
# The data grid spans from 80 to 920 in both axes (width = 840)
GRID_START = 80
GRID_SPAN = 840
# Standard grid dimensions scanned during auto-detection
DEFAULT_CANDIDATE_GRIDS = [64, 48, 80, 96, 128, 32, 40, 56, 72, 112, 144, 160]

def detect_fiducials(frame):
    """
    Detects the 4 corner fiducials using hybrid HSV and chromatic-difference segmentation:
    - TL: Red marker with min(x + y)
    - TR: Green marker with max(x - y)
    - BL: Blue marker with min(x - y)
    - BR: Magenta marker with max(x + y)
    Returns: (tl, tr, br, bl) coordinates or None.
    """
    def get_extreme_pt(mask, key_fn):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pts = []
        for c in cnts:
            if cv2.contourArea(c) > 6:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    pts.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
        return key_fn(pts) if pts else None

    # 1. First attempt: Standard HSV color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    masks_hsv = {
        "TL": cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 80, 70]), np.array([12, 255, 255])),
            cv2.inRange(hsv, np.array([168, 80, 70]), np.array([180, 255, 255]))
        ),
        "TR": cv2.inRange(hsv, np.array([35, 80, 70]), np.array([85, 255, 255])),
        "BL": cv2.inRange(hsv, np.array([95, 80, 70]), np.array([135, 255, 255])),
        "BR": cv2.inRange(hsv, np.array([140, 80, 70]), np.array([168, 255, 255]))
    }

    tl = get_extreme_pt(masks_hsv["TL"], lambda pts: min(pts, key=lambda p: p[0] + p[1]))
    tr = get_extreme_pt(masks_hsv["TR"], lambda pts: max(pts, key=lambda p: p[0] - p[1]))
    bl = get_extreme_pt(masks_hsv["BL"], lambda pts: min(pts, key=lambda p: p[0] - p[1]))
    br = get_extreme_pt(masks_hsv["BR"], lambda pts: max(pts, key=lambda p: p[0] + p[1]))

    if tl and tr and bl and br:
        if tl[1] < bl[1] and tl[0] < tr[0] and tr[1] < br[1] and bl[0] < br[0]:
            return refine_subpixel_fiducials(frame, (tl, tr, br, bl))

    # 2. Second attempt: Robust chromatic difference (immune to lighting & auto-exposure)
    b = frame[:, :, 0].astype(np.int16)
    g = frame[:, :, 1].astype(np.int16)
    r = frame[:, :, 2].astype(np.int16)

    r_diff = np.clip(r - np.maximum(g, b), 0, 255).astype(np.uint8)
    g_diff = np.clip(g - np.maximum(r, b), 0, 255).astype(np.uint8)
    b_diff = np.clip(b - np.maximum(r, g), 0, 255).astype(np.uint8)
    m_diff = np.clip(np.minimum(r, b) - g, 0, 255).astype(np.uint8)

    def get_chroma_pt(diff_img, key_fn):
        val = np.percentile(diff_img, 99.5)
        if val < 20:
            return None
        _, mask = cv2.threshold(diff_img, int(val * 0.6), 255, cv2.THRESH_BINARY)
        return get_extreme_pt(mask, key_fn)

    tl = get_chroma_pt(r_diff, lambda pts: min(pts, key=lambda p: p[0] + p[1]))
    tr = get_chroma_pt(g_diff, lambda pts: max(pts, key=lambda p: p[0] - p[1]))
    bl = get_chroma_pt(b_diff, lambda pts: min(pts, key=lambda p: p[0] - p[1]))
    br = get_chroma_pt(m_diff, lambda pts: max(pts, key=lambda p: p[0] + p[1]))

    if tl and tr and bl and br:
        if tl[1] < bl[1] and tl[0] < tr[0] and tr[1] < br[1] and bl[0] < br[0]:
            return refine_subpixel_fiducials(frame, (tl, tr, br, bl))

    return None

def refine_subpixel_fiducials(frame, fids):
    """Refines corner fiducial coordinates to subpixel accuracy using cv2.cornerSubPix."""
    if not fids or len(fids) != 4:
        return fids
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pts = np.float32(fids).reshape(-1, 1, 2)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        refined = cv2.cornerSubPix(gray, pts.copy(), (5, 5), (-1, -1), criteria).reshape(-1, 2)
        out = []
        for i in range(4):
            orig_pt = fids[i]
            ref_pt = (float(refined[i, 0]), float(refined[i, 1]))
            dist = np.hypot(ref_pt[0] - orig_pt[0], ref_pt[1] - orig_pt[1])
            # Guard against cornerSubPix drifting away to external window borders
            if dist > 3.0:
                out.append((float(orig_pt[0]), float(orig_pt[1])))
            else:
                out.append(ref_pt)
        return tuple(out)
    except Exception:
        return fids

def calibrate_fiducial_thresholds(warped):
    """
    Samples the 4 corner chromatic fiducials plus white/black borders to
    dynamically compute empirical per-channel thresholds (BGR).
    Eliminates color distortion caused by display gamma, night-light, and chroma compression.
    Works seamlessly across both GUI Window mode and TTY terminal transmissions.
    """
    try:
        red_val = np.mean(warped[36:45, 36:45], axis=(0, 1))
        green_val = np.mean(warped[36:45, 956:965], axis=(0, 1))
        blue_val = np.mean(warped[956:965, 36:45], axis=(0, 1))
        mag_val = np.mean(warped[956:965, 956:965], axis=(0, 1))

        # Sample white from fiducial outer borders (guaranteed white in both GUI and TTY)
        white_tl = np.mean(warped[18:25, 36:44], axis=(0, 1))
        white_tr = np.mean(warped[18:25, 956:964], axis=(0, 1))
        white_val = np.maximum(white_tl, white_tr)

        # Sample black from canonical black margins (outside fiducials and between grid)
        # Avoid sampling within fiducial cores where TTY lacks a middle black ring
        black_top = np.mean(warped[15:35, 450:550], axis=(0, 1))
        black_diag = np.mean(warped[58:65, 58:65], axis=(0, 1))
        black_val = np.minimum(black_top, black_diag)

        th_b = (np.mean([blue_val[0], mag_val[0], white_val[0]]) + np.mean([black_val[0], red_val[0], green_val[0]])) / 2.0
        th_g = (np.mean([green_val[1], white_val[1]]) + np.mean([black_val[1], red_val[1], blue_val[1], mag_val[1]])) / 2.0
        th_r = (np.mean([red_val[2], mag_val[2], white_val[2]]) + np.mean([black_val[2], green_val[2], blue_val[2]])) / 2.0

        return float(np.clip(th_b, 15, 240)), float(np.clip(th_g, 15, 240)), float(np.clip(th_r, 15, 240))
    except Exception:
        return 128.0, 128.0, 128.0

def get_candidate_geometries(grid_size):
    """
    Returns candidate (name, grid_start, grid_span) layout tuples for a given grid_size.
    Supports standard GUI layout, symmetric TTY layout, and legacy TTY layout.
    """
    # 1. Standard GUI mode:
    # Fiducials at 40 and 960 (distance 920). Data grid starts at 80, spans 840.
    gui_geom = ("gui", 80.0, 840.0)

    # 2. Symmetric TTY mode:
    # Margin M units, Distance D = grid_size + 2 * M.
    M = max(2, int(round(grid_size / 21.0)))
    D = grid_size + 2 * M
    unit = 920.0 / float(D)
    g_start_tty = 40.0 + (M - 0.5) * unit
    g_span_tty = grid_size * unit
    tty_geom = ("tty", g_start_tty, g_span_tty)

    # 3. Legacy TTY mode (backward compatibility with earlier TTY encoders):
    D_leg = int(round(grid_size * 23.0 / 21.0))
    unit_leg = 920.0 / float(D_leg)
    g_start_leg = 40.0 + 1.5 * unit_leg
    g_span_leg = grid_size * unit_leg
    tty_leg_geom = ("tty_legacy", g_start_leg, g_span_leg)

    return [gui_geom, tty_geom, tty_leg_geom]

def sample_dewarped_grid(warped, grid_size, mode, th_b=128, th_g=128, th_r=128, return_mean=False, grid_start=None, grid_span=None):
    """
    High-speed vectorized cell sampling from the canonical 1000x1000 dewarped image.
    Uses 3x3 patch averaging and fast bit packing. 25-50x faster than pure Python loops.
    Supports customizable canonical layout geometry for GUI and TTY modes.
    Returns: raw bytes string (or (raw bytes, mean_bgr) if return_mean=True).
    """
    g_start = GRID_START if grid_start is None else float(grid_start)
    g_span = GRID_SPAN if grid_span is None else float(grid_span)

    canon_cell = g_span / float(grid_size)
    c_indices = np.arange(grid_size)
    r_indices = np.arange(grid_size)
    cx = (g_start + (c_indices + 0.5) * canon_cell).astype(np.int32)
    cy = (g_start + (r_indices + 0.5) * canon_cell).astype(np.int32)
    cy_grid, cx_grid = np.meshgrid(cy, cx, indexing="ij")

    # Vectorized 3x3 patch sampling
    patches = np.zeros((grid_size, grid_size, 3), dtype=np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            patches += warped[cy_grid + dy, cx_grid + dx]
    mean_bgr = patches / 9.0

    if mode == "rgb":
        b_bit = (mean_bgr[..., 0] > th_b).astype(np.uint8)
        g_bit = (mean_bgr[..., 1] > th_g).astype(np.uint8)
        r_bit = (mean_bgr[..., 2] > th_r).astype(np.uint8)
        val = (r_bit << 2) | (g_bit << 1) | b_bit
        flat_vals = val.ravel()
        bits = np.unpackbits(flat_vals[:, None].astype(np.uint8), axis=1)[:, 5:8].ravel()
    else:
        luma = np.mean(mean_bgr, axis=2)
        luma_bit = (luma > th_b).astype(np.uint8)
        bits = luma_bit.ravel()

    n_bytes = len(bits) // 8
    bits = bits[:n_bytes * 8]
    raw_bytes = np.packbits(bits).tobytes()

    if return_mean:
        return raw_bytes, mean_bgr
    return raw_bytes

def parse_and_validate_frame(raw_bytes):
    """
    Validates the 16-byte frame header and IEEE 802.3 CRC32 checksum.
    Returns: (idx, total_chunks, payload, grid_size, mode_str) or None.
    """
    if len(raw_bytes) < 16:
        return None

    magic, ver, mode_byte, grid_size, flags, idx, total, plen, crc = struct.unpack(">2sBBBBHHHI", raw_bytes[:16])

    if magic != b"Q2" or ver != 2:
        return None

    is_parity = (flags & 2) != 0
    if total == 0:
        return None
    if not is_parity and idx >= total:
        return None
    if is_parity and (idx < total or idx >= total * 2 + 32):
        return None

    if len(raw_bytes) < 16 + plen:
        return None

    payload = raw_bytes[16:16 + plen]
    computed_crc = zlib.crc32(payload) & 0xffffffff

    if computed_crc != crc:
        return None

    mode_str = "rgb" if mode_byte == 3 else "bw"
    return idx, total, payload, grid_size, mode_str

class FiducialTracker:
    """
    High-Performance Visual Tracker & Decoder Session:
    - Lucas-Kanade optical flow tracking (cv2.calcOpticalFlowPyrLK) at < 0.5ms per frame
    - Temporal Exponential Moving Average (EMA) corner stabilization to eliminate jitter
    - Duplicate frame bypass for fast 30/60 FPS captures
    - Adaptive per-channel percentile thresholding for dim/gamma-distorted displays
    - CRC-driven threshold perturbation to salvage boundary-error frames
    - Optional OpenCV DNN neural model acceleration via --model / --onnx
    """
    def __init__(self, model_path=None, ema_alpha=0.85):
        # Auto-detect default model if not explicitly specified
        if model_path is None:
            default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "otd_tracker.onnx")
            if os.path.exists(default_path):
                model_path = default_path
        elif str(model_path).lower() in ("none", "false", "off", "0", ""):
            model_path = None

        self.model_path = model_path
        self.net = None
        if model_path and os.path.exists(model_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(model_path)
                print(f"[ML Tracker] Loaded ONNX tracking model: {model_path}")
            except Exception as e:
                print(f"[ML Tracker] Warning: Could not load ONNX model '{model_path}': {e}")

        self.ema_alpha = ema_alpha
        self.prev_gray = None
        self.prev_pts = None
        self.ema_pts = None
        self.tracking_confidence = 0
        self.prev_thumb = None
        self.last_res = None
        self.last_warped = None
        self.last_fiducials = None
        self.prev_warped = None
        self.locked_geometry = None

    def reset(self):
        self.prev_gray = None
        self.prev_pts = None
        self.ema_pts = None
        self.tracking_confidence = 0
        self.prev_thumb = None
        self.last_res = None
        self.last_warped = None
        self.last_fiducials = None
        self.prev_warped = None
        self.locked_geometry = None

    def verify_fiducials(self, frame, fiducials):
        """Verifies quadrilateral geometry and chromatic corner colors."""
        if not fiducials:
            return False
        tl, tr, br, bl = fiducials
        h, w = frame.shape[:2]
        for pt in (tl, tr, br, bl):
            if pt[0] < 0 or pt[0] >= w or pt[1] < 0 or pt[1] >= h:
                return False
        if not (tl[1] < bl[1] and tl[0] < tr[0] and tr[1] < br[1] and bl[0] < br[0]):
            return False

        def get_color(pt):
            x, y = int(pt[0]), int(pt[1])
            patch = frame[max(0, y - 2):min(h, y + 3), max(0, x - 2):min(w, x + 3)]
            return np.mean(patch, axis=(0, 1))  # BGR

        b_tl, g_tl, r_tl = get_color(tl)
        b_tr, g_tr, r_tr = get_color(tr)
        b_bl, g_bl, r_bl = get_color(bl)
        b_br, g_br, r_br = get_color(br)

        # Red: R > G and R > B
        if not (r_tl > g_tl and r_tl > b_tl):
            return False
        # Green: G > R and G > B
        if not (g_tr > r_tr and g_tr > b_tr):
            return False
        # Blue: B > R and B > G
        if not (b_bl > r_bl and b_bl > g_bl):
            return False
        # Magenta: R > G and B > G
        if not (r_br > g_br and b_br > g_br):
            return False
        return True

    def detect_ml(self, frame):
        """Runs optional ONNX keypoint model inference via cv2.dnn."""
        if self.net is None:
            return None
        try:
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (128, 128), (0, 0, 0), swapRB=True, crop=False)
            self.net.setInput(blob)
            out = self.net.forward()
            coords = out.flatten()
            if len(coords) >= 8:
                tl = (float(coords[0] * w), float(coords[1] * h))
                tr = (float(coords[2] * w), float(coords[3] * h))
                br = (float(coords[4] * w), float(coords[5] * h))
                bl = (float(coords[6] * w), float(coords[7] * h))
                cand = (tl, tr, br, bl)
                if self.verify_fiducials(frame, cand):
                    return cand
        except Exception:
            pass
        return None

    def get_fiducials(self, frame):
        """
        Locates the 4 corner fiducials using optical flow if active,
        falling back to ML/chromatic detection upon signal loss.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Active optical flow tracking (< 0.5ms) - requires matching dimensions
        if (
            self.prev_pts is not None 
            and self.prev_gray is not None 
            and self.prev_gray.shape == gray.shape
        ):
            try:
                next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, self.prev_pts, None,
                    winSize=(25, 25), maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
                )
                if status is not None and np.all(status == 1) and np.all(err < 20.0):
                    pts = next_pts.reshape(-1, 2)
                    tl, tr, br, bl = pts[0], pts[1], pts[2], pts[3]
                    if tl[1] < bl[1] and tl[0] < tr[0] and tr[1] < br[1] and bl[0] < br[0]:
                        # Smooth jitter with exponential moving average
                        self.ema_pts = self.ema_alpha * pts + (1.0 - self.ema_alpha) * self.ema_pts
                        self.prev_pts = self.ema_pts.reshape(-1, 1, 2).astype(np.float32)
                        self.prev_gray = gray
                        self.tracking_confidence += 1
                        smoothed = tuple((float(p[0]), float(p[1])) for p in self.ema_pts)
                        self.last_fiducials = smoothed
                        return smoothed
            except cv2.error:
                # If optical flow fails or ROI crop changes dynamically, reset tracking
                self.prev_pts = None
                self.prev_gray = None
        else:
            self.prev_pts = None
            self.prev_gray = None

        # 2. ML model inference (if configured)
        pts = None
        if self.net is not None:
            pts = self.detect_ml(frame)

        # 3. Hybrid chromatic / HSV detection fallback
        if pts is None:
            pts = detect_fiducials(frame)

        if pts is not None:
            pts_arr = np.array(pts, dtype=np.float32)
            self.prev_pts = pts_arr.reshape(-1, 1, 2)
            self.ema_pts = pts_arr.copy()
            self.prev_gray = gray
            self.tracking_confidence = 1
            pts_floats = tuple((float(p[0]), float(p[1])) for p in pts)
            self.last_fiducials = pts_floats
            return pts_floats

        # Signal lost
        self.prev_pts = None
        self.prev_gray = None
        self.tracking_confidence = 0
        self.last_fiducials = None
        return None

    def decode_frame(self, frame, locked_grid_size=None, locked_mode="rgb"):
        """
        Decodes a single visual frame with optical flow tracking, vectorized sampling,
        adaptive thresholding, and duplicate frame skipping.
        """
        # Fast duplicate frame check (< 0.1ms)
        thumb = cv2.resize(frame, (64, 64))
        if self.prev_thumb is not None:
            diff = np.mean(cv2.absdiff(thumb, self.prev_thumb))
            if diff < 0.6 and self.last_res is not None:
                return self.last_res, self.last_fiducials, self.last_warped

        self.prev_thumb = thumb

        fiducials = self.get_fiducials(frame)
        if not fiducials:
            self.last_res = None
            return None, None, None

        tl, tr, br, bl = fiducials
        src_pts = np.float32([tl, tr, br, bl])
        M_warp = cv2.getPerspectiveTransform(src_pts, DST_PTS)
        warped = cv2.warpPerspective(frame, M_warp, (CANONICAL_DIM, CANONICAL_DIM))
        self.last_warped = warped

        candidate_grids = [locked_grid_size] if locked_grid_size else DEFAULT_CANDIDATE_GRIDS
        modes = (locked_mode, "bw" if locked_mode == "rgb" else "rgb") if locked_grid_size else ("rgb", "bw")

        # Calibrate optimal per-channel thresholds using fiducial reference colors
        th_b_cal, th_g_cal, th_r_cal = calibrate_fiducial_thresholds(warped)

        for gs in candidate_grids:
            geoms = [self.locked_geometry] if self.locked_geometry else get_candidate_geometries(gs)
            for mode_cand in modes:
                for geom_entry in geoms:
                    geom_name, g_start, g_span = geom_entry

                    # Pass 1: Calibrated threshold directly from fiducials (dynamic auto-exposure/gamma)
                    raw_b, mean_bgr = sample_dewarped_grid(
                        warped, gs, mode_cand, th_b_cal, th_g_cal, th_r_cal,
                        return_mean=True, grid_start=g_start, grid_span=g_span
                    )
                    res = parse_and_validate_frame(raw_b)
                    if res:
                        self.locked_geometry = geom_entry
                        self.last_res = res
                        self.prev_warped = warped.copy()
                        return res, fiducials, warped

                    # Pass 2: Standard threshold (128) if calibrated had significant deviation
                    if abs(th_b_cal - 128) > 3 or abs(th_g_cal - 128) > 3 or abs(th_r_cal - 128) > 3:
                        raw_b_std = sample_dewarped_grid(
                            warped, gs, mode_cand, 128, 128, 128,
                            grid_start=g_start, grid_span=g_span
                        )
                        res = parse_and_validate_frame(raw_b_std)
                        if res:
                            self.locked_geometry = geom_entry
                            self.last_res = res
                            self.prev_warped = warped.copy()
                            return res, fiducials, warped

                    # Pass 3: Adaptive per-channel percentile threshold (recovers dim / gamma-shifted screens)
                    if mode_cand == "rgb":
                        th_b = (np.percentile(mean_bgr[..., 0], 25) + np.percentile(mean_bgr[..., 0], 75)) / 2.0
                        th_g = (np.percentile(mean_bgr[..., 1], 25) + np.percentile(mean_bgr[..., 1], 75)) / 2.0
                        th_r = (np.percentile(mean_bgr[..., 2], 25) + np.percentile(mean_bgr[..., 2], 75)) / 2.0
                    else:
                        luma = np.mean(mean_bgr, axis=2)
                        th_b = th_g = th_r = (np.percentile(luma, 25) + np.percentile(luma, 75)) / 2.0

                    if abs(th_b - 128) > 5 or abs(th_g - 128) > 5 or abs(th_r - 128) > 5:
                        raw_b_ad, _ = sample_dewarped_grid(
                            warped, gs, mode_cand, th_b, th_g, th_r,
                            return_mean=True, grid_start=g_start, grid_span=g_span
                        )
                        res = parse_and_validate_frame(raw_b_ad)
                        if res:
                            self.locked_geometry = geom_entry
                            self.last_res = res
                            self.prev_warped = warped.copy()
                            return res, fiducials, warped

                    # Pass 4: Perturbation fallback (salvages compression noise / edge boundary bits)
                    for delta in (-12, 12):
                        raw_b_pert, _ = sample_dewarped_grid(
                            warped, gs, mode_cand,
                            np.clip(th_b + delta, 10, 245),
                            np.clip(th_g + delta, 10, 245),
                            np.clip(th_r + delta, 10, 245),
                            return_mean=True,
                            grid_start=g_start, grid_span=g_span
                        )
                        res = parse_and_validate_frame(raw_b_pert)
                        if res:
                            self.locked_geometry = geom_entry
                            self.last_res = res
                            self.prev_warped = warped.copy()
                            return res, fiducials, warped

                    # Pass 5: Multi-frame temporal super-resolution averaging (kills sensor/DCT compression noise)
                    if self.prev_warped is not None and self.prev_warped.shape == warped.shape:
                        warped_avg = cv2.addWeighted(warped, 0.5, self.prev_warped, 0.5, 0)
                        raw_b_avg = sample_dewarped_grid(
                            warped_avg, gs, mode_cand, th_b_cal, th_g_cal, th_r_cal,
                            grid_start=g_start, grid_span=g_span
                        )
                        res = parse_and_validate_frame(raw_b_avg)
                        if res:
                            self.locked_geometry = geom_entry
                            self.last_res = res
                            self.prev_warped = warped.copy()
                            return res, fiducials, warped

        # Pass 6: Fallback fast probe for arbitrary non-standard grid sizes (16 to 160)
        if not locked_grid_size:
            candidate_set = set(DEFAULT_CANDIDATE_GRIDS)
            for probe_gs in range(16, 161):
                if probe_gs in candidate_set:
                    continue
                geoms_probe = [self.locked_geometry] if self.locked_geometry else get_candidate_geometries(probe_gs)
                for mode_cand in ("rgb", "bw"):
                    for geom_entry in geoms_probe:
                        geom_name, g_start, g_span = geom_entry
                        canon_cell = g_span / float(probe_gs)
                        cy = int(g_start + 0.5 * canon_cell)
                        if mode_cand == "rgb":
                            cxs = (g_start + (np.arange(6) + 0.5) * canon_cell).astype(np.int32)
                            mean_bgr = warped[cy, cxs]
                            b_bit = (mean_bgr[..., 0] > th_b_cal).astype(np.uint8)
                            g_bit = (mean_bgr[..., 1] > th_g_cal).astype(np.uint8)
                            r_bit = (mean_bgr[..., 2] > th_r_cal).astype(np.uint8)
                            val = (r_bit << 2) | (g_bit << 1) | b_bit
                            bits = np.unpackbits(val[:, None], axis=1)[:, 5:8].ravel()[:16]
                        else:
                            cxs = (g_start + (np.arange(16) + 0.5) * canon_cell).astype(np.int32)
                            luma = np.mean(warped[cy, cxs], axis=1)
                            bits = (luma > 128).astype(np.uint8)

                        if np.packbits(bits).tobytes() == b"Q2":
                            raw_b = sample_dewarped_grid(
                                warped, probe_gs, mode_cand, th_b_cal, th_g_cal, th_r_cal,
                                grid_start=g_start, grid_span=g_span
                            )
                            res = parse_and_validate_frame(raw_b)
                            if res:
                                self.locked_geometry = geom_entry
                                self.last_res = res
                                self.prev_warped = warped.copy()
                                return res, fiducials, warped

        self.last_res = None
        self.prev_warped = warped.copy()
        return None, fiducials, warped

_default_tracker = None

def decode_single_frame(frame, locked_grid_size=None, locked_mode="rgb", tracker=None):
    """
    Decodes a single visual frame:
    1. Tracks/detects fiducials
    2. Perspective dewarps to 1000x1000 canonical plane
    3. Samples cells with vectorized speed and checks CRC32
    Returns: (decoded_result, fiducials, warped)
    """
    global _default_tracker
    active_tracker = tracker
    if active_tracker is None:
        if _default_tracker is None:
            _default_tracker = FiducialTracker()
        active_tracker = _default_tracker
    return active_tracker.decode_frame(frame, locked_grid_size, locked_mode)

# =====================================================================
# FORWARD ERROR CORRECTION (CAUCHY REED-SOLOMON GF(2^8) ERASURE DECODER)
# =====================================================================
_GF_EXP = [0] * 512
_GF_LOG = [0] * 256
_x = 1
for _i in range(255):
    _GF_EXP[_i] = _x
    _GF_EXP[_i + 255] = _x
    _GF_LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11d
_GF_LOG[0] = 511

def _gf_mul(a, b):
    return 0 if a == 0 or b == 0 else _GF_EXP[_GF_LOG[a] + _GF_LOG[b]]

def _gf_inv(a):
    return _GF_EXP[255 - _GF_LOG[a]]

def _cauchy_coeff(r, c):
    return _gf_inv((255 - r) ^ c)

def recover_fec_chunks(collected_chunks, K):
    """
    Recovers missing data chunks (0..K-1) from parity chunks (idx >= K)
    using systematic Cauchy Reed-Solomon erasure decoding.
    """
    missing = [i for i in range(K) if i not in collected_chunks]
    if not missing:
        return {i: collected_chunks[i] for i in range(K)}

    parities_avail = [i for i in collected_chunks if i >= K]
    if len(parities_avail) < len(missing):
        return None

    used_parities = sorted(parities_avail)[:len(missing)]
    L = len(missing)
    chunk_len = max(len(v) for v in collected_chunks.values())
    S = []
    R = []
    for p_idx in used_parities:
        p_row = p_idx - K
        S.append([_cauchy_coeff(p_row, m) for m in missing])
        rhs = bytearray(collected_chunks[p_idx].ljust(chunk_len, b"\x00"))
        for j in range(K):
            if j not in missing:
                coeff = _cauchy_coeff(p_row, j)
                if coeff == 0:
                    continue
                d = collected_chunks[j].ljust(chunk_len, b"\x00")
                for b_i in range(chunk_len):
                    if d[b_i]:
                        rhs[b_i] ^= _gf_mul(coeff, d[b_i])
        R.append(rhs)

    for i in range(L):
        if S[i][i] == 0:
            for r in range(i + 1, L):
                if S[r][i] != 0:
                    S[i], S[r] = S[r], S[i]
                    R[i], R[r] = R[r], R[i]
                    break
        pivot = S[i][i]
        if pivot == 0:
            return None
        inv_p = _gf_inv(pivot)
        for c in range(i, L):
            S[i][c] = _gf_mul(S[i][c], inv_p)
        for b_i in range(chunk_len):
            R[i][b_i] = _gf_mul(R[i][b_i], inv_p)
        for r in range(L):
            if r != i and S[r][i] != 0:
                f = S[r][i]
                for c in range(i, L):
                    S[r][c] ^= _gf_mul(f, S[i][c])
                for b_i in range(chunk_len):
                    R[r][b_i] ^= _gf_mul(f, R[i][b_i])

    result = dict(collected_chunks)
    for m_i, m in enumerate(missing):
        result[m] = bytes(R[m_i])
    print(f"\n[FEC Recovery] Reconstructed {len(missing)} dropped chunks ({missing}) from parity frames!")
    return result

def reassemble_stream(collected_chunks, total_chunks, output_path=None, start_time=None):
    """
    Reassembles full stream from collected chunks, verifies SHA-256,
    and writes original file to disk.
    Returns: (success: bool, info: dict)
    """
    missing = [i for i in range(total_chunks) if i not in collected_chunks]
    if missing:
        recovered = recover_fec_chunks(collected_chunks, total_chunks)
        if recovered is not None:
            collected_chunks = recovered
        else:
            return False, {"error": f"Missing {len(missing)} chunks and insufficient parity chunks."}

    full_stream = b"".join(collected_chunks[i] for i in range(total_chunks))

    if len(full_stream) < 54:
        return False, {"error": "Stream corrupted, metadata header too short."}

    meta_magic, orig_size, comp_size, orig_sha, fname_len = struct.unpack(">4sQQ32sH", full_stream[:54])

    if meta_magic != b"QSMD":
        return False, {"error": "Invalid stream metadata magic."}

    embedded_fname = full_stream[54:54 + fname_len].decode("utf-8", errors="replace")
    compressed_bytes = full_stream[54 + fname_len:54 + fname_len + comp_size]

    try:
        raw_data = zlib.decompress(compressed_bytes)
    except Exception as e:
        return False, {"error": f"Error during zlib decompression: {e}"}

    computed_sha = hashlib.sha256(raw_data).digest()
    if computed_sha != orig_sha:
        return False, {"error": "CRITICAL ERROR: SHA-256 mismatch! Data corrupted."}

    final_output = output_path if output_path else embedded_fname
    with open(final_output, "wb") as f:
        f.write(raw_data)

    elapsed = (time.time() - start_time) if start_time else 0.0
    eff_speed = (orig_size / 1024.0) / elapsed if elapsed > 0 else 0.0
    raw_speed = (len(full_stream) / 1024.0) / elapsed if elapsed > 0 else 0.0

    return True, {
        "filename": embedded_fname,
        "saved_path": os.path.abspath(final_output),
        "orig_size": orig_size,
        "comp_size": comp_size,
        "elapsed_sec": round(elapsed, 2),
        "eff_speed_kbps": round(eff_speed, 1),
        "channel_speed_kbps": round(raw_speed, 1),
        "sha256": computed_sha.hex()
    }

def print_transfer_summary(info):
    print("=" * 65)
    print("   TRANSFER COMPLETE - VERIFICATION SUCCESSFUL")
    print("=" * 65)
    print(f"File Saved:       {info['saved_path']}")
    print(f"Original Size:    {info['orig_size']:,} bytes ({info['orig_size'] / 1024 / 1024:.2f} MB)")
    print(f"Compressed Size:  {info['comp_size']:,} bytes ({info['comp_size'] / 1024 / 1024:.2f} MB)")
    print(f"Elapsed Time:     {info['elapsed_sec']:.2f} seconds")
    print(f"Effective Speed:  {info['eff_speed_kbps']:.1f} KB/s (uncompressed throughput)")
    print(f"Channel Speed:    {info['channel_speed_kbps']:.1f} KB/s")
    print(f"SHA-256 Checksum: {info['sha256']} (VERIFIED MATCH)")
    print("=" * 65)

def decode_stream(frame_generator, output_path=None, show_preview=True, model_path=None, grid_size=None):
    tracker = FiducialTracker(model_path=model_path)
    collected_chunks = {}
    total_chunks = None
    locked_grid_size = grid_size
    locked_mode = "rgb"
    debug_saved = False

    start_time = time.time()
    frames_processed = 0
    valid_frames_decoded = 0

    print("=" * 65)
    print("   OTD (OPTICAL DATA TRANSFER) v2 RECEIVER")
    if model_path:
        print(f"   [ML Acceleration Active: {os.path.basename(model_path)}]")
    print("=" * 65)
    print("Waiting for visual transmission signal...")
    if show_preview:
        print("Press 'q' in preview window to abort capture.\n")

    for frame in frame_generator:
        frames_processed += 1
        res, fiducials, warped = tracker.decode_frame(frame, locked_grid_size, locked_mode)

        preview_frame = frame.copy() if show_preview else None

        if fiducials and show_preview:
            tl, tr, br, bl = fiducials
            cv2.circle(preview_frame, (int(tl[0]), int(tl[1])), 8, (0, 0, 255), -1)   # Red
            cv2.circle(preview_frame, (int(tr[0]), int(tr[1])), 8, (0, 255, 0), -1)   # Green
            cv2.circle(preview_frame, (int(bl[0]), int(bl[1])), 8, (255, 0, 0), -1)   # Blue
            cv2.circle(preview_frame, (int(br[0]), int(br[1])), 8, (255, 0, 255), -1) # Magenta
            cv2.line(preview_frame, (int(tl[0]), int(tl[1])), (int(tr[0]), int(tr[1])), (0, 255, 255), 2)
            cv2.line(preview_frame, (int(tr[0]), int(tr[1])), (int(br[0]), int(br[1])), (0, 255, 255), 2)
            cv2.line(preview_frame, (int(br[0]), int(br[1])), (int(bl[0]), int(bl[1])), (0, 255, 255), 2)
            cv2.line(preview_frame, (int(bl[0]), int(bl[1])), (int(tl[0]), int(tl[1])), (0, 255, 255), 2)

        if res:
            idx, total, payload, gs, mode_str = res
            total_chunks = total
            locked_grid_size = gs
            locked_mode = mode_str
            valid_frames_decoded += 1

            if not debug_saved and warped is not None:
                cv2.imwrite("debug_sample_v2.png", warped)
                print(f"[Debug] Signal locked! Grid: {gs}x{gs} ({mode_str.upper()}). Wrote dewarped sample to 'debug_sample_v2.png'")
                debug_saved = True

            if idx not in collected_chunks:
                collected_chunks[idx] = payload
                is_parity = idx >= total_chunks
                tag = f"Parity {idx - total_chunks + 1}" if is_parity else f"Chunk {idx + 1:3d}/{total_chunks:3d}"
                pct = min(100.0, (len(collected_chunks) / total_chunks) * 100.0)
                elapsed = time.time() - start_time
                speed = (len(collected_chunks) * len(payload) / 1024.0) / elapsed if elapsed > 0 else 0
                print(f"[Frame {frames_processed:4d}] Captured {tag} ({pct:5.1f}%) | "
                      f"Speed: {speed:5.1f} KB/s | {len(collected_chunks)}/{total_chunks} frames")

            if total_chunks and len(collected_chunks) >= total_chunks:
                missing = [i for i in range(total_chunks) if i not in collected_chunks]
                if not missing or recover_fec_chunks(collected_chunks, total_chunks) is not None:
                    print("\nAll frames successfully captured (or FEC reconstructed) with 100% integrity!")
                    break

        if show_preview and preview_frame is not None:
            status_str = f"Captured: {len(collected_chunks)}/{total_chunks if total_chunks else '?'}"
            cv2.putText(preview_frame, status_str, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("otd v2 Receiver", cv2.resize(preview_frame, (800, 600)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nCapture aborted by user.")
                break

    if show_preview:
        cv2.destroyAllWindows()

    if not total_chunks or len(collected_chunks) < total_chunks:
        missing = set(range(total_chunks)) - set(collected_chunks.keys()) if total_chunks else "All"
        print(f"\nTransfer incomplete. Collected {len(collected_chunks)} of {total_chunks if total_chunks else '?'} chunks.")
        print(f"Missing chunk indices: {missing}")
        return False

    print("\nReassembling and verifying file...")
    success, info = reassemble_stream(collected_chunks, total_chunks, output_path, start_time)
    if not success:
        print(f"Error: {info.get('error', 'Reassembly failed')}")
        return False

    print_transfer_summary(info)
    return True

def video_file_generator(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file '{video_path}'")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()

def live_screen_generator(monitor_index=1):
    import mss
    with mss.MSS() as sct:
        if monitor_index >= len(sct.monitors):
            print(f"Error: Monitor index {monitor_index} out of range (available: 1..{len(sct.monitors)-1})")
            return
        mon = sct.monitors[monitor_index]
        warned_wayland = False

        while True:
            sct_img = sct.grab(mon)
            frame = np.array(sct_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Check if Wayland returned all-black frames
            if not warned_wayland and frame.mean() < 0.05:
                session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
                if "wayland" in session_type:
                    print("\n" + "=" * 65)
                    print("⚠️  WARNING: Screen capture returned an all-black image!")
                    print("=" * 65)
                    print("You are running on a Wayland desktop session (e.g. GNOME on Wayland).")
                    print("Wayland security isolates windows and blocks background X11 capture (mss).\n")
                    print("Recommended solutions for Wayland:")
                    print("1. Use Web Capture mode (supports Wayland window/screen sharing):")
                    print("     python3 decoder_v2.py --http\n")
                    print("2. Record a video of the screen (e.g. via GNOME screen recorder")
                    print("   using PrtScn / Ctrl+Alt+Shift+R or OBS), then run:")
                    print("     python3 decoder_v2.py <recording.mp4>\n")
                    print("3. Point an external webcam / camera at the screen:")
                    print("     python3 decoder_v2.py --camera 0\n")
                    print("4. Log in with 'GNOME on Xorg' at your login screen to enable")
                    print("   native real-time desktop screen capture.")
                    print("=" * 65 + "\n")
                    warned_wayland = True

            yield frame

def camera_stream_generator(device_index=0):
    print(f"Opening camera device {device_index}...")
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        print(f"Error: Unable to open camera device {device_index}")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()

# =====================================================================
# WEB CAPTURE RECEIVER (--http)
# =====================================================================

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>otd v2 | Web Optical Receiver</title>
<style>
  :root {
    --bg-base: #0a0e17;
    --bg-card: #121a2a;
    --bg-card-sub: #1a2438;
    --border: #233149;
    --border-light: #2d3f5d;
    --text-main: #f1f5f9;
    --text-muted: #94a3b8;
    --text-dim: #64748b;
    --primary: #10b981;
    --primary-glow: rgba(16, 185, 129, 0.25);
    --accent-cyan: #06b6d4;
    --accent-blue: #3b82f6;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background-color: var(--bg-base);
    color: var(--text-main);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    background: var(--bg-card);
    border-bottom: 1px solid var(--border);
    padding: 14px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .brand-logo {
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, var(--primary), var(--accent-cyan));
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    color: #051410;
    font-size: 16px;
  }
  .brand-title {
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .badge-ver {
    font-size: 11px;
    background: var(--accent-cyan);
    color: #032128;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 700;
  }
  .badge-tag {
    font-size: 11px;
    background: rgba(16, 185, 129, 0.15);
    color: var(--primary);
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 2px 8px;
    border-radius: 9999px;
    font-weight: 600;
  }
  .header-actions {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .conn-status {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    color: var(--text-muted);
  }
  .dot-online {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--primary);
    box-shadow: 0 0 8px var(--primary);
  }
  .btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 9px 16px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    border: none;
    transition: all 0.15s ease;
    text-decoration: none;
  }
  .btn-primary {
    background: var(--primary);
    color: #042116;
    box-shadow: 0 2px 8px var(--primary-glow);
  }
  .btn-primary:hover {
    background: #0ea372;
  }
  .btn-danger {
    background: var(--accent-rose);
    color: #ffffff;
  }
  .btn-danger:hover {
    background: #e11d48;
  }
  .btn-secondary {
    background: var(--bg-card-sub);
    color: var(--text-main);
    border: 1px solid var(--border-light);
  }
  .btn-secondary:hover {
    background: #25334e;
  }
  .btn-sm {
    padding: 5px 10px;
    font-size: 12px;
    border-radius: 6px;
  }
  .btn-outline {
    background: transparent;
    border: 1px solid var(--border-light);
    color: var(--text-main);
  }
  .btn-outline:hover {
    background: rgba(6, 182, 212, 0.15);
    border-color: var(--accent-cyan);
  }
  .btn-outline.active {
    background: rgba(6, 182, 212, 0.25);
    border-color: var(--accent-cyan);
    color: var(--accent-cyan);
  }
  .main-container {
    max-width: 1400px;
    width: 100%;
    margin: 0 auto;
    padding: 24px;
    display: grid;
    grid-template-columns: 1.15fr 0.85fr;
    gap: 24px;
    flex: 1;
  }
  @media (max-width: 1024px) {
    .main-container {
      grid-template-columns: 1fr;
    }
  }
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
  .card-header {
    background: var(--bg-card-header);
    border-bottom: 1px solid var(--border);
    padding: 12px 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .card-title {
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .card-body {
    padding: 18px;
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .mask-toolbar {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    background: var(--bg-card-sub);
    border: 1px solid var(--border-light);
    border-radius: 8px;
    padding: 8px 12px;
  }
  .auto-snap-label {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--text-muted);
    cursor: pointer;
    user-select: none;
    margin-left: 4px;
  }
  .mask-status-badge {
    font-size: 11px;
    font-family: ui-monospace, monospace;
    color: var(--accent-cyan);
    margin-left: auto;
    background: rgba(6, 182, 212, 0.1);
    border: 1px solid rgba(6, 182, 212, 0.25);
    padding: 3px 8px;
    border-radius: 4px;
  }
  .video-stage {
    position: relative;
    width: 100%;
    aspect-ratio: 16 / 10;
    background: #000000;
    border-radius: 8px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid var(--border);
    user-select: none;
  }
  .video-stage.draw-mode {
    cursor: crosshair;
  }
  #videoEl {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: none;
  }
  #overlayCanvas {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: none;
    pointer-events: auto;
  }
  .empty-stage {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    color: var(--text-muted);
    text-align: center;
    padding: 24px;
  }
  .empty-icon {
    width: 54px;
    height: 54px;
    color: var(--text-dim);
  }
  .tip-box {
    background: rgba(6, 182, 212, 0.08);
    border: 1px solid rgba(6, 182, 212, 0.25);
    border-radius: 8px;
    padding: 12px 14px;
    font-size: 13px;
    line-height: 1.5;
    color: #bae6fd;
  }
  .status-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 5px 12px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.03em;
  }
  .status-waiting {
    background: rgba(148, 163, 184, 0.15);
    color: var(--text-muted);
    border: 1px solid var(--border);
  }
  .status-searching {
    background: rgba(245, 158, 11, 0.15);
    color: var(--accent-amber);
    border: 1px solid rgba(245, 158, 11, 0.3);
    animation: pulse 1.5s infinite;
  }
  .status-locked {
    background: rgba(6, 182, 212, 0.15);
    color: var(--accent-cyan);
    border: 1px solid rgba(6, 182, 212, 0.3);
  }
  .status-receiving {
    background: rgba(16, 185, 129, 0.15);
    color: var(--primary);
    border: 1px solid rgba(16, 185, 129, 0.3);
    animation: pulse 1.2s infinite;
  }
  .status-complete {
    background: rgba(16, 185, 129, 0.25);
    color: var(--primary);
    border: 1px solid var(--primary);
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
  }
  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
  }
  .metric-card {
    background: var(--bg-card-sub);
    border: 1px solid var(--border-light);
    border-radius: 8px;
    padding: 12px 14px;
  }
  .metric-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 4px;
  }
  .metric-val {
    font-size: 19px;
    font-weight: 700;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    color: var(--text-main);
  }
  .progress-wrap {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .progress-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
  }
  .progress-pct {
    font-size: 26px;
    font-weight: 800;
    font-family: ui-monospace, monospace;
    color: var(--primary);
  }
  .progress-bar-bg {
    width: 100%;
    height: 12px;
    background: var(--bg-card-sub);
    border: 1px solid var(--border-light);
    border-radius: 9999px;
    overflow: hidden;
  }
  .progress-bar-fill {
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, var(--accent-cyan), var(--primary));
    box-shadow: 0 0 10px var(--primary-glow);
    transition: width 0.15s ease;
    border-radius: 9999px;
  }
  .chunk-matrix {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(12px, 1fr));
    gap: 3px;
    max-height: 120px;
    overflow-y: auto;
    background: #0d131f;
    padding: 8px;
    border-radius: 6px;
    border: 1px solid var(--border);
  }
  .chunk-tile {
    aspect-ratio: 1;
    background: #1e293b;
    border-radius: 2px;
    transition: background 0.2s;
  }
  .chunk-tile.active {
    background: var(--primary);
    box-shadow: 0 0 4px var(--primary);
  }
  .log-box {
    background: #080c14;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 12px;
    font-family: ui-monospace, monospace;
    font-size: 12px;
    color: var(--text-muted);
    max-height: 110px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .log-entry {
    line-height: 1.4;
  }
  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(5, 8, 15, 0.85);
    backdrop-filter: blur(6px);
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 999;
    padding: 20px;
  }
  .modal-overlay.show {
    display: flex;
  }
  .modal-card {
    background: var(--bg-card);
    border: 1px solid var(--primary);
    box-shadow: 0 10px 40px rgba(16, 185, 129, 0.2);
    border-radius: 14px;
    max-width: 580px;
    width: 100%;
    padding: 28px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  .modal-hero {
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .modal-icon {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    background: rgba(16, 185, 129, 0.2);
    color: var(--primary);
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .modal-title {
    font-size: 20px;
    font-weight: 700;
  }
  .modal-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }
  .modal-table td {
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
  }
  .modal-table td:first-child {
    color: var(--text-muted);
    width: 130px;
  }
  .modal-table td:last-child {
    font-family: ui-monospace, monospace;
    color: var(--text-main);
    word-break: break-all;
  }
  .modal-actions {
    display: flex;
    gap: 12px;
    justify-content: flex-end;
  }
</style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="brand-logo">OTD</div>
      <div>
        <div class="brand-title">
          otd <span class="badge-ver">v2</span>
          <span class="badge-tag">Optical Data Transfer</span>
        </div>
      </div>
    </div>
    <div class="header-actions">
      <div class="conn-status">
        <div class="dot-online"></div>
        <span>Connected</span>
      </div>
      <button id="btnShare" class="btn btn-primary" onclick="toggleCapture()">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
        <span id="btnShareText">Share Window / Screen</span>
      </button>
      <button class="btn btn-secondary" onclick="resetCapture()">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M3 21v-5h5"/></svg>
        Reset
      </button>
    </div>
  </header>

  <div class="main-container">
    <!-- Left Column: Video & Alignment -->
    <div class="card">
      <div class="card-header">
        <div class="card-title">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>
          Live Optical View
        </div>
        <div id="statusBadge" class="status-badge status-waiting">Waiting for Screen Share</div>
      </div>
      <div class="card-body">
        <!-- Mask / ROI Toolbar -->
        <div class="mask-toolbar">
          <button id="btnDrawMask" class="btn btn-sm btn-outline" onclick="toggleDrawMaskMode()" title="Click and drag on video to draw transmission boundary">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" stroke-dasharray="4 2"/><circle cx="12" cy="12" r="3"/></svg>
            <span id="btnDrawMaskText">Draw Mask</span>
          </button>
          <button id="btnAutoMask" class="btn btn-sm btn-outline" onclick="autoSnapMask()" title="Automatically snap crop mask around detected fiducials">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4m0 12v4M2 12h4m12 0h4"/><circle cx="12" cy="12" r="7"/></svg>
            Auto-Snap
          </button>
          <label class="auto-snap-label" title="Automatically lock mask to signal when detected">
            <input type="checkbox" id="chkAutoSnap" checked>
            <span>Auto-Lock</span>
          </label>
          <button id="btnClearMask" class="btn btn-sm btn-outline" style="display:none;" onclick="clearMask()" title="Reset to capturing full screen/window">
            ✖ Clear Mask
          </button>
          <div id="maskStatus" class="mask-status-badge">Full Feed (No Mask)</div>
        </div>

        <div id="videoStage" class="video-stage">
          <video id="videoEl" autoplay playsinline muted></video>
          <canvas id="overlayCanvas"></canvas>
          <div id="emptyStage" class="empty-stage">
            <svg class="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
            <div style="font-size:16px; font-weight:600; color:var(--text-main);">No Video Feed Selected</div>
            <div style="font-size:13px; max-width:380px;">Click "Share Window / Screen" above. In the browser dialog, select the otd transmitter window.</div>
          </div>
        </div>

        <div class="tip-box">
          <strong>Optical Masking Tip:</strong> If sharing your entire desktop, use <strong>"Draw Mask"</strong> or click <strong>"Auto-Snap"</strong> to isolate the transmitter window. The web app will only capture and encode the masked region, dramatically reducing network bandwidth and CPU usage.
        </div>
      </div>
    </div>

    <!-- Right Column: Telemetry & Progress -->
    <div class="card">
      <div class="card-header">
        <div class="card-title">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
          Reception Telemetry
        </div>
        <div id="streamResolution" style="font-size:12px; color:var(--text-muted); font-family:monospace;">0x0 @ 0 FPS</div>
      </div>
      <div class="card-body">
        <div class="progress-wrap">
          <div class="progress-header">
            <div style="font-size:12px; font-weight:600; text-transform:uppercase; color:var(--text-muted);">Transfer Progress</div>
            <div id="pctLabel" class="progress-pct">0%</div>
          </div>
          <div class="progress-bar-bg">
            <div id="progressBar" class="progress-bar-fill"></div>
          </div>
        </div>

        <div class="metrics-grid">
          <div class="metric-card">
            <div class="metric-label">Chunks Captured</div>
            <div id="chunksVal" class="metric-val">0 / 0</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Transfer Speed</div>
            <div id="speedVal" class="metric-val">0.0 KB/s</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Signal Lock</div>
            <div id="signalVal" class="metric-val">--</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Elapsed Time</div>
            <div id="timeVal" class="metric-val">0.0s</div>
          </div>
          <div class="metric-card" style="grid-column: 1 / -1;">
            <div class="metric-label">Active Transmission Mask</div>
            <div id="maskMetricVal" class="metric-val" style="font-size:15px; color:var(--accent-cyan);">Full Screen (Unmasked)</div>
          </div>
        </div>

        <div>
          <div style="font-size:11px; font-weight:600; text-transform:uppercase; color:var(--text-muted); margin-bottom:6px;">Chunk Block Map</div>
          <div id="chunkMatrix" class="chunk-matrix">
            <div style="color:var(--text-dim); font-size:12px; grid-column:1/-1; text-align:center; padding:10px;">Awaiting signal lock...</div>
          </div>
        </div>

        <div>
          <div style="font-size:11px; font-weight:600; text-transform:uppercase; color:var(--text-muted); margin-bottom:6px;">Activity Log</div>
          <div id="logBox" class="log-box">
            <div class="log-entry">[Ready] Web capture receiver initialized.</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Complete Modal -->
  <div id="completeModal" class="modal-overlay">
    <div class="modal-card">
      <div class="modal-hero">
        <div class="modal-icon">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
        </div>
        <div>
          <div class="modal-title">Transfer Complete & Verified!</div>
          <div style="font-size:13px; color:var(--primary);">100% CRC integrity & cryptographic SHA-256 matched</div>
        </div>
      </div>

      <table class="modal-table">
        <tr><td>Filename</td><td id="resFilename">--</td></tr>
        <tr><td>Original Size</td><td id="resOrigSize">--</td></tr>
        <tr><td>Compressed</td><td id="resCompSize">--</td></tr>
        <tr><td>Elapsed Time</td><td id="resElapsed">--</td></tr>
        <tr><td>Throughput</td><td id="resSpeed">--</td></tr>
        <tr><td>Saved On Host</td><td id="resSavedPath" style="font-size:11px; color:#38bdf8;">--</td></tr>
        <tr><td>SHA-256</td><td id="resSha" style="font-size:11px;">--</td></tr>
      </table>

      <div class="modal-actions">
        <a id="btnDownload" href="/api/download" class="btn btn-primary" download>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
          Download in Browser
        </a>
        <button class="btn btn-secondary" onclick="resetCapture(); hideModal();">
          Receive Another File
        </button>
      </div>
    </div>
  </div>

  <canvas id="offscreenCanvas" style="display:none;"></canvas>

  <script>
    let captureStream = null;
    let isCapturing = false;
    let isSending = false;
    let knownTotalChunks = null;
    let chimePlayed = false;

    // Transmission area masking / ROI state
    let cropRegion = null; // { x, y, w, h } in video pixel coordinates
    let isDrawingMaskMode = false;
    let isDragging = false;
    let dragStart = null;
    let dragCurrent = null;
    let lastDetectedFiducials = null;
    let autoSnapTriggered = false;

    const videoEl = document.getElementById('videoEl');
    const videoStage = document.getElementById('videoStage');
    const overlayCanvas = document.getElementById('overlayCanvas');
    const overlayCtx = overlayCanvas.getContext('2d');
    const offscreenCanvas = document.getElementById('offscreenCanvas');
    const offscreenCtx = offscreenCanvas.getContext('2d');

    const emptyStage = document.getElementById('emptyStage');
    const statusBadge = document.getElementById('statusBadge');
    const btnShareText = document.getElementById('btnShareText');
    const pctLabel = document.getElementById('pctLabel');
    const progressBar = document.getElementById('progressBar');
    const chunksVal = document.getElementById('chunksVal');
    const speedVal = document.getElementById('speedVal');
    const signalVal = document.getElementById('signalVal');
    const timeVal = document.getElementById('timeVal');
    const chunkMatrix = document.getElementById('chunkMatrix');
    const logBox = document.getElementById('logBox');
    const resRes = document.getElementById('streamResolution');

    const btnDrawMask = document.getElementById('btnDrawMask');
    const btnDrawMaskText = document.getElementById('btnDrawMaskText');
    const btnClearMask = document.getElementById('btnClearMask');
    const maskStatus = document.getElementById('maskStatus');
    const maskMetricVal = document.getElementById('maskMetricVal');
    const chkAutoSnap = document.getElementById('chkAutoSnap');

    function logMsg(msg) {
      const now = new Date().toTimeString().split(' ')[0];
      const div = document.createElement('div');
      div.className = 'log-entry';
      div.textContent = `[${now}] ${msg}`;
      logBox.appendChild(div);
      logBox.scrollTop = logBox.scrollHeight;
    }

    // Geometry transform helper: maps video container to letterboxed video element
    function getVideoDisplayRect() {
      const stageW = videoStage.clientWidth;
      const stageH = videoStage.clientHeight;
      const vw = videoEl.videoWidth;
      const vh = videoEl.videoHeight;
      if (!vw || !vh) return null;

      const stageAspect = stageW / stageH;
      const videoAspect = vw / vh;

      let dw, dh, dx, dy;
      if (stageAspect > videoAspect) {
        // Pillarbox (bars left & right)
        dh = stageH;
        dw = stageH * videoAspect;
        dx = (stageW - dw) / 2;
        dy = 0;
      } else {
        // Letterbox (bars top & bottom)
        dw = stageW;
        dh = stageW / videoAspect;
        dx = 0;
        dy = (stageH - dh) / 2;
      }
      return { dx, dy, dw, dh, videoW: vw, videoH: vh, stageW, stageH };
    }

    function getPointerVideoCoords(e) {
      const rect = getVideoDisplayRect();
      if (!rect) return null;
      const canvasRect = overlayCanvas.getBoundingClientRect();
      const pointerX = e.clientX - canvasRect.left;
      const pointerY = e.clientY - canvasRect.top;

      // Clamped inside the actual displayed video area
      const clampedX = Math.max(rect.dx, Math.min(rect.dx + rect.dw, pointerX));
      const clampedY = Math.max(rect.dy, Math.min(rect.dy + rect.dh, pointerY));

      // Convert from display pixels to video native pixels
      const vx = ((clampedX - rect.dx) / rect.dw) * rect.videoW;
      const vy = ((clampedY - rect.dy) / rect.dh) * rect.videoH;
      return { x: vx, y: vy };
    }

    // Pointer event listeners for interactive drag-to-mask
    overlayCanvas.addEventListener('pointerdown', (e) => {
      if (!isCapturing || videoEl.videoWidth === 0) return;
      const pt = getPointerVideoCoords(e);
      if (!pt) return;

      isDragging = true;
      dragStart = pt;
      dragCurrent = pt;
      overlayCanvas.setPointerCapture(e.pointerId);
    });

    window.addEventListener('pointermove', (e) => {
      if (!isDragging) return;
      const pt = getPointerVideoCoords(e);
      if (!pt) return;
      dragCurrent = pt;
      redrawOverlay();
    });

    window.addEventListener('pointerup', (e) => {
      if (!isDragging) return;
      isDragging = false;
      const pt = getPointerVideoCoords(e) || dragCurrent;

      const x1 = Math.min(dragStart.x, pt.x);
      const y1 = Math.min(dragStart.y, pt.y);
      const x2 = Math.max(dragStart.x, pt.x);
      const y2 = Math.max(dragStart.y, pt.y);
      const w = x2 - x1;
      const h = y2 - y1;

      // Require at least 40x40 pixel box to avoid accidental clicks
      if (w > 40 && h > 40) {
        cropRegion = { x: Math.round(x1), y: Math.round(y1), w: Math.round(w), h: Math.round(h) };
        logMsg(`Transmission mask applied: ${cropRegion.w}×${cropRegion.h} px`);
        updateMaskUI();
      }

      if (isDrawingMaskMode) {
        toggleDrawMaskMode();
      }
      redrawOverlay();
    });

    function toggleDrawMaskMode() {
      isDrawingMaskMode = !isDrawingMaskMode;
      if (isDrawingMaskMode) {
        btnDrawMask.classList.add('active');
        btnDrawMaskText.textContent = 'Drawing...';
        videoStage.classList.add('draw-mode');
        logMsg('Drag a rectangle on the video feed to set the transmission mask.');
      } else {
        btnDrawMask.classList.remove('active');
        btnDrawMaskText.textContent = 'Draw Mask';
        videoStage.classList.remove('draw-mode');
      }
    }

    function autoSnapMask() {
      if (!lastDetectedFiducials || lastDetectedFiducials.length !== 4) {
        logMsg('Cannot auto-snap: waiting for optical signal to be detected first.');
        return;
      }
      const xs = lastDetectedFiducials.map(p => p[0]);
      const ys = lastDetectedFiducials.map(p => p[1]);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);

      const w = maxX - minX;
      const h = maxY - minY;
      const padX = w * 0.15;
      const padY = h * 0.15;

      const cx = Math.max(0, Math.round(minX - padX));
      const cy = Math.max(0, Math.round(minY - padY));
      const cw = Math.min(videoEl.videoWidth - cx, Math.round(w + padX * 2));
      const ch = Math.min(videoEl.videoHeight - cy, Math.round(h + padY * 2));

      cropRegion = { x: cx, y: cy, w: cw, h: ch };
      autoSnapTriggered = true;
      updateMaskUI();
      logMsg(`Auto-snapped mask to transmission signal: ${cw}×${ch} px`);
      redrawOverlay();
    }

    function clearMask() {
      cropRegion = null;
      autoSnapTriggered = false;
      updateMaskUI();
      logMsg('Transmission mask cleared. Capturing full video feed.');
      redrawOverlay();
    }

    function updateMaskUI() {
      if (cropRegion && videoEl.videoWidth > 0) {
        btnClearMask.style.display = 'inline-flex';
        const totalPixels = videoEl.videoWidth * videoEl.videoHeight;
        const maskPixels = cropRegion.w * cropRegion.h;
        const savedPct = Math.max(0, Math.round((1 - (maskPixels / totalPixels)) * 100));

        maskStatus.textContent = `Mask: ${cropRegion.w}×${cropRegion.h} (${savedPct}% saved)`;
        maskStatus.style.color = 'var(--primary)';
        maskStatus.style.borderColor = 'rgba(16, 185, 129, 0.4)';
        maskStatus.style.background = 'rgba(16, 185, 129, 0.12)';

        maskMetricVal.textContent = `${cropRegion.w}×${cropRegion.h} px (${savedPct}% data saved)`;
        maskMetricVal.style.color = 'var(--primary)';
      } else {
        btnClearMask.style.display = 'none';
        maskStatus.textContent = 'Full Feed (No Mask)';
        maskStatus.style.color = 'var(--accent-cyan)';
        maskStatus.style.borderColor = 'rgba(6, 182, 212, 0.25)';
        maskStatus.style.background = 'rgba(6, 182, 212, 0.1)';

        maskMetricVal.textContent = 'Full Screen (Unmasked)';
        maskMetricVal.style.color = 'var(--text-muted)';
      }
    }

    async function toggleCapture() {
      if (isCapturing) {
        stopCapture();
      } else {
        await startCapture();
      }
    }

    async function startCapture() {
      try {
        captureStream = await navigator.mediaDevices.getDisplayMedia({
          video: {
            displaySurface: "window",
            width: { ideal: 1920, max: 3840 },
            height: { ideal: 1080, max: 2160 },
            frameRate: { ideal: 30, max: 60 }
          },
          audio: false
        });

        videoEl.srcObject = captureStream;
        videoEl.style.display = 'block';
        overlayCanvas.style.display = 'block';
        emptyStage.style.display = 'none';

        await videoEl.play();
        isCapturing = true;
        autoSnapTriggered = false;
        btnShareText.textContent = 'Stop Sharing';
        document.getElementById('btnShare').classList.replace('btn-primary', 'btn-danger');

        captureStream.getVideoTracks()[0].onended = () => {
          stopCapture();
        };

        logMsg('Screen capture started. Looking for optical transmission...');
        setStatus('searching', 'Searching for Signal');
        updateMaskUI();
        runCaptureLoop();
      } catch (err) {
        if (err.name !== 'AbortError' && err.name !== 'NotAllowedError') {
          alert('Screen capture failed: ' + err.message);
        }
      }
    }

    function stopCapture() {
      isCapturing = false;
      if (ws) {
        try { ws.close(); } catch (e) {}
        ws = null;
      }
      if (captureStream) {
        captureStream.getTracks().forEach(t => t.stop());
        captureStream = null;
      }
      videoEl.style.display = 'none';
      overlayCanvas.style.display = 'none';
      emptyStage.style.display = 'flex';
      btnShareText.textContent = 'Share Window / Screen';
      document.getElementById('btnShare').classList.replace('btn-danger', 'btn-primary');
      setStatus('waiting', 'Waiting for Screen Share');
      logMsg('Capture stopped.');
    }

    function setStatus(type, text) {
      statusBadge.className = `status-badge status-${type}`;
      statusBadge.textContent = text;
    }

    let ws = null;
    let wsConnecting = false;

    function setupWebSocket() {
      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        return;
      }
      wsConnecting = true;
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      try {
        ws = new WebSocket(`${proto}//${location.host}/ws`);
        ws.binaryType = 'arraybuffer';
        ws.onopen = () => {
          wsConnecting = false;
          logMsg('Connected to high-speed WebSocket pipeline (45-60 FPS).');
        };
        ws.onmessage = (evt) => {
          try {
            const data = JSON.parse(evt.data);
            handleTelemetry(data);
          } catch (e) {}
          isSending = false;
        };
        ws.onerror = () => {
          wsConnecting = false;
        };
        ws.onclose = () => {
          wsConnecting = false;
          ws = null;
        };
      } catch (e) {
        wsConnecting = false;
        ws = null;
      }
    }

    async function runCaptureLoop() {
      setupWebSocket();
      while (isCapturing) {
        if (!isSending && videoEl.readyState >= 2 && videoEl.videoWidth > 0) {
          isSending = true;
          try {
            const vw = videoEl.videoWidth;
            const vh = videoEl.videoHeight;
            resRes.textContent = `${vw}x${vh}`;

            // Check if active crop mask is set
            if (cropRegion && cropRegion.w > 40 && cropRegion.h > 40) {
              // Ensure cropRegion stays within actual video bounds
              const cx = Math.max(0, Math.min(vw - 20, cropRegion.x));
              const cy = Math.max(0, Math.min(vh - 20, cropRegion.y));
              const cw = Math.max(20, Math.min(vw - cx, cropRegion.w));
              const ch = Math.max(20, Math.min(vh - cy, cropRegion.h));

              offscreenCanvas.width = cw;
              offscreenCanvas.height = ch;
              offscreenCtx.drawImage(videoEl, cx, cy, cw, ch, 0, 0, cw, ch);
            } else {
              // Full video (cap max dimension to 1920 to keep compression low-latency)
              const maxDim = Math.max(vw, vh);
              const scale = maxDim > 1920 ? (1920 / maxDim) : 1;
              offscreenCanvas.width = Math.round(vw * scale);
              offscreenCanvas.height = Math.round(vh * scale);
              offscreenCtx.drawImage(videoEl, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
            }

            const blob = await new Promise(r => offscreenCanvas.toBlob(r, 'image/jpeg', 0.94));
            if (blob && isCapturing) {
              if (ws && ws.readyState === WebSocket.OPEN) {
                const buf = await blob.arrayBuffer();
                ws.send(buf);
              } else {
                const res = await fetch('/api/frame', {
                  method: 'POST',
                  body: blob,
                  headers: { 'Content-Type': 'image/jpeg' }
                });
                if (res.ok) {
                  const data = await res.json();
                  handleTelemetry(data);
                }
                isSending = false;
              }
            } else {
              isSending = false;
            }
          } catch (e) {
            console.warn('Frame processing error:', e);
            isSending = false;
          }
        }
        await new Promise(r => setTimeout(r, 12));
      }
    }

    function handleTelemetry(data) {
      // Map fiducials from cropped coordinates back to full video coordinates
      let adjustedFiducials = null;
      if (data.fiducials && data.fiducials.length === 4) {
        if (cropRegion && cropRegion.w > 40) {
          adjustedFiducials = data.fiducials.map(pt => [
            pt[0] + cropRegion.x,
            pt[1] + cropRegion.y
          ]);
        } else {
          const scaleX = videoEl.videoWidth / offscreenCanvas.width;
          const scaleY = videoEl.videoHeight / offscreenCanvas.height;
          adjustedFiducials = data.fiducials.map(pt => [
            pt[0] * scaleX,
            pt[1] * scaleY
          ]);
        }
        lastDetectedFiducials = adjustedFiducials;

        // Auto-lock mask if enabled and not already snapped
        if (chkAutoSnap.checked && !cropRegion && !autoSnapTriggered) {
          if (data.auto_roi && data.auto_roi.length === 4) {
            const scaleX = videoEl.videoWidth / offscreenCanvas.width;
            const scaleY = videoEl.videoHeight / offscreenCanvas.height;
            const [rx, ry, rw, rh] = data.auto_roi;
            const cx = Math.max(0, Math.round(rx * scaleX));
            const cy = Math.max(0, Math.round(ry * scaleY));
            const cw = Math.min(videoEl.videoWidth - cx, Math.round(rw * scaleX));
            const ch = Math.min(videoEl.videoHeight - cy, Math.round(rh * scaleY));
            if (cw > 40 && ch > 40) {
              cropRegion = { x: cx, y: cy, w: cw, h: ch };
              autoSnapTriggered = true;
              updateMaskUI();
              logMsg(`Auto-snapped transmission ROI: ${cw}×${ch} px`);
            }
          } else {
            autoSnapMask();
          }
        }
      }

      redrawOverlay(adjustedFiducials);

      if (data.grid_size && data.mode) {
        signalVal.textContent = `${data.grid_size}x${data.grid_size} ${data.mode.toUpperCase()}`;
      }

      if (data.total_chunks) {
        chunksVal.textContent = `${data.captured_chunks} / ${data.total_chunks}`;
        pctLabel.textContent = `${data.percent}%`;
        progressBar.style.width = `${data.percent}%`;

        if (knownTotalChunks !== data.total_chunks) {
          knownTotalChunks = data.total_chunks;
          buildChunkMatrix(data.total_chunks);
          logMsg(`Signal locked! Grid: ${data.grid_size}x${data.grid_size} (${data.mode.toUpperCase()}) | Total: ${data.total_chunks} chunks`);
        }
      }

      if (data.speed_kbps !== undefined) {
        speedVal.textContent = `${data.speed_kbps.toFixed(1)} KB/s`;
      }
      if (data.elapsed_sec !== undefined) {
        timeVal.textContent = `${data.elapsed_sec.toFixed(1)}s`;
      }

      if (data.collected_indices) {
        updateChunkMatrix(data.collected_indices);
      }

      if (data.state === 'receiving') {
        setStatus('receiving', `Receiving (${data.captured_chunks}/${data.total_chunks})`);
      } else if (data.state === 'locked') {
        setStatus('locked', 'Signal Locked');
      }

      if (data.is_complete && data.result) {
        setStatus('complete', 'Transfer Complete');
        showModal(data.result);
        if (!chimePlayed) {
          playSuccessChime();
          chimePlayed = true;
        }
      }
    }

    function redrawOverlay(fiducials = lastDetectedFiducials) {
      const rect = getVideoDisplayRect();
      if (!rect) return;

      // Sync overlay canvas internal dimensions to display container
      if (overlayCanvas.width !== rect.stageW || overlayCanvas.height !== rect.stageH) {
        overlayCanvas.width = rect.stageW;
        overlayCanvas.height = rect.stageH;
      }

      overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

      // 1. Draw Transmission Area Mask (dimming outside region)
      if (cropRegion && cropRegion.w > 20 && cropRegion.h > 20) {
        const dx = rect.dx + (cropRegion.x / rect.videoW) * rect.dw;
        const dy = rect.dy + (cropRegion.y / rect.videoH) * rect.dh;
        const dw = (cropRegion.w / rect.videoW) * rect.dw;
        const dh = (cropRegion.h / rect.videoH) * rect.dh;

        overlayCtx.save();
        // Darken outside mask using evenodd cutout
        overlayCtx.fillStyle = 'rgba(10, 14, 23, 0.70)';
        overlayCtx.beginPath();
        overlayCtx.rect(0, 0, overlayCanvas.width, overlayCanvas.height);
        overlayCtx.rect(dx, dy, dw, dh);
        overlayCtx.fill('evenodd');

        // Bounding outline
        overlayCtx.strokeStyle = '#06b6d4';
        overlayCtx.lineWidth = 2;
        overlayCtx.strokeRect(dx, dy, dw, dh);

        // High-visibility green corner reticle brackets
        const clen = Math.min(22, dw / 4, dh / 4);
        overlayCtx.strokeStyle = '#10b981';
        overlayCtx.lineWidth = 3;

        // TL
        overlayCtx.beginPath();
        overlayCtx.moveTo(dx, dy + clen);
        overlayCtx.lineTo(dx, dy);
        overlayCtx.lineTo(dx + clen, dy);
        overlayCtx.stroke();
        // TR
        overlayCtx.beginPath();
        overlayCtx.moveTo(dx + dw - clen, dy);
        overlayCtx.lineTo(dx + dw, dy);
        overlayCtx.lineTo(dx + dw, dy + clen);
        overlayCtx.stroke();
        // BR
        overlayCtx.beginPath();
        overlayCtx.moveTo(dx + dw, dy + dh - clen);
        overlayCtx.lineTo(dx + dw, dy + dh);
        overlayCtx.lineTo(dx + dw - clen, dy + dh);
        overlayCtx.stroke();
        // BL
        overlayCtx.beginPath();
        overlayCtx.moveTo(dx + clen, dy + dh);
        overlayCtx.lineTo(dx, dy + dh);
        overlayCtx.lineTo(dx, dy + dh - clen);
        overlayCtx.stroke();

        // Label above mask
        overlayCtx.fillStyle = '#06b6d4';
        overlayCtx.font = 'bold 11px monospace';
        const tagY = dy - 6 > 14 ? dy - 6 : dy + 16;
        overlayCtx.fillText(`MASKED TRANSMISSION AREA (${cropRegion.w}×${cropRegion.h})`, dx + 4, tagY);

        overlayCtx.restore();
      }

      // 2. Draw active dragging selection box
      if (isDragging && dragStart && dragCurrent) {
        const x1 = Math.min(dragStart.x, dragCurrent.x);
        const y1 = Math.min(dragStart.y, dragCurrent.y);
        const w = Math.abs(dragCurrent.x - dragStart.x);
        const h = Math.abs(dragCurrent.y - dragStart.y);

        const dx = rect.dx + (x1 / rect.videoW) * rect.dw;
        const dy = rect.dy + (y1 / rect.videoH) * rect.dh;
        const dw = (w / rect.videoW) * rect.dw;
        const dh = (h / rect.videoH) * rect.dh;

        overlayCtx.save();
        overlayCtx.strokeStyle = '#38bdf8';
        overlayCtx.lineWidth = 2;
        overlayCtx.setLineDash([6, 4]);
        overlayCtx.fillStyle = 'rgba(56, 189, 248, 0.15)';
        overlayCtx.fillRect(dx, dy, dw, dh);
        overlayCtx.strokeRect(dx, dy, dw, dh);

        overlayCtx.setLineDash([]);
        overlayCtx.fillStyle = '#38bdf8';
        overlayCtx.font = 'bold 11px monospace';
        overlayCtx.fillText(`${Math.round(w)} × ${Math.round(h)} px`, dx + 6, dy + 18);
        overlayCtx.restore();
      }

      // 3. Draw detected optical fiducials (mapped to display coordinates)
      if (fiducials && fiducials.length === 4) {
        const toDisplayPt = (pt) => [
          rect.dx + (pt[0] / rect.videoW) * rect.dw,
          rect.dy + (pt[1] / rect.videoH) * rect.dh
        ];

        const [tl, tr, br, bl] = fiducials.map(toDisplayPt);

        overlayCtx.save();
        overlayCtx.beginPath();
        overlayCtx.moveTo(tl[0], tl[1]);
        overlayCtx.lineTo(tr[0], tr[1]);
        overlayCtx.lineTo(br[0], br[1]);
        overlayCtx.lineTo(bl[0], bl[1]);
        overlayCtx.closePath();
        overlayCtx.lineWidth = 2.5;
        overlayCtx.strokeStyle = '#06b6d4';
        overlayCtx.stroke();
        overlayCtx.fillStyle = 'rgba(6, 182, 212, 0.08)';
        overlayCtx.fill();

        const corners = [
          { pt: tl, col: '#ef4444' }, // TL: Red
          { pt: tr, col: '#22c55e' }, // TR: Green
          { pt: bl, col: '#3b82f6' }, // BL: Blue
          { pt: br, col: '#d946ef' }  // BR: Magenta
        ];
        corners.forEach(c => {
          overlayCtx.beginPath();
          overlayCtx.arc(c.pt[0], c.pt[1], 6, 0, 2 * Math.PI);
          overlayCtx.fillStyle = c.col;
          overlayCtx.fill();
          overlayCtx.lineWidth = 2;
          overlayCtx.strokeStyle = '#ffffff';
          overlayCtx.stroke();
        });
        overlayCtx.restore();
      }
    }

    function buildChunkMatrix(total) {
      chunkMatrix.innerHTML = '';
      for (let i = 0; i < total; i++) {
        const div = document.createElement('div');
        div.className = 'chunk-tile';
        div.id = `chunk-${i}`;
        div.title = `Chunk ${i + 1}`;
        chunkMatrix.appendChild(div);
      }
    }

    function updateChunkMatrix(indices) {
      indices.forEach(idx => {
        const el = document.getElementById(`chunk-${idx}`);
        if (el && !el.classList.contains('active')) {
          el.classList.add('active');
        }
      });
    }

    function showModal(res) {
      document.getElementById('resFilename').textContent = res.filename;
      document.getElementById('resOrigSize').textContent = `${res.orig_size.toLocaleString()} bytes (${(res.orig_size/1024/1024).toFixed(2)} MB)`;
      document.getElementById('resCompSize').textContent = `${res.comp_size.toLocaleString()} bytes`;
      document.getElementById('resElapsed').textContent = `${res.elapsed_sec}s`;
      document.getElementById('resSpeed').textContent = `${res.eff_speed_kbps} KB/s`;
      document.getElementById('resSavedPath').textContent = res.saved_path;
      document.getElementById('resSha').textContent = res.sha256;
      document.getElementById('completeModal').classList.add('show');
    }

    function hideModal() {
      document.getElementById('completeModal').classList.remove('show');
    }

    async function resetCapture() {
      try {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ action: 'reset' }));
        } else {
          await fetch('/api/reset', { method: 'POST' });
        }
        knownTotalChunks = null;
        chimePlayed = false;
        autoSnapTriggered = false;
        lastDetectedFiducials = null;
        pctLabel.textContent = '0%';
        progressBar.style.width = '0%';
        chunksVal.textContent = '0 / 0';
        speedVal.textContent = '0.0 KB/s';
        signalVal.textContent = '--';
        timeVal.textContent = '0.0s';
        chunkMatrix.innerHTML = '<div style="color:var(--text-dim); font-size:12px; grid-column:1/-1; text-align:center; padding:10px;">Awaiting signal lock...</div>';
        redrawOverlay();
        logMsg('Transfer reset. Ready for next transmission.');
      } catch (e) {
        console.error('Reset error:', e);
      }
    }

    function playSuccessChime() {
      try {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtx) return;
        const actx = new AudioCtx();
        const freqs = [523.25, 659.25, 783.99, 1046.50];
        freqs.forEach((f, i) => {
          const osc = actx.createOscillator();
          const gain = actx.createGain();
          osc.type = 'sine';
          osc.frequency.setValueAtTime(f, actx.currentTime + i * 0.1);
          gain.gain.setValueAtTime(0.15, actx.currentTime + i * 0.1);
          gain.gain.exponentialRampToValueAtTime(0.001, actx.currentTime + i * 0.1 + 0.3);
          osc.connect(gain);
          gain.connect(actx.destination);
          osc.start(actx.currentTime + i * 0.1);
          osc.stop(actx.currentTime + i * 0.1 + 0.35);
        });
      } catch (e) {}
    }
  </script>
</body>
</html>
"""

class HttpDecoderSession:
    def __init__(self, output_path=None, model_path=None, grid_size=None):
        self.output_path = output_path
        self.model_path = model_path
        self.initial_grid_size = grid_size
        self.lock = threading.Lock()
        self.tracker = FiducialTracker(model_path=model_path)
        self.reset()

    def reset(self):
        with self.lock:
            self.collected_chunks = {}
            self.total_chunks = None
            self.locked_grid_size = self.initial_grid_size
            self.locked_mode = "rgb"
            self.debug_saved = False
            self.start_time = None
            self.frames_processed = 0
            self.valid_frames_decoded = 0
            self.is_complete = False
            self.result_info = None
            self.error = None
            self.last_fiducials = None
            self.latest_chunk_idx = None
            if hasattr(self, "tracker") and self.tracker is not None:
                self.tracker.reset()

    def process_image_bytes(self, img_bytes):
        with self.lock:
            if self.is_complete:
                return self.get_status_dict()

            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return self.get_status_dict(error="Invalid image format")

            if self.start_time is None:
                self.start_time = time.time()

            self.frames_processed += 1
            res, fiducials, warped = self.tracker.decode_frame(
                frame, self.locked_grid_size, self.locked_mode
            )

            if fiducials is not None:
                self.last_fiducials = [[float(pt[0]), float(pt[1])] for pt in fiducials]
            else:
                self.last_fiducials = None

            if res:
                idx, total, payload, gs, mode_str = res
                self.total_chunks = total
                self.locked_grid_size = gs
                self.locked_mode = mode_str
                self.valid_frames_decoded += 1
                self.latest_chunk_idx = idx

                if not self.debug_saved and warped is not None:
                    cv2.imwrite("debug_sample_v2.png", warped)
                    print(f"[Debug] Signal locked! Grid: {gs}x{gs} ({mode_str.upper()}). Wrote dewarped sample to 'debug_sample_v2.png'")
                    self.debug_saved = True

                if idx not in self.collected_chunks:
                    self.collected_chunks[idx] = payload
                    is_parity = idx >= total
                    tag = f"Parity {idx - total + 1}" if is_parity else f"Chunk {idx + 1:3d}/{total:3d}"
                    pct = min(100.0, (len(self.collected_chunks) / total) * 100.0)
                    elapsed = time.time() - self.start_time
                    speed = (len(self.collected_chunks) * len(payload) / 1024.0) / elapsed if elapsed > 0 else 0
                    print(f"[Frame {self.frames_processed:4d}] Captured {tag} ({pct:5.1f}%) | "
                          f"Speed: {speed:5.1f} KB/s | {len(self.collected_chunks)}/{total} frames")

                if self.total_chunks and len(self.collected_chunks) >= self.total_chunks:
                    missing = [i for i in range(self.total_chunks) if i not in self.collected_chunks]
                    if not missing or recover_fec_chunks(self.collected_chunks, self.total_chunks) is not None:
                        print("\nAll frames successfully captured (or FEC reconstructed) with 100% CRC integrity!")
                        print("Reassembling and verifying file...")
                        success, info = reassemble_stream(
                            self.collected_chunks, self.total_chunks, self.output_path, self.start_time
                        )
                        if success:
                            self.is_complete = True
                            self.result_info = info
                            print_transfer_summary(info)
                        else:
                            self.error = info.get("error", "Reassembly failed")
                            print(f"Error: {self.error}")

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

        auto_roi = None
        if self.last_fiducials is not None:
            tl, tr, br, bl = self.last_fiducials
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
            "geometry": self.tracker.locked_geometry[0] if getattr(self.tracker, "locked_geometry", None) else None,
            "elapsed_sec": round(elapsed, 1),
            "fiducials": self.last_fiducials,
            "auto_roi": auto_roi,
            "collected_indices": list(self.collected_chunks.keys()),
            "error": error or self.error
        }
        if self.is_complete and self.result_info:
            d["result"] = self.result_info
        return d

class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

def make_http_handler(session, shutdown_callback):
    class OTDDecoderHTTPHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/ws" and self.headers.get("Upgrade", "").lower() == "websocket":
                self.handle_websocket()
            elif self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()
                self.wfile.write(HTML_PAGE.encode("utf-8"))
            elif self.path == "/api/status":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(json.dumps(session.get_status_dict()).encode("utf-8"))
            elif self.path == "/api/download":
                if session.is_complete and session.result_info:
                    path = session.result_info["saved_path"]
                    fname = session.result_info["filename"]
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            content = f.read()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/octet-stream")
                        self.send_header("Content-Disposition", f'attachment; filename="{fname}"')
                        self.send_header("Content-Length", str(len(content)))
                        self.end_headers()
                        self.wfile.write(content)
                        return
                self.send_response(404)
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            def _json_default(obj):
                if isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return str(obj)

            if self.path == "/api/frame":
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                resp = session.process_image_bytes(body)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(resp, default=_json_default).encode("utf-8"))
            elif self.path == "/api/reset":
                session.reset()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
            elif self.path == "/api/stop":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "stopping"}).encode("utf-8"))
                threading.Thread(target=shutdown_callback).start()
            else:
                self.send_response(404)
                self.end_headers()

        def handle_websocket(self):
            key = self.headers.get("Sec-WebSocket-Key", "")
            if not key:
                self.send_error(400, "Missing Sec-WebSocket-Key")
                return

            guid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
            accept_val = base64.b64encode(
                hashlib.sha1((key.strip() + guid).encode("utf-8")).digest()
            ).decode("utf-8")

            handshake = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept_val}\r\n"
                "\r\n"
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
                    except (socket.timeout, OSError, socket.error):
                        return None
                return bytes(buf)

            def send_ws_frame(opcode, payload_bytes):
                header = bytearray([0x80 | (opcode & 0x0F)])
                plen = len(payload_bytes)
                if plen < 126:
                    header.append(plen)
                elif plen <= 65535:
                    header.append(126)
                    header.extend(struct.pack(">H", plen))
                else:
                    header.append(127)
                    header.extend(struct.pack(">Q", plen))
                try:
                    sock.sendall(bytes(header) + payload_bytes)
                except (OSError, socket.error):
                    pass

            def _json_default(obj):
                if isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return str(obj)

            try:
                while True:
                    head = recv_exact(2)
                    if not head or len(head) < 2:
                        break
                    b1, b2 = head[0], head[1]
                    opcode = b1 & 0x0F
                    masked = (b2 & 0x80) != 0
                    plen = b2 & 0x7F

                    if opcode == 0x8:  # Close
                        send_ws_frame(0x8, b"")
                        break
                    elif opcode == 0x9:  # Ping
                        send_ws_frame(0xA, b"")  # Pong
                        continue

                    if plen == 126:
                        ext = recv_exact(2)
                        if not ext:
                            break
                        plen = struct.unpack(">H", ext)[0]
                    elif plen == 127:
                        ext = recv_exact(8)
                        if not ext:
                            break
                        plen = struct.unpack(">Q", ext)[0]

                    mask_key = recv_exact(4) if masked else None
                    if masked and not mask_key:
                        break

                    raw = recv_exact(plen)
                    if raw is None or len(raw) < plen:
                        break

                    if masked and mask_key:
                        raw_arr = np.frombuffer(raw, dtype=np.uint8)
                        k_arr = np.frombuffer(mask_key, dtype=np.uint8)
                        payload = (raw_arr ^ np.resize(k_arr, len(raw_arr))).tobytes()
                    else:
                        payload = raw

                    if opcode == 0x2:  # Binary frame (image data)
                        resp = session.process_image_bytes(payload)
                        resp_json = json.dumps(resp, default=_json_default).encode("utf-8")
                        send_ws_frame(0x1, resp_json)
                    elif opcode == 0x1:  # Text frame (JSON command)
                        try:
                            cmd_obj = json.loads(payload.decode("utf-8"))
                            action = cmd_obj.get("action")
                            if action == "reset":
                                session.reset()
                                send_ws_frame(0x1, b'{"status":"ok"}')
                            elif action == "status":
                                status_data = json.dumps(session.get_status_dict(), default=_json_default).encode("utf-8")
                                send_ws_frame(0x1, status_data)
                        except Exception:
                            pass
            except Exception:
                pass

        def log_message(self, format, *args):
            # Suppress HTTP access logging to preserve clean receiver CLI output
            return

    return OTDDecoderHTTPHandler

def run_http_receiver(port=8080, output_path=None, open_browser=True, model_path=None, grid_size=None):
    session = HttpDecoderSession(output_path=output_path, model_path=model_path, grid_size=grid_size)
    shutdown_event = threading.Event()

    def do_shutdown():
        time.sleep(0.5)
        shutdown_event.set()

    # Find available port
    target_port = port
    server = None
    for p in range(target_port, target_port + 20):
        try:
            handler_cls = make_http_handler(session, do_shutdown)
            server = ThreadedHTTPServer(("127.0.0.1", p), handler_cls)
            target_port = p
            break
        except OSError:
            continue

    if server is None:
        print(f"Error: Unable to bind to port {port} or next 20 ports.")
        return False

    url = f"http://127.0.0.1:{target_port}"

    print("=" * 65)
    print("   OTD (OPTICAL DATA TRANSFER) v2 RECEIVER (WEB CAPTURE MODE)")
    if model_path:
        print(f"   [ML Acceleration Active: {os.path.basename(model_path)}]")
    if grid_size:
        print(f"   [Locked Grid Dimension: {grid_size}x{grid_size}]")
    print("=" * 65)
    print(f"Web Capture Server: {url}")
    if open_browser:
        print("Launching default browser automatically...")
    else:
        print(f"Open {url} in your browser to begin.")
    print("\nWayland Tip:")
    print("  When your browser displays the sharing dialog:")
    print("  1. Click the 'Window' tab.")
    print("  2. Select the 'otd Transmitter v2' window.")
    print("  The receiver will lock onto the signal and reconstruct the file.")
    print("=" * 65)
    print("Waiting for web capture stream (Press Ctrl+C to stop)...\n")

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        while not shutdown_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[Shutting down web receiver server...]")
    finally:
        server.shutdown()
        server.server_close()

    print("Server stopped.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="otd (Optical Data Transfer) v2 High-Speed Optical Receiver")
    parser.add_argument("input", nargs="?", default=None,
                        help="Path to input video file (or leave empty if using --http / --live / --camera)")
    parser.add_argument("output", nargs="?", default=None,
                        help="Optional output filepath (default: embedded original filename)")
    parser.add_argument("-o", "--output-file", dest="output_file", default=None,
                        help="Optional output filepath (alternative to positional argument)")
    parser.add_argument("-g", "--grid", dest="grid_size", type=int, default=None,
                        help="Lock to specific grid size (e.g. 32, 48, 64, 80, 96, 128; auto-detected if omitted)")
    parser.add_argument("--http", nargs="?", const=8080, type=int, default=None,
                        help="Start web-based screen/window capture receiver on specified port (default: 8080)")
    parser.add_argument("--port", type=int, default=None,
                        help="Custom port for --http web capture server (default: 8080)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not automatically launch web browser when using --http")
    parser.add_argument("--live", action="store_true",
                        help="Capture live from screen instead of reading a video file (X11 only; use --http on Wayland)")
    parser.add_argument("--camera", "--cam", type=int, nargs="?", const=0, default=None,
                        help="Capture live from webcam/camera index (e.g. --camera 0)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Disable OpenCV preview window")
    parser.add_argument("--monitor", type=int, default=1,
                        help="Monitor index for live capture (default: 1)")
    parser.add_argument("--model", "--onnx", dest="model_path", default=None,
                        help="Optional ONNX tracking/keypoint model for ML-accelerated fiducial tracking")

    args = parser.parse_args()

    # Determine output filepath
    output_path = args.output_file
    if not output_path:
        if args.http is not None or args.port is not None or args.live or args.camera is not None:
            # In live / web capture mode, if one positional argument is provided, treat it as output path
            output_path = args.output if args.output else args.input
        else:
            output_path = args.output

    if args.http is not None or args.port is not None:
        p = args.port if args.port is not None else (args.http if isinstance(args.http, int) else 8080)
        run_http_receiver(port=p, output_path=output_path, open_browser=not args.no_browser, model_path=args.model_path, grid_size=args.grid_size)
        sys.exit(0)
    elif args.camera is not None:
        generator = camera_stream_generator(args.camera)
    elif args.live:
        generator = live_screen_generator(args.monitor)
    elif args.input:
        if not os.path.exists(args.input):
            print(f"Error: video file '{args.input}' not found.")
            sys.exit(1)
        generator = video_file_generator(args.input)
    else:
        parser.print_help()
        print("\nError: Please provide a video file path, --http, --live, or --camera.")
        sys.exit(1)

    show_preview = not args.no_preview
    success = decode_stream(generator, output_path=output_path, show_preview=show_preview, model_path=args.model_path, grid_size=args.grid_size)
    sys.exit(0 if success else 1)
