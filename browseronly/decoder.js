/**
 * browseronly/decoder.js - Complete In-Browser Optical Decoder Engine
 * Pure client-side computer vision, perspective mapping, vectorized cell sampling,
 * CRC32 validation, Cauchy Reed-Solomon FEC, and SHA-256 reassembly.
 */

(function(exports) {
  'use strict';

  // 1. IEEE 802.3 CRC32 Table
  const CRC32_TABLE = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) {
      c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
    }
    CRC32_TABLE[i] = c >>> 0;
  }

  function computeCRC32(bytes) {
    let crc = 0xffffffff;
    const len = bytes.length;
    for (let i = 0; i < len; i++) {
      crc = CRC32_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
    }
    return (crc ^ 0xffffffff) >>> 0;
  }

  // 2. Candidate Grids & Geometries
  const DEFAULT_CANDIDATE_GRIDS = [64, 48, 80, 96, 128, 32, 40, 56, 72, 112, 144, 160];

  function getCandidateGeometries(gridSize) {
    // 1. Standard GUI mode: Data grid spans 80 to 920 in 1000x1000 canonical plane
    const guiGeom = { name: "gui", start: 80.0, span: 840.0 };

    // 2. Symmetric TTY mode:
    const M = Math.max(2, Math.round(gridSize / 21.0));
    const D = gridSize + 2 * M;
    const unit = 920.0 / D;
    const ttyGeom = { name: "tty", start: 40.0 + (M - 0.5) * unit, span: gridSize * unit };

    // 3. Legacy TTY mode
    const D_leg = Math.round(gridSize * 23.0 / 21.0);
    const unit_leg = 920.0 / D_leg;
    const ttyLegGeom = { name: "tty_legacy", start: 40.0 + 1.5 * unit_leg, span: gridSize * unit_leg };

    return [guiGeom, ttyGeom, ttyLegGeom];
  }

  // 3. Exact Closed-Form Projective Homography (Unit Square [0,1]^2 -> Arbitrary Convex Quadrilateral)
  class QuadHomography {
    constructor(tl, tr, br, bl) {
      this.tl = tl;
      this.tr = tr;
      this.br = br;
      this.bl = bl;

      const x0 = tl[0], y0 = tl[1];
      const x1 = tr[0], y1 = tr[1];
      const x2 = br[0], y2 = br[1];
      const x3 = bl[0], y3 = bl[1];

      const dx1 = x1 - x2;
      const dx2 = x3 - x2;
      const sx = x0 - x1 + x2 - x3;
      const dy1 = y1 - y2;
      const dy2 = y3 - y2;
      const sy = y0 - y1 + y2 - y3;

      if (Math.abs(sx) < 1e-6 && Math.abs(sy) < 1e-6) {
        // Affine
        this.a11 = x1 - x0;
        this.a12 = x2 - x1;
        this.a13 = x0;
        this.a21 = y1 - y0;
        this.a22 = y2 - y1;
        this.a23 = y0;
        this.a31 = 0;
        this.a32 = 0;
      } else {
        // General Projective Quadrilateral
        const det = dx1 * dy2 - dx2 * dy1;
        if (Math.abs(det) < 1e-8) {
          this.valid = false;
          return;
        }
        this.a31 = (sx * dy2 - sy * dx2) / det;
        this.a32 = (dx1 * sy - dy1 * sx) / det;
        this.a11 = x1 - x0 + this.a31 * x1;
        this.a12 = x3 - x0 + this.a32 * x3;
        this.a13 = x0;
        this.a21 = y1 - y0 + this.a31 * y1;
        this.a22 = y3 - y0 + this.a32 * y3;
        this.a23 = y0;
      }
      this.valid = true;
    }

    /**
     * Projects normalized coordinate (u, v) in [0, 1] x [0, 1] to source image coordinate (x, y).
     */
    project(u, v) {
      const W = this.a31 * u + this.a32 * v + 1.0;
      const x = (this.a11 * u + this.a12 * v + this.a13) / W;
      const y = (this.a21 * u + this.a22 * v + this.a23) / W;
      return [x, y];
    }

    /**
     * Projects canonical coordinate (xCanon, yCanon) in [0, 1000] x [0, 1000] to source image coordinate (x, y).
     * The 4 corner fiducials are canonical points (40, 40), (960, 40), (960, 960), (40, 960).
     */
    projectCanonical(xCanon, yCanon) {
      return this.project((xCanon - 40.0) / 920.0, (yCanon - 40.0) / 920.0);
    }
  }

  // 4. Optical Tracker & Fiducial Detector
  class BrowserFiducialTracker {
    constructor() {
      this.prevFiducials = null;
      this.emaFiducials = null;
      this.emaAlpha = 0.85;
      this.confidence = 0;
      this.lockedGridSize = null;
      this.lockedMode = "rgb";
      this.lockedGeom = null;
    }

    reset() {
      this.prevFiducials = null;
      this.emaFiducials = null;
      this.confidence = 0;
      this.lockedGridSize = null;
      this.lockedMode = "rgb";
      this.lockedGeom = null;
    }

    /**
     * Fast chromatic difference fiducial detection on ImageData.
     * Searches for Red (TL), Green (TR), Blue (BL), Magenta (BR).
     */
    detectFiducials(imgData) {
      const w = imgData.width;
      const h = imgData.height;
      const data = imgData.data;

      // 1. If we have locked corners with high confidence, use fast local tracking (±18 px)
      if (this.prevFiducials && this.confidence > 0) {
        const tracked = this.trackLocalFiducials(imgData, this.prevFiducials);
        if (tracked) {
          return this.applyEma(tracked);
        }
      }

      // 2. Full Frame Scan
      // Step size 2 for speed on large frames (>1000px)
      const step = (w > 1200 || h > 1200) ? 3 : (w > 640 ? 2 : 1);

      let bestTL = null, minSum = Infinity;
      let bestTR = null, maxDiffTR = -Infinity;
      let bestBL = null, minDiffBL = Infinity;
      let bestBR = null, maxSum = -Infinity;

      for (let y = 4; y < h - 4; y += step) {
        const rowOffset = y * w * 4;
        for (let x = 4; x < w - 4; x += step) {
          const idx = rowOffset + x * 4;
          const r = data[idx];
          const g = data[idx + 1];
          const b = data[idx + 2];

          // TL Red: r - max(g, b)
          const rDiff = r - Math.max(g, b);
          if (rDiff > 35) {
            const sum = x + y;
            if (sum < minSum) {
              minSum = sum;
              bestTL = [x, y, rDiff];
            }
          }

          // TR Green: g - max(r, b)
          const gDiff = g - Math.max(r, b);
          if (gDiff > 35) {
            const diff = x - y;
            if (diff > maxDiffTR) {
              maxDiffTR = diff;
              bestTR = [x, y, gDiff];
            }
          }

          // BL Blue: b - max(r, g)
          const bDiff = b - Math.max(r, g);
          if (bDiff > 35) {
            const diff = x - y;
            if (diff < minDiffBL) {
              minDiffBL = diff;
              bestBL = [x, y, bDiff];
            }
          }

          // BR Magenta: min(r, b) - g
          const mDiff = Math.min(r, b) - g;
          if (mDiff > 35) {
            const sum = x + y;
            if (sum > maxSum) {
              maxSum = sum;
              bestBR = [x, y, mDiff];
            }
          }
        }
      }

      if (!bestTL || !bestTR || !bestBR || !bestBL) {
        this.confidence = 0;
        return null;
      }

      // Geometry check
      if (!(bestTL[1] < bestBL[1] && bestTL[0] < bestTR[0] &&
            bestTR[1] < bestBR[1] && bestBL[0] < bestBR[0])) {
        this.confidence = 0;
        return null;
      }

      const rawFids = [
        this.refineSubpixelCentroid(imgData, bestTL[0], bestTL[1], 'r'),
        this.refineSubpixelCentroid(imgData, bestTR[0], bestTR[1], 'g'),
        this.refineSubpixelCentroid(imgData, bestBR[0], bestBR[1], 'm'),
        this.refineSubpixelCentroid(imgData, bestBL[0], bestBL[1], 'b')
      ];

      return this.applyEma(rawFids);
    }

    /**
     * Local patch tracker around previous corners (< 0.05ms).
     */
    trackLocalFiducials(imgData, prevFids) {
      const w = imgData.width;
      const h = imgData.height;
      const data = imgData.data;
      const radius = 22;
      const types = ['r', 'g', 'm', 'b'];
      const updated = [];

      for (let i = 0; i < 4; i++) {
        const [px, py] = prevFids[i];
        const type = types[i];
        const minX = Math.max(2, Math.round(px - radius));
        const maxX = Math.min(w - 3, Math.round(px + radius));
        const minY = Math.max(2, Math.round(py - radius));
        const maxY = Math.min(h - 3, Math.round(py + radius));

        let bestScore = -1, bx = px, by = py;
        for (let y = minY; y <= maxY; y++) {
          const rowOffset = y * w * 4;
          for (let x = minX; x <= maxX; x++) {
            const idx = rowOffset + x * 4;
            const r = data[idx], g = data[idx + 1], b = data[idx + 2];
            let score = 0;
            if (type === 'r') score = r - Math.max(g, b);
            else if (type === 'g') score = g - Math.max(r, b);
            else if (type === 'b') score = b - Math.max(r, g);
            else if (type === 'm') score = Math.min(r, b) - g;

            if (score > bestScore) {
              bestScore = score;
              bx = x;
              by = y;
            }
          }
        }

        if (bestScore < 20) {
          return null; // Lost tracking
        }

        updated.push(this.refineSubpixelCentroid(imgData, bx, by, type));
      }

      // Geometry sanity check
      if (!(updated[0][1] < updated[3][1] && updated[0][0] < updated[1][0] &&
            updated[1][1] < updated[2][1] && updated[3][0] < updated[2][0])) {
        return null;
      }

      return updated;
    }

    /**
     * 2-Pass Subpixel center-of-mass refinement around peak.
     * Pass 1: Wide radius (16px) captures the full blob even if cx,cy hit an outer edge.
     * Pass 2: Focused radius (10px) refines to true optical centroid (< 0.2 px error).
     */
    refineSubpixelCentroid(imgData, cx, cy, type) {
      const w = imgData.width, h = imgData.height, data = imgData.data;

      const calcCentroid = (x0, y0, rad, minScore) => {
        const minX = Math.max(0, Math.floor(x0 - rad)), maxX = Math.min(w - 1, Math.ceil(x0 + rad));
        const minY = Math.max(0, Math.floor(y0 - rad)), maxY = Math.min(h - 1, Math.ceil(y0 + rad));

        let sumWeight = 0, sumX = 0, sumY = 0;
        for (let y = minY; y <= maxY; y++) {
          const rowOffset = y * w * 4;
          for (let x = minX; x <= maxX; x++) {
            const idx = rowOffset + x * 4;
            const r = data[idx], g = data[idx + 1], b = data[idx + 2];
            let score = 0;
            if (type === 'r') score = r - Math.max(g, b);
            else if (type === 'g') score = g - Math.max(r, b);
            else if (type === 'b') score = b - Math.max(r, g);
            else if (type === 'm') score = Math.min(r, b) - g;

            if (score > minScore) {
              const wt = score;
              sumWeight += wt;
              sumX += x * wt;
              sumY += y * wt;
            }
          }
        }
        if (sumWeight > 0) {
          return [sumX / sumWeight, sumY / sumWeight];
        }
        return [x0, y0];
      };

      const c1 = calcCentroid(cx, cy, 16, 20);
      return calcCentroid(c1[0], c1[1], 10, 20);
    }

    applyEma(fids) {
      if (!this.emaFiducials) {
        this.emaFiducials = fids.map(pt => [pt[0], pt[1]]);
      } else {
        const alpha = this.emaAlpha;
        for (let i = 0; i < 4; i++) {
          this.emaFiducials[i][0] = alpha * fids[i][0] + (1 - alpha) * this.emaFiducials[i][0];
          this.emaFiducials[i][1] = alpha * fids[i][1] + (1 - alpha) * this.emaFiducials[i][1];
        }
      }
      this.prevFiducials = this.emaFiducials;
      this.confidence = Math.min(10, this.confidence + 1);
      return this.emaFiducials;
    }

    /**
     * Calibrate dynamic RGB thresholds by sampling canonical fiducial references & borders.
     * Matches decoder_v2.py calibrate_fiducial_thresholds with patch averaging.
     */
    calibrateThresholds(imgData, homography) {
      const w = imgData.width, h = imgData.height, data = imgData.data;

      const sampleCanonPatch = (cx, cy) => {
        let rSum = 0, gSum = 0, bSum = 0, cnt = 0;
        for (let dy = -2; dy <= 2; dy += 2) {
          for (let dx = -2; dx <= 2; dx += 2) {
            const [px, py] = homography.projectCanonical(cx + dx, cy + dy);
            const ix = Math.max(0, Math.min(w - 1, Math.round(px)));
            const iy = Math.max(0, Math.min(h - 1, Math.round(py)));
            const idx = (iy * w + ix) * 4;
            rSum += data[idx];
            gSum += data[idx + 1];
            bSum += data[idx + 2];
            cnt++;
          }
        }
        return [rSum / cnt, gSum / cnt, bSum / cnt];
      };

      try {
        const rFid = sampleCanonPatch(40, 40);     // Red (TL)
        const gFid = sampleCanonPatch(960, 40);    // Green (TR)
        const bFid = sampleCanonPatch(40, 960);    // Blue (BL)
        const mFid = sampleCanonPatch(960, 960);   // Magenta (BR)

        // Outer white borders
        const w1 = sampleCanonPatch(40, 21);
        const w2 = sampleCanonPatch(960, 21);
        const whiteVal = [Math.max(w1[0], w2[0]), Math.max(w1[1], w2[1]), Math.max(w1[2], w2[2])];

        // Black margins
        const b1 = sampleCanonPatch(500, 25);
        const b2 = sampleCanonPatch(61, 61);
        const blackVal = [Math.min(b1[0], b2[0]), Math.min(b1[1], b2[1]), Math.min(b1[2], b2[2])];

        const thR = ((rFid[0] + mFid[0] + whiteVal[0]) / 3.0 + (blackVal[0] + gFid[0] + bFid[0]) / 3.0) / 2.0;
        const thG = ((gFid[1] + whiteVal[1]) / 2.0 + (blackVal[1] + rFid[1] + bFid[1] + mFid[1]) / 4.0) / 2.0;
        const thB = ((bFid[2] + mFid[2] + whiteVal[2]) / 3.0 + (blackVal[2] + rFid[2] + gFid[2]) / 3.0) / 2.0;

        return [
          Math.max(15, Math.min(240, thR)),
          Math.max(15, Math.min(240, thG)),
          Math.max(15, Math.min(240, thB))
        ];
      } catch (e) {
        return [128, 128, 128];
      }
    }

    /**
     * Vectorized sampling of grid cell RGB colors using homography.projectCanonical.
     */
    sampleGridColors(imgData, homography, gridSize, geom) {
      const canonCell = geom.span / gridSize;
      const totCells = gridSize * gridSize;
      const rArr = new Uint8Array(totCells);
      const gArr = new Uint8Array(totCells);
      const bArr = new Uint8Array(totCells);
      const w = imgData.width, h = imgData.height, data = imgData.data;

      let idx = 0;
      for (let r = 0; r < gridSize; r++) {
        const cy = geom.start + (r + 0.5) * canonCell;
        for (let c = 0; c < gridSize; c++) {
          const cx = geom.start + (c + 0.5) * canonCell;
          const [px, py] = homography.projectCanonical(cx, cy);
          const ix = Math.max(0, Math.min(w - 1, Math.round(px)));
          const iy = Math.max(0, Math.min(h - 1, Math.round(py)));
          const pIdx = (iy * w + ix) * 4;

          rArr[idx] = data[pIdx];
          gArr[idx] = data[pIdx + 1];
          bArr[idx] = data[pIdx + 2];
          idx++;
        }
      }
      return { rArr, gArr, bArr, totCells };
    }

    binarizeAndPackRgb(rArr, gArr, bArr, totCells, thR, thG, thB) {
      const totalBits = totCells * 3;
      const numBytes = Math.floor(totalBits / 8);
      const out = new Uint8Array(numBytes);
      let currentByte = 0;
      let bitsInByte = 0;
      let byteIdx = 0;

      for (let i = 0; i < totCells && byteIdx < numBytes; i++) {
        const rBit = rArr[i] > thR ? 1 : 0;
        const gBit = gArr[i] > thG ? 1 : 0;
        const bBit = bArr[i] > thB ? 1 : 0;
        const val = (rBit << 2) | (gBit << 1) | bBit;

        for (let shift = 2; shift >= 0; shift--) {
          const bit = (val >> shift) & 1;
          currentByte = (currentByte << 1) | bit;
          bitsInByte++;
          if (bitsInByte === 8) {
            out[byteIdx++] = currentByte;
            currentByte = 0;
            bitsInByte = 0;
          }
        }
      }
      return out;
    }

    binarizeAndPackBw(rArr, gArr, bArr, totCells, thLuma) {
      const numBytes = Math.floor(totCells / 8);
      const out = new Uint8Array(numBytes);
      let currentByte = 0;
      let bitsInByte = 0;
      let byteIdx = 0;

      for (let i = 0; i < totCells && byteIdx < numBytes; i++) {
        const luma = (rArr[i] * 299 + gArr[i] * 587 + bArr[i] * 114) / 1000;
        const bit = luma > thLuma ? 1 : 0;
        currentByte = (currentByte << 1) | bit;
        bitsInByte++;
        if (bitsInByte === 8) {
          out[byteIdx++] = currentByte;
          currentByte = 0;
          bitsInByte = 0;
        }
      }
      return out;
    }

    computePercentileThreshold(arr, totCells) {
      const hist = new Int32Array(256);
      for (let i = 0; i < totCells; i++) hist[arr[i]]++;
      const p25Count = totCells * 0.25;
      const p75Count = totCells * 0.75;
      let sum = 0, p25 = 128, p75 = 128;
      let found25 = false;
      for (let i = 0; i < 256; i++) {
        sum += hist[i];
        if (!found25 && sum >= p25Count) {
          p25 = i;
          found25 = true;
        }
        if (sum >= p75Count) {
          p75 = i;
          break;
        }
      }
      return (p25 + p75) / 2.0;
    }

    /**
     * Decodes a single video frame across candidate grids, geometries, and adaptive passes.
     */
    decodeFrame(imgData) {
      const fiducials = this.detectFiducials(imgData);
      if (!fiducials) {
        return { parsedFrame: null, fiducials: null, homography: null };
      }

      const [tl, tr, br, bl] = fiducials;
      const homography = new QuadHomography(tl, tr, br, bl);
      if (!homography.valid) {
        return { parsedFrame: null, fiducials, homography: null };
      }

      const candidateGrids = this.lockedGridSize ? [this.lockedGridSize] : DEFAULT_CANDIDATE_GRIDS;
      const modes = this.lockedGridSize ? [this.lockedMode] : ["rgb", "bw"];
      const [thR, thG, thB] = this.calibrateThresholds(imgData, homography);

      for (const gs of candidateGrids) {
        const geoms = this.lockedGeom ? [this.lockedGeom] : getCandidateGeometries(gs);
        for (const geom of geoms) {
          const { rArr, gArr, bArr, totCells } = this.sampleGridColors(imgData, homography, gs, geom);

          for (const mode of modes) {
            if (mode === "rgb") {
              // Pass 1: Calibrated thresholds
              let rawBytes = this.binarizeAndPackRgb(rArr, gArr, bArr, totCells, thR, thG, thB);
              let parsed = this.parseFrameHeader(rawBytes);
              if (parsed) {
                this.lockedGridSize = gs;
                this.lockedMode = mode;
                this.lockedGeom = geom;
                this.consecutiveFailures = 0;
                return { parsedFrame: parsed, fiducials, homography };
              }

              // Pass 2: Standard threshold (128) fallback
              if (Math.abs(thR - 128) > 3 || Math.abs(thG - 128) > 3 || Math.abs(thB - 128) > 3) {
                rawBytes = this.binarizeAndPackRgb(rArr, gArr, bArr, totCells, 128, 128, 128);
                parsed = this.parseFrameHeader(rawBytes);
                if (parsed) {
                  this.lockedGridSize = gs;
                  this.lockedMode = mode;
                  this.lockedGeom = geom;
                  this.consecutiveFailures = 0;
                  return { parsedFrame: parsed, fiducials, homography };
                }
              }

              // Pass 3: Adaptive percentile thresholds (handles dimmed, night-light, and gamma-shifted screens)
              const adR = this.computePercentileThreshold(rArr, totCells);
              const adG = this.computePercentileThreshold(gArr, totCells);
              const adB = this.computePercentileThreshold(bArr, totCells);
              if (Math.abs(adR - 128) > 5 || Math.abs(adG - 128) > 5 || Math.abs(adB - 128) > 5) {
                rawBytes = this.binarizeAndPackRgb(rArr, gArr, bArr, totCells, adR, adG, adB);
                parsed = this.parseFrameHeader(rawBytes);
                if (parsed) {
                  this.lockedGridSize = gs;
                  this.lockedMode = mode;
                  this.lockedGeom = geom;
                  this.consecutiveFailures = 0;
                  return { parsedFrame: parsed, fiducials, homography };
                }
              }
            } else {
              // mode === "bw"
              const thLuma = (thR * 299 + thG * 587 + thB * 114) / 1000;
              // Pass 1
              let rawBytes = this.binarizeAndPackBw(rArr, gArr, bArr, totCells, thLuma);
              let parsed = this.parseFrameHeader(rawBytes);
              if (parsed) {
                this.lockedGridSize = gs;
                this.lockedMode = mode;
                this.lockedGeom = geom;
                this.consecutiveFailures = 0;
                return { parsedFrame: parsed, fiducials, homography };
              }

              // Pass 2: 128
              if (Math.abs(thLuma - 128) > 3) {
                rawBytes = this.binarizeAndPackBw(rArr, gArr, bArr, totCells, 128);
                parsed = this.parseFrameHeader(rawBytes);
                if (parsed) {
                  this.lockedGridSize = gs;
                  this.lockedMode = mode;
                  this.lockedGeom = geom;
                  this.consecutiveFailures = 0;
                  return { parsedFrame: parsed, fiducials, homography };
                }
              }
            }
          }
        }
      }

      // Unlock if locked parameters failed consecutively (sender changed resolution/grid)
      if (this.lockedGridSize) {
        this.consecutiveFailures = (this.consecutiveFailures || 0) + 1;
        if (this.consecutiveFailures > 45) {
          this.lockedGridSize = null;
          this.lockedGeom = null;
          this.consecutiveFailures = 0;
        }
      }

      return { parsedFrame: null, fiducials, homography };
    }

    parseFrameHeader(rawBytes) {
      if (!rawBytes || rawBytes.length < 16) return null;
      const view = new DataView(rawBytes.buffer, rawBytes.byteOffset, rawBytes.byteLength);
      const m0 = view.getUint8(0);
      const m1 = view.getUint8(1);
      if (m0 !== 0x51 || m1 !== 0x32) return null; // 'Q', '2'
      const ver = view.getUint8(2);
      if (ver !== 2) return null;
      const modeByte = view.getUint8(3);
      const gridSize = view.getUint8(4);
      const flags = view.getUint8(5);
      const idx = view.getUint16(6, false);
      const total = view.getUint16(8, false);
      const plen = view.getUint16(10, false);
      const expectedCrc = view.getUint32(12, false);

      const isParity = (flags & 2) !== 0;
      if (total === 0) return null;
      if (!isParity && idx >= total) return null;
      if (isParity && (idx < total || idx >= total * 2 + 32)) return null;
      if (rawBytes.length < 16 + plen) return null;

      const payload = rawBytes.subarray(16, 16 + plen);
      const calcCrc = computeCRC32(payload);
      if (calcCrc !== expectedCrc) return null;

      let ackInfo = null;
      if (plen >= 118 && payload[0] === 0x41 && payload[1] === 0x43 && payload[2] === 0x4b && payload[3] === 0x32) {
        ackInfo = unpackOpticalAckPayload(payload);
      }

      return {
        idx,
        total,
        payload,
        gridSize,
        mode: modeByte === 3 ? "rgb" : "bw",
        isParity,
        ackInfo
      };
    }
  }

  const STATE_NAMES = {
    0: "WAITING_FOR_FOCUS",
    1: "KEYBOARD_FOCUSED",
    2: "RECEIVING_DATA",
    3: "VERIFIED_COMPLETE",
    4: "TRANSFER_ERROR"
  };

  function unpackOpticalAckPayload(payload) {
    if (!payload || payload.length < 118) return null;
    const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
    const m0 = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
    if (m0 !== "ACK2") return null;

    const state = view.getUint8(4);
    const lastAck = view.getUint16(5, false);
    const totalChunks = view.getUint16(7, false);
    const receivedCount = view.getUint32(9, false);
    const bytesReceived = view.getUint32(13, false);
    const bitmask = payload.subarray(17, 17 + 64);
    const shaBytes = payload.subarray(81, 81 + 32);
    const errByte = view.getUint8(113);
    const expectedCrc = view.getUint32(114, false);

    const calcCrc = computeCRC32(payload.subarray(0, 114));
    if (calcCrc !== expectedCrc) return null;

    const focusSeq = (errByte >> 4) & 0x0f;
    const errorCode = errByte & 0x0f;

    let shaHex = "";
    for (let i = 0; i < 32; i++) shaHex += shaBytes[i].toString(16).padStart(2, "0");

    return {
      state,
      stateName: STATE_NAMES[state] || ("STATE_" + state),
      lastAck,
      totalChunks,
      receivedCount,
      bytesReceived,
      bitmask: Array.from(bitmask),
      sha256: shaHex,
      errorCode,
      focusSeq
    };
  }

  // 5. Complete Optical Session Receiver
  class BrowserOpticalReceiver {
    constructor() {
      this.tracker = new BrowserFiducialTracker();
      this.reset();
    }

    reset() {
      this.tracker.reset();
      this.collectedChunks = {};
      this.totalChunks = null;
      this.startTime = null;
      this.framesProcessed = 0;
      this.validFrames = 0;
      this.isComplete = false;
      this.completedResult = null;
      this.lastFiducials = null;
    }

    async processFrame(imgData) {
      this.framesProcessed++;
      const { parsedFrame, fiducials, homography } = this.tracker.decodeFrame(imgData);
      this.lastFiducials = fiducials;

      if (!parsedFrame) {
        return {
          status: fiducials ? "locked" : "searching",
          fiducials,
          homography,
          isComplete: this.isComplete,
          result: this.completedResult,
          stats: this.getStats()
        };
      }

      this.validFrames++;
      if (!this.startTime) {
        this.startTime = performance.now();
      }

      this.totalChunks = parsedFrame.total;
      const idx = parsedFrame.idx;

      if (!(idx in this.collectedChunks)) {
        this.collectedChunks[idx] = parsedFrame.payload;
      }

      // Check if we have collected all data chunks or have enough parities for FEC recovery
      if (!this.isComplete && this.totalChunks) {
        const collectedCount = Object.keys(this.collectedChunks).length;
        if (collectedCount >= this.totalChunks) {
          const res = await this.tryReassembly();
          if (res) {
            this.isComplete = true;
            this.completedResult = res;
          }
        }
      }

      return {
        status: this.isComplete ? "complete" : "receiving",
        parsedFrame,
        fiducials,
        homography,
        isComplete: this.isComplete,
        result: this.completedResult,
        stats: this.getStats()
      };
    }

    getStats() {
      const count = Object.keys(this.collectedChunks).length;
      const total = this.totalChunks || 0;
      const pct = total > 0 ? Math.min(100.0, (count / total) * 100.0) : 0;
      const elapsedSec = this.startTime ? (performance.now() - this.startTime) / 1000 : 0;
      let totalBytes = 0;
      for (const k in this.collectedChunks) {
        totalBytes += this.collectedChunks[k].length;
      }
      const speedKbps = elapsedSec > 0 ? (totalBytes / 1024) / elapsedSec : 0;

      return {
        collectedChunks: count,
        totalChunks: total,
        percent: pct.toFixed(1),
        elapsedSec: elapsedSec.toFixed(1),
        speedKbps: speedKbps.toFixed(1),
        framesProcessed: this.framesProcessed,
        validFrames: this.validFrames,
        gridSize: this.tracker.lockedGridSize,
        mode: this.tracker.lockedMode
      };
    }

    async tryReassembly() {
      const K = this.totalChunks;
      let chunks = this.collectedChunks;

      // Check missing data chunks
      const missing = [];
      for (let i = 0; i < K; i++) {
        if (!chunks[i]) missing.push(i);
      }

      if (missing.length > 0) {
        // Run Cauchy RS FEC recovery
        if (window.OTDFec && window.OTDFec.recoverFecChunks) {
          const recovered = window.OTDFec.recoverFecChunks(chunks, K);
          if (recovered) {
            chunks = recovered;
          } else {
            return null; // Need more parity frames
          }
        } else {
          return null;
        }
      }

      // Concatenate data chunks 0..K-1
      let totalLen = 0;
      for (let i = 0; i < K; i++) totalLen += chunks[i].length;
      const fullStream = new Uint8Array(totalLen);
      let offset = 0;
      for (let i = 0; i < K; i++) {
        fullStream.set(chunks[i], offset);
        offset += chunks[i].length;
      }

      if (fullStream.length < 54) return null;

      // Unpack QSMD metadata header (54 bytes)
      const view = new DataView(fullStream.buffer, fullStream.byteOffset, fullStream.byteLength);
      const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
      if (magic !== "QSMD") return null;

      const origSize = Number(view.getBigUint64(4, false));
      const compSize = Number(view.getBigUint64(12, false));
      const expectedShaBytes = fullStream.subarray(20, 52);
      const fnameLen = view.getUint16(52, false);

      const decoder = new TextDecoder("utf-8");
      const filename = decoder.decode(fullStream.subarray(54, 54 + fnameLen));
      const compressedData = fullStream.subarray(54 + fnameLen, 54 + fnameLen + compSize);

      // Decompress
      let rawData = null;
      if (window.OTDInflate && window.OTDInflate.decompressZlib) {
        rawData = await window.OTDInflate.decompressZlib(compressedData);
      } else {
        throw new Error("Decompression engine unavailable");
      }

      // Verify SHA-256 using browser Web Crypto API
      const hashBuffer = await crypto.subtle.digest("SHA-256", rawData);
      const hashArray = Array.from(new Uint8Array(hashBuffer));
      const computedShaHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');

      const expectedShaHex = Array.from(expectedShaBytes)
        .map(b => b.toString(16).padStart(2, '0')).join('');

      const shaMatches = (computedShaHex.toLowerCase() === expectedShaHex.toLowerCase());

      return {
        filename,
        origSize,
        compSize,
        sha256: computedShaHex,
        expectedSha: expectedShaHex,
        verified: shaMatches,
        data: rawData,
        elapsedSec: ((performance.now() - this.startTime) / 1000).toFixed(2),
        speedKbps: (((origSize / 1024) / ((performance.now() - this.startTime) / 1000))).toFixed(1)
      };
    }
  }

  exports.computeCRC32 = computeCRC32;
  exports.QuadHomography = QuadHomography;
  exports.BrowserFiducialTracker = BrowserFiducialTracker;
  exports.BrowserOpticalReceiver = BrowserOpticalReceiver;
  exports.unpackOpticalAckPayload = unpackOpticalAckPayload;
  exports.STATE_NAMES = STATE_NAMES;

})(typeof module !== 'undefined' && module.exports ? module.exports : (window.OTDDecoder = {}));

