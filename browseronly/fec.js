/**
 * browseronly/fec.js - Cauchy Reed-Solomon GF(2^8) Erasure Decoder in Pure JavaScript
 * Matches Python implementation in decoder_v2.py / encoder_min.py with 100% mathematical fidelity.
 */

(function(exports) {
  'use strict';

  const GF_EXP = new Uint16Array(512);
  const GF_LOG = new Uint16Array(256);

  let x = 1;
  for (let i = 0; i < 255; i++) {
    GF_EXP[i] = x;
    GF_EXP[i + 255] = x;
    GF_LOG[x] = i;
    x <<= 1;
    if (x & 0x100) x ^= 0x11d;
  }
  GF_LOG[0] = 511;

  function gfMul(a, b) {
    if (a === 0 || b === 0) return 0;
    return GF_EXP[GF_LOG[a] + GF_LOG[b]];
  }

  function gfInv(a) {
    if (a === 0) return 0;
    return GF_EXP[255 - GF_LOG[a]];
  }

  function cauchyCoeff(r, c) {
    return gfInv((255 - r) ^ c);
  }

  /**
   * Recovers missing data chunks (0..K-1) from parity chunks (idx >= K)
   * using systematic Cauchy Reed-Solomon erasure decoding.
   *
   * @param {Object} collectedChunks - Map of chunkIndex (number) to Uint8Array payload
   * @param {number} K - Total original data chunks
   * @returns {Object|null} - Reconstructed map containing all chunks 0..K-1, or null if insufficient parities
   */
  function recoverFecChunks(collectedChunks, K) {
    const missing = [];
    for (let i = 0; i < K; i++) {
      if (!collectedChunks[i]) missing.push(i);
    }
    if (missing.length === 0) {
      const res = {};
      for (let i = 0; i < K; i++) res[i] = collectedChunks[i];
      return res;
    }

    const paritiesAvail = Object.keys(collectedChunks)
      .map(Number)
      .filter(idx => idx >= K)
      .sort((a, b) => a - b);

    if (paritiesAvail.length < missing.length) {
      return null;
    }

    const usedParities = paritiesAvail.slice(0, missing.length);
    const L = missing.length;
    let chunkLen = 0;
    for (const idx in collectedChunks) {
      if (collectedChunks[idx].length > chunkLen) {
        chunkLen = collectedChunks[idx].length;
      }
    }

    const S = [];
    const R = [];

    for (const pIdx of usedParities) {
      const pRow = pIdx - K;
      const rowS = [];
      for (const m of missing) {
        rowS.push(cauchyCoeff(pRow, m));
      }
      S.push(rowS);

      const origP = collectedChunks[pIdx];
      const rhs = new Uint8Array(chunkLen);
      rhs.set(origP);

      for (let j = 0; j < K; j++) {
        if (!missing.includes(j)) {
          const coeff = cauchyCoeff(pRow, j);
          if (coeff === 0) continue;
          const d = collectedChunks[j];
          const dLen = d.length;
          for (let b_i = 0; b_i < dLen; b_i++) {
            if (d[b_i] !== 0) {
              rhs[b_i] ^= gfMul(coeff, d[b_i]);
            }
          }
        }
      }
      R.push(rhs);
    }

    // Gaussian elimination over GF(2^8)
    for (let i = 0; i < L; i++) {
      if (S[i][i] === 0) {
        for (let r = i + 1; r < L; r++) {
          if (S[r][i] !== 0) {
            const tempS = S[i]; S[i] = S[r]; S[r] = tempS;
            const tempR = R[i]; R[i] = R[r]; R[r] = tempR;
            break;
          }
        }
      }
      const pivot = S[i][i];
      if (pivot === 0) return null;
      const invP = gfInv(pivot);

      for (let c = i; c < L; c++) {
        S[i][c] = gfMul(S[i][c], invP);
      }
      for (let b_i = 0; b_i < chunkLen; b_i++) {
        R[i][b_i] = gfMul(R[i][b_i], invP);
      }

      for (let r = 0; r < L; r++) {
        if (r !== i && S[r][i] !== 0) {
          const factor = S[r][i];
          for (let c = i; c < L; c++) {
            S[r][c] ^= gfMul(factor, S[i][c]);
          }
          for (let b_i = 0; b_i < chunkLen; b_i++) {
            R[r][b_i] ^= gfMul(factor, R[i][b_i]);
          }
        }
      }
    }

    const result = Object.assign({}, collectedChunks);
    for (let m_i = 0; m_i < missing.length; m_i++) {
      result[missing[m_i]] = R[m_i];
    }
    return result;
  }

  /**
   * Generates M Cauchy Reed-Solomon parity chunks from K data chunks.
   *
   * @param {Array<Uint8Array>} chunks - List of K data chunks
   * @param {number} M - Number of parity chunks to generate
   * @returns {Array<Uint8Array>} - List of M parity chunks
   */
  function generateFecParities(chunks, M) {
    const K = chunks.length;
    let maxLen = 0;
    for (let i = 0; i < K; i++) {
      if (chunks[i].length > maxLen) maxLen = chunks[i].length;
    }
    const parities = [];
    for (let r = 0; r < M; r++) {
      const p = new Uint8Array(maxLen);
      for (let c = 0; c < K; c++) {
        const coeff = cauchyCoeff(r, c);
        const d = chunks[c];
        const dLen = d.length;
        for (let i = 0; i < dLen; i++) {
          if (d[i] !== 0) {
            p[i] ^= gfMul(coeff, d[i]);
          }
        }
      }
      parities.push(p);
    }
    return parities;
  }

  exports.gfMul = gfMul;
  exports.gfInv = gfInv;
  exports.cauchyCoeff = cauchyCoeff;
  exports.recoverFecChunks = recoverFecChunks;
  exports.generateFecParities = generateFecParities;

})(typeof module !== 'undefined' && module.exports ? module.exports : (window.OTDFec = {}));

