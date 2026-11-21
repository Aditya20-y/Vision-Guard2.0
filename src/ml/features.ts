import { RGBTensor } from './imageOps';

export type FeatureLayerSpec = 'L1' | 'L2' | 'L1+L2' | 'L1+L2+L3';

export interface SpatialFeatureMap {
  gridW: number;
  gridH: number;
  dim: number;
  /** Row-major [gridH * gridW * dim] */
  vectors: Float32Array;
}

/**
 * Multi-scale deterministic convolutional feature extractor inspired by
 * intermediate WideResNet hierarchies (Layer 1: fine edges & color,
 * Layer 2: mid-scale oriented texture & Laplacian blobs,
 * Layer 3: coarse neighborhood contextual statistics).
 *
 * Used consistently by PaDiM, PatchCore, and FastFlow so feature comparisons
 * and PatchCore layer ablations (Experiment D) are strictly controlled.
 */
export function extractSpatialFeatures(
  tensor: RGBTensor,
  layers: FeatureLayerSpec = 'L1+L2',
  cellSize = 4,
  locallyAwarePool = false
): SpatialFeatureMap {
  const { width, height, data } = tensor;
  const nPix = width * height;

  // Luminance and opponent color channels
  const gray = new Float32Array(nPix);
  const rg = new Float32Array(nPix);
  const by = new Float32Array(nPix);
  for (let i = 0; i < nPix; i++) {
    const r = data[i * 3];
    const g = data[i * 3 + 1];
    const b = data[i * 3 + 2];
    gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    rg[i] = r - g;
    by[i] = 0.5 * (r + g) - b;
  }

  // Compute Sobel gradients & Laplacian at pixel level
  const gx = new Float32Array(nPix);
  const gy = new Float32Array(nPix);
  const gmag = new Float32Array(nPix);
  const lap = new Float32Array(nPix);
  const diag1 = new Float32Array(nPix);
  const diag2 = new Float32Array(nPix);

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      const tl = gray[idx - width - 1];
      const tc = gray[idx - width];
      const tr = gray[idx - width + 1];
      const ml = gray[idx - 1];
      const mc = gray[idx];
      const mr = gray[idx + 1];
      const bl = gray[idx + width - 1];
      const bc = gray[idx + width];
      const br = gray[idx + width + 1];

      const sx = -tl + tr - 2 * ml + 2 * mr - bl + br;
      const sy = -tl - 2 * tc - tr + bl + 2 * bc + br;
      gx[idx] = sx;
      gy[idx] = sy;
      gmag[idx] = Math.sqrt(sx * sx + sy * sy);
      lap[idx] = Math.abs(tc + bc + ml + mr - 4 * mc);
      // 45° and 135° oriented filter responses
      diag1[idx] = Math.abs(tl + br - 2 * mc);
      diag2[idx] = Math.abs(tr + bl - 2 * mc);
    }
  }

  const gridW = Math.max(4, Math.floor(width / cellSize));
  const gridH = Math.max(4, Math.floor(height / cellSize));

  const useL1 = layers === 'L1' || layers === 'L1+L2' || layers === 'L1+L2+L3';
  const useL2 = layers === 'L2' || layers === 'L1+L2' || layers === 'L1+L2+L3';
  const useL3 = layers === 'L1+L2+L3';

  // L1 = 8 dims, L2 = 8 dims, L3 = 6 dims
  const dimL1 = useL1 ? 8 : 0;
  const dimL2 = useL2 ? 8 : 0;
  const dimL3 = useL3 ? 6 : 0;
  const dim = dimL1 + dimL2 + dimL3;

  const raw = new Float32Array(gridH * gridW * dim);

  for (let gyIdx = 0; gyIdx < gridH; gyIdx++) {
    const y0 = gyIdx * cellSize;
    const y1 = Math.min(height, y0 + cellSize);
    for (let gxIdx = 0; gxIdx < gridW; gxIdx++) {
      const x0 = gxIdx * cellSize;
      const x1 = Math.min(width, x0 + cellSize);

      let cnt = 0;
      let sumR = 0,
        sumG = 0,
        sumB = 0,
        sumLum = 0,
        sumLumSq = 0;
      let sumGmag = 0,
        maxGmag = 0,
        sumGxAbs = 0,
        sumGyAbs = 0;
      let sumLap = 0,
        maxLap = 0,
        sumD1 = 0,
        sumD2 = 0;
      let sumRG = 0,
        sumBY = 0,
        minLum = 1,
        maxLum = 0;

      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const i = y * width + x;
          const r = data[i * 3];
          const g = data[i * 3 + 1];
          const b = data[i * 3 + 2];
          const l = gray[i];
          sumR += r;
          sumG += g;
          sumB += b;
          sumLum += l;
          sumLumSq += l * l;
          if (l < minLum) minLum = l;
          if (l > maxLum) maxLum = l;
          const gm = gmag[i];
          sumGmag += gm;
          if (gm > maxGmag) maxGmag = gm;
          sumGxAbs += Math.abs(gx[i]);
          sumGyAbs += Math.abs(gy[i]);
          const lp = lap[i];
          sumLap += lp;
          if (lp > maxLap) maxLap = lp;
          sumD1 += diag1[i];
          sumD2 += diag2[i];
          sumRG += rg[i];
          sumBY += by[i];
          cnt++;
        }
      }

      const inv = 1 / Math.max(1, cnt);
      const meanLum = sumLum * inv;
      const stdLum = Math.sqrt(Math.max(0, sumLumSq * inv - meanLum * meanLum));

      let offset = (gyIdx * gridW + gxIdx) * dim;
      if (useL1) {
        raw[offset++] = sumR * inv;
        raw[offset++] = sumG * inv;
        raw[offset++] = sumB * inv;
        raw[offset++] = stdLum * 2.5;
        raw[offset++] = sumGmag * inv * 0.65;
        raw[offset++] = maxGmag * 0.35;
        raw[offset++] = sumGxAbs * inv * 0.7;
        raw[offset++] = sumGyAbs * inv * 0.7;
      }

      if (useL2) {
        raw[offset++] = sumLap * inv * 1.4;
        raw[offset++] = maxLap * 0.75;
        raw[offset++] = sumD1 * inv * 1.3;
        raw[offset++] = sumD2 * inv * 1.3;
        raw[offset++] = (maxLum - minLum) * 1.2;
        raw[offset++] = sumRG * inv * 1.5;
        raw[offset++] = sumBY * inv * 1.5;
        raw[offset++] = Math.abs(sumGxAbs - sumGyAbs) * inv * 0.8;
      }

      if (useL3) {
        // Coarse 3x cell receptive field contextual mean & contrast around cell center
        const cy0 = Math.max(0, y0 - cellSize);
        const cy1 = Math.min(height, y1 + cellSize);
        const cx0 = Math.max(0, x0 - cellSize);
        const cx1 = Math.min(width, x1 + cellSize);
        let cCnt = 0,
          cLum = 0,
          cLumSq = 0,
          cGrad = 0,
          cR = 0,
          cB = 0;
        for (let y = cy0; y < cy1; y += 2) {
          for (let x = cx0; x < cx1; x += 2) {
            const i = y * width + x;
            const l = gray[i];
            cLum += l;
            cLumSq += l * l;
            cGrad += gmag[i];
            cR += data[i * 3];
            cB += data[i * 3 + 2];
            cCnt++;
          }
        }
        const cInv = 1 / Math.max(1, cCnt);
        const cMean = cLum * cInv;
        const cStd = Math.sqrt(Math.max(0, cLumSq * cInv - cMean * cMean));
        raw[offset++] = cMean;
        raw[offset++] = cStd * 2.2;
        raw[offset++] = cGrad * cInv * 0.6;
        raw[offset++] = (meanLum - cMean) * 2.4; // local-to-context contrast residual
        raw[offset++] = cR * cInv;
        raw[offset++] = cB * cInv;
      }
    }
  }

  if (!locallyAwarePool) {
    return { gridW, gridH, dim, vectors: raw };
  }

  // PatchCore locally-aware neighborhood aggregation (3x3 weighted spatial pooling)
  const pooled = new Float32Array(gridH * gridW * dim);
  for (let y = 0; y < gridH; y++) {
    for (let x = 0; x < gridW; x++) {
      const dstBase = (y * gridW + x) * dim;
      let wSum = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = Math.max(0, Math.min(gridH - 1, y + dy));
        for (let dx = -1; dx <= 1; dx++) {
          const nx = Math.max(0, Math.min(gridW - 1, x + dx));
          const weight = dx === 0 && dy === 0 ? 0.52 : 0.06;
          const srcBase = (ny * gridW + nx) * dim;
          for (let d = 0; d < dim; d++) {
            pooled[dstBase + d] += raw[srcBase + d] * weight;
          }
          wSum += weight;
        }
      }
      for (let d = 0; d < dim; d++) pooled[dstBase + d] /= wSum;
    }
  }

  return { gridW, gridH, dim, vectors: pooled };
}
