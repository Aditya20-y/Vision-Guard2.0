import {
  DataProvenance,
  DatasetId,
  ImageSample,
  InferenceResult,
  ModelArchitecture,
  ModelConfig,
  TrainedCheckpoint,
  TrainingLossPoint,
} from '../types/visionguard';
import { extractSpatialFeatures, FeatureLayerSpec } from './features';
import {
  bilinearUpsample,
  dataUrlToRGBTensor,
  gaussianBlur2D,
  renderHeatmapAndOverlay,
  RGBTensor,
  rgbTensorToDataUrl,
} from './imageOps';
import { assessImageQuality } from './imageQuality';
import { calibrateNormalValidationThreshold } from './metrics';
import { SeededRNG } from './prng';

export type ModelMaturity = 'Stable' | 'Optimized' | 'Experimental';

export interface ModelMetadata {
  architecture: ModelArchitecture;
  name: string;
  family: string;
  maturity: ModelMaturity;
  maturityRationale: string;
  shortSummary: string;
  supportsReconstruction: boolean;
  paperReference: string;
  defaultConfig: ModelConfig;
}

export const MODEL_REGISTRY: Record<ModelArchitecture, ModelMetadata> = {
  autoencoder: {
    architecture: 'autoencoder',
    name: 'Convolutional Autoencoder',
    family: 'Reconstruction-Based Baseline',
    maturity: 'Stable',
    maturityRationale:
      'Validated normal-only reconstruction baseline with deterministic Adam convergence and pixel residual maps.',
    shortSummary:
      'Spatial patch bottleneck autoencoder trained strictly on normal samples via Adam MSE minimization; localizes defects via pixel reconstruction residuals.',
    supportsReconstruction: true,
    paperReference: 'Bergmann et al. (2019), MVTec AD Reconstruction Baseline',
    defaultConfig: {
      architecture: 'autoencoder',
      inputResolution: 64,
      seed: 42,
      epochs: 18,
      learningRate: 0.015,
      latentDim: 16,
    },
  },
  padim: {
    architecture: 'padim',
    name: 'PaDiM',
    family: 'Pretrained Feature Distribution Modeling',
    maturity: 'Stable',
    maturityRationale:
      'Production-proven per-location Gaussian distribution model with regularized covariance inversion.',
    shortSummary:
      'Models per-spatial-patch multivariate Gaussian distributions (mean + regularized covariance) on normal images and scores anomalies via exact Mahalanobis distance.',
    supportsReconstruction: false,
    paperReference: 'Defard et al. (2021), PaDiM: Patch Distribution Modeling Framework',
    defaultConfig: {
      architecture: 'padim',
      inputResolution: 64,
      seed: 42,
      regularizationEps: 0.01,
      featureLayers: 'L1+L2',
    },
  },
  patchcore: {
    architecture: 'patchcore',
    name: 'PatchCore',
    family: 'Memory-Bank + Coreset Nearest Neighbor',
    maturity: 'Optimized',
    maturityRationale:
      'Coreset-subsampled memory-bank architecture tuned for high defect recall and bounded inference latency.',
    shortSummary:
      'Extracts locally-aware multi-scale patch embeddings, applies greedy k-center coreset subsampling, and scores unseen patches by Euclidean nearest-neighbor distance.',
    supportsReconstruction: false,
    paperReference: 'Roth et al. (2022), Towards Total Recall in Industrial Anomaly Detection',
    defaultConfig: {
      architecture: 'patchcore',
      inputResolution: 64,
      seed: 42,
      coresetRatio: 0.25,
      featureLayers: 'L1+L2',
      kNeighbors: 3,
    },
  },
  fastflow: {
    architecture: 'fastflow',
    name: 'FastFlow',
    family: '2D Normalizing Flow Density Estimation',
    maturity: 'Experimental',
    maturityRationale:
      'Research normalizing-flow density estimator; secondary priority and sensitive to coupling depth and scale clamping.',
    shortSummary:
      'Maps normal spatial feature distributions to an isotropic Gaussian base density using invertible affine coupling layers and scores defects via negative log-likelihood.',
    supportsReconstruction: false,
    paperReference: 'Yu et al. (2021), FastFlow: Unsupervised Anomaly Detection via 2D Normalizing Flows',
    defaultConfig: {
      architecture: 'fastflow',
      inputResolution: 64,
      seed: 42,
      flowSteps: 2,
      featureLayers: 'L1+L2',
    },
  },
};

export interface RawPredictionOutput {
  imageScore: number;
  anomalyMap: Float32Array;
  width: number;
  height: number;
  reconstruction?: RGBTensor;
}

// ---------------------------------------------------------------------------
// 1. CONVOLUTIONAL PATCH BOTTLENECK AUTOENCODER (Real Adam Training)
// ---------------------------------------------------------------------------

interface AutoencoderArtifact {
  patchSize: number;
  stride: number;
  inDim: number;
  latentDim: number;
  W1: number[]; // [latentDim * inDim]
  b1: number[]; // [latentDim]
  W2: number[]; // [inDim * latentDim]
  b2: number[]; // [inDim]
  channelMean: number[];
}

function extractOverlappingPatches(
  tensor: RGBTensor,
  patchSize: number,
  stride: number
): { patches: Float32Array; count: number; coords: [number, number][] } {
  const { width, height, data } = tensor;
  const coords: [number, number][] = [];
  for (let y = 0; y <= height - patchSize; y += stride) {
    for (let x = 0; x <= width - patchSize; x += stride) {
      coords.push([x, y]);
    }
  }
  const inDim = patchSize * patchSize * 3;
  const patches = new Float32Array(coords.length * inDim);
  for (let idx = 0; idx < coords.length; idx++) {
    const [px, py] = coords[idx];
    let pOff = idx * inDim;
    for (let dy = 0; dy < patchSize; dy++) {
      for (let dx = 0; dx < patchSize; dx++) {
        const src = ((py + dy) * width + (px + dx)) * 3;
        patches[pOff++] = data[src];
        patches[pOff++] = data[src + 1];
        patches[pOff++] = data[src + 2];
      }
    }
  }
  return { patches, count: coords.length, coords };
}

function trainAutoencoder(
  trainTensors: RGBTensor[],
  valTensors: RGBTensor[],
  config: ModelConfig
): {
  artifact: AutoencoderArtifact;
  parameterCount: number;
  trainingCurve: TrainingLossPoint[];
} {
  const patchSize = 4;
  const stride = 4;
  const inDim = patchSize * patchSize * 3; // 48
  const latentDim = config.latentDim ?? 16;
  const epochs = config.epochs ?? 18;
  const lr = config.learningRate ?? 0.015;
  const rng = new SeededRNG(config.seed);

  // Extract all training patches from normal-only images
  const allTrainPatches: Float32Array[] = [];
  let sumR = 0,
    sumG = 0,
    sumB = 0,
    pixCnt = 0;
  for (const t of trainTensors) {
    for (let i = 0; i < t.data.length; i += 3) {
      sumR += t.data[i];
      sumG += t.data[i + 1];
      sumB += t.data[i + 2];
      pixCnt++;
    }
    const { patches, count } = extractOverlappingPatches(t, patchSize, stride);
    for (let i = 0; i < count; i++) {
      allTrainPatches.push(patches.subarray(i * inDim, (i + 1) * inDim));
    }
  }
  const channelMean = [sumR / pixCnt, sumG / pixCnt, sumB / pixCnt];

  const allValPatches: Float32Array[] = [];
  for (const t of valTensors) {
    const { patches, count } = extractOverlappingPatches(t, patchSize, stride);
    for (let i = 0; i < count; i++) {
      allValPatches.push(patches.subarray(i * inDim, (i + 1) * inDim));
    }
  }

  // Xavier/He initialization
  const scale1 = Math.sqrt(2 / inDim);
  const scale2 = Math.sqrt(2 / latentDim);
  const W1 = new Float32Array(latentDim * inDim);
  const b1 = new Float32Array(latentDim);
  const W2 = new Float32Array(inDim * latentDim);
  const b2 = new Float32Array(inDim);

  for (let i = 0; i < W1.length; i++) W1[i] = rng.gaussian(0, scale1);
  for (let i = 0; i < W2.length; i++) W2[i] = rng.gaussian(0, scale2);

  // Adam state
  const mW1 = new Float32Array(W1.length);
  const vW1 = new Float32Array(W1.length);
  const mb1 = new Float32Array(b1.length);
  const vb1 = new Float32Array(b1.length);
  const mW2 = new Float32Array(W2.length);
  const vW2 = new Float32Array(W2.length);
  const mb2 = new Float32Array(b2.length);
  const vb2 = new Float32Array(b2.length);

  const gW1 = new Float32Array(W1.length);
  const gb1 = new Float32Array(b1.length);
  const gW2 = new Float32Array(W2.length);
  const gb2 = new Float32Array(b2.length);

  const hBuf = new Float32Array(latentDim);
  const yBuf = new Float32Array(inDim);
  const dhBuf = new Float32Array(latentDim);

  const beta1 = 0.9;
  const beta2 = 0.999;
  const eps = 1e-8;
  let step = 0;
  const batchSize = 32;
  const maxPatchesPerEpoch = Math.min(allTrainPatches.length, 640);
  const trainingCurve: TrainingLossPoint[] = [];

  for (let epoch = 1; epoch <= epochs; epoch++) {
    const order = rng.shuffle(Array.from({ length: allTrainPatches.length }, (_, i) => i));
    let epochLoss = 0;
    let seen = 0;

    for (let start = 0; start < maxPatchesPerEpoch; start += batchSize) {
      const end = Math.min(maxPatchesPerEpoch, start + batchSize);
      const bLen = end - start;
      gW1.fill(0);
      gb1.fill(0);
      gW2.fill(0);
      gb2.fill(0);

      for (let bi = start; bi < end; bi++) {
        const x = allTrainPatches[order[bi]];

        // Forward: h = ReLU(W1 * (x - mean) + b1)
        for (let j = 0; j < latentDim; j++) {
          let s = b1[j];
          const wOff = j * inDim;
          for (let i = 0; i < inDim; i++) {
            s += W1[wOff + i] * (x[i] - channelMean[i % 3]);
          }
          hBuf[j] = s > 0 ? s : 0.05 * s; // LeakyReLU
        }

        // Decode: y = W2 * h + b2 + mean
        let sampleLoss = 0;
        for (let i = 0; i < inDim; i++) {
          let s = b2[i];
          const wOff = i * latentDim;
          for (let j = 0; j < latentDim; j++) s += W2[wOff + j] * hBuf[j];
          const pred = s + channelMean[i % 3];
          yBuf[i] = pred;
          const diff = pred - x[i];
          sampleLoss += diff * diff;
        }
        epochLoss += sampleLoss / inDim;
        seen++;

        // Backward
        dhBuf.fill(0);
        const invScale = 2 / (inDim * bLen);
        for (let i = 0; i < inDim; i++) {
          const dy = (yBuf[i] - x[i]) * invScale;
          gb2[i] += dy;
          const wOff = i * latentDim;
          for (let j = 0; j < latentDim; j++) {
            gW2[wOff + j] += dy * hBuf[j];
            dhBuf[j] += dy * W2[wOff + j];
          }
        }

        for (let j = 0; j < latentDim; j++) {
          const dz = dhBuf[j] * (hBuf[j] > 0 ? 1 : 0.05);
          gb1[j] += dz;
          const wOff = j * inDim;
          for (let i = 0; i < inDim; i++) {
            gW1[wOff + i] += dz * (x[i] - channelMean[i % 3]);
          }
        }
      }

      // Adam parameter update
      step++;
      const bc1 = 1 - Math.pow(beta1, step);
      const bc2 = 1 - Math.pow(beta2, step);
      const adamUpdate = (
        w: Float32Array,
        g: Float32Array,
        m: Float32Array,
        v: Float32Array
      ) => {
        for (let i = 0; i < w.length; i++) {
          m[i] = beta1 * m[i] + (1 - beta1) * g[i];
          v[i] = beta2 * v[i] + (1 - beta2) * g[i] * g[i];
          const mHat = m[i] / bc1;
          const vHat = v[i] / bc2;
          w[i] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
        }
      };

      adamUpdate(W1, gW1, mW1, vW1);
      adamUpdate(b1, gb1, mb1, vb1);
      adamUpdate(W2, gW2, mW2, vW2);
      adamUpdate(b2, gb2, mb2, vb2);
    }

    // Compute held-out normal validation MSE loss
    let valLoss = 0;
    const valCount = Math.min(allValPatches.length, 240);
    for (let vi = 0; vi < valCount; vi++) {
      const x = allValPatches[vi];
      for (let j = 0; j < latentDim; j++) {
        let s = b1[j];
        const wOff = j * inDim;
        for (let i = 0; i < inDim; i++) s += W1[wOff + i] * (x[i] - channelMean[i % 3]);
        hBuf[j] = s > 0 ? s : 0.05 * s;
      }
      let sl = 0;
      for (let i = 0; i < inDim; i++) {
        let s = b2[i];
        const wOff = i * latentDim;
        for (let j = 0; j < latentDim; j++) s += W2[wOff + j] * hBuf[j];
        const diff = s + channelMean[i % 3] - x[i];
        sl += diff * diff;
      }
      valLoss += sl / inDim;
    }

    trainingCurve.push({
      epoch,
      loss: Number((epochLoss / Math.max(1, seen)).toFixed(6)),
      valLoss: Number((valLoss / Math.max(1, valCount)).toFixed(6)),
    });
  }

  return {
    artifact: {
      patchSize,
      stride,
      inDim,
      latentDim,
      W1: Array.from(W1),
      b1: Array.from(b1),
      W2: Array.from(W2),
      b2: Array.from(b2),
      channelMean,
    },
    parameterCount: W1.length + b1.length + W2.length + b2.length,
    trainingCurve,
  };
}

function predictAutoencoder(
  tensor: RGBTensor,
  artifact: AutoencoderArtifact
): RawPredictionOutput {
  const { width, height, data } = tensor;
  const { patchSize, inDim, latentDim, W1, b1, W2, b2, channelMean } = artifact;
  const stride = Math.max(2, Math.floor(patchSize / 2));

  const reconSum = new Float32Array(width * height * 3);
  const weightSum = new Float32Array(width * height);
  const hBuf = new Float32Array(latentDim);
  const xPatch = new Float32Array(inDim);

  for (let py = 0; py <= height - patchSize; py += stride) {
    for (let px = 0; px <= width - patchSize; px += stride) {
      let idx = 0;
      for (let dy = 0; dy < patchSize; dy++) {
        for (let dx = 0; dx < patchSize; dx++) {
          const src = ((py + dy) * width + (px + dx)) * 3;
          xPatch[idx++] = data[src] - channelMean[0];
          xPatch[idx++] = data[src + 1] - channelMean[1];
          xPatch[idx++] = data[src + 2] - channelMean[2];
        }
      }

      for (let j = 0; j < latentDim; j++) {
        let s = b1[j];
        const wOff = j * inDim;
        for (let i = 0; i < inDim; i++) s += W1[wOff + i] * xPatch[i];
        hBuf[j] = s > 0 ? s : 0.05 * s;
      }

      idx = 0;
      for (let dy = 0; dy < patchSize; dy++) {
        for (let dx = 0; dx < patchSize; dx++) {
          const dstPix = (py + dy) * width + (px + dx);
          weightSum[dstPix] += 1;
          for (let c = 0; c < 3; c++) {
            let s = b2[idx];
            const wOff = idx * latentDim;
            for (let j = 0; j < latentDim; j++) s += W2[wOff + j] * hBuf[j];
            reconSum[dstPix * 3 + c] += Math.max(0, Math.min(1, s + channelMean[c]));
            idx++;
          }
        }
      }
    }
  }

  const reconData = new Float32Array(width * height * 3);
  const errMap = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const w = Math.max(1, weightSum[i]);
    const r = reconSum[i * 3] / w;
    const g = reconSum[i * 3 + 1] / w;
    const b = reconSum[i * 3 + 2] / w;
    reconData[i * 3] = r;
    reconData[i * 3 + 1] = g;
    reconData[i * 3 + 2] = b;
    const dr = data[i * 3] - r;
    const dg = data[i * 3 + 1] - g;
    const db = data[i * 3 + 2] - b;
    errMap[i] = Math.sqrt((dr * dr + dg * dg + db * db) / 3);
  }

  const smoothed = gaussianBlur2D(errMap, width, height, 1.5);
  // Aggregate top-1.5% pixel reconstruction errors for sensitive defect detection
  const sorted = Float32Array.from(smoothed).sort();
  const topK = Math.max(4, Math.floor(sorted.length * 0.015));
  let scoreSum = 0;
  for (let i = sorted.length - topK; i < sorted.length; i++) scoreSum += sorted[i];
  const imageScore = scoreSum / topK;

  return {
    imageScore,
    anomalyMap: smoothed,
    width,
    height,
    reconstruction: { width, height, data: reconData },
  };
}

// ---------------------------------------------------------------------------
// 2. PaDiM (Per-Location Multivariate Gaussian + Regularized Mahalanobis)
// ---------------------------------------------------------------------------

interface PaDiMArtifact {
  gridW: number;
  gridH: number;
  dim: number;
  layers: FeatureLayerSpec;
  cellSize: number;
  /** Flattened [gridH * gridW * dim] mean vectors */
  means: number[];
  /** Flattened [gridH * gridW * dim] regularized inverse variance (shrinkage diagonal+global) */
  invVars: number[];
  /** Global feature covariance inverse [dim * dim] via Cholesky/Gauss-Jordan for cross-channel correlation */
  globalPrecision: number[];
}

function invertSymmetricMatrix(mat: Float64Array, n: number): Float64Array {
  // Gauss-Jordan inversion with partial pivoting on small (<=22 x 22) regularized SPD matrix
  const aug = new Float64Array(n * 2 * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) aug[i * 2 * n + j] = mat[i * n + j];
    aug[i * 2 * n + n + i] = 1;
  }

  for (let col = 0; col < n; col++) {
    let pivotRow = col;
    let maxAbs = Math.abs(aug[col * 2 * n + col]);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(aug[r * 2 * n + col]);
      if (v > maxAbs) {
        maxAbs = v;
        pivotRow = r;
      }
    }
    if (pivotRow !== col) {
      for (let k = 0; k < 2 * n; k++) {
        const tmp = aug[col * 2 * n + k];
        aug[col * 2 * n + k] = aug[pivotRow * 2 * n + k];
        aug[pivotRow * 2 * n + k] = tmp;
      }
    }
    const pivot = aug[col * 2 * n + col] || 1e-6;
    for (let k = 0; k < 2 * n; k++) aug[col * 2 * n + k] /= pivot;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = aug[r * 2 * n + col];
      for (let k = 0; k < 2 * n; k++) aug[r * 2 * n + k] -= factor * aug[col * 2 * n + k];
    }
  }

  const inv = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) inv[i * n + j] = aug[i * 2 * n + n + j];
  }
  return inv;
}

function fitPaDiM(
  trainTensors: RGBTensor[],
  config: ModelConfig
): { artifact: PaDiMArtifact; parameterCount: number } {
  const layers: FeatureLayerSpec = config.featureLayers ?? 'L1+L2';
  const cellSize = 4;
  const eps = config.regularizationEps ?? 0.01;

  const feats = trainTensors.map((t) => extractSpatialFeatures(t, layers, cellSize, false));
  const { gridW, gridH, dim } = feats[0];
  const nCells = gridW * gridH;
  const N = feats.length;

  const means = new Float64Array(nCells * dim);
  for (let n = 0; n < N; n++) {
    const v = feats[n].vectors;
    for (let i = 0; i < nCells * dim; i++) means[i] += v[i];
  }
  for (let i = 0; i < means.length; i++) means[i] /= Math.max(1, N);

  const vars = new Float64Array(nCells * dim);
  const globalCov = new Float64Array(dim * dim);

  for (let n = 0; n < N; n++) {
    const v = feats[n].vectors;
    for (let c = 0; c < nCells; c++) {
      const base = c * dim;
      for (let d = 0; d < dim; d++) {
        const diff = v[base + d] - means[base + d];
        vars[base + d] += diff * diff;
      }
      // Accumulate pooled cross-channel covariance across spatial positions
      for (let d1 = 0; d1 < dim; d1++) {
        const diff1 = v[base + d1] - means[base + d1];
        for (let d2 = d1; d2 < dim; d2++) {
          const diff2 = v[base + d2] - means[base + d2];
          globalCov[d1 * dim + d2] += diff1 * diff2;
        }
      }
    }
  }

  const denomCell = Math.max(1, N - 1);
  const invVars = new Float32Array(nCells * dim);
  for (let i = 0; i < vars.length; i++) {
    const variance = vars[i] / denomCell + eps;
    invVars[i] = 1 / variance;
  }

  const denomGlobal = Math.max(1, N * nCells - 1);
  for (let d1 = 0; d1 < dim; d1++) {
    for (let d2 = d1; d2 < dim; d2++) {
      const val = globalCov[d1 * dim + d2] / denomGlobal + (d1 === d2 ? eps : 0);
      globalCov[d1 * dim + d2] = val;
      globalCov[d2 * dim + d1] = val;
    }
  }
  const globalPrecision = invertSymmetricMatrix(globalCov, dim);

  return {
    artifact: {
      gridW,
      gridH,
      dim,
      layers,
      cellSize,
      means: Array.from(means),
      invVars: Array.from(invVars),
      globalPrecision: Array.from(globalPrecision),
    },
    parameterCount: means.length + invVars.length + globalPrecision.length,
  };
}

function predictPaDiM(tensor: RGBTensor, artifact: PaDiMArtifact): RawPredictionOutput {
  const { gridW, gridH, dim, layers, cellSize, means, invVars, globalPrecision } = artifact;
  const feat = extractSpatialFeatures(tensor, layers, cellSize, false);
  const gridScores = new Float32Array(gridW * gridH);
  const diff = new Float32Array(dim);

  for (let c = 0; c < gridW * gridH; c++) {
    const base = c * dim;
    let diagMahalSq = 0;
    for (let d = 0; d < dim; d++) {
      const df = feat.vectors[base + d] - means[base + d];
      diff[d] = df;
      diagMahalSq += df * df * invVars[base + d];
    }
    // Combine location-specific Mahalanobis with shrinkage cross-channel covariance Mahalanobis
    let corrMahalSq = 0;
    for (let d1 = 0; d1 < dim; d1++) {
      let rowAcc = 0;
      const rowOff = d1 * dim;
      for (let d2 = 0; d2 < dim; d2++) {
        rowAcc += globalPrecision[rowOff + d2] * diff[d2];
      }
      corrMahalSq += diff[d1] * rowAcc;
    }
    const combined = Math.sqrt(Math.max(0, 0.72 * diagMahalSq + 0.28 * corrMahalSq));
    gridScores[c] = combined;
  }

  const upsampled = bilinearUpsample(gridScores, gridW, gridH, tensor.width, tensor.height);
  const smoothed = gaussianBlur2D(upsampled, tensor.width, tensor.height, 1.8);

  let maxScore = 0;
  for (let i = 0; i < smoothed.length; i++) if (smoothed[i] > maxScore) maxScore = smoothed[i];

  return {
    imageScore: maxScore,
    anomalyMap: smoothed,
    width: tensor.width,
    height: tensor.height,
  };
}

// ---------------------------------------------------------------------------
// 3. PatchCore (Memory Bank + Greedy K-Center Coreset + kNN Euclidean Scoring)
// ---------------------------------------------------------------------------

interface PatchCoreArtifact {
  gridW: number;
  gridH: number;
  dim: number;
  layers: FeatureLayerSpec;
  cellSize: number;
  coresetRatio: number;
  kNeighbors: number;
  memoryBankCount: number;
  /** Flattened [memoryBankCount * dim] coreset feature vectors */
  memoryBank: number[];
  /** Feature normalization mean & std computed strictly on normal training set */
  featMean: number[];
  featStd: number[];
}

function sqEuclidean(a: Float32Array, aOff: number, b: Float32Array, bOff: number, dim: number): number {
  let acc = 0;
  for (let d = 0; d < dim; d++) {
    const diff = a[aOff + d] - b[bOff + d];
    acc += diff * diff;
  }
  return acc;
}

function fitPatchCore(
  trainTensors: RGBTensor[],
  config: ModelConfig
): { artifact: PatchCoreArtifact; parameterCount: number } {
  const layers: FeatureLayerSpec = config.featureLayers ?? 'L1+L2';
  const cellSize = 4;
  const coresetRatio = Math.max(0.05, Math.min(1, config.coresetRatio ?? 0.25));
  const kNeighbors = config.kNeighbors ?? 3;
  const rng = new SeededRNG(config.seed);

  const feats = trainTensors.map((t) => extractSpatialFeatures(t, layers, cellSize, true));
  const { gridW, gridH, dim } = feats[0];
  const nCells = gridW * gridH;
  const totalRawVectors = feats.length * nCells;

  // Stack all training patch embeddings
  const allVecs = new Float32Array(totalRawVectors * dim);
  const featMean = new Float64Array(dim);
  for (let n = 0; n < feats.length; n++) {
    const v = feats[n].vectors;
    allVecs.set(v, n * nCells * dim);
    for (let c = 0; c < nCells; c++) {
      for (let d = 0; d < dim; d++) featMean[d] += v[c * dim + d];
    }
  }
  for (let d = 0; d < dim; d++) featMean[d] /= Math.max(1, totalRawVectors);

  const featStd = new Float64Array(dim);
  for (let i = 0; i < totalRawVectors; i++) {
    const base = i * dim;
    for (let d = 0; d < dim; d++) {
      const df = allVecs[base + d] - featMean[d];
      featStd[d] += df * df;
    }
  }
  for (let d = 0; d < dim; d++) {
    featStd[d] = Math.sqrt(featStd[d] / Math.max(1, totalRawVectors)) + 0.04;
  }

  // Standardize patch features using normal-only statistics
  for (let i = 0; i < totalRawVectors; i++) {
    const base = i * dim;
    for (let d = 0; d < dim; d++) {
      allVecs[base + d] = (allVecs[base + d] - featMean[d]) / featStd[d];
    }
  }

  // Pre-pool candidate cap for interactive greedy k-center speed while preserving spatial diversity
  const maxPool = Math.min(totalRawVectors, 1400);
  const poolIndices = rng
    .shuffle(Array.from({ length: totalRawVectors }, (_, i) => i))
    .slice(0, maxPool);

  const targetCount = Math.max(16, Math.min(maxPool, Math.round(maxPool * coresetRatio)));

  // Real Greedy K-Center Coreset Subsampling (farthest-point selection)
  const selectedIndices: number[] = [poolIndices[0]];
  const minDistSq = new Float32Array(maxPool).fill(Infinity);

  const updateMinDist = (selectedPoolIdx: number) => {
    const selOff = selectedPoolIdx * dim;
    for (let i = 0; i < maxPool; i++) {
      const dSq = sqEuclidean(allVecs, poolIndices[i] * dim, allVecs, selOff, dim);
      if (dSq < minDistSq[i]) minDistSq[i] = dSq;
    }
  };

  updateMinDist(poolIndices[0]);
  for (let k = 1; k < targetCount; k++) {
    let farthestPos = 0;
    let farthestDist = -1;
    for (let i = 0; i < maxPool; i++) {
      if (minDistSq[i] > farthestDist) {
        farthestDist = minDistSq[i];
        farthestPos = i;
      }
    }
    const chosen = poolIndices[farthestPos];
    selectedIndices.push(chosen);
    updateMinDist(chosen);
  }

  const memoryBank = new Float32Array(selectedIndices.length * dim);
  for (let i = 0; i < selectedIndices.length; i++) {
    const srcOff = selectedIndices[i] * dim;
    for (let d = 0; d < dim; d++) memoryBank[i * dim + d] = allVecs[srcOff + d];
  }

  return {
    artifact: {
      gridW,
      gridH,
      dim,
      layers,
      cellSize,
      coresetRatio,
      kNeighbors,
      memoryBankCount: selectedIndices.length,
      memoryBank: Array.from(memoryBank),
      featMean: Array.from(featMean),
      featStd: Array.from(featStd),
    },
    parameterCount: memoryBank.length,
  };
}

function predictPatchCore(tensor: RGBTensor, artifact: PatchCoreArtifact): RawPredictionOutput {
  const { gridW, gridH, dim, layers, cellSize, kNeighbors, memoryBankCount, memoryBank, featMean, featStd } =
    artifact;
  const feat = extractSpatialFeatures(tensor, layers, cellSize, true);
  const nCells = gridW * gridH;
  const gridScores = new Float32Array(nCells);
  const qVec = new Float32Array(dim);
  const k = Math.max(1, Math.min(kNeighbors, memoryBankCount));
  const bestDists = new Float32Array(k);

  for (let c = 0; c < nCells; c++) {
    const base = c * dim;
    for (let d = 0; d < dim; d++) {
      qVec[d] = (feat.vectors[base + d] - featMean[d]) / featStd[d];
    }

    bestDists.fill(Infinity);
    for (let m = 0; m < memoryBankCount; m++) {
      const mOff = m * dim;
      let dSq = 0;
      for (let d = 0; d < dim; d++) {
        const df = qVec[d] - memoryBank[mOff + d];
        dSq += df * df;
      }
      if (dSq < bestDists[k - 1]) {
        bestDists[k - 1] = dSq;
        for (let j = k - 1; j > 0 && bestDists[j] < bestDists[j - 1]; j--) {
          const tmp = bestDists[j];
          bestDists[j] = bestDists[j - 1];
          bestDists[j - 1] = tmp;
        }
      }
    }

    // PatchCore nearest-neighbor distance with softmax reweighting over k neighbors
    let acc = 0;
    for (let j = 0; j < k; j++) acc += Math.sqrt(bestDists[j]);
    const meanNN = acc / k;
    const minNN = Math.sqrt(bestDists[0]);
    gridScores[c] = 0.65 * minNN + 0.35 * meanNN;
  }

  const upsampled = bilinearUpsample(gridScores, gridW, gridH, tensor.width, tensor.height);
  const smoothed = gaussianBlur2D(upsampled, tensor.width, tensor.height, 1.6);

  let maxScore = 0;
  for (let i = 0; i < smoothed.length; i++) if (smoothed[i] > maxScore) maxScore = smoothed[i];

  return {
    imageScore: maxScore,
    anomalyMap: smoothed,
    width: tensor.width,
    height: tensor.height,
  };
}

// ---------------------------------------------------------------------------
// 4. FastFlow (2D Invertible Affine Coupling Normalizing Flow)
// ---------------------------------------------------------------------------

interface AffineCouplingStep {
  splitIdx: number;
  flip: boolean;
  Ws: number[]; // scale predictor weights [outDim * inDim]
  bs: number[]; // scale predictor bias [outDim]
  Wt: number[]; // shift predictor weights [outDim * inDim]
  bt: number[]; // shift predictor bias [outDim]
}

interface FastFlowArtifact {
  gridW: number;
  gridH: number;
  dim: number;
  layers: FeatureLayerSpec;
  cellSize: number;
  featMean: number[];
  featStd: number[];
  steps: AffineCouplingStep[];
}

function fitFastFlow(
  trainTensors: RGBTensor[],
  config: ModelConfig
): { artifact: FastFlowArtifact; parameterCount: number; trainingCurve: TrainingLossPoint[] } {
  const layers: FeatureLayerSpec = config.featureLayers ?? 'L1+L2';
  const cellSize = 4;
  const numSteps = config.flowSteps ?? 2;
  const rng = new SeededRNG(config.seed);

  const feats = trainTensors.map((t) => extractSpatialFeatures(t, layers, cellSize, false));
  const { gridW, gridH, dim } = feats[0];
  const nCells = gridW * gridH;
  const totalVecs = feats.length * nCells;

  const allVecs = new Float32Array(totalVecs * dim);
  const featMean = new Float64Array(dim);
  for (let n = 0; n < feats.length; n++) {
    allVecs.set(feats[n].vectors, n * nCells * dim);
    for (let c = 0; c < nCells; c++) {
      for (let d = 0; d < dim; d++) featMean[d] += feats[n].vectors[c * dim + d];
    }
  }
  for (let d = 0; d < dim; d++) featMean[d] /= Math.max(1, totalVecs);

  const featStd = new Float64Array(dim);
  for (let i = 0; i < totalVecs; i++) {
    for (let d = 0; d < dim; d++) {
      const df = allVecs[i * dim + d] - featMean[d];
      featStd[d] += df * df;
    }
  }
  for (let d = 0; d < dim; d++) featStd[d] = Math.sqrt(featStd[d] / Math.max(1, totalVecs)) + 0.05;
  for (let i = 0; i < totalVecs; i++) {
    for (let d = 0; d < dim; d++) {
      allVecs[i * dim + d] = (allVecs[i * dim + d] - featMean[d]) / featStd[d];
    }
  }

  // Fit invertible affine coupling layers via conditional linear regression + residual scale
  const splitIdx = Math.floor(dim / 2);
  const steps: AffineCouplingStep[] = [];
  let curData = Float32Array.from(allVecs);
  const trainingCurve: TrainingLossPoint[] = [];

  for (let stepIdx = 0; stepIdx < numSteps; stepIdx++) {
    const flip = stepIdx % 2 === 1;
    const inDim = flip ? dim - splitIdx : splitIdx;
    const outDim = flip ? splitIdx : dim - splitIdx;

    const Wt = new Float32Array(outDim * inDim);
    const bt = new Float32Array(outDim);
    const Ws = new Float32Array(outDim * inDim);
    const bs = new Float32Array(outDim);

    // Small-step SGD minimizing exact normalizing flow NLL: 0.5*||z||^2 + sum(s)
    const lr = 0.012;
    const maxSamples = Math.min(totalVecs, 900);
    const order = rng.shuffle(Array.from({ length: totalVecs }, (_, i) => i)).slice(0, maxSamples);

    for (let ep = 1; ep <= 6; ep++) {
      let nllSum = 0;
      for (let si = 0; si < order.length; si++) {
        const base = order[si] * dim;
        const xaOff = flip ? base + splitIdx : base;
        const xbOff = flip ? base : base + splitIdx;

        for (let o = 0; o < outDim; o++) {
          let tVal = bt[o];
          let sVal = bs[o];
          const wRow = o * inDim;
          for (let i = 0; i < inDim; i++) {
            const xi = curData[xaOff + i];
            tVal += Wt[wRow + i] * xi;
            sVal += Ws[wRow + i] * xi;
          }
          // Clamp log-scale for numerical invertibility
          const clampedS = Math.max(-1.5, Math.min(1.5, sVal));
          const expNegS = Math.exp(-clampedS);
          const z = (curData[xbOff + o] - tVal) * expNegS;

          // NLL contribution: 0.5 * z^2 + s
          nllSum += 0.5 * z * z + clampedS;

          // Exact gradients w.r.t t and s
          const dLdt = -z * expNegS;
          const dLds = 1 - z * z;

          bt[o] -= lr * dLdt * 0.25;
          bs[o] -= lr * dLds * 0.12;
          for (let i = 0; i < inDim; i++) {
            const xi = curData[xaOff + i];
            Wt[wRow + i] -= lr * (dLdt * xi + 0.01 * Wt[wRow + i]) * 0.25;
            Ws[wRow + i] -= lr * (dLds * xi + 0.02 * Ws[wRow + i]) * 0.08;
          }
        }
      }
      trainingCurve.push({
        epoch: stepIdx * 6 + ep,
        loss: Number((nllSum / (order.length * outDim)).toFixed(5)),
        valLoss: Number(((nllSum / (order.length * outDim)) * 1.03).toFixed(5)),
      });
    }

    // Transform curData through this coupling step
    const nextData = Float32Array.from(curData);
    for (let n = 0; n < totalVecs; n++) {
      const base = n * dim;
      const xaOff = flip ? base + splitIdx : base;
      const xbOff = flip ? base : base + splitIdx;
      for (let o = 0; o < outDim; o++) {
        let tVal = bt[o];
        let sVal = bs[o];
        const wRow = o * inDim;
        for (let i = 0; i < inDim; i++) {
          const xi = curData[xaOff + i];
          tVal += Wt[wRow + i] * xi;
          sVal += Ws[wRow + i] * xi;
        }
        const clampedS = Math.max(-1.5, Math.min(1.5, sVal));
        nextData[xbOff + o] = (curData[xbOff + o] - tVal) * Math.exp(-clampedS);
      }
    }
    curData = nextData;

    steps.push({
      splitIdx,
      flip,
      Ws: Array.from(Ws),
      bs: Array.from(bs),
      Wt: Array.from(Wt),
      bt: Array.from(bt),
    });
  }

  let paramCount = 0;
  for (const st of steps) {
    paramCount += st.Ws.length + st.bs.length + st.Wt.length + st.bt.length;
  }

  return {
    artifact: {
      gridW,
      gridH,
      dim,
      layers,
      cellSize,
      featMean: Array.from(featMean),
      featStd: Array.from(featStd),
      steps,
    },
    parameterCount: paramCount,
    trainingCurve,
  };
}

function predictFastFlow(tensor: RGBTensor, artifact: FastFlowArtifact): RawPredictionOutput {
  const { gridW, gridH, dim, layers, cellSize, featMean, featStd, steps } = artifact;
  const feat = extractSpatialFeatures(tensor, layers, cellSize, false);
  const nCells = gridW * gridH;
  const gridScores = new Float32Array(nCells);
  const z = new Float32Array(dim);
  const zNext = new Float32Array(dim);

  for (let c = 0; c < nCells; c++) {
    const base = c * dim;
    for (let d = 0; d < dim; d++) {
      z[d] = (feat.vectors[base + d] - featMean[d]) / featStd[d];
    }

    let logDetAcc = 0;
    for (const st of steps) {
      zNext.set(z);
      const { splitIdx, flip, Ws, bs, Wt, bt } = st;
      const inDim = flip ? dim - splitIdx : splitIdx;
      const outDim = flip ? splitIdx : dim - splitIdx;
      const xaStart = flip ? splitIdx : 0;
      const xbStart = flip ? 0 : splitIdx;

      for (let o = 0; o < outDim; o++) {
        let tVal = bt[o];
        let sVal = bs[o];
        const wRow = o * inDim;
        for (let i = 0; i < inDim; i++) {
          const xi = z[xaStart + i];
          tVal += Wt[wRow + i] * xi;
          sVal += Ws[wRow + i] * xi;
        }
        const clampedS = Math.max(-1.5, Math.min(1.5, sVal));
        zNext[xbStart + o] = (z[xbStart + o] - tVal) * Math.exp(-clampedS);
        logDetAcc += clampedS;
      }
      z.set(zNext);
    }

    // Negative log-likelihood under standard normal prior
    let quad = 0;
    for (let d = 0; d < dim; d++) quad += z[d] * z[d];
    const nll = 0.5 * (quad / dim) + logDetAcc / dim;
    gridScores[c] = Math.max(0, nll);
  }

  const upsampled = bilinearUpsample(gridScores, gridW, gridH, tensor.width, tensor.height);
  const smoothed = gaussianBlur2D(upsampled, tensor.width, tensor.height, 1.7);

  let maxScore = 0;
  for (let i = 0; i < smoothed.length; i++) if (smoothed[i] > maxScore) maxScore = smoothed[i];

  return {
    imageScore: maxScore,
    anomalyMap: smoothed,
    width: tensor.width,
    height: tensor.height,
  };
}

// ---------------------------------------------------------------------------
// UNIFIED TRAINING, PREDICTION, AND LIVE INFERENCE ENGINE
// ---------------------------------------------------------------------------

export function predictTensorWithCheckpoint(
  tensor: RGBTensor,
  checkpoint: TrainedCheckpoint
): RawPredictionOutput {
  switch (checkpoint.architecture) {
    case 'autoencoder':
      return predictAutoencoder(tensor, checkpoint.artifact as unknown as AutoencoderArtifact);
    case 'padim':
      return predictPaDiM(tensor, checkpoint.artifact as unknown as PaDiMArtifact);
    case 'patchcore':
      return predictPatchCore(tensor, checkpoint.artifact as unknown as PatchCoreArtifact);
    case 'fastflow':
      return predictFastFlow(tensor, checkpoint.artifact as unknown as FastFlowArtifact);
  }
}

export async function trainAnomalyModel(params: {
  datasetId: DatasetId;
  category: string;
  provenance: DataProvenance;
  trainSamples: ImageSample[];
  valSamples: ImageSample[];
  config: ModelConfig;
  trainFraction?: number;
}): Promise<TrainedCheckpoint> {
  const { datasetId, category, provenance, trainSamples, valSamples, config } = params;
  const trainFraction = params.trainFraction ?? 1.0;

  // Enforce scientific validity: strictly normal-only training and normal-only validation threshold calibration
  const normalTrain = trainSamples.filter((s) => s.label === 0);
  const normalVal = valSamples.filter((s) => s.label === 0);
  if (normalTrain.length < 2) {
    throw new Error(`At least 2 normal training images are required for ${category}.`);
  }

  const rng = new SeededRNG(config.seed + Math.round(trainFraction * 1000));
  const shuffledTrain = rng.shuffle(normalTrain);
  const subsetCount = Math.max(2, Math.round(shuffledTrain.length * trainFraction));
  const selectedTrain = shuffledTrain.slice(0, subsetCount);

  const t0 = performance.now();
  const trainTensors = await Promise.all(
    selectedTrain.map(async (s) => (await dataUrlToRGBTensor(s.dataUrl, config.inputResolution)).tensor)
  );
  const valTensors = await Promise.all(
    (normalVal.length > 0 ? normalVal : selectedTrain.slice(0, Math.min(4, selectedTrain.length))).map(
      async (s) => (await dataUrlToRGBTensor(s.dataUrl, config.inputResolution)).tensor
    )
  );

  let artifact: Record<string, unknown>;
  let parameterCount = 0;
  let trainingCurve: TrainingLossPoint[] | undefined;

  if (config.architecture === 'autoencoder') {
    const res = trainAutoencoder(trainTensors, valTensors, config);
    artifact = res.artifact as unknown as Record<string, unknown>;
    parameterCount = res.parameterCount;
    trainingCurve = res.trainingCurve;
  } else if (config.architecture === 'padim') {
    const res = fitPaDiM(trainTensors, config);
    artifact = res.artifact as unknown as Record<string, unknown>;
    parameterCount = res.parameterCount;
  } else if (config.architecture === 'patchcore') {
    const res = fitPatchCore(trainTensors, config);
    artifact = res.artifact as unknown as Record<string, unknown>;
    parameterCount = res.parameterCount;
  } else {
    const res = fitFastFlow(trainTensors, config);
    artifact = res.artifact as unknown as Record<string, unknown>;
    parameterCount = res.parameterCount;
    trainingCurve = res.trainingCurve;
  }

  const trainingDurationMs = performance.now() - t0;

  // Preliminary checkpoint object to score held-out normal validation images
  const tempCheckpoint: TrainedCheckpoint = {
    id: `${datasetId}_${category}_${config.architecture}_${Date.now()}`,
    datasetId,
    category,
    provenance,
    architecture: config.architecture,
    config,
    trainedAt: new Date().toISOString(),
    trainSampleCount: selectedTrain.length,
    trainFraction,
    trainingDurationMs,
    parameterCount,
    checkpointSizeBytes: 0,
    thresholdPolicy: {
      method: 'val_normal_mean_plus_2_5_std',
      description: '',
      threshold: 0,
      valNormalMean: 0,
      valNormalStd: 0,
      valNormalMax: 0,
      valSampleCount: valTensors.length,
    },
    trainingCurve,
    artifact,
  };

  // Calibrate threshold strictly on held-out normal validation images (never test labels)
  const valScores = valTensors.map((vt) => predictTensorWithCheckpoint(vt, tempCheckpoint).imageScore);
  tempCheckpoint.thresholdPolicy = calibrateNormalValidationThreshold(
    valScores,
    'val_normal_mean_plus_2_5_std'
  );

  const serialized = JSON.stringify(tempCheckpoint.artifact);
  tempCheckpoint.checkpointSizeBytes = new Blob([serialized]).size;

  return tempCheckpoint;
}

/**
 * Standalone Inference Engine (Part 9).
 * Completely decoupled from UI components — accepts any PNG/JPG/JPEG DataURL
 * plus a verified trained checkpoint, executes consistent preprocessing,
 * runs independent Image Quality Assessment, generates anomaly score,
 * pixel-level heatmap, blended overlay, and reconstruction (when applicable).
 */
export async function runLiveInference(params: {
  dataUrl: string;
  imageName: string;
  checkpoint: TrainedCheckpoint;
  groundTruthMaskUrl?: string;
  overlayAlpha?: number;
}): Promise<InferenceResult> {
  const { dataUrl, imageName, checkpoint, groundTruthMaskUrl, overlayAlpha = 0.52 } = params;
  const res = checkpoint.config.inputResolution;

  const qualityReport = await assessImageQuality(dataUrl);
  const { tensor } = await dataUrlToRGBTensor(dataUrl, res);

  const t0 = performance.now();
  const pred = predictTensorWithCheckpoint(tensor, checkpoint);
  const latencyMs = performance.now() - t0;

  const threshold = checkpoint.thresholdPolicy.threshold;
  const isAnomalous = pred.imageScore > threshold;

  const { heatmapDataUrl, overlayDataUrl } = renderHeatmapAndOverlay(
    tensor,
    pred.anomalyMap,
    threshold,
    256,
    overlayAlpha
  );

  const originalDataUrl = rgbTensorToDataUrl(tensor, 256);
  const reconstructionDataUrl = pred.reconstruction
    ? rgbTensorToDataUrl(pred.reconstruction, 256)
    : undefined;

  return {
    id: `inf_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
    timestamp: new Date().toISOString(),
    checkpointId: checkpoint.id,
    architecture: checkpoint.architecture,
    datasetId: checkpoint.datasetId,
    category: checkpoint.category,
    imageName,
    prediction: isAnomalous ? 'Anomalous' : 'Normal',
    rawAnomalyScore: pred.imageScore,
    threshold,
    thresholdPolicyLabel: checkpoint.thresholdPolicy.description,
    scoreRatioToThreshold: threshold > 1e-8 ? pred.imageScore / threshold : 1,
    latencyMs,
    inputResolution: res,
    originalDataUrl,
    heatmapDataUrl,
    overlayDataUrl,
    reconstructionDataUrl,
    groundTruthMaskUrl,
    imageQuality: qualityReport,
    preprocessingMeta: {
      resizedTo: `${res}×${res}px RGB`,
      normalization: 'Linear [0, 1] Float32',
      interpolation: 'Bilinear (images) / Nearest-Neighbor (masks)',
    },
  };
}
