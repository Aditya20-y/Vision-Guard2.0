import { ConfusionMatrix, CurvePoint, ThresholdPolicy } from '../types/visionguard';
import { extractConnectedComponents } from './imageOps';

/**
 * Compute exact ROC curve and Area Under ROC Curve (AUROC) via trapezoidal rule
 * on continuous anomaly scores (Part 6 & Part 8).
 */
export function computeAurocAndRoc(
  labels: number[],
  scores: number[]
): { auroc: number; rocCurve: CurvePoint[] } {
  const n = labels.length;
  if (n === 0) return { auroc: 0, rocCurve: [] };

  let totalPos = 0;
  let totalNeg = 0;
  for (let i = 0; i < n; i++) {
    if (labels[i] === 1) totalPos++;
    else totalNeg++;
  }
  if (totalPos === 0 || totalNeg === 0) {
    return { auroc: 0.5, rocCurve: [{ x: 0, y: 0 }, { x: 1, y: 1 }] };
  }

  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => scores[b] - scores[a]);

  const roc: CurvePoint[] = [{ x: 0, y: 0 }];
  let tp = 0;
  let fp = 0;
  let i = 0;

  while (i < n) {
    const curScore = scores[order[i]];
    // Group tied scores so trapezoidal integration handles ties exactly
    while (i < n && scores[order[i]] === curScore) {
      if (labels[order[i]] === 1) tp++;
      else fp++;
      i++;
    }
    roc.push({
      x: fp / totalNeg,
      y: tp / totalPos,
      threshold: curScore,
    });
  }

  let area = 0;
  for (let k = 1; k < roc.length; k++) {
    const dx = roc[k].x - roc[k - 1].x;
    const avgY = 0.5 * (roc[k].y + roc[k - 1].y);
    area += dx * avgY;
  }

  // Downsample ROC curve to at most 80 points for smooth plotting while preserving exact AUROC
  const sampledRoc: CurvePoint[] =
    roc.length <= 80
      ? roc
      : Array.from({ length: 80 }, (_, idx) => roc[Math.round((idx / 79) * (roc.length - 1))]);

  return { auroc: Math.max(0, Math.min(1, area)), rocCurve: sampledRoc };
}

/**
 * Compute Precision-Recall Curve on continuous scores.
 */
export function computePrCurve(labels: number[], scores: number[]): CurvePoint[] {
  const n = labels.length;
  let totalPos = 0;
  for (let i = 0; i < n; i++) if (labels[i] === 1) totalPos++;
  if (totalPos === 0) return [{ x: 0, y: 1 }, { x: 1, y: 0 }];

  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => scores[b] - scores[a]);
  const curve: CurvePoint[] = [{ x: 0, y: 1 }];
  let tp = 0;
  let fp = 0;
  let i = 0;
  while (i < n) {
    const curScore = scores[order[i]];
    while (i < n && scores[order[i]] === curScore) {
      if (labels[order[i]] === 1) tp++;
      else fp++;
      i++;
    }
    const recall = tp / totalPos;
    const precision = tp / Math.max(1, tp + fp);
    curve.push({ x: recall, y: precision, threshold: curScore });
  }
  return curve.length <= 80
    ? curve
    : Array.from({ length: 80 }, (_, idx) => curve[Math.round((idx / 79) * (curve.length - 1))]);
}

/**
 * Compute Pixel-Level AUROC across all evaluated test pixels using histogram-binned
 * exact-rank approximation (2048 quantile/linear bins) for fast, high-precision pixel scoring.
 */
export function computePixelAuroc(
  masks: Uint8Array[],
  anomalyMaps: Float32Array[]
): number | null {
  let totalPos = 0;
  let totalNeg = 0;
  let minScore = Infinity;
  let maxScore = -Infinity;

  for (let k = 0; k < masks.length; k++) {
    const m = masks[k];
    const s = anomalyMaps[k];
    for (let i = 0; i < m.length; i++) {
      if (m[i] === 1) totalPos++;
      else totalNeg++;
      if (s[i] < minScore) minScore = s[i];
      if (s[i] > maxScore) maxScore = s[i];
    }
  }

  if (totalPos === 0 || totalNeg === 0 || maxScore <= minScore) return null;

  const BINS = 2048;
  const posHist = new Float64Array(BINS);
  const negHist = new Float64Array(BINS);
  const scale = (BINS - 1) / (maxScore - minScore);

  for (let k = 0; k < masks.length; k++) {
    const m = masks[k];
    const s = anomalyMaps[k];
    for (let i = 0; i < m.length; i++) {
      const bin = Math.max(0, Math.min(BINS - 1, Math.floor((s[i] - minScore) * scale)));
      if (m[i] === 1) posHist[bin]++;
      else negHist[bin]++;
    }
  }

  let tp = 0;
  let fp = 0;
  let prevFpr = 0;
  let prevTpr = 0;
  let area = 0;

  for (let b = BINS - 1; b >= 0; b--) {
    tp += posHist[b];
    fp += negHist[b];
    const fpr = fp / totalNeg;
    const tpr = tp / totalPos;
    area += (fpr - prevFpr) * 0.5 * (tpr + prevTpr);
    prevFpr = fpr;
    prevTpr = tpr;
  }

  return Math.max(0, Math.min(1, area));
}

/**
 * Compute AUPRO (Area Under Per-Region Overlap curve up to FPR = 0.30).
 * Normalizes integration over [0, 0.30] by dividing by 0.30 as in MVTec AD standard protocol.
 */
export function computeAupro(
  masks: Uint8Array[],
  anomalyMaps: Float32Array[],
  width: number,
  height: number,
  maxFpr = 0.3
): number | null {
  interface RegionRef {
    imageIdx: number;
    pixels: number[];
  }
  const regions: RegionRef[] = [];
  let totalNormalPixels = 0;
  let minScore = Infinity;
  let maxScore = -Infinity;

  for (let k = 0; k < masks.length; k++) {
    const m = masks[k];
    const s = anomalyMaps[k];
    const comps = extractConnectedComponents(m, width, height);
    for (const c of comps) regions.push({ imageIdx: k, pixels: c });
    for (let i = 0; i < m.length; i++) {
      if (m[i] === 0) totalNormalPixels++;
      if (s[i] < minScore) minScore = s[i];
      if (s[i] > maxScore) maxScore = s[i];
    }
  }

  if (regions.length === 0 || totalNormalPixels === 0 || maxScore <= minScore) return null;

  const STEPS = 50;
  const pts: { fpr: number; pro: number }[] = [];

  for (let step = 0; step <= STEPS; step++) {
    // Sweeping threshold from high to low
    const t = maxScore - (step / STEPS) * (maxScore - minScore);
    let fp = 0;
    for (let k = 0; k < masks.length; k++) {
      const m = masks[k];
      const s = anomalyMaps[k];
      for (let i = 0; i < m.length; i++) {
        if (m[i] === 0 && s[i] >= t) fp++;
      }
    }
    const fpr = fp / totalNormalPixels;

    let proSum = 0;
    for (const reg of regions) {
      const s = anomalyMaps[reg.imageIdx];
      let hit = 0;
      for (let j = 0; j < reg.pixels.length; j++) {
        if (s[reg.pixels[j]] >= t) hit++;
      }
      proSum += hit / reg.pixels.length;
    }
    const pro = proSum / regions.length;
    pts.push({ fpr, pro });
    if (fpr >= maxFpr) break;
  }

  if (pts.length < 2) return null;

  // Integrate PRO over FPR in [0, maxFpr]
  let area = 0;
  for (let i = 1; i < pts.length; i++) {
    const x0 = Math.min(maxFpr, pts[i - 1].fpr);
    const x1 = Math.min(maxFpr, pts[i].fpr);
    if (x1 > x0) {
      area += (x1 - x0) * 0.5 * (pts[i - 1].pro + pts[i].pro);
    }
  }
  return Math.max(0, Math.min(1, area / maxFpr));
}

/**
 * Calibrate decision threshold strictly using held-out NORMAL validation scores.
 * Never touches anomalous or test labels (Part 6 & Part 8 Scientific Validity).
 */
export function calibrateNormalValidationThreshold(
  valNormalScores: number[],
  method: ThresholdPolicy['method'] = 'val_normal_mean_plus_2_5_std'
): ThresholdPolicy {
  const n = valNormalScores.length;
  if (n === 0) {
    return {
      method,
      description: 'Fallback threshold (no validation normals provided)',
      threshold: 0.5,
      valNormalMean: 0,
      valNormalStd: 0,
      valNormalMax: 0,
      valSampleCount: 0,
    };
  }

  const sorted = valNormalScores.slice().sort((a, b) => a - b);
  const mean = sorted.reduce((a, b) => a + b, 0) / n;
  const variance = sorted.reduce((a, b) => a + (b - mean) * (b - mean), 0) / Math.max(1, n - 1);
  const std = Math.sqrt(variance);
  const max = sorted[n - 1];

  let threshold: number;
  let description: string;
  if (method === 'val_normal_p95') {
    const idx = Math.min(n - 1, Math.floor(0.95 * (n - 1)));
    threshold = sorted[idx] * 1.05;
    description = 'Held-out normal validation P95 × 1.05 (zero test-label leakage)';
  } else {
    threshold = mean + 2.5 * Math.max(std, mean * 0.04);
    description = 'Held-out normal validation μ + 2.5σ (zero test-label leakage)';
  }

  return {
    method,
    description,
    threshold,
    valNormalMean: mean,
    valNormalStd: std,
    valNormalMax: max,
    valSampleCount: n,
  };
}

/**
 * Compute threshold-dependent classification metrics strictly at the validation-derived threshold.
 */
export function computeThresholdMetrics(
  labels: number[],
  scores: number[],
  threshold: number
): {
  precision: number;
  recall: number;
  f1: number;
  confusionMatrix: ConfusionMatrix;
} {
  let tp = 0,
    fp = 0,
    tn = 0,
    fn = 0;

  for (let i = 0; i < labels.length; i++) {
    const pred = scores[i] > threshold ? 1 : 0;
    if (pred === 1 && labels[i] === 1) tp++;
    else if (pred === 1 && labels[i] === 0) fp++;
    else if (pred === 0 && labels[i] === 0) tn++;
    else fn++;
  }

  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

  return {
    precision,
    recall,
    f1,
    confusionMatrix: { tp, fp, tn, fn },
  };
}
