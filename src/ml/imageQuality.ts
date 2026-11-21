import { ImageQualityReport } from '../types/visionguard';
import { loadImageElement } from './imageOps';

/**
 * Independent Industrial Image Quality Assessment (Part 10).
 * Strictly decoupled from anomaly classification — never mistakes a blurry
 * or poorly illuminated image for a physical product defect.
 * Does not claim calibrated quality probabilities.
 */
export async function assessImageQuality(dataUrl: string): Promise<ImageQualityReport> {
  const img = await loadImageElement(dataUrl);
  const origW = img.naturalWidth || img.width;
  const origH = img.naturalHeight || img.height;

  // Analyze at up to 128x128 for deterministic speed while inspecting actual pixels
  const w = Math.min(128, Math.max(32, origW));
  const h = Math.min(128, Math.max(32, origH));
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) {
    throw new Error('Canvas context unavailable for image quality assessment.');
  }
  ctx.drawImage(img, 0, 0, w, h);
  const rgba = ctx.getImageData(0, 0, w, h).data;

  const gray = new Float32Array(w * h);
  let sumLum = 0;
  for (let i = 0, p = 0; i < rgba.length; i += 4, p++) {
    const lum = 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
    gray[p] = lum;
    sumLum += lum;
  }
  const meanLuminance = sumLum / (w * h);

  // RMS Contrast
  let sumSqDiff = 0;
  for (let i = 0; i < gray.length; i++) {
    const d = gray[i] - meanLuminance;
    sumSqDiff += d * d;
  }
  const rmsContrast = Math.sqrt(sumSqDiff / gray.length);

  // Variance of Laplacian (4-neighbor) for sharpness / blur estimation
  let lapSum = 0;
  let lapSqSum = 0;
  let lapCount = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = y * w + x;
      const lap =
        gray[idx - 1] +
        gray[idx + 1] +
        gray[idx - w] +
        gray[idx + w] -
        4 * gray[idx];
      lapSum += lap;
      lapSqSum += lap * lap;
      lapCount++;
    }
  }
  const lapMean = lapSum / Math.max(1, lapCount);
  const blurVariance = lapSqSum / Math.max(1, lapCount) - lapMean * lapMean;

  // 8x8 block boundary gradient ratio to detect severe JPEG blocking artifacts
  let boundaryDiff = 0;
  let boundaryCnt = 0;
  let interiorDiff = 0;
  let interiorCnt = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const d = Math.abs(gray[y * w + x] - gray[y * w + x - 1]);
      if (x % 8 === 0) {
        boundaryDiff += d;
        boundaryCnt++;
      } else {
        interiorDiff += d;
        interiorCnt++;
      }
    }
  }
  const avgBoundary = boundaryDiff / Math.max(1, boundaryCnt);
  const avgInterior = interiorDiff / Math.max(1, interiorCnt);
  const blockinessScore = avgInterior > 0.5 ? avgBoundary / avgInterior : 1.0;

  const warnings: string[] = [];
  const checks: ImageQualityReport['checks'] = {
    blur: 'ok',
    illumination: 'ok',
    contrast: 'ok',
    resolution: 'ok',
    compression: 'ok',
  };

  if (blurVariance < 32) {
    checks.blur = 'warning';
    warnings.push(
      `Excessive blur detected (Laplacian variance ${blurVariance.toFixed(1)} < 32). Fine surface defects may be obscured.`
    );
  }
  if (meanLuminance < 35) {
    checks.illumination = 'warning';
    warnings.push(
      `Poor illumination / underexposure detected (mean luminance ${meanLuminance.toFixed(1)}/255).`
    );
  } else if (meanLuminance > 225) {
    checks.illumination = 'warning';
    warnings.push(
      `Extreme brightness / overexposure detected (mean luminance ${meanLuminance.toFixed(1)}/255).`
    );
  }
  if (rmsContrast < 14) {
    checks.contrast = 'warning';
    warnings.push(
      `Extremely low contrast detected (RMS contrast ${rmsContrast.toFixed(1)}). Subtle defects may blend into background.`
    );
  } else if (rmsContrast > 105) {
    checks.contrast = 'warning';
    warnings.push(
      `Extreme contrast detected (RMS contrast ${rmsContrast.toFixed(1)}). Clipping may distort anomaly scores.`
    );
  }
  if (origW < 96 || origH < 96) {
    checks.resolution = 'warning';
    warnings.push(
      `Low input resolution (${origW}×${origH}px). Recommended minimum is 128×128px for pixel localization.`
    );
  }
  if (blockinessScore > 1.42) {
    checks.compression = 'warning';
    warnings.push(
      `Severe JPEG compression block artifacts detected (blockiness ratio ${blockinessScore.toFixed(2)}).`
    );
  }

  const overallStatus: ImageQualityReport['overallStatus'] =
    warnings.length === 0 ? 'acceptable' : warnings.length === 1 ? 'caution' : 'poor';

  return {
    overallStatus,
    warnings,
    blurVariance,
    meanLuminance,
    rmsContrast,
    resolution: { width: origW, height: origH },
    blockinessScore,
    checks,
  };
}
