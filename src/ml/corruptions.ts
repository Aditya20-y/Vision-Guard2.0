import { CorruptionType } from '../types/visionguard';
import { RGBTensor, gaussianBlur2D, rgbTensorToDataUrl, dataUrlToRGBTensor } from './imageOps';
import { SeededRNG } from './prng';

export const CORRUPTION_METADATA: Record<
  CorruptionType,
  { label: string; description: string; unit: string }
> = {
  gaussian_noise: {
    label: 'Gaussian Noise',
    description: 'Additive zero-mean sensor noise applied to RGB pixel intensities.',
    unit: 'σ (intensity)',
  },
  gaussian_blur: {
    label: 'Gaussian Blur',
    description: 'Defocus optical blur via separable Gaussian kernel.',
    unit: 'σ (px)',
  },
  brightness_shift: {
    label: 'Brightness Variation',
    description: 'Additive illumination shift simulating factory lighting drift.',
    unit: 'Δ luminance',
  },
  contrast_shift: {
    label: 'Contrast Variation',
    description: 'Multiplicative contrast compression around mid-gray.',
    unit: 'contrast scale',
  },
  jpeg_compression: {
    label: 'JPEG Compression',
    description: 'Lossy DCT re-encoding at reduced quality factors.',
    unit: 'JPEG quality',
  },
  resize_resample: {
    label: 'Resize & Resample',
    description: 'Downsampling followed by bilinear upsampling back to inspection resolution.',
    unit: 'scale factor',
  },
};

/**
 * Deterministic reproducible image degradation suite (Part 7 — Experiment C).
 * Corrupts ONLY the input image tensor; ground-truth masks are never modified.
 * Severity ranges from 1 (mild) to 5 (severe).
 */
export async function applyCorruption(
  tensor: RGBTensor,
  type: CorruptionType,
  severity: number,
  seed: number
): Promise<RGBTensor> {
  const sev = Math.max(1, Math.min(5, Math.round(severity)));
  const { width, height, data } = tensor;
  const out = new Float32Array(data.length);
  const rng = new SeededRNG(seed + sev * 997);

  if (type === 'gaussian_noise') {
    const std = [0.03, 0.06, 0.1, 0.15, 0.22][sev - 1];
    for (let i = 0; i < data.length; i++) {
      out[i] = Math.max(0, Math.min(1, data[i] + rng.gaussian(0, std)));
    }
    return { width, height, data: out };
  }

  if (type === 'gaussian_blur') {
    const sigma = [0.6, 1.1, 1.7, 2.4, 3.2][sev - 1];
    for (let c = 0; c < 3; c++) {
      const ch = new Float32Array(width * height);
      for (let i = 0; i < width * height; i++) ch[i] = data[i * 3 + c];
      const blurred = gaussianBlur2D(ch, width, height, sigma);
      for (let i = 0; i < width * height; i++) out[i * 3 + c] = blurred[i];
    }
    return { width, height, data: out };
  }

  if (type === 'brightness_shift') {
    const delta = [0.08, 0.15, -0.2, 0.26, -0.34][sev - 1];
    for (let i = 0; i < data.length; i++) {
      out[i] = Math.max(0, Math.min(1, data[i] + delta));
    }
    return { width, height, data: out };
  }

  if (type === 'contrast_shift') {
    const factor = [0.82, 0.68, 0.54, 0.42, 0.3][sev - 1];
    for (let i = 0; i < data.length; i++) {
      out[i] = Math.max(0, Math.min(1, 0.5 + (data[i] - 0.5) * factor));
    }
    return { width, height, data: out };
  }

  if (type === 'jpeg_compression') {
    const quality = [0.65, 0.42, 0.25, 0.14, 0.07][sev - 1];
    const pngUrl = rgbTensorToDataUrl(tensor, width);
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d')!;
    const img = new Image();
    await new Promise<void>((res, rej) => {
      img.onload = () => res();
      img.onerror = () => rej();
      img.src = pngUrl;
    });
    ctx.drawImage(img, 0, 0, width, height);
    const jpegUrl = canvas.toDataURL('image/jpeg', quality);
    const decoded = await dataUrlToRGBTensor(jpegUrl, width);
    return decoded.tensor;
  }

  // resize_resample
  const scale = [0.75, 0.6, 0.48, 0.35, 0.25][sev - 1];
  const smallSize = Math.max(8, Math.round(width * scale));
  const pngUrl = rgbTensorToDataUrl(tensor, width);
  const down = await dataUrlToRGBTensor(pngUrl, smallSize);
  const downUrl = rgbTensorToDataUrl(down.tensor, smallSize);
  const up = await dataUrlToRGBTensor(downUrl, width);
  return up.tensor;
}
