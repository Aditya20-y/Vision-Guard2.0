/**
 * Image processing, tensor conversion, mask resizing (nearest-neighbor),
 * Gaussian smoothing, perceptual colormap heatmap rendering, and connected components.
 */

export interface RGBTensor {
  width: number;
  height: number;
  /** Length = width * height * 3, values normalized in [0, 1] */
  data: Float32Array;
}

export interface GrayTensor {
  width: number;
  height: number;
  /** Length = width * height */
  data: Float32Array;
}

export function loadImageElement(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Failed to decode image data.'));
    img.src = src;
  });
}

/**
 * Decode any PNG/JPG/JPEG DataURL into a normalized RGB Float32Array tensor [0, 1]
 * using consistent bilinear resizing.
 */
export async function dataUrlToRGBTensor(
  dataUrl: string,
  targetSize: number
): Promise<{ tensor: RGBTensor; originalWidth: number; originalHeight: number }> {
  const img = await loadImageElement(dataUrl);
  const canvas = document.createElement('canvas');
  canvas.width = targetSize;
  canvas.height = targetSize;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('2D canvas context unavailable.');
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(img, 0, 0, targetSize, targetSize);
  const imgData = ctx.getImageData(0, 0, targetSize, targetSize).data;

  const data = new Float32Array(targetSize * targetSize * 3);
  for (let i = 0, p = 0; i < imgData.length; i += 4, p += 3) {
    data[p] = imgData[i] / 255;
    data[p + 1] = imgData[i + 1] / 255;
    data[p + 2] = imgData[i + 2] / 255;
  }

  return {
    tensor: { width: targetSize, height: targetSize, data },
    originalWidth: img.naturalWidth || img.width,
    originalHeight: img.naturalHeight || img.height,
  };
}

/**
 * Decode a ground-truth binary mask using strict nearest-neighbor interpolation
 * so binary label boundaries are never blurred or interpolated into soft values.
 */
export async function dataUrlToBinaryMask(
  maskDataUrl: string | undefined,
  targetSize: number
): Promise<Uint8Array> {
  const mask = new Uint8Array(targetSize * targetSize);
  if (!maskDataUrl) return mask;

  const img = await loadImageElement(maskDataUrl);
  const canvas = document.createElement('canvas');
  canvas.width = targetSize;
  canvas.height = targetSize;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return mask;
  // Nearest-neighbor mask resizing
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(img, 0, 0, targetSize, targetSize);
  const imgData = ctx.getImageData(0, 0, targetSize, targetSize).data;

  for (let i = 0, p = 0; i < imgData.length; i += 4, p++) {
    const lum = 0.299 * imgData[i] + 0.587 * imgData[i + 1] + 0.114 * imgData[i + 2];
    mask[p] = lum >= 127 ? 1 : 0;
  }
  return mask;
}

export function rgbTensorToDataUrl(tensor: RGBTensor, displaySize = 256): string {
  const srcCanvas = document.createElement('canvas');
  srcCanvas.width = tensor.width;
  srcCanvas.height = tensor.height;
  const sctx = srcCanvas.getContext('2d');
  if (!sctx) return '';
  const imgData = sctx.createImageData(tensor.width, tensor.height);
  for (let p = 0, i = 0; p < tensor.data.length; p += 3, i += 4) {
    imgData.data[i] = Math.max(0, Math.min(255, Math.round(tensor.data[p] * 255)));
    imgData.data[i + 1] = Math.max(0, Math.min(255, Math.round(tensor.data[p + 1] * 255)));
    imgData.data[i + 2] = Math.max(0, Math.min(255, Math.round(tensor.data[p + 2] * 255)));
    imgData.data[i + 3] = 255;
  }
  sctx.putImageData(imgData, 0, 0);

  if (displaySize === tensor.width) return srcCanvas.toDataURL('image/png');
  const outCanvas = document.createElement('canvas');
  outCanvas.width = displaySize;
  outCanvas.height = displaySize;
  const octx = outCanvas.getContext('2d');
  if (!octx) return srcCanvas.toDataURL('image/png');
  octx.imageSmoothingEnabled = true;
  octx.drawImage(srcCanvas, 0, 0, displaySize, displaySize);
  return outCanvas.toDataURL('image/png');
}

/**
 * Separable 2D Gaussian blur for anomaly score map smoothing (sigma in pixels).
 */
export function gaussianBlur2D(
  map: Float32Array,
  width: number,
  height: number,
  sigma: number
): Float32Array {
  if (sigma <= 0.1) return map.slice();
  const radius = Math.max(1, Math.ceil(sigma * 2.5));
  const kSize = radius * 2 + 1;
  const kernel = new Float32Array(kSize);
  let sum = 0;
  const inv2s2 = 1 / (2 * sigma * sigma);
  for (let i = -radius; i <= radius; i++) {
    const val = Math.exp(-i * i * inv2s2);
    kernel[i + radius] = val;
    sum += val;
  }
  for (let i = 0; i < kSize; i++) kernel[i] /= sum;

  const temp = new Float32Array(width * height);
  // Horizontal pass
  for (let y = 0; y < height; y++) {
    const rowOffset = y * width;
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        const nx = Math.max(0, Math.min(width - 1, x + k));
        acc += map[rowOffset + nx] * kernel[k + radius];
      }
      temp[rowOffset + x] = acc;
    }
  }

  const out = new Float32Array(width * height);
  // Vertical pass
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        const ny = Math.max(0, Math.min(height - 1, y + k));
        acc += temp[ny * width + x] * kernel[k + radius];
      }
      out[y * width + x] = acc;
    }
  }
  return out;
}

/**
 * Bilinear upsampling of a low-resolution patch anomaly grid to image resolution.
 */
export function bilinearUpsample(
  grid: Float32Array,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number
): Float32Array {
  const out = new Float32Array(dstW * dstH);
  for (let y = 0; y < dstH; y++) {
    const gy = ((y + 0.5) / dstH) * srcH - 0.5;
    const y0 = Math.max(0, Math.min(srcH - 1, Math.floor(gy)));
    const y1 = Math.max(0, Math.min(srcH - 1, y0 + 1));
    const wy = Math.max(0, Math.min(1, gy - y0));

    for (let x = 0; x < dstW; x++) {
      const gx = ((x + 0.5) / dstW) * srcW - 0.5;
      const x0 = Math.max(0, Math.min(srcW - 1, Math.floor(gx)));
      const x1 = Math.max(0, Math.min(srcW - 1, x0 + 1));
      const wx = Math.max(0, Math.min(1, gx - x0));

      const v00 = grid[y0 * srcW + x0];
      const v10 = grid[y0 * srcW + x1];
      const v01 = grid[y1 * srcW + x0];
      const v11 = grid[y1 * srcW + x1];

      const top = v00 * (1 - wx) + v10 * wx;
      const bot = v01 * (1 - wx) + v11 * wx;
      out[y * dstW + x] = top * (1 - wy) + bot * wy;
    }
  }
  return out;
}

/**
 * Perceptually uniform Inferno-inspired industrial thermal colormap.
 * Input t in [0, 1] -> [r, g, b] in [0..255].
 */
export function infernoColor(t: number): [number, number, number] {
  const v = Math.max(0, Math.min(1, t));
  const stops: [number, number, number, number][] = [
    [0.0, 12, 16, 36],
    [0.22, 56, 26, 108],
    [0.45, 147, 38, 103],
    [0.68, 221, 81, 58],
    [0.86, 251, 164, 10],
    [1.0, 252, 255, 164],
  ];
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, r0, g0, b0] = stops[i];
    const [t1, r1, g1, b1] = stops[i + 1];
    if (v >= t0 && v <= t1) {
      const f = (v - t0) / (t1 - t0 || 1);
      return [
        Math.round(r0 + (r1 - r0) * f),
        Math.round(g0 + (g1 - g0) * f),
        Math.round(b0 + (b1 - b0) * f),
      ];
    }
  }
  return [252, 255, 164];
}

/**
 * Render standalone anomaly heatmap and blended overlay over the original RGB tensor.
 * Normalization uses both the model's threshold anchor and map dynamic range so
 * normal images do not artificially saturate bright yellow!
 */
export function renderHeatmapAndOverlay(
  original: RGBTensor,
  anomalyMap: Float32Array,
  referenceScale: number,
  displaySize = 256,
  overlayAlpha = 0.52
): { heatmapDataUrl: string; overlayDataUrl: string } {
  const { width, height } = original;
  let minV = Infinity;
  let maxV = -Infinity;
  for (let i = 0; i < anomalyMap.length; i++) {
    if (anomalyMap[i] < minV) minV = anomalyMap[i];
    if (anomalyMap[i] > maxV) maxV = anomalyMap[i];
  }
  // Anchor upper scale to at least 1.35 * referenceScale so clean normal samples stay dark/cool
  const denom = Math.max(1e-6, Math.max(maxV, referenceScale * 1.35) - minV);

  const heatCanvas = document.createElement('canvas');
  heatCanvas.width = width;
  heatCanvas.height = height;
  const hctx = heatCanvas.getContext('2d')!;
  const hImg = hctx.createImageData(width, height);

  const overCanvas = document.createElement('canvas');
  overCanvas.width = width;
  overCanvas.height = height;
  const octx = overCanvas.getContext('2d')!;
  const oImg = octx.createImageData(width, height);

  for (let i = 0; i < width * height; i++) {
    const norm = Math.max(0, Math.min(1, (anomalyMap[i] - minV) / denom));
    const [hr, hg, hb] = infernoColor(norm);
    const idx = i * 4;
    hImg.data[idx] = hr;
    hImg.data[idx + 1] = hg;
    hImg.data[idx + 2] = hb;
    hImg.data[idx + 3] = 255;

    const or = Math.round(original.data[i * 3] * 255);
    const og = Math.round(original.data[i * 3 + 1] * 255);
    const ob = Math.round(original.data[i * 3 + 2] * 255);

    // Weight alpha by normalized anomaly intensity so background remains crisp
    const localAlpha = overlayAlpha * Math.min(1, norm * 1.35);
    oImg.data[idx] = Math.round(or * (1 - localAlpha) + hr * localAlpha);
    oImg.data[idx + 1] = Math.round(og * (1 - localAlpha) + hg * localAlpha);
    oImg.data[idx + 2] = Math.round(ob * (1 - localAlpha) + hb * localAlpha);
    oImg.data[idx + 3] = 255;
  }

  hctx.putImageData(hImg, 0, 0);
  octx.putImageData(oImg, 0, 0);

  const upscale = (src: HTMLCanvasElement): string => {
    if (displaySize === width) return src.toDataURL('image/png');
    const c = document.createElement('canvas');
    c.width = displaySize;
    c.height = displaySize;
    const ctx = c.getContext('2d')!;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(src, 0, 0, displaySize, displaySize);
    return c.toDataURL('image/png');
  };

  return {
    heatmapDataUrl: upscale(heatCanvas),
    overlayDataUrl: upscale(overCanvas),
  };
}

/**
 * 8-connected components labeling on a binary mask (used for real AUPRO computation).
 */
export function extractConnectedComponents(
  mask: Uint8Array,
  width: number,
  height: number
): number[][] {
  const visited = new Uint8Array(width * height);
  const components: number[][] = [];
  const stack: number[] = [];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (mask[idx] === 1 && !visited[idx]) {
        const comp: number[] = [];
        visited[idx] = 1;
        stack.push(idx);
        while (stack.length > 0) {
          const cur = stack.pop()!;
          comp.push(cur);
          const cy = Math.floor(cur / width);
          const cx = cur % width;
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              if (dx === 0 && dy === 0) continue;
              const nx = cx + dx;
              const ny = cy + dy;
              if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                const nIdx = ny * width + nx;
                if (mask[nIdx] === 1 && !visited[nIdx]) {
                  visited[nIdx] = 1;
                  stack.push(nIdx);
                }
              }
            }
          }
        }
        if (comp.length >= 2) components.push(comp);
      }
    }
  }
  return components;
}
