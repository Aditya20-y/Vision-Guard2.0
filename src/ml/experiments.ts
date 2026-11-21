import {
  CategoryDifficultyRow,
  CorruptionType,
  DataEfficiencyPoint,
  DatasetId,
  EvaluationMetrics,
  ModelArchitecture,
  ModelConfig,
  PatchCoreAblationRow,
  RobustnessResultRow,
  TrainedCheckpoint,
} from '../types/visionguard';
import { applyCorruption, CORRUPTION_METADATA } from './corruptions';
import { CategoryDatasetBundle, getCategoryDatasetBundle } from './datasets';
import { dataUrlToBinaryMask, dataUrlToRGBTensor } from './imageOps';
import {
  computeAupro,
  computeAurocAndRoc,
  computePixelAuroc,
  computePrCurve,
  computeThresholdMetrics,
} from './metrics';
import { MODEL_REGISTRY, predictTensorWithCheckpoint, trainAnomalyModel } from './models';

/**
 * Evaluate a trained checkpoint on a dataset bundle's test split.
 * Computes exact Image AUROC, Pixel AUROC, AUPRO, F1, Precision, Recall,
 * ROC curve, PR curve, Confusion Matrix, and real measured inference latency.
 */
export async function evaluateCheckpoint(
  checkpoint: TrainedCheckpoint,
  bundle: CategoryDatasetBundle
): Promise<EvaluationMetrics> {
  const res = checkpoint.config.inputResolution;
  const testSamples = bundle.test;

  const labels: number[] = [];
  const scores: number[] = [];
  const masks: Uint8Array[] = [];
  const anomalyMaps: Float32Array[] = [];
  const latencies: number[] = [];

  const structuralLabels: number[] = [];
  const structuralScores: number[] = [];
  const logicalLabels: number[] = [];
  const logicalScores: number[] = [];

  for (const sample of testSamples) {
    const { tensor } = await dataUrlToRGBTensor(sample.dataUrl, res);
    const t0 = performance.now();
    const pred = predictTensorWithCheckpoint(tensor, checkpoint);
    latencies.push(performance.now() - t0);

    labels.push(sample.label);
    scores.push(pred.imageScore);

    if (sample.label === 0 || sample.maskDataUrl) {
      const mask = await dataUrlToBinaryMask(sample.maskDataUrl, res);
      masks.push(mask);
      anomalyMaps.push(pred.anomalyMap);
    }

    if (bundle.datasetId === 'mvtec_loco') {
      if (sample.label === 0) {
        structuralLabels.push(0);
        structuralScores.push(pred.imageScore);
        logicalLabels.push(0);
        logicalScores.push(pred.imageScore);
      } else if (sample.anomalyKind === 'logical') {
        logicalLabels.push(1);
        logicalScores.push(pred.imageScore);
      } else {
        structuralLabels.push(1);
        structuralScores.push(pred.imageScore);
      }
    }
  }

  const { auroc: imageAuroc, rocCurve } = computeAurocAndRoc(labels, scores);
  const prCurve = computePrCurve(labels, scores);
  const pixelAuroc = computePixelAuroc(masks, anomalyMaps);
  const aupro = computeAupro(masks, anomalyMaps, res, res, 0.3);
  const { precision, recall, f1, confusionMatrix } = computeThresholdMetrics(
    labels,
    scores,
    checkpoint.thresholdPolicy.threshold
  );

  const sortedLat = latencies.slice().sort((a, b) => a - b);
  const meanLatencyMs =
    sortedLat.reduce((a, b) => a + b, 0) / Math.max(1, sortedLat.length);
  const p95LatencyMs =
    sortedLat[Math.min(sortedLat.length - 1, Math.floor(sortedLat.length * 0.95))] ??
    meanLatencyMs;

  let structuralAuroc: number | null | undefined;
  let logicalAuroc: number | null | undefined;
  if (bundle.datasetId === 'mvtec_loco') {
    structuralAuroc = structuralLabels.includes(1)
      ? computeAurocAndRoc(structuralLabels, structuralScores).auroc
      : null;
    logicalAuroc = logicalLabels.includes(1)
      ? computeAurocAndRoc(logicalLabels, logicalScores).auroc
      : null;
  }

  return {
    id: `eval_${checkpoint.id}_${Date.now()}`,
    checkpointId: checkpoint.id,
    datasetId: checkpoint.datasetId,
    category: checkpoint.category,
    provenance: checkpoint.provenance,
    architecture: checkpoint.architecture,
    evaluatedAt: new Date().toISOString(),
    trainFraction: checkpoint.trainFraction,
    seed: checkpoint.config.seed,
    inputResolution: res,
    imageAuroc,
    pixelAuroc,
    aupro,
    imageF1: f1,
    precision,
    recall,
    confusionMatrix,
    thresholdPolicy: checkpoint.thresholdPolicy,
    meanLatencyMs,
    p95LatencyMs,
    parameterCount: checkpoint.parameterCount,
    checkpointSizeBytes: checkpoint.checkpointSizeBytes,
    testNormalCount: testSamples.filter((s) => s.label === 0).length,
    testAnomalousCount: testSamples.filter((s) => s.label === 1).length,
    rocCurve,
    prCurve,
    structuralAuroc,
    logicalAuroc,
  };
}

/**
 * EXPERIMENT B — Data Efficiency (10%, 25%, 50%, 100% normal training fractions).
 */
export async function runDataEfficiencyExperiment(params: {
  datasetId: DatasetId;
  category: string;
  architectures: ModelArchitecture[];
  seed?: number;
  onProgress?: (msg: string) => void;
}): Promise<DataEfficiencyPoint[]> {
  const { datasetId, category, architectures, seed = 42, onProgress } = params;
  const bundle = getCategoryDatasetBundle(datasetId, category);
  const fractions = [0.1, 0.25, 0.5, 1.0];
  const results: DataEfficiencyPoint[] = [];

  for (const arch of architectures) {
    for (const fraction of fractions) {
      onProgress?.(`Data Efficiency: ${MODEL_REGISTRY[arch].name} @ ${Math.round(fraction * 100)}% normal data`);
      await new Promise((r) => setTimeout(r, 10));

      const baseCfg: ModelConfig = {
        ...MODEL_REGISTRY[arch].defaultConfig,
        seed,
        inputResolution: 64,
      };
      const ckpt = await trainAnomalyModel({
        datasetId,
        category,
        provenance: bundle.provenance,
        trainSamples: bundle.train,
        valSamples: bundle.val,
        config: baseCfg,
        trainFraction: fraction,
      });
      const metrics = await evaluateCheckpoint(ckpt, bundle);
      results.push({
        fraction,
        trainSamples: ckpt.trainSampleCount,
        architecture: arch,
        imageAuroc: metrics.imageAuroc,
        pixelAuroc: metrics.pixelAuroc,
        imageF1: metrics.imageF1,
        trainingDurationMs: ckpt.trainingDurationMs,
      });
    }
  }
  return results;
}

/**
 * EXPERIMENT C — Robustness Under Image Degradation.
 * Evaluates trained checkpoints on clean vs corrupted test splits across severities 1..5.
 * Ground-truth masks are never corrupted.
 */
export async function runRobustnessExperiment(params: {
  checkpoints: TrainedCheckpoint[];
  datasetId: DatasetId;
  category: string;
  corruptions?: CorruptionType[];
  severities?: number[];
  seed?: number;
  onProgress?: (msg: string) => void;
}): Promise<RobustnessResultRow[]> {
  const {
    checkpoints,
    datasetId,
    category,
    corruptions = [
      'gaussian_noise',
      'gaussian_blur',
      'brightness_shift',
      'contrast_shift',
      'jpeg_compression',
      'resize_resample',
    ],
    severities = [1, 3, 5],
    seed = 42,
    onProgress,
  } = params;

  const bundle = getCategoryDatasetBundle(datasetId, category);
  const rows: RobustnessResultRow[] = [];

  for (const ckpt of checkpoints) {
    const res = ckpt.config.inputResolution;
    // Clean baseline
    const cleanEval = await evaluateCheckpoint(ckpt, bundle);

    // Pre-decode clean test tensors and untouched GT masks
    const cleanTensors = await Promise.all(
      bundle.test.map(async (s) => (await dataUrlToRGBTensor(s.dataUrl, res)).tensor)
    );
    const masks = await Promise.all(
      bundle.test.map(async (s) => dataUrlToBinaryMask(s.maskDataUrl, res))
    );
    const labels = bundle.test.map((s) => s.label);

    for (const corr of corruptions) {
      for (const sev of severities) {
        onProgress?.(
          `Robustness: ${MODEL_REGISTRY[ckpt.architecture].name} — ${CORRUPTION_METADATA[corr].label} (Severity ${sev})`
        );
        await new Promise((r) => setTimeout(r, 8));

        const corrScores: number[] = [];
        const corrMaps: Float32Array[] = [];
        for (let i = 0; i < cleanTensors.length; i++) {
          const degraded = await applyCorruption(cleanTensors[i], corr, sev, seed + i * 13);
          const pred = predictTensorWithCheckpoint(degraded, ckpt);
          corrScores.push(pred.imageScore);
          corrMaps.push(pred.anomalyMap);
        }

        const { auroc: corrImgAuroc } = computeAurocAndRoc(labels, corrScores);
        const corrPixAuroc = computePixelAuroc(masks, corrMaps);

        rows.push({
          corruption: corr,
          corruptionLabel: CORRUPTION_METADATA[corr].label,
          severity: sev,
          architecture: ckpt.architecture,
          cleanImageAuroc: cleanEval.imageAuroc,
          corruptedImageAuroc: corrImgAuroc,
          aurocDrop: cleanEval.imageAuroc - corrImgAuroc,
          cleanPixelAuroc: cleanEval.pixelAuroc,
          corruptedPixelAuroc: corrPixAuroc,
        });
      }
    }
  }
  return rows;
}

/**
 * EXPERIMENT D — PatchCore Ablations (Coreset Ratio, Feature Layers, Input Resolution).
 */
export async function runPatchCoreAblations(params: {
  datasetId: DatasetId;
  category: string;
  seed?: number;
  onProgress?: (msg: string) => void;
}): Promise<PatchCoreAblationRow[]> {
  const { datasetId, category, seed = 42, onProgress } = params;
  const bundle = getCategoryDatasetBundle(datasetId, category);
  const rows: PatchCoreAblationRow[] = [];

  const configs: {
    group: PatchCoreAblationRow['experimentGroup'];
    label: string;
    coresetRatio: number;
    featureLayers: 'L1' | 'L2' | 'L1+L2' | 'L1+L2+L3';
    inputResolution: 64 | 96 | 128;
  }[] = [
    // Coreset ratio ablation
    { group: 'coreset_ratio', label: 'Coreset 10%', coresetRatio: 0.1, featureLayers: 'L1+L2', inputResolution: 64 },
    { group: 'coreset_ratio', label: 'Coreset 25%', coresetRatio: 0.25, featureLayers: 'L1+L2', inputResolution: 64 },
    { group: 'coreset_ratio', label: 'Coreset 50%', coresetRatio: 0.5, featureLayers: 'L1+L2', inputResolution: 64 },
    { group: 'coreset_ratio', label: 'Coreset 100%', coresetRatio: 1.0, featureLayers: 'L1+L2', inputResolution: 64 },
    // Feature hierarchy ablation
    { group: 'feature_layers', label: 'Layer L1 Only', coresetRatio: 0.25, featureLayers: 'L1', inputResolution: 64 },
    { group: 'feature_layers', label: 'Layer L2 Only', coresetRatio: 0.25, featureLayers: 'L2', inputResolution: 64 },
    { group: 'feature_layers', label: 'Layers L1+L2', coresetRatio: 0.25, featureLayers: 'L1+L2', inputResolution: 64 },
    { group: 'feature_layers', label: 'Layers L1+L2+L3', coresetRatio: 0.25, featureLayers: 'L1+L2+L3', inputResolution: 64 },
    // Resolution ablation
    { group: 'input_resolution', label: '64 × 64 px', coresetRatio: 0.25, featureLayers: 'L1+L2', inputResolution: 64 },
    { group: 'input_resolution', label: '96 × 96 px', coresetRatio: 0.25, featureLayers: 'L1+L2', inputResolution: 96 },
  ];

  for (const cfg of configs) {
    onProgress?.(`PatchCore Ablation: ${cfg.label}`);
    await new Promise((r) => setTimeout(r, 10));

    const ckpt = await trainAnomalyModel({
      datasetId,
      category,
      provenance: bundle.provenance,
      trainSamples: bundle.train,
      valSamples: bundle.val,
      config: {
        architecture: 'patchcore',
        inputResolution: cfg.inputResolution,
        seed,
        coresetRatio: cfg.coresetRatio,
        featureLayers: cfg.featureLayers,
        kNeighbors: 3,
      },
    });
    const metrics = await evaluateCheckpoint(ckpt, bundle);
    const mbCount = Number((ckpt.artifact as Record<string, unknown>).memoryBankCount ?? 0);

    rows.push({
      experimentGroup: cfg.group,
      settingLabel: cfg.label,
      coresetRatio: cfg.coresetRatio,
      featureLayers: cfg.featureLayers,
      inputResolution: cfg.inputResolution,
      memoryBankVectors: mbCount,
      checkpointSizeBytes: ckpt.checkpointSizeBytes,
      imageAuroc: metrics.imageAuroc,
      pixelAuroc: metrics.pixelAuroc,
      meanLatencyMs: metrics.meanLatencyMs,
    });
  }

  return rows;
}

/**
 * EXPERIMENT F — Category Difficulty Analysis across multiple categories & datasets.
 */
export async function runCategoryDifficultyAnalysis(params: {
  targets: { datasetId: DatasetId; category: string }[];
  architecture: ModelArchitecture;
  seed?: number;
  onProgress?: (msg: string) => void;
}): Promise<CategoryDifficultyRow[]> {
  const { targets, architecture, seed = 42, onProgress } = params;
  const rows: CategoryDifficultyRow[] = [];

  for (const t of targets) {
    onProgress?.(`Category Difficulty: ${t.datasetId} / ${t.category} (${MODEL_REGISTRY[architecture].name})`);
    await new Promise((r) => setTimeout(r, 10));

    const bundle = getCategoryDatasetBundle(t.datasetId, t.category);
    const ckpt = await trainAnomalyModel({
      datasetId: t.datasetId,
      category: t.category,
      provenance: bundle.provenance,
      trainSamples: bundle.train,
      valSamples: bundle.val,
      config: { ...MODEL_REGISTRY[architecture].defaultConfig, seed, inputResolution: 64 },
    });
    const metrics = await evaluateCheckpoint(ckpt, bundle);

    // Measure actual average defect mask area fraction & contrast delta
    let totalMaskRatio = 0;
    let anomCount = 0;
    let totalContrastDelta = 0;
    for (const s of bundle.test) {
      if (s.label === 1 && s.maskDataUrl) {
        const mask = await dataUrlToBinaryMask(s.maskDataUrl, 64);
        const { tensor } = await dataUrlToRGBTensor(s.dataUrl, 64);
        let posPix = 0;
        let posLum = 0;
        let bgLum = 0;
        let bgPix = 0;
        for (let i = 0; i < mask.length; i++) {
          const lum =
            0.299 * tensor.data[i * 3] +
            0.587 * tensor.data[i * 3 + 1] +
            0.114 * tensor.data[i * 3 + 2];
          if (mask[i] === 1) {
            posPix++;
            posLum += lum;
          } else {
            bgPix++;
            bgLum += lum;
          }
        }
        totalMaskRatio += posPix / mask.length;
        if (posPix > 0 && bgPix > 0) {
          totalContrastDelta += Math.abs(posLum / posPix - bgLum / bgPix);
        }
        anomCount++;
      }
    }

    const defectAreaFraction = anomCount > 0 ? totalMaskRatio / anomCount : 0;
    const meanContrastDelta = anomCount > 0 ? totalContrastDelta / anomCount : 0;

    let difficultyReason = 'Distinct structural appearance contrast';
    if (t.datasetId === 'mvtec_loco') {
      difficultyReason =
        'Logical constraint violations (missing/extra components without local texture break)';
    } else if (t.datasetId === 'mvtec_ad2') {
      difficultyReason = 'Illumination & contrast shift across public evaluation split';
    } else if (defectAreaFraction < 0.025) {
      difficultyReason = 'Small fine-grained defect area (<2.5% of image)';
    } else if (meanContrastDelta < 0.14) {
      difficultyReason = 'Low defect-to-background luminance contrast';
    }

    rows.push({
      datasetId: t.datasetId,
      category: t.category,
      provenance: bundle.provenance,
      architecture,
      imageAuroc: metrics.imageAuroc,
      pixelAuroc: metrics.pixelAuroc,
      defectAreaFraction,
      meanContrastDelta,
      difficultyReason,
    });
  }

  return rows.sort((a, b) => a.imageAuroc - b.imageAuroc);
}
