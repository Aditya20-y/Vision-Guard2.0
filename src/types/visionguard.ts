export type DatasetId = 'mvtec_ad' | 'visa' | 'mvtec_loco' | 'mvtec_ad2' | 'btad';

export type ModelArchitecture = 'autoencoder' | 'padim' | 'patchcore' | 'fastflow';

export type DataProvenance = 'synthetic_verification' | 'local_real_dataset';

export type SplitType = 'train' | 'val' | 'test';

export type AnomalyKind = 'normal' | 'structural' | 'logical' | 'shift';

export interface DatasetMetadata {
  id: DatasetId;
  name: string;
  role: string;
  officialUrl: string;
  docsUrl?: string;
  license: string;
  totalOfficialImages: string;
  officialCategories: string[];
  supportedCategories: string[];
  description: string;
  citation: string;
  folderStructureNote: string;
  acquisitionNote: string;
  status: 'ready_synthetic_fixture' | 'awaiting_local_import' | 'local_verified';
}

export interface ImageSample {
  id: string;
  datasetId: DatasetId;
  category: string;
  split: SplitType;
  label: 0 | 1; // 0 = normal, 1 = anomalous
  defectType: string;
  anomalyKind: AnomalyKind;
  width: number;
  height: number;
  dataUrl: string;
  maskDataUrl?: string;
  provenance: DataProvenance;
  fileName: string;
}

export interface DatasetValidationReport {
  datasetId: DatasetId;
  category: string;
  provenance: DataProvenance;
  validatedAt: string;
  trainNormalCount: number;
  valNormalCount: number;
  testNormalCount: number;
  testAnomalousCount: number;
  masksPresent: number;
  masksAligned: boolean;
  readableImages: number;
  corruptedImages: number;
  missingAnnotations: number;
  splitLeakageCheck: 'passed' | 'failed';
  notes: string[];
}

export interface ModelConfig {
  architecture: ModelArchitecture;
  inputResolution: 64 | 96 | 128;
  seed: number;
  // Autoencoder specific
  epochs?: number;
  learningRate?: number;
  latentDim?: number;
  // PaDiM specific
  regularizationEps?: number;
  // PatchCore specific
  coresetRatio?: number;
  featureLayers?: 'L1' | 'L2' | 'L1+L2' | 'L1+L2+L3';
  kNeighbors?: number;
  // FastFlow specific
  flowSteps?: number;
}

export interface ThresholdPolicy {
  method: 'val_normal_p95' | 'val_normal_mean_plus_2_5_std';
  description: string;
  threshold: number;
  valNormalMean: number;
  valNormalStd: number;
  valNormalMax: number;
  valSampleCount: number;
}

export interface TrainingLossPoint {
  epoch: number;
  loss: number;
  valLoss: number;
}

export interface TrainedCheckpoint {
  id: string;
  datasetId: DatasetId;
  category: string;
  provenance: DataProvenance;
  architecture: ModelArchitecture;
  config: ModelConfig;
  trainedAt: string;
  trainSampleCount: number;
  trainFraction: number;
  trainingDurationMs: number;
  parameterCount: number;
  checkpointSizeBytes: number;
  thresholdPolicy: ThresholdPolicy;
  trainingCurve?: TrainingLossPoint[];
  // Serialized model weights / statistics / memory bank
  artifact: Record<string, unknown>;
}

export interface CurvePoint {
  x: number;
  y: number;
  threshold?: number;
}

export interface ConfusionMatrix {
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

export interface EvaluationMetrics {
  id: string;
  checkpointId: string;
  datasetId: DatasetId;
  category: string;
  provenance: DataProvenance;
  architecture: ModelArchitecture;
  evaluatedAt: string;
  trainFraction: number;
  seed: number;
  inputResolution: number;
  imageAuroc: number;
  pixelAuroc: number | null;
  aupro: number | null;
  imageF1: number;
  precision: number;
  recall: number;
  confusionMatrix: ConfusionMatrix;
  thresholdPolicy: ThresholdPolicy;
  meanLatencyMs: number;
  p95LatencyMs: number;
  parameterCount: number;
  checkpointSizeBytes: number;
  testNormalCount: number;
  testAnomalousCount: number;
  rocCurve: CurvePoint[];
  prCurve: CurvePoint[];
  structuralAuroc?: number | null;
  logicalAuroc?: number | null;
  notes?: string;
}

export interface ImageQualityReport {
  overallStatus: 'acceptable' | 'caution' | 'poor';
  warnings: string[];
  blurVariance: number;
  meanLuminance: number;
  rmsContrast: number;
  resolution: { width: number; height: number };
  blockinessScore: number;
  checks: {
    blur: 'ok' | 'warning';
    illumination: 'ok' | 'warning';
    contrast: 'ok' | 'warning';
    resolution: 'ok' | 'warning';
    compression: 'ok' | 'warning';
  };
}

export interface InferenceResult {
  id: string;
  timestamp: string;
  checkpointId: string;
  architecture: ModelArchitecture;
  datasetId: DatasetId;
  category: string;
  imageName: string;
  prediction: 'Normal' | 'Anomalous';
  rawAnomalyScore: number;
  threshold: number;
  thresholdPolicyLabel: string;
  scoreRatioToThreshold: number;
  latencyMs: number;
  inputResolution: number;
  originalDataUrl: string;
  heatmapDataUrl: string;
  overlayDataUrl: string;
  reconstructionDataUrl?: string;
  groundTruthMaskUrl?: string;
  imageQuality: ImageQualityReport;
  preprocessingMeta: {
    resizedTo: string;
    normalization: string;
    interpolation: string;
  };
}

export interface InspectionHistoryRecord {
  id: string;
  timestamp: string;
  architecture: ModelArchitecture;
  datasetId: DatasetId;
  category: string;
  imageReference: string;
  prediction: 'Normal' | 'Anomalous';
  rawAnomalyScore: number;
  threshold: number;
  latencyMs: number;
  qualityStatus: 'acceptable' | 'caution' | 'poor';
  qualityWarningCount: number;
}

export type CorruptionType =
  | 'gaussian_noise'
  | 'gaussian_blur'
  | 'brightness_shift'
  | 'contrast_shift'
  | 'jpeg_compression'
  | 'resize_resample';

export interface DataEfficiencyPoint {
  fraction: number;
  trainSamples: number;
  architecture: ModelArchitecture;
  imageAuroc: number;
  pixelAuroc: number | null;
  imageF1: number;
  trainingDurationMs: number;
}

export interface RobustnessResultRow {
  corruption: CorruptionType;
  corruptionLabel: string;
  severity: number;
  architecture: ModelArchitecture;
  cleanImageAuroc: number;
  corruptedImageAuroc: number;
  aurocDrop: number;
  cleanPixelAuroc: number | null;
  corruptedPixelAuroc: number | null;
}

export interface PatchCoreAblationRow {
  experimentGroup: 'coreset_ratio' | 'feature_layers' | 'input_resolution';
  settingLabel: string;
  coresetRatio: number;
  featureLayers: 'L1' | 'L2' | 'L1+L2' | 'L1+L2+L3';
  inputResolution: 64 | 96 | 128;
  memoryBankVectors: number;
  checkpointSizeBytes: number;
  imageAuroc: number;
  pixelAuroc: number | null;
  meanLatencyMs: number;
}

export interface CategoryDifficultyRow {
  datasetId: DatasetId;
  category: string;
  provenance: DataProvenance;
  architecture: ModelArchitecture;
  imageAuroc: number;
  pixelAuroc: number | null;
  defectAreaFraction: number;
  meanContrastDelta: number;
  difficultyReason: string;
}

export type DashboardPageId =
  | 'overview'
  | 'live_inspection'
  | 'model_lab'
  | 'dataset_explorer'
  | 'experiments'
  | 'history'
  | 'methodology';
