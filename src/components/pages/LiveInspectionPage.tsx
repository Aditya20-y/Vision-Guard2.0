import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  Play,
  ScanSearch,
  ShieldAlert,
  Upload,
} from 'lucide-react';
import { DATASET_REGISTRY } from '../../ml/datasets';
import { MODEL_REGISTRY, runLiveInference } from '../../ml/models';
import { useVisionGuard } from '../../store/useVisionGuardStore';
import {
  DatasetId,
  ImageSample,
  InferenceResult,
  ModelArchitecture,
} from '../../types/visionguard';

const MAX_UPLOAD_BYTES = 10 * 1024 * 1024; // 10 MB upload safety limit (Part 18)

export const LiveInspectionPage: React.FC = () => {
  const {
    checkpoints,
    getBundle,
    trainAndEvaluateModel,
    recordInspection,
    busyMessage,
  } = useVisionGuard();

  const [datasetId, setDatasetId] = useState<DatasetId>('mvtec_ad');
  const [category, setCategory] = useState<string>('bottle');
  const [architecture, setArchitecture] = useState<ModelArchitecture>('patchcore');

  const [uploadedDataUrl, setUploadedDataUrl] = useState<string | null>(null);
  const [uploadedName, setUploadedName] = useState<string>('');
  const [selectedTestSample, setSelectedTestSample] = useState<ImageSample | null>(null);
  const [overlayAlpha, setOverlayAlpha] = useState<number>(0.55);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const bundle = useMemo(() => getBundle(datasetId, category), [getBundle, datasetId, category]);

  useEffect(() => {
    const cats = DATASET_REGISTRY[datasetId].supportedCategories;
    if (!cats.includes(category)) {
      setCategory(cats[0]);
    }
  }, [datasetId, category]);

  // Auto-select first anomalous test sample when switching category (if no manual upload)
  useEffect(() => {
    if (!uploadedDataUrl) {
      const firstAnom = bundle.test.find((s) => s.label === 1) ?? bundle.test[0] ?? null;
      setSelectedTestSample(firstAnom);
      setResult(null);
    }
  }, [bundle, uploadedDataUrl]);

  const matchingCheckpoint = useMemo(
    () =>
      checkpoints.find(
        (c) =>
          c.datasetId === datasetId &&
          c.category === category &&
          c.architecture === architecture
      ) ?? null,
    [checkpoints, datasetId, category, architecture]
  );

  const activeImageUrl = uploadedDataUrl ?? selectedTestSample?.dataUrl ?? null;
  const activeImageName = uploadedDataUrl
    ? uploadedName
    : selectedTestSample?.fileName ?? 'No image selected';

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    setLocalError(null);
    const file = e.target.files?.[0];
    if (!file) return;

    if (!/\.(png|jpe?g)$/i.test(file.name) && !['image/png', 'image/jpeg'].includes(file.type)) {
      setLocalError('Unsupported file format. Please upload a PNG, JPG, or JPEG image.');
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      setLocalError('Image exceeds the 10 MB upload safety limit.');
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      setUploadedDataUrl(String(reader.result));
      setUploadedName(file.name);
      setSelectedTestSample(null);
      setResult(null);
    };
    reader.onerror = () => setLocalError('Failed to read uploaded image file.');
    reader.readAsDataURL(file);
  };

  const handleTrainSelected = async () => {
    setLocalError(null);
    try {
      await trainAndEvaluateModel({
        datasetId,
        category,
        config: { ...MODEL_REGISTRY[architecture].defaultConfig },
      });
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : 'Training failed.');
    }
  };

  const handleRunInspection = async () => {
    setLocalError(null);
    if (!matchingCheckpoint) {
      setLocalError(
        `No trained ${MODEL_REGISTRY[architecture].name} checkpoint exists for ${DATASET_REGISTRY[datasetId].name} / ${category}. Train the model first — VisionGuard never fabricates inference outputs.`
      );
      return;
    }
    if (!activeImageUrl) {
      setLocalError('Upload an image or select a test sample to inspect.');
      return;
    }

    setRunning(true);
    try {
      const inf = await runLiveInference({
        dataUrl: activeImageUrl,
        imageName: activeImageName,
        checkpoint: matchingCheckpoint,
        groundTruthMaskUrl: uploadedDataUrl ? undefined : selectedTestSample?.maskDataUrl,
        overlayAlpha,
      });
      setResult(inf);
      recordInspection({
        id: inf.id,
        timestamp: inf.timestamp,
        architecture: inf.architecture,
        datasetId: inf.datasetId,
        category: inf.category,
        imageReference: inf.imageName,
        prediction: inf.prediction,
        rawAnomalyScore: inf.rawAnomalyScore,
        threshold: inf.threshold,
        latencyMs: inf.latencyMs,
        qualityStatus: inf.imageQuality.overallStatus,
        qualityWarningCount: inf.imageQuality.warnings.length,
      });
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : 'Inference failed.');
    } finally {
      setRunning(false);
    }
  };

  const handleDownloadReport = () => {
    if (!result) return;
    const reportPayload = {
      reportType: 'VisionGuard Visual Inspection Report',
      generatedAt: new Date().toISOString(),
      inspectionId: result.id,
      model: {
        architecture: result.architecture,
        name: MODEL_REGISTRY[result.architecture].name,
        checkpointId: result.checkpointId,
      },
      context: {
        dataset: DATASET_REGISTRY[result.datasetId].name,
        category: result.category,
        imageReference: result.imageName,
        preprocessing: result.preprocessingMeta,
      },
      decision: {
        prediction: result.prediction,
        rawAnomalyScore: Number(result.rawAnomalyScore.toFixed(6)),
        threshold: Number(result.threshold.toFixed(6)),
        thresholdPolicy: result.thresholdPolicyLabel,
        note: 'Raw anomaly score is a continuous distance/error magnitude, not a calibrated probability.',
      },
      performance: {
        measuredLatencyMs: Number(result.latencyMs.toFixed(2)),
      },
      imageQualityAssessment: result.imageQuality,
    };

    const blob = new Blob([JSON.stringify(reportPayload, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `visionguard_inspection_${result.category}_${result.architecture}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Inspection Configuration Bar */}
      <section className="bg-[#111827] border border-slate-800 rounded-xl p-5">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
          <div>
            <label htmlFor="insp-dataset" className="block text-xs font-medium text-slate-400 mb-1.5">
              Dataset Context
            </label>
            <select
              id="insp-dataset"
              value={datasetId}
              onChange={(e) => {
                setDatasetId(e.target.value as DatasetId);
                setUploadedDataUrl(null);
              }}
              className="w-full h-10 px-3 rounded-lg bg-[#0B0F17] border border-slate-700 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-sky-500"
            >
              {(['mvtec_ad', 'visa', 'mvtec_loco', 'mvtec_ad2'] as DatasetId[]).map((id) => (
                <option key={id} value={id}>
                  {DATASET_REGISTRY[id].name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label htmlFor="insp-category" className="block text-xs font-medium text-slate-400 mb-1.5">
              Product Category
            </label>
            <select
              id="insp-category"
              value={category}
              onChange={(e) => {
                setCategory(e.target.value);
                setUploadedDataUrl(null);
              }}
              className="w-full h-10 px-3 rounded-lg bg-[#0B0F17] border border-slate-700 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-sky-500"
            >
              {DATASET_REGISTRY[datasetId].supportedCategories.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label htmlFor="insp-model" className="block text-xs font-medium text-slate-400 mb-1.5">
              Model Architecture
            </label>
            <select
              id="insp-model"
              value={architecture}
              onChange={(e) => setArchitecture(e.target.value as ModelArchitecture)}
              className="w-full h-10 px-3 rounded-lg bg-[#0B0F17] border border-slate-700 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-sky-500"
            >
              {(['patchcore', 'padim', 'autoencoder', 'fastflow'] as ModelArchitecture[]).map(
                (arch) => (
                  <option key={arch} value={arch}>
                    {MODEL_REGISTRY[arch].name}
                  </option>
                )
              )}
            </select>
          </div>

          <div className="flex items-center gap-2">
            {!matchingCheckpoint ? (
              <button
                type="button"
                disabled={Boolean(busyMessage)}
                onClick={handleTrainSelected}
                className="w-full h-10 inline-flex items-center justify-center gap-2 px-4 rounded-lg text-sm font-medium bg-amber-600 text-white hover:bg-amber-500 disabled:opacity-50 transition-colors whitespace-nowrap"
              >
                <Play className="w-4 h-4" />
                Train Model First
              </button>
            ) : (
              <button
                type="button"
                disabled={running || !activeImageUrl}
                onClick={handleRunInspection}
                className="w-full h-10 inline-flex items-center justify-center gap-2 px-4 rounded-lg text-sm font-medium bg-sky-600 text-white hover:bg-sky-500 disabled:opacity-50 transition-colors whitespace-nowrap"
              >
                <ScanSearch className="w-4 h-4" />
                {running ? 'Running Inference...' : 'Run Live Inspection'}
              </button>
            )}
          </div>
        </div>

        {/* Checkpoint availability message */}
        <div className="mt-4 pt-3.5 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-4 text-xs">
          <div className="text-slate-400">
            Checkpoint Status:{' '}
            {matchingCheckpoint ? (
              <span className="text-emerald-400 font-medium">
                Ready ({matchingCheckpoint.trainSampleCount} normal train images · Threshold{' '}
                <span className="font-mono tabular-nums">
                  {matchingCheckpoint.thresholdPolicy.threshold.toFixed(4)}
                </span>
                )
              </span>
            ) : (
              <span className="text-amber-400 font-medium">
                Model not trained for {category} — click &ldquo;Train Model First&rdquo; to fit on
                normal images
              </span>
            )}
          </div>
          <div className="flex items-center gap-3 text-slate-400">
            <label htmlFor="overlay-alpha" className="whitespace-nowrap">
              Overlay Intensity ({Math.round(overlayAlpha * 100)}%)
            </label>
            <input
              id="overlay-alpha"
              type="range"
              min={0.2}
              max={0.85}
              step={0.05}
              value={overlayAlpha}
              onChange={(e) => setOverlayAlpha(Number(e.target.value))}
              className="w-28 accent-sky-500"
            />
          </div>
        </div>
      </section>

      {localError && (
        <div className="bg-rose-950/50 border border-rose-800 rounded-xl px-4 py-3 text-sm text-rose-200 flex items-center gap-2.5">
          <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
          <span>{localError}</span>
        </div>
      )}

      {/* Main Split: Input Selection + Visual Inspection Viewport */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
        {/* Left column: Upload & Unseen Test Sample Picker */}
        <section className="xl:col-span-4 bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-5">
          <div className="flex items-center justify-between gap-2">
            <h2 className="font-display text-base font-semibold text-slate-100">
              Inspection Input
            </h2>
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border border-slate-700 text-slate-200 hover:bg-slate-800 transition-colors whitespace-nowrap"
            >
              <Upload className="w-3.5 h-3.5" />
              Upload PNG / JPG
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".png,.jpg,.jpeg,image/png,image/jpeg"
              onChange={handleFileUpload}
              className="hidden"
            />
          </div>

          {activeImageUrl ? (
            <div className="space-y-2.5">
              <div className="aspect-square w-full bg-[#0B0F17] border border-slate-800 rounded-lg overflow-hidden flex items-center justify-center">
                <img
                  src={activeImageUrl}
                  alt={`Input preview for ${activeImageName}`}
                  referrerPolicy="no-referrer"
                  className="w-full h-full object-contain"
                />
              </div>
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span className="font-mono truncate max-w-[210px]">{activeImageName}</span>
                <span>{uploadedDataUrl ? 'User Upload' : 'Unseen Test Split'}</span>
              </div>
            </div>
          ) : (
            <div className="aspect-square w-full border border-dashed border-slate-700 rounded-lg flex items-center justify-center text-xs text-slate-500">
              Select a test sample or upload an image
            </div>
          )}

          {/* Unseen Test Sample Strip */}
          <div className="space-y-2 pt-3 border-t border-slate-800">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span>Unseen {category} Test Samples</span>
              <span className="font-mono tabular-nums">{bundle.test.length} samples</span>
            </div>
            <div className="grid grid-cols-5 gap-2 max-h-44 overflow-y-auto pr-1">
              {bundle.test.map((sample) => {
                const isSelected = !uploadedDataUrl && selectedTestSample?.id === sample.id;
                return (
                  <button
                    key={sample.id}
                    type="button"
                    title={`${sample.fileName} (${sample.label === 1 ? sample.defectType : 'normal'})`}
                    onClick={() => {
                      setUploadedDataUrl(null);
                      setSelectedTestSample(sample);
                      setResult(null);
                    }}
                    className={`relative aspect-square rounded-md overflow-hidden border transition-colors ${
                      isSelected
                        ? 'border-sky-400 ring-1 ring-sky-400'
                        : 'border-slate-800 hover:border-slate-600'
                    }`}
                  >
                    <img
                      src={sample.dataUrl}
                      alt={sample.fileName}
                      referrerPolicy="no-referrer"
                      className="w-full h-full object-cover"
                    />
                    <span
                      className={`absolute bottom-0 inset-x-0 text-[9px] font-mono text-center leading-tight py-0.5 ${
                        sample.label === 1
                          ? 'bg-rose-950/85 text-rose-200'
                          : 'bg-emerald-950/85 text-emerald-200'
                      }`}
                    >
                      {sample.label === 1 ? 'DEF' : 'OK'}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        </section>

        {/* Right column: Real Inference Outputs, Heatmaps, Quality Warning */}
        <section className="xl:col-span-8 bg-[#111827] border border-slate-800 rounded-xl p-6 space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <h2 className="font-display text-base font-semibold text-slate-100">
                Anomaly Localization & Decision Output
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Pixel-level score map, blended localization overlay, and decoupled image-quality audit
              </p>
            </div>
            <button
              type="button"
              disabled={!result}
              onClick={handleDownloadReport}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg text-xs font-medium border border-slate-700 text-slate-200 hover:bg-slate-800 disabled:opacity-40 transition-colors whitespace-nowrap"
            >
              <Download className="w-3.5 h-3.5" />
              Download Inspection Report
            </button>
          </div>

          {!result ? (
            <div className="py-16 border border-dashed border-slate-800 rounded-lg text-center space-y-2 px-6">
              <p className="text-sm font-medium text-slate-300">
                Ready for Live Inference
              </p>
              <p className="text-xs text-slate-500 max-w-md mx-auto">
                {matchingCheckpoint
                  ? 'Click "Run Live Inspection" to execute the trained model, compute raw anomaly scores, localize pixel-level defects, and evaluate input image quality.'
                  : 'Train the selected model first, then click "Run Live Inspection" to run real inference without simulation.'}
              </p>
            </div>
          ) : (
            <>
              {/* Decision & Measured Metrics Strip */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pb-5 border-b border-slate-800">
                <div>
                  <div className="text-xs text-slate-400">Prediction</div>
                  <div className="mt-1 flex items-center gap-2">
                    {result.prediction === 'Normal' ? (
                      <>
                        <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0" />
                        <span className="font-display text-lg font-bold text-emerald-400">
                          Normal
                        </span>
                      </>
                    ) : (
                      <>
                        <ShieldAlert className="w-5 h-5 text-rose-400 shrink-0" />
                        <span className="font-display text-lg font-bold text-rose-400">
                          Anomalous
                        </span>
                      </>
                    )}
                  </div>
                </div>

                <div>
                  <div className="text-xs text-slate-400">Raw Anomaly Score</div>
                  <div className="mt-1 font-mono text-lg font-semibold text-slate-100 tabular-nums">
                    {result.rawAnomalyScore.toFixed(4)}
                  </div>
                  <div className="text-[11px] text-slate-500">Not a probability</div>
                </div>

                <div>
                  <div className="text-xs text-slate-400">Validation Threshold</div>
                  <div className="mt-1 font-mono text-lg font-semibold text-slate-100 tabular-nums">
                    {result.threshold.toFixed(4)}
                  </div>
                  <div className="text-[11px] text-slate-500 tabular-nums">
                    Ratio: {result.scoreRatioToThreshold.toFixed(2)}×
                  </div>
                </div>

                <div>
                  <div className="text-xs text-slate-400">Measured Latency</div>
                  <div className="mt-1 font-mono text-lg font-semibold text-slate-100 tabular-nums">
                    {result.latencyMs.toFixed(1)} ms
                  </div>
                  <div className="text-[11px] text-slate-500">
                    {result.preprocessingMeta.resizedTo}
                  </div>
                </div>
              </div>

              {/* Independent Image Quality Warning Panel (Part 10) */}
              <div
                className={`rounded-lg border p-4 text-xs space-y-2 ${
                  result.imageQuality.overallStatus === 'acceptable'
                    ? 'bg-slate-900/60 border-slate-800 text-slate-300'
                    : 'bg-amber-950/35 border-amber-800/70 text-amber-100'
                }`}
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="font-medium flex items-center gap-2">
                    {result.imageQuality.overallStatus !== 'acceptable' && (
                      <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
                    )}
                    <span>
                      Image Quality Assessment (Independent of Defect Prediction):{' '}
                      <strong className="uppercase">{result.imageQuality.overallStatus}</strong>
                    </span>
                  </div>
                  <div className="font-mono text-[11px] text-slate-400 tabular-nums">
                    Laplacian Var {result.imageQuality.blurVariance.toFixed(1)} · Lum{' '}
                    {result.imageQuality.meanLuminance.toFixed(0)} · Contrast{' '}
                    {result.imageQuality.rmsContrast.toFixed(1)} · Blockiness{' '}
                    {result.imageQuality.blockinessScore.toFixed(2)}
                  </div>
                </div>
                {result.imageQuality.warnings.length > 0 ? (
                  <div className="space-y-1 text-amber-200/90">
                    <div className="font-medium">
                      Image quality may affect inspection reliability:
                    </div>
                    <ul className="list-disc pl-5 space-y-0.5">
                      {result.imageQuality.warnings.map((w, i) => (
                        <li key={i}>{w}</li>
                      ))}
                    </ul>
                  </div>
                ) : (
                  <p className="text-slate-400">
                    No excessive blur, illumination drift, extreme contrast, low resolution, or
                    severe JPEG blockiness detected.
                  </p>
                )}
              </div>

              {/* Visual Output Grid: Original / Heatmap / Overlay / Reconstruction or GT Mask */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="space-y-2">
                  <div className="aspect-square rounded-lg overflow-hidden border border-slate-800 bg-[#0B0F17]">
                    <img
                      src={result.originalDataUrl}
                      alt="Preprocessed input"
                      referrerPolicy="no-referrer"
                      className="w-full h-full object-contain"
                    />
                  </div>
                  <div className="text-xs font-medium text-slate-300">1. Input ({result.inputResolution}px)</div>
                </div>

                <div className="space-y-2">
                  <div className="aspect-square rounded-lg overflow-hidden border border-slate-800 bg-[#0B0F17]">
                    <img
                      src={result.heatmapDataUrl}
                      alt="Pixel-level anomaly heatmap"
                      referrerPolicy="no-referrer"
                      className="w-full h-full object-contain"
                    />
                  </div>
                  <div className="text-xs font-medium text-slate-300">2. Anomaly Heatmap</div>
                </div>

                <div className="space-y-2">
                  <div className="aspect-square rounded-lg overflow-hidden border border-slate-800 bg-[#0B0F17]">
                    <img
                      src={result.overlayDataUrl}
                      alt="Anomaly heatmap overlay"
                      referrerPolicy="no-referrer"
                      className="w-full h-full object-contain"
                    />
                  </div>
                  <div className="text-xs font-medium text-slate-300">3. Localization Overlay</div>
                </div>

                <div className="space-y-2">
                  <div className="aspect-square rounded-lg overflow-hidden border border-slate-800 bg-[#0B0F17] flex items-center justify-center">
                    {result.reconstructionDataUrl ? (
                      <img
                        src={result.reconstructionDataUrl}
                        alt="Autoencoder reconstruction"
                        referrerPolicy="no-referrer"
                        className="w-full h-full object-contain"
                      />
                    ) : result.groundTruthMaskUrl ? (
                      <img
                        src={result.groundTruthMaskUrl}
                        alt="Ground truth defect mask"
                        referrerPolicy="no-referrer"
                        className="w-full h-full object-contain"
                      />
                    ) : (
                      <span className="text-xs text-slate-500 text-center px-3">
                        Reconstruction available for Autoencoder; GT mask available for test split
                      </span>
                    )}
                  </div>
                  <div className="text-xs font-medium text-slate-300">
                    {result.reconstructionDataUrl
                      ? '4. AE Reconstruction'
                      : result.groundTruthMaskUrl
                        ? '4. Ground-Truth Mask'
                        : '4. Reference Mask'}
                  </div>
                </div>
              </div>

              <div className="text-xs text-slate-500 pt-2 border-t border-slate-800/80 flex flex-wrap justify-between gap-2">
                <span>Threshold Policy: {result.thresholdPolicyLabel}</span>
                <span>Preprocessing: {result.preprocessingMeta.normalization}</span>
              </div>
            </>
          )}
        </section>
      </div>
    </div>
  );
};
