import React, { useMemo, useRef, useState } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { CheckCircle2, Download, FlaskConical, Play, Trash2, Upload, Zap } from 'lucide-react';
import { DATASET_REGISTRY } from '../../ml/datasets';
import { MODEL_REGISTRY, ModelMaturity } from '../../ml/models';
import { useVisionGuard } from '../../store/useVisionGuardStore';
import {
  DatasetId,
  EvaluationMetrics,
  ModelArchitecture,
  ModelConfig,
} from '../../types/visionguard';

const MATURITY_STYLES: Record<
  ModelMaturity,
  { badgeClass: string; Icon: React.ComponentType<{ className?: string }> }
> = {
  Stable: {
    badgeClass: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30',
    Icon: CheckCircle2,
  },
  Optimized: {
    badgeClass: 'bg-sky-500/10 text-sky-300 border-sky-500/30',
    Icon: Zap,
  },
  Experimental: {
    badgeClass: 'bg-amber-500/10 text-amber-300 border-amber-500/30',
    Icon: FlaskConical,
  },
};

const ModelMaturityBadge: React.FC<{ maturity: ModelMaturity; title?: string }> = ({
  maturity,
  title,
}) => {
  const { badgeClass, Icon } = MATURITY_STYLES[maturity];
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md border text-[11px] font-medium whitespace-nowrap shrink-0 ${badgeClass}`}
    >
      <Icon className="w-3 h-3 shrink-0" aria-hidden="true" />
      <span>{maturity}</span>
    </span>
  );
};

const ENGINE_ORDER: ModelArchitecture[] = ['patchcore', 'padim', 'autoencoder', 'fastflow'];

export const ModelLaboratoryPage: React.FC = () => {
  const {
    checkpoints,
    evaluations,
    busyMessage,
    trainAndEvaluateModel,
    runBaselineComparison,
    importCheckpointJson,
    deleteCheckpoint,
  } = useVisionGuard();

  const [datasetId, setDatasetId] = useState<DatasetId>('mvtec_ad');
  const [category, setCategory] = useState<string>('bottle');
  const [architecture, setArchitecture] = useState<ModelArchitecture>('patchcore');
  const [resolution, setResolution] = useState<64 | 96 | 128>(64);
  const [seed, setSeed] = useState<number>(42);
  const [coresetRatio, setCoresetRatio] = useState<number>(0.25);
  const [featureLayers, setFeatureLayers] = useState<'L1' | 'L2' | 'L1+L2' | 'L1+L2+L3'>('L1+L2');
  const [aeEpochs, setAeEpochs] = useState<number>(18);

  const [filterDataset, setFilterDataset] = useState<string>('all');
  const [filterMaturity, setFilterMaturity] = useState<'all' | ModelMaturity>('all');
  const [selectedEvalId, setSelectedEvalId] = useState<string | null>(null);
  const ckptInputRef = useRef<HTMLInputElement>(null);

  const categories = DATASET_REGISTRY[datasetId].supportedCategories;
  const activeCategory = categories.includes(category) ? category : categories[0];

  const filteredEvaluations = useMemo(
    () =>
      evaluations.filter((e) => {
        if (filterDataset !== 'all' && e.datasetId !== filterDataset) return false;
        if (filterMaturity !== 'all' && MODEL_REGISTRY[e.architecture].maturity !== filterMaturity) {
          return false;
        }
        return true;
      }),
    [evaluations, filterDataset, filterMaturity]
  );

  const selectedEval: EvaluationMetrics | null =
    filteredEvaluations.find((e) => e.id === selectedEvalId) ?? filteredEvaluations[0] ?? null;

  const selectedCheckpoint = useMemo(
    () => (selectedEval ? checkpoints.find((c) => c.id === selectedEval.checkpointId) ?? null : null),
    [checkpoints, selectedEval]
  );

  const handleTrainSingle = async () => {
    const cfg: ModelConfig = {
      ...MODEL_REGISTRY[architecture].defaultConfig,
      architecture,
      inputResolution: resolution,
      seed,
      coresetRatio,
      featureLayers,
      epochs: aeEpochs,
    };
    const { evaluation } = await trainAndEvaluateModel({
      datasetId,
      category: activeCategory,
      config: cfg,
    });
    setSelectedEvalId(evaluation.id);
  };

  const handleSaveCheckpoint = (ckptId: string) => {
    const ckpt = checkpoints.find((c) => c.id === ckptId);
    if (!ckpt) return;
    const blob = new Blob([JSON.stringify(ckpt)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${ckpt.datasetId}_${ckpt.category}_${ckpt.architecture}_ckpt.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImportCheckpoint = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => importCheckpointJson(String(reader.result));
    reader.readAsText(file);
  };

  const exportCsvTable = () => {
    const headers = [
      'dataset',
      'category',
      'provenance',
      'model',
      'maturity',
      'seed',
      'resolution',
      'image_auroc',
      'pixel_auroc',
      'aupro',
      'image_f1',
      'precision',
      'recall',
      'mean_latency_ms',
      'checkpoint_kb',
    ];
    const lines = [headers.join(',')];
    for (const e of filteredEvaluations) {
      lines.push(
        [
          e.datasetId,
          e.category,
          e.provenance,
          e.architecture,
          MODEL_REGISTRY[e.architecture].maturity,
          e.seed,
          e.inputResolution,
          e.imageAuroc.toFixed(4),
          e.pixelAuroc !== null ? e.pixelAuroc.toFixed(4) : '',
          e.aupro !== null ? e.aupro.toFixed(4) : '',
          e.imageF1.toFixed(4),
          e.precision.toFixed(4),
          e.recall.toFixed(4),
          e.meanLatencyMs.toFixed(2),
          (e.checkpointSizeBytes / 1024).toFixed(1),
        ].join(',')
      );
    }
    const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'visionguard_model_comparison.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Training & Benchmark Workbench */}
      <section className="bg-[#111827] border border-slate-800 rounded-xl p-6 space-y-5">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="font-display text-base font-semibold text-slate-100">
              Train & Evaluate Model Architectures
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Fits strictly on normal training images, calibrates threshold on held-out normal
              validation images, and evaluates on unseen test splits.
            </p>
          </div>
          <div className="flex items-center gap-2.5">
            <button
              type="button"
              onClick={() => ckptInputRef.current?.click()}
              className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium border border-slate-700 text-slate-200 hover:bg-slate-800 transition-colors whitespace-nowrap"
            >
              <Upload className="w-3.5 h-3.5" />
              Load Checkpoint JSON
            </button>
            <input
              ref={ckptInputRef}
              type="file"
              accept=".json,application/json"
              onChange={handleImportCheckpoint}
              className="hidden"
            />
            <button
              type="button"
              disabled={Boolean(busyMessage)}
              onClick={() => runBaselineComparison(datasetId, activeCategory, undefined, seed)}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg text-xs font-medium bg-slate-800 text-sky-300 border border-slate-700 hover:bg-slate-700 disabled:opacity-50 transition-colors whitespace-nowrap"
            >
              <Play className="w-3.5 h-3.5" />
              Compare All 4 Models
            </button>
          </div>
        </div>

        {/* Model Engine Catalog with Maturity Badges & Legend */}
        <div className="space-y-2.5">
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400">
            <span className="font-medium text-slate-300">Model Engine Maturity Catalog</span>
            <div className="flex flex-wrap items-center gap-3">
              <span className="inline-flex items-center gap-1.5">
                <ModelMaturityBadge maturity="Optimized" />
                <span className="text-[11px] text-slate-500">Speed &amp; recall tuned</span>
              </span>
              <span className="inline-flex items-center gap-1.5">
                <ModelMaturityBadge maturity="Stable" />
                <span className="text-[11px] text-slate-500">Validated baseline</span>
              </span>
              <span className="inline-flex items-center gap-1.5">
                <ModelMaturityBadge maturity="Experimental" />
                <span className="text-[11px] text-slate-500">Active research</span>
              </span>
            </div>
          </div>

          <div
            role="radiogroup"
            aria-label="Select anomaly detection model engine"
            className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3"
          >
          {ENGINE_ORDER.map((arch) => {
            const meta = MODEL_REGISTRY[arch];
            const isSelected = architecture === arch;
            return (
              <button
                key={arch}
                type="button"
                role="radio"
                aria-checked={isSelected}
                onClick={() => setArchitecture(arch)}
                className={`text-left p-3.5 rounded-lg border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500 ${
                  isSelected
                    ? 'bg-sky-950/35 border-sky-500/60'
                    : 'bg-[#0B0F17] border-slate-800 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="font-display text-sm font-semibold text-slate-100 truncate">
                    {meta.name}
                  </span>
                  <ModelMaturityBadge maturity={meta.maturity} title={meta.maturityRationale} />
                </div>
                <div className="text-[11px] text-slate-400 mt-1 truncate">{meta.family}</div>
                <p className="text-[11px] text-slate-500 mt-1.5 line-clamp-2 leading-relaxed">
                  {meta.maturityRationale}
                </p>
              </button>
            );
          })}
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-4 items-end">
          <div>
            <label htmlFor="lab-ds" className="block text-xs text-slate-400 mb-1.5">
              Dataset
            </label>
            <select
              id="lab-ds"
              value={datasetId}
              onChange={(e) => {
                const next = e.target.value as DatasetId;
                setDatasetId(next);
                setCategory(DATASET_REGISTRY[next].supportedCategories[0]);
              }}
              className="w-full h-10 px-3 rounded-lg bg-[#0B0F17] border border-slate-700 text-sm text-slate-100"
            >
              {(['mvtec_ad', 'visa', 'mvtec_loco', 'mvtec_ad2'] as DatasetId[]).map((id) => (
                <option key={id} value={id}>
                  {DATASET_REGISTRY[id].name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label htmlFor="lab-cat" className="block text-xs text-slate-400 mb-1.5">
              Category
            </label>
            <select
              id="lab-cat"
              value={activeCategory}
              onChange={(e) => setCategory(e.target.value)}
              className="w-full h-10 px-3 rounded-lg bg-[#0B0F17] border border-slate-700 text-sm text-slate-100"
            >
              {categories.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label htmlFor="lab-arch" className="block text-xs text-slate-400 mb-1.5">
              Algorithm
            </label>
            <select
              id="lab-arch"
              value={architecture}
              onChange={(e) => setArchitecture(e.target.value as ModelArchitecture)}
              className="w-full h-10 px-3 rounded-lg bg-[#0B0F17] border border-slate-700 text-sm text-slate-100"
            >
              {(['autoencoder', 'padim', 'patchcore', 'fastflow'] as ModelArchitecture[]).map(
                (a) => (
                  <option key={a} value={a}>
                    {MODEL_REGISTRY[a].name} ({MODEL_REGISTRY[a].maturity})
                  </option>
                )
              )}
            </select>
          </div>

          <div>
            <label htmlFor="lab-res" className="block text-xs text-slate-400 mb-1.5">
              Input Resolution
            </label>
            <select
              id="lab-res"
              value={resolution}
              onChange={(e) => setResolution(Number(e.target.value) as 64 | 96 | 128)}
              className="w-full h-10 px-3 rounded-lg bg-[#0B0F17] border border-slate-700 text-sm text-slate-100"
            >
              <option value={64}>64 × 64 px</option>
              <option value={96}>96 × 96 px</option>
              <option value={128}>128 × 128 px</option>
            </select>
          </div>

          <div>
            <label htmlFor="lab-param" className="block text-xs text-slate-400 mb-1.5">
              {architecture === 'autoencoder'
                ? 'Adam Epochs'
                : architecture === 'patchcore'
                  ? 'Coreset Ratio'
                  : 'Feature Layers'}
            </label>
            {architecture === 'autoencoder' ? (
              <input
                id="lab-param"
                type="number"
                min={6}
                max={40}
                value={aeEpochs}
                onChange={(e) => setAeEpochs(Number(e.target.value))}
                className="w-full h-10 px-3 rounded-lg bg-[#0B0F17] border border-slate-700 text-sm font-mono text-slate-100"
              />
            ) : architecture === 'patchcore' ? (
              <select
                id="lab-param"
                value={coresetRatio}
                onChange={(e) => setCoresetRatio(Number(e.target.value))}
                className="w-full h-10 px-3 rounded-lg bg-[#0B0F17] border border-slate-700 text-sm font-mono text-slate-100"
              >
                <option value={0.1}>10% Coreset</option>
                <option value={0.25}>25% Coreset</option>
                <option value={0.5}>50% Coreset</option>
                <option value={1.0}>100% Full Bank</option>
              </select>
            ) : (
              <select
                id="lab-param"
                value={featureLayers}
                onChange={(e) =>
                  setFeatureLayers(e.target.value as 'L1' | 'L2' | 'L1+L2' | 'L1+L2+L3')
                }
                className="w-full h-10 px-3 rounded-lg bg-[#0B0F17] border border-slate-700 text-sm font-mono text-slate-100"
              >
                <option value="L1">L1</option>
                <option value="L2">L2</option>
                <option value="L1+L2">L1 + L2</option>
                <option value="L1+L2+L3">L1 + L2 + L3</option>
              </select>
            )}
          </div>

          <button
            type="button"
            disabled={Boolean(busyMessage)}
            onClick={handleTrainSingle}
            className="h-10 px-4 rounded-lg text-sm font-medium bg-sky-600 text-white hover:bg-sky-500 disabled:opacity-50 transition-colors whitespace-nowrap"
          >
            Train & Evaluate
          </button>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-4 pt-3 border-t border-slate-800/80 text-xs text-slate-400">
          <div className="flex flex-wrap items-center gap-2 min-w-0">
            <ModelMaturityBadge
              maturity={MODEL_REGISTRY[architecture].maturity}
              title={MODEL_REGISTRY[architecture].maturityRationale}
            />
            <span>
              <strong className="text-slate-200 font-medium">
                {MODEL_REGISTRY[architecture].family}
              </strong>{' '}
              — {MODEL_REGISTRY[architecture].shortSummary}
            </span>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <label htmlFor="lab-seed">Random Seed:</label>
            <input
              id="lab-seed"
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
              className="w-20 h-7 px-2 rounded bg-[#0B0F17] border border-slate-700 font-mono text-xs text-slate-100"
            />
          </div>
        </div>
      </section>

      {/* Verified Model Comparison Table */}
      <section className="bg-[#111827] border border-slate-800 rounded-xl p-6 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h3 className="font-display text-base font-semibold text-slate-100">
              Verified Benchmark Comparison Table
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Click any row to inspect its ROC curve, PR curve, confusion matrix, and training curve.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2.5">
            <div
              role="group"
              aria-label="Filter evaluations by model maturity"
              className="flex items-center gap-1 p-1 bg-[#0B0F17] border border-slate-800 rounded-lg"
            >
              {(['all', 'Optimized', 'Stable', 'Experimental'] as const).map((tier) => (
                <button
                  key={tier}
                  type="button"
                  onClick={() => setFilterMaturity(tier)}
                  className={`px-2.5 py-1 rounded text-xs font-medium transition-colors whitespace-nowrap ${
                    filterMaturity === tier
                      ? 'bg-slate-800 text-sky-300'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {tier === 'all' ? 'All Tiers' : tier}
                </button>
              ))}
            </div>
            <select
              aria-label="Filter evaluations by dataset"
              value={filterDataset}
              onChange={(e) => setFilterDataset(e.target.value)}
              className="h-9 px-3 rounded-lg bg-[#0B0F17] border border-slate-700 text-xs text-slate-200"
            >
              <option value="all">All Datasets</option>
              <option value="mvtec_ad">MVTec AD</option>
              <option value="visa">VisA</option>
              <option value="mvtec_loco">MVTec LOCO AD</option>
              <option value="mvtec_ad2">MVTec AD 2</option>
            </select>
            <button
              type="button"
              disabled={filteredEvaluations.length === 0}
              onClick={exportCsvTable}
              className="inline-flex items-center gap-1.5 h-9 px-3 rounded-lg text-xs font-medium border border-slate-700 text-slate-200 hover:bg-slate-800 disabled:opacity-40 whitespace-nowrap"
            >
              <Download className="w-3.5 h-3.5" />
              Export CSV
            </button>
          </div>
        </div>

        {filteredEvaluations.length === 0 ? (
          <div className="py-12 border border-dashed border-slate-800 rounded-lg text-center text-sm text-slate-500">
            Not evaluated yet. Train a model or click &ldquo;Compare All 4 Models&rdquo; above to
            record real benchmark metrics.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th className="py-2.5 pr-3 font-medium">Dataset / Category</th>
                  <th className="py-2.5 pr-3 font-medium">Model</th>
                  <th className="py-2.5 pr-3 font-medium text-right">Image AUROC</th>
                  <th className="py-2.5 pr-3 font-medium text-right">Pixel AUROC</th>
                  <th className="py-2.5 pr-3 font-medium text-right">AUPRO</th>
                  <th className="py-2.5 pr-3 font-medium text-right">F1</th>
                  <th className="py-2.5 pr-3 font-medium text-right">Prec / Rec</th>
                  <th className="py-2.5 pr-3 font-medium text-right">Latency</th>
                  <th className="py-2.5 pr-3 font-medium text-right">Size</th>
                  <th className="py-2.5 font-medium text-right">Checkpoint</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70">
                {filteredEvaluations.map((ev) => {
                  const isSel = selectedEval?.id === ev.id;
                  return (
                    <tr
                      key={ev.id}
                      onClick={() => setSelectedEvalId(ev.id)}
                      className={`cursor-pointer transition-colors ${
                        isSel ? 'bg-sky-950/35' : 'hover:bg-slate-800/40'
                      }`}
                    >
                      <td className="py-3 pr-3 whitespace-nowrap">
                        <div className="text-slate-200 font-medium">
                          {DATASET_REGISTRY[ev.datasetId].name} · {ev.category}
                        </div>
                        <div className="text-[11px] text-slate-500">
                          {ev.provenance === 'local_real_dataset'
                            ? 'Local Real Data'
                            : 'Verification Fixture'}{' '}
                          · {ev.inputResolution}px
                        </div>
                      </td>
                      <td className="py-3 pr-3 text-slate-200 font-medium whitespace-nowrap">
                        <div className="flex items-center gap-2">
                          <span>{MODEL_REGISTRY[ev.architecture].name}</span>
                          <ModelMaturityBadge
                            maturity={MODEL_REGISTRY[ev.architecture].maturity}
                            title={MODEL_REGISTRY[ev.architecture].maturityRationale}
                          />
                        </div>
                      </td>
                      <td className="py-3 pr-3 font-mono text-right tabular-nums text-sky-300 font-medium">
                        {(ev.imageAuroc * 100).toFixed(1)}%
                      </td>
                      <td className="py-3 pr-3 font-mono text-right tabular-nums text-emerald-300">
                        {ev.pixelAuroc !== null ? `${(ev.pixelAuroc * 100).toFixed(1)}%` : 'N/A'}
                      </td>
                      <td className="py-3 pr-3 font-mono text-right tabular-nums text-slate-200">
                        {ev.aupro !== null ? `${(ev.aupro * 100).toFixed(1)}%` : 'N/A'}
                      </td>
                      <td className="py-3 pr-3 font-mono text-right tabular-nums text-slate-200">
                        {(ev.imageF1 * 100).toFixed(1)}%
                      </td>
                      <td className="py-3 pr-3 font-mono text-right tabular-nums text-slate-400">
                        {(ev.precision * 100).toFixed(0)}% / {(ev.recall * 100).toFixed(0)}%
                      </td>
                      <td className="py-3 pr-3 font-mono text-right tabular-nums text-slate-300">
                        {ev.meanLatencyMs.toFixed(1)} ms
                      </td>
                      <td className="py-3 pr-3 font-mono text-right tabular-nums text-slate-400">
                        {(ev.checkpointSizeBytes / 1024).toFixed(1)} KB
                      </td>
                      <td
                        className="py-3 text-right whitespace-nowrap"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <button
                          type="button"
                          title="Save checkpoint JSON"
                          onClick={() => handleSaveCheckpoint(ev.checkpointId)}
                          className="p-1.5 text-slate-400 hover:text-sky-400 transition-colors"
                        >
                          <Download className="w-3.5 h-3.5" />
                        </button>
                        <button
                          type="button"
                          title="Remove checkpoint from memory"
                          onClick={() => deleteCheckpoint(ev.checkpointId)}
                          className="p-1.5 text-slate-400 hover:text-rose-400 transition-colors"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Diagnostic Curves & Confusion Matrix for Selected Run */}
      {selectedEval && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* ROC Curve */}
          <section className="bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-3">
            <div>
              <div className="flex items-center justify-between gap-2">
                <h4 className="font-display text-sm font-semibold text-slate-100 truncate">
                  ROC Curve — {MODEL_REGISTRY[selectedEval.architecture].name}
                </h4>
                <ModelMaturityBadge
                  maturity={MODEL_REGISTRY[selectedEval.architecture].maturity}
                  title={MODEL_REGISTRY[selectedEval.architecture].maturityRationale}
                />
              </div>
              <p className="text-xs text-slate-400 font-mono tabular-nums">
                AUROC = {(selectedEval.imageAuroc * 100).toFixed(2)}% ({selectedEval.category})
              </p>
            </div>
            <div className="h-52">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={selectedEval.rocCurve} margin={{ top: 6, right: 8, left: -16, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                  <XAxis dataKey="x" type="number" domain={[0, 1]} stroke="#94A3B8" tick={{ fontSize: 11 }} name="FPR" />
                  <YAxis domain={[0, 1]} stroke="#94A3B8" tick={{ fontSize: 11 }} name="TPR" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', fontSize: 11 }}
                  />
                  <Line type="monotone" dataKey="y" name="TPR" stroke="#0EA5E9" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>

          {/* Precision-Recall Curve */}
          <section className="bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-3">
            <div>
              <h4 className="font-display text-sm font-semibold text-slate-100">
                Precision-Recall Curve
              </h4>
              <p className="text-xs text-slate-400 font-mono tabular-nums">
                F1 = {(selectedEval.imageF1 * 100).toFixed(1)}% at Val Threshold{' '}
                {selectedEval.thresholdPolicy.threshold.toFixed(4)}
              </p>
            </div>
            <div className="h-52">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={selectedEval.prCurve} margin={{ top: 6, right: 8, left: -16, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                  <XAxis dataKey="x" type="number" domain={[0, 1]} stroke="#94A3B8" tick={{ fontSize: 11 }} name="Recall" />
                  <YAxis domain={[0, 1]} stroke="#94A3B8" tick={{ fontSize: 11 }} name="Precision" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', fontSize: 11 }}
                  />
                  <Line type="monotone" dataKey="y" name="Precision" stroke="#10B981" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>

          {/* Confusion Matrix & Training Curve / LOCO Breakdown */}
          <section className="bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-4">
            <div>
              <h4 className="font-display text-sm font-semibold text-slate-100">
                Confusion Matrix & Policy
              </h4>
              <p className="text-xs text-slate-400 mt-0.5">
                {selectedEval.thresholdPolicy.description}
              </p>
            </div>

            <div className="grid grid-cols-2 gap-2 text-center font-mono text-xs tabular-nums">
              <div className="p-3 rounded-lg bg-[#0B0F17] border border-slate-800">
                <div className="text-slate-500 text-[11px]">True Negative (Normal)</div>
                <div className="text-lg font-semibold text-emerald-400 mt-0.5">
                  {selectedEval.confusionMatrix.tn}
                </div>
              </div>
              <div className="p-3 rounded-lg bg-[#0B0F17] border border-slate-800">
                <div className="text-slate-500 text-[11px]">False Positive</div>
                <div className="text-lg font-semibold text-amber-400 mt-0.5">
                  {selectedEval.confusionMatrix.fp}
                </div>
              </div>
              <div className="p-3 rounded-lg bg-[#0B0F17] border border-slate-800">
                <div className="text-slate-500 text-[11px]">False Negative</div>
                <div className="text-lg font-semibold text-rose-400 mt-0.5">
                  {selectedEval.confusionMatrix.fn}
                </div>
              </div>
              <div className="p-3 rounded-lg bg-[#0B0F17] border border-slate-800">
                <div className="text-slate-500 text-[11px]">True Positive (Defect)</div>
                <div className="text-lg font-semibold text-sky-400 mt-0.5">
                  {selectedEval.confusionMatrix.tp}
                </div>
              </div>
            </div>

            {selectedEval.datasetId === 'mvtec_loco' && (
              <div className="pt-2 border-t border-slate-800 text-xs space-y-1">
                <div className="flex justify-between">
                  <span className="text-slate-400">Structural Anomaly AUROC</span>
                  <span className="font-mono text-emerald-300 tabular-nums">
                    {selectedEval.structuralAuroc != null
                      ? `${(selectedEval.structuralAuroc * 100).toFixed(1)}%`
                      : 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Logical Anomaly AUROC</span>
                  <span className="font-mono text-amber-300 tabular-nums">
                    {selectedEval.logicalAuroc != null
                      ? `${(selectedEval.logicalAuroc * 100).toFixed(1)}%`
                      : 'N/A'}
                  </span>
                </div>
              </div>
            )}

            {selectedCheckpoint?.trainingCurve && selectedCheckpoint.trainingCurve.length > 0 && (
              <div className="h-28 pt-2 border-t border-slate-800">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={selectedCheckpoint.trainingCurve}
                    margin={{ top: 4, right: 6, left: -20, bottom: 0 }}
                  >
                    <XAxis dataKey="epoch" stroke="#64748B" tick={{ fontSize: 10 }} />
                    <YAxis stroke="#64748B" tick={{ fontSize: 10 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', fontSize: 11 }}
                    />
                    <Legend wrapperStyle={{ fontSize: 10 }} />
                    <Line type="monotone" dataKey="loss" name="Train Loss" stroke="#38BDF8" dot={false} />
                    <Line type="monotone" dataKey="valLoss" name="Val Normal Loss" stroke="#F59E0B" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </section>
        </div>
      )}
    </div>
  );
};
