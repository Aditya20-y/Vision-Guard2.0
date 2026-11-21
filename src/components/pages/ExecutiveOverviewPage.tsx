import React from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { ArrowRight, CheckCircle2, Cpu, Play, ShieldAlert } from 'lucide-react';
import { DATASET_REGISTRY } from '../../ml/datasets';
import { MODEL_REGISTRY } from '../../ml/models';
import { useVisionGuard } from '../../store/useVisionGuardStore';
import { DashboardPageId, DatasetId, ModelArchitecture } from '../../types/visionguard';

interface Props {
  onNavigate: (page: DashboardPageId) => void;
}

export const ExecutiveOverviewPage: React.FC<Props> = ({ onNavigate }) => {
  const {
    checkpoints,
    evaluations,
    history,
    dataEfficiencyResults,
    robustnessResults,
    ablationResults,
    difficultyResults,
    importedBundles,
    busyMessage,
    runBaselineComparison,
  } = useVisionGuard();

  const completedExperimentCount =
    (evaluations.length > 0 ? 1 : 0) +
    (dataEfficiencyResults.length > 0 ? 1 : 0) +
    (robustnessResults.length > 0 ? 1 : 0) +
    (ablationResults.length > 0 ? 1 : 0) +
    (difficultyResults.length > 0 ? 1 : 0);

  // Latest verified evaluation per architecture
  const latestByArch: Partial<Record<ModelArchitecture, (typeof evaluations)[number]>> = {};
  for (const ev of evaluations) {
    if (!latestByArch[ev.architecture]) {
      latestByArch[ev.architecture] = ev;
    }
  }

  const comparisonChartData = (['autoencoder', 'padim', 'patchcore', 'fastflow'] as ModelArchitecture[])
    .map((arch) => {
      const ev = latestByArch[arch];
      if (!ev) return null;
      return {
        model: MODEL_REGISTRY[arch].name,
        imageAuroc: Number((ev.imageAuroc * 100).toFixed(1)),
        pixelAuroc: ev.pixelAuroc !== null ? Number((ev.pixelAuroc * 100).toFixed(1)) : 0,
        f1: Number((ev.imageF1 * 100).toFixed(1)),
      };
    })
    .filter((d): d is NonNullable<typeof d> => d !== null);

  const realImportedCount = Object.keys(importedBundles).length;
  const bestEval =
    evaluations.length > 0
      ? evaluations.reduce((best, cur) => (cur.imageAuroc > best.imageAuroc ? cur : best), evaluations[0])
      : null;

  return (
    <div className="space-y-8">
      {/* Research Brief & Action Anchor */}
      <section className="bg-[#111827] border border-slate-800 rounded-xl p-6 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
        <div className="space-y-2.5 max-w-3xl">
          <div className="text-xs text-slate-400 flex flex-wrap items-center gap-2">
            <span>Industrial Anomaly Detection Platform</span>
            <span aria-hidden="true">·</span>
            <span>Normal-Only Training Protocol</span>
            <span aria-hidden="true">·</span>
            <span>Zero Fabricated Metrics</span>
          </div>
          <h2 className="font-display text-2xl font-bold text-slate-50 tracking-tight">
            Systematic Defect Detection, Pixel Localization, and Robustness Benchmarking
          </h2>
          <p className="text-sm text-slate-300 leading-relaxed">
            Primary Research Question: How effectively do reconstruction-based (Autoencoder) and
            feature-based (PaDiM, PatchCore, FastFlow) anomaly detection methods identify and localize
            unseen industrial defects across categories and datasets, and how does their performance
            change under limited normal training data and realistic image-quality degradation?
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 shrink-0">
          <button
            type="button"
            disabled={Boolean(busyMessage)}
            onClick={() => runBaselineComparison('mvtec_ad', 'bottle')}
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium bg-sky-600 text-white hover:bg-sky-500 disabled:opacity-50 transition-colors whitespace-nowrap"
          >
            <Play className="w-4 h-4" />
            Train & Evaluate 4 Models
          </button>
          <button
            type="button"
            onClick={() => onNavigate('live_inspection')}
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium border border-slate-700 text-slate-200 hover:bg-slate-800 transition-colors whitespace-nowrap"
          >
            Open Live Inspection
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </section>

      {/* Primary KPI Strip */}
      <section className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-1.5">
          <div className="text-xs text-slate-400">Trained Checkpoints</div>
          <div className="font-mono text-2xl font-semibold text-slate-50 tabular-nums">
            {checkpoints.length}
          </div>
          <div className="text-xs text-slate-500">
            {checkpoints.length === 0
              ? 'No models trained in current session'
              : `Across ${new Set(checkpoints.map((c) => c.category)).size} category(s)`}
          </div>
        </div>

        <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-1.5">
          <div className="text-xs text-slate-400">Highest Verified Image AUROC</div>
          <div className="font-mono text-2xl font-semibold text-slate-50 tabular-nums">
            {bestEval ? `${(bestEval.imageAuroc * 100).toFixed(1)}%` : 'Not evaluated'}
          </div>
          <div className="text-xs text-slate-500">
            {bestEval
              ? `${MODEL_REGISTRY[bestEval.architecture].name} · ${bestEval.category}`
              : 'Run Model Laboratory to compute'}
          </div>
        </div>

        <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-1.5">
          <div className="text-xs text-slate-400">Research Experiment Suites</div>
          <div className="font-mono text-2xl font-semibold text-slate-50 tabular-nums">
            {completedExperimentCount} / 5
          </div>
          <div className="text-xs text-slate-500">
            Comparison · Efficiency · Robustness · Ablation · Difficulty
          </div>
        </div>

        <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-1.5">
          <div className="text-xs text-slate-400">Dataset Adapters Ready</div>
          <div className="font-mono text-2xl font-semibold text-slate-50 tabular-nums">
            5
          </div>
          <div className="text-xs text-slate-500">
            {realImportedCount > 0
              ? `${realImportedCount} local real dataset folder(s) verified`
              : 'Verification fixtures ready · Local import supported'}
          </div>
        </div>
      </section>

      {/* Model Comparison Overview & Dataset Status */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-stretch">
        {/* Left: Verified Model Comparison Chart */}
        <section className="xl:col-span-7 bg-[#111827] border border-slate-800 rounded-xl p-6 flex flex-col justify-between">
          <div className="flex items-center justify-between gap-4 mb-5">
            <div>
              <h3 className="font-display text-base font-semibold text-slate-100">
                Latest Verified Model Performance
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Measured strictly on unseen test splits with normal-validation-calibrated thresholds
              </p>
            </div>
            <button
              type="button"
              onClick={() => onNavigate('model_lab')}
              className="text-xs font-medium text-sky-400 hover:text-sky-300 whitespace-nowrap"
            >
              Open Model Lab
            </button>
          </div>

          {comparisonChartData.length === 0 ? (
            <div className="h-64 border border-dashed border-slate-800 rounded-lg flex flex-col items-center justify-center text-center p-6">
              <p className="text-sm font-medium text-slate-300">Not evaluated yet</p>
              <p className="text-xs text-slate-500 max-w-md mt-1.5">
                VisionGuard never displays fabricated benchmark numbers. Click &ldquo;Train &amp;
                Evaluate 4 Models&rdquo; above to fit real Autoencoder, PaDiM, PatchCore, and
                FastFlow models and measure actual AUROC, Pixel AUROC, and F1 scores.
              </p>
            </div>
          ) : (
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={comparisonChartData} margin={{ top: 8, right: 12, left: -10, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                  <XAxis dataKey="model" stroke="#94A3B8" tick={{ fontSize: 12 }} />
                  <YAxis domain={[0, 100]} stroke="#94A3B8" tick={{ fontSize: 12 }} unit="%" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0F172A',
                      borderColor: '#334155',
                      borderRadius: 8,
                      fontSize: 12,
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="imageAuroc" name="Image AUROC (%)" fill="#0EA5E9" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="pixelAuroc" name="Pixel AUROC (%)" fill="#10B981" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="f1" name="Image F1 (%)" fill="#F59E0B" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </section>

        {/* Right: Dataset Availability & Provenance */}
        <section className="xl:col-span-5 bg-[#111827] border border-slate-800 rounded-xl p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between gap-4 mb-4">
              <div>
                <h3 className="font-display text-base font-semibold text-slate-100">
                  Industrial Dataset Registry
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Independent adapters with strict split isolation
                </p>
              </div>
              <button
                type="button"
                onClick={() => onNavigate('dataset_explorer')}
                className="text-xs font-medium text-sky-400 hover:text-sky-300 whitespace-nowrap"
              >
                Explore Datasets
              </button>
            </div>

            <div className="divide-y divide-slate-800/80">
              {(['mvtec_ad', 'visa', 'mvtec_loco', 'mvtec_ad2', 'btad'] as DatasetId[]).map(
                (id) => {
                  const ds = DATASET_REGISTRY[id];
                  const hasRealImport = Object.keys(importedBundles).some((k) =>
                    k.startsWith(`${id}::`)
                  );
                  return (
                    <div key={id} className="py-3 flex items-center justify-between gap-4">
                      <div className="min-w-0">
                        <div className="text-sm font-medium text-slate-200 truncate">{ds.name}</div>
                        <div className="text-xs text-slate-500 truncate">
                          {ds.license} · {ds.officialCategories.length} categories
                        </div>
                      </div>
                      <div className="text-xs font-mono text-right shrink-0">
                        {hasRealImport ? (
                          <span className="text-emerald-400">Local Real Data Verified</span>
                        ) : ds.status === 'ready_synthetic_fixture' ? (
                          <span className="text-slate-400">Verification Fixture</span>
                        ) : (
                          <span className="text-amber-400">Optional / Not Loaded</span>
                        )}
                      </div>
                    </div>
                  );
                }
              )}
            </div>
          </div>

          <div className="mt-4 pt-4 border-t border-slate-800 flex items-start gap-2.5 text-xs text-slate-400">
            <Cpu className="w-4 h-4 text-slate-500 shrink-0 mt-0.5" />
            <span>
              Runtime: Client-side Float32Array ML Engine (CPU/Canvas) + Full Python 3.11 PyTorch /
              Anomalib / Streamlit repository packaged under <code className="text-slate-300">python/</code>.
            </span>
          </div>
        </section>
      </div>

      {/* Recent Inspection Activity */}
      <section className="bg-[#111827] border border-slate-800 rounded-xl p-6">
        <div className="flex items-center justify-between gap-4 mb-4">
          <div>
            <h3 className="font-display text-base font-semibold text-slate-100">
              Recent Inspection Activity
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Locally persisted inspection log (metadata only, raw uploads excluded)
            </p>
          </div>
          <button
            type="button"
            onClick={() => onNavigate('history')}
            className="text-xs font-medium text-sky-400 hover:text-sky-300 whitespace-nowrap"
          >
            View All History
          </button>
        </div>

        {history.length === 0 ? (
          <div className="py-8 text-center text-sm text-slate-500 border border-dashed border-slate-800 rounded-lg">
            No inspections recorded yet. Train a model and inspect an image in Live Inspection.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-sm">
              <thead>
                <tr className="border-b border-slate-800 text-xs text-slate-400">
                  <th className="py-2.5 pr-4 font-medium">Timestamp</th>
                  <th className="py-2.5 pr-4 font-medium">Dataset / Category</th>
                  <th className="py-2.5 pr-4 font-medium">Model</th>
                  <th className="py-2.5 pr-4 font-medium">Image Reference</th>
                  <th className="py-2.5 pr-4 font-medium">Prediction</th>
                  <th className="py-2.5 pr-4 font-medium text-right">Raw Score</th>
                  <th className="py-2.5 font-medium text-right">Latency</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70">
                {history.slice(0, 5).map((rec) => (
                  <tr key={rec.id} className="hover:bg-slate-800/30">
                    <td className="py-2.5 pr-4 font-mono text-xs text-slate-400 tabular-nums whitespace-nowrap">
                      {new Date(rec.timestamp).toLocaleTimeString()}
                    </td>
                    <td className="py-2.5 pr-4 text-slate-200 whitespace-nowrap">
                      {DATASET_REGISTRY[rec.datasetId].name} · {rec.category}
                    </td>
                    <td className="py-2.5 pr-4 text-slate-300 whitespace-nowrap">
                      {MODEL_REGISTRY[rec.architecture].name}
                    </td>
                    <td className="py-2.5 pr-4 font-mono text-xs text-slate-400 truncate max-w-[180px]">
                      {rec.imageReference}
                    </td>
                    <td className="py-2.5 pr-4 whitespace-nowrap">
                      {rec.prediction === 'Normal' ? (
                        <span className="inline-flex items-center gap-1.5 text-emerald-400 text-xs font-medium">
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          Normal
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1.5 text-rose-400 text-xs font-medium">
                          <ShieldAlert className="w-3.5 h-3.5" />
                          Anomalous
                        </span>
                      )}
                    </td>
                    <td className="py-2.5 pr-4 font-mono text-xs text-slate-200 text-right tabular-nums">
                      {rec.rawAnomalyScore.toFixed(4)}
                    </td>
                    <td className="py-2.5 font-mono text-xs text-slate-400 text-right tabular-nums">
                      {rec.latencyMs.toFixed(1)} ms
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
};
