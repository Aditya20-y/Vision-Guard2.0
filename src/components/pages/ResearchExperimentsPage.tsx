import React, { useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Download, Play } from 'lucide-react';
import { DATASET_REGISTRY } from '../../ml/datasets';
import { MODEL_REGISTRY } from '../../ml/models';
import { useVisionGuard } from '../../store/useVisionGuardStore';
import { DatasetId, ModelArchitecture } from '../../types/visionguard';

type ExperimentTab = 'efficiency' | 'robustness' | 'ablations' | 'difficulty';

export const ResearchExperimentsPage: React.FC = () => {
  const {
    dataEfficiencyResults,
    robustnessResults,
    ablationResults,
    difficultyResults,
    busyMessage,
    executeDataEfficiency,
    executeRobustness,
    executePatchCoreAblations,
    executeCategoryDifficulty,
  } = useVisionGuard();

  const [tab, setTab] = useState<ExperimentTab>('efficiency');
  const [datasetId, setDatasetId] = useState<DatasetId>('mvtec_ad');
  const [category, setCategory] = useState<string>('bottle');
  const [seed, setSeed] = useState<number>(42);
  const [difficultyArch, setDifficultyArch] = useState<ModelArchitecture>('patchcore');
  const [selectedRobustnessArch, setSelectedRobustnessArch] =
    useState<ModelArchitecture>('patchcore');

  const categories = DATASET_REGISTRY[datasetId].supportedCategories;
  const activeCategory = categories.includes(category) ? category : categories[0];

  // Pivot Data Efficiency rows by training fraction for multi-line Recharts plotting
  const efficiencyChartData = useMemo(() => {
    const map = new Map<number, Record<string, number | string>>();
    for (const row of dataEfficiencyResults) {
      const pct = Math.round(row.fraction * 100);
      if (!map.has(pct)) map.set(pct, { fractionLabel: `${pct}%`, fraction: pct });
      const entry = map.get(pct)!;
      entry[row.architecture] = Number((row.imageAuroc * 100).toFixed(1));
    }
    return Array.from(map.values()).sort((a, b) => Number(a.fraction) - Number(b.fraction));
  }, [dataEfficiencyResults]);

  // Pivot Robustness rows for selected model across corruptions and severities
  const robustnessChartData = useMemo(() => {
    const filtered = robustnessResults.filter((r) => r.architecture === selectedRobustnessArch);
    const map = new Map<string, Record<string, number | string>>();
    for (const r of filtered) {
      if (!map.has(r.corruption)) {
        map.set(r.corruption, {
          corruption: r.corruptionLabel,
          clean: Number((r.cleanImageAuroc * 100).toFixed(1)),
        });
      }
      const entry = map.get(r.corruption)!;
      entry[`sev${r.severity}`] = Number((r.corruptedImageAuroc * 100).toFixed(1));
    }
    return Array.from(map.values());
  }, [robustnessResults, selectedRobustnessArch]);

  const exportExperimentJson = () => {
    const payload = {
      exportedAt: new Date().toISOString(),
      dataEfficiency: dataEfficiencyResults,
      robustness: robustnessResults,
      patchCoreAblations: ablationResults,
      categoryDifficulty: difficultyResults,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `visionguard_experiments_${tab}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Experiment Selector & Controlled Seed Header */}
      <section className="bg-[#111827] border border-slate-800 rounded-xl p-5 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div className="flex flex-wrap items-center gap-1.5 p-1 bg-[#0B0F17] border border-slate-800 rounded-lg">
          {(
            [
              ['efficiency', 'Exp B: Data Efficiency'],
              ['robustness', 'Exp C: Robustness'],
              ['ablations', 'Exp D: PatchCore Ablations'],
              ['difficulty', 'Exp E/F: Cross-Dataset & Difficulty'],
            ] as [ExperimentTab, string][]
          ).map(([id, label]) => (
            <button
              key={id}
              type="button"
              onClick={() => setTab(id)}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors whitespace-nowrap ${
                tab === id
                  ? 'bg-sky-600 text-white'
                  : 'text-slate-400 hover:text-slate-100'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        <div className="flex flex-wrap items-center gap-3 text-xs">
          {tab !== 'difficulty' && (
            <>
              <select
                aria-label="Experiment dataset"
                value={datasetId}
                onChange={(e) => {
                  const next = e.target.value as DatasetId;
                  setDatasetId(next);
                  setCategory(DATASET_REGISTRY[next].supportedCategories[0]);
                }}
                className="h-9 px-3 rounded-lg bg-[#0B0F17] border border-slate-700 text-slate-200"
              >
                {(['mvtec_ad', 'visa', 'mvtec_loco', 'mvtec_ad2'] as DatasetId[]).map((id) => (
                  <option key={id} value={id}>
                    {DATASET_REGISTRY[id].name}
                  </option>
                ))}
              </select>

              <select
                aria-label="Experiment category"
                value={activeCategory}
                onChange={(e) => setCategory(e.target.value)}
                className="h-9 px-3 rounded-lg bg-[#0B0F17] border border-slate-700 text-slate-200"
              >
                {categories.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </>
          )}

          <div className="flex items-center gap-1.5 text-slate-400">
            <label htmlFor="exp-seed">Seed</label>
            <input
              id="exp-seed"
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
              className="w-16 h-9 px-2 rounded-lg bg-[#0B0F17] border border-slate-700 font-mono text-slate-100"
            />
          </div>

          <button
            type="button"
            onClick={exportExperimentJson}
            className="inline-flex items-center gap-1.5 h-9 px-3 rounded-lg border border-slate-700 text-slate-200 hover:bg-slate-800 whitespace-nowrap"
          >
            <Download className="w-3.5 h-3.5" />
            Export JSON
          </button>
        </div>
      </section>

      {/* EXPERIMENT B — DATA EFFICIENCY */}
      {tab === 'efficiency' && (
        <section className="bg-[#111827] border border-slate-800 rounded-xl p-6 space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="max-w-2xl">
              <h2 className="font-display text-base font-semibold text-slate-100">
                Experiment B — Normal Training Data Efficiency (10% · 25% · 50% · 100%)
              </h2>
              <p className="text-xs text-slate-400 mt-1">
                Measures how reconstruction-based (Autoencoder) and feature-based (PaDiM,
                PatchCore) models degrade when normal training samples are deterministically
                subsampled.
              </p>
            </div>
            <button
              type="button"
              disabled={Boolean(busyMessage)}
              onClick={() =>
                executeDataEfficiency(
                  datasetId,
                  activeCategory,
                  ['autoencoder', 'padim', 'patchcore'],
                  seed
                )
              }
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium bg-sky-600 text-white hover:bg-sky-500 disabled:opacity-50 whitespace-nowrap"
            >
              <Play className="w-3.5 h-3.5" />
              Run Data Efficiency Sweep
            </button>
          </div>

          {efficiencyChartData.length === 0 ? (
            <div className="py-14 border border-dashed border-slate-800 rounded-lg text-center text-sm text-slate-500">
              Not evaluated yet. Run the sweep to train and measure real AUROC across 10%, 25%, 50%,
              and 100% normal subsets.
            </div>
          ) : (
            <>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={efficiencyChartData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                    <XAxis dataKey="fractionLabel" stroke="#94A3B8" tick={{ fontSize: 12 }} />
                    <YAxis domain={[0, 100]} stroke="#94A3B8" tick={{ fontSize: 12 }} unit="%" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', fontSize: 12 }}
                    />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Line type="monotone" dataKey="patchcore" name="PatchCore AUROC (%)" stroke="#0EA5E9" strokeWidth={2.5} />
                    <Line type="monotone" dataKey="padim" name="PaDiM AUROC (%)" stroke="#10B981" strokeWidth={2.5} />
                    <Line type="monotone" dataKey="autoencoder" name="Autoencoder AUROC (%)" stroke="#F59E0B" strokeWidth={2.5} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400">
                      <th className="py-2 pr-4 font-medium">Model</th>
                      <th className="py-2 pr-4 font-medium text-right">Fraction</th>
                      <th className="py-2 pr-4 font-medium text-right">Train Images</th>
                      <th className="py-2 pr-4 font-medium text-right">Image AUROC</th>
                      <th className="py-2 pr-4 font-medium text-right">Pixel AUROC</th>
                      <th className="py-2 pr-4 font-medium text-right">Image F1</th>
                      <th className="py-2 font-medium text-right">Train Duration</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/70 font-mono tabular-nums">
                    {dataEfficiencyResults.map((r, i) => (
                      <tr key={i} className="hover:bg-slate-800/30">
                        <td className="py-2 pr-4 font-sans text-slate-200">
                          {MODEL_REGISTRY[r.architecture].name}
                        </td>
                        <td className="py-2 pr-4 text-right text-slate-300">
                          {Math.round(r.fraction * 100)}%
                        </td>
                        <td className="py-2 pr-4 text-right text-slate-400">{r.trainSamples}</td>
                        <td className="py-2 pr-4 text-right text-sky-300">
                          {(r.imageAuroc * 100).toFixed(1)}%
                        </td>
                        <td className="py-2 pr-4 text-right text-emerald-300">
                          {r.pixelAuroc !== null ? `${(r.pixelAuroc * 100).toFixed(1)}%` : 'N/A'}
                        </td>
                        <td className="py-2 pr-4 text-right text-slate-200">
                          {(r.imageF1 * 100).toFixed(1)}%
                        </td>
                        <td className="py-2 text-right text-slate-400">
                          {r.trainingDurationMs.toFixed(0)} ms
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </section>
      )}

      {/* EXPERIMENT C — ROBUSTNESS UNDER IMAGE DEGRADATION */}
      {tab === 'robustness' && (
        <section className="bg-[#111827] border border-slate-800 rounded-xl p-6 space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="max-w-2xl">
              <h2 className="font-display text-base font-semibold text-slate-100">
                Experiment C — Robustness Under Realistic Image Degradation
              </h2>
              <p className="text-xs text-slate-400 mt-1">
                Evaluates clean-trained models on corrupted test images across 6 industrial
                degradations (noise, blur, brightness, contrast, JPEG compression, resize/resample)
                at severities 1, 3, and 5. Ground-truth masks are never corrupted.
              </p>
            </div>
            <div className="flex items-center gap-2.5">
              <select
                aria-label="Filter robustness chart by model"
                value={selectedRobustnessArch}
                onChange={(e) => setSelectedRobustnessArch(e.target.value as ModelArchitecture)}
                className="h-9 px-3 rounded-lg bg-[#0B0F17] border border-slate-700 text-xs text-slate-200"
              >
                <option value="patchcore">PatchCore</option>
                <option value="padim">PaDiM</option>
                <option value="autoencoder">Autoencoder</option>
              </select>
              <button
                type="button"
                disabled={Boolean(busyMessage)}
                onClick={() => executeRobustness(datasetId, activeCategory, seed)}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium bg-sky-600 text-white hover:bg-sky-500 disabled:opacity-50 whitespace-nowrap"
              >
                <Play className="w-3.5 h-3.5" />
                Run Robustness Suite
              </button>
            </div>
          </div>

          {robustnessChartData.length === 0 ? (
            <div className="py-14 border border-dashed border-slate-800 rounded-lg text-center text-sm text-slate-500">
              Not evaluated yet. Run the robustness suite to measure clean vs corrupted AUROC
              degradation across all 6 corruption families.
            </div>
          ) : (
            <>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={robustnessChartData} margin={{ top: 10, right: 16, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                    <XAxis dataKey="corruption" stroke="#94A3B8" tick={{ fontSize: 11 }} />
                    <YAxis domain={[0, 100]} stroke="#94A3B8" tick={{ fontSize: 12 }} unit="%" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', fontSize: 12 }}
                    />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Bar dataKey="clean" name="Clean Baseline (%)" fill="#10B981" radius={[3, 3, 0, 0]} />
                    <Bar dataKey="sev1" name="Severity 1 (%)" fill="#38BDF8" radius={[3, 3, 0, 0]} />
                    <Bar dataKey="sev3" name="Severity 3 (%)" fill="#F59E0B" radius={[3, 3, 0, 0]} />
                    <Bar dataKey="sev5" name="Severity 5 (%)" fill="#EF4444" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="overflow-x-auto max-h-72 overflow-y-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400">
                      <th className="py-2 pr-4 font-medium">Model</th>
                      <th className="py-2 pr-4 font-medium">Corruption</th>
                      <th className="py-2 pr-4 font-medium text-right">Severity</th>
                      <th className="py-2 pr-4 font-medium text-right">Clean AUROC</th>
                      <th className="py-2 pr-4 font-medium text-right">Corrupted AUROC</th>
                      <th className="py-2 font-medium text-right">Delta (Drop)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/70 font-mono tabular-nums">
                    {robustnessResults
                      .filter((r) => r.architecture === selectedRobustnessArch)
                      .map((r, i) => (
                        <tr key={i} className="hover:bg-slate-800/30">
                          <td className="py-2 pr-4 font-sans text-slate-200">
                            {MODEL_REGISTRY[r.architecture].name}
                          </td>
                          <td className="py-2 pr-4 font-sans text-slate-300">{r.corruptionLabel}</td>
                          <td className="py-2 pr-4 text-right text-slate-400">L{r.severity}</td>
                          <td className="py-2 pr-4 text-right text-emerald-300">
                            {(r.cleanImageAuroc * 100).toFixed(1)}%
                          </td>
                          <td className="py-2 pr-4 text-right text-sky-300">
                            {(r.corruptedImageAuroc * 100).toFixed(1)}%
                          </td>
                          <td
                            className={`py-2 text-right ${
                              r.aurocDrop > 0.1
                                ? 'text-rose-400'
                                : r.aurocDrop > 0.03
                                  ? 'text-amber-300'
                                  : 'text-slate-400'
                            }`}
                          >
                            {r.aurocDrop >= 0 ? '-' : '+'}
                            {(Math.abs(r.aurocDrop) * 100).toFixed(1)}%
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </section>
      )}

      {/* EXPERIMENT D — PATCHCORE ABLATIONS */}
      {tab === 'ablations' && (
        <section className="bg-[#111827] border border-slate-800 rounded-xl p-6 space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="max-w-2xl">
              <h2 className="font-display text-base font-semibold text-slate-100">
                Experiment D — PatchCore Controlled Ablation Study
              </h2>
              <p className="text-xs text-slate-400 mt-1">
                Systematically varies Coreset Subsampling Ratio (10%–100%), Multi-Scale Feature
                Layers (L1 / L2 / L1+L2 / L1+L2+L3), and Input Resolution (64px vs 96px) while
                measuring memory bank size, latency, and detection accuracy.
              </p>
            </div>
            <button
              type="button"
              disabled={Boolean(busyMessage)}
              onClick={() => executePatchCoreAblations(datasetId, activeCategory, seed)}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium bg-sky-600 text-white hover:bg-sky-500 disabled:opacity-50 whitespace-nowrap"
            >
              <Play className="w-3.5 h-3.5" />
              Run PatchCore Ablations
            </button>
          </div>

          {ablationResults.length === 0 ? (
            <div className="py-14 border border-dashed border-slate-800 rounded-lg text-center text-sm text-slate-500">
              Not evaluated yet. Execute the controlled PatchCore ablation sweep to measure memory
              bank vs latency vs AUROC trade-offs.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400">
                    <th className="py-2.5 pr-4 font-medium">Ablation Axis</th>
                    <th className="py-2.5 pr-4 font-medium">Setting</th>
                    <th className="py-2.5 pr-4 font-medium text-right">Memory Bank</th>
                    <th className="py-2.5 pr-4 font-medium text-right">Checkpoint KB</th>
                    <th className="py-2.5 pr-4 font-medium text-right">Image AUROC</th>
                    <th className="py-2.5 pr-4 font-medium text-right">Pixel AUROC</th>
                    <th className="py-2.5 font-medium text-right">Latency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/70 font-mono tabular-nums">
                  {ablationResults.map((row, idx) => (
                    <tr key={idx} className="hover:bg-slate-800/30">
                      <td className="py-2.5 pr-4 font-sans text-slate-400">
                        {row.experimentGroup.replace('_', ' ')}
                      </td>
                      <td className="py-2.5 pr-4 font-sans text-slate-100 font-medium">
                        {row.settingLabel}
                      </td>
                      <td className="py-2.5 pr-4 text-right text-slate-300">
                        {row.memoryBankVectors} vecs
                      </td>
                      <td className="py-2.5 pr-4 text-right text-slate-400">
                        {(row.checkpointSizeBytes / 1024).toFixed(1)} KB
                      </td>
                      <td className="py-2.5 pr-4 text-right text-sky-300">
                        {(row.imageAuroc * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 pr-4 text-right text-emerald-300">
                        {row.pixelAuroc !== null ? `${(row.pixelAuroc * 100).toFixed(1)}%` : 'N/A'}
                      </td>
                      <td className="py-2.5 text-right text-slate-300">
                        {row.meanLatencyMs.toFixed(1)} ms
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}

      {/* EXPERIMENT E & F — CROSS-DATASET & CATEGORY DIFFICULTY */}
      {tab === 'difficulty' && (
        <section className="bg-[#111827] border border-slate-800 rounded-xl p-6 space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="max-w-2xl">
              <h2 className="font-display text-base font-semibold text-slate-100">
                Experiments E &amp; F — Cross-Dataset Generalization &amp; Category Difficulty Analysis
              </h2>
              <p className="text-xs text-slate-400 mt-1">
                Evaluates a single model pipeline independently across MVTec AD, VisA, MVTec LOCO
                AD (structural + logical anomalies), and MVTec AD 2 (lighting shift) without mixing
                datasets, ranking hardest categories by observed AUROC, defect area, and contrast.
              </p>
            </div>
            <div className="flex items-center gap-2.5">
              <select
                aria-label="Difficulty analysis model"
                value={difficultyArch}
                onChange={(e) => setDifficultyArch(e.target.value as ModelArchitecture)}
                className="h-9 px-3 rounded-lg bg-[#0B0F17] border border-slate-700 text-xs text-slate-200"
              >
                <option value="patchcore">PatchCore</option>
                <option value="padim">PaDiM</option>
                <option value="autoencoder">Autoencoder</option>
                <option value="fastflow">FastFlow</option>
              </select>
              <button
                type="button"
                disabled={Boolean(busyMessage)}
                onClick={() => executeCategoryDifficulty(difficultyArch, seed)}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium bg-sky-600 text-white hover:bg-sky-500 disabled:opacity-50 whitespace-nowrap"
              >
                <Play className="w-3.5 h-3.5" />
                Run Cross-Dataset Analysis
              </button>
            </div>
          </div>

          {difficultyResults.length === 0 ? (
            <div className="py-14 border border-dashed border-slate-800 rounded-lg text-center text-sm text-slate-500">
              Not evaluated yet. Execute the multi-dataset sweep across 8 independent categories to
              identify hard defect regimes and logical anomaly failure modes.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400">
                    <th className="py-2.5 pr-4 font-medium">Dataset</th>
                    <th className="py-2.5 pr-4 font-medium">Category</th>
                    <th className="py-2.5 pr-4 font-medium text-right">Image AUROC</th>
                    <th className="py-2.5 pr-4 font-medium text-right">Pixel AUROC</th>
                    <th className="py-2.5 pr-4 font-medium text-right">Mean Defect Area</th>
                    <th className="py-2.5 font-medium">Attributed Difficulty Factor</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/70">
                  {difficultyResults.map((row, idx) => (
                    <tr key={idx} className="hover:bg-slate-800/30">
                      <td className="py-2.5 pr-4 text-slate-300 whitespace-nowrap">
                        {DATASET_REGISTRY[row.datasetId].name}
                      </td>
                      <td className="py-2.5 pr-4 font-mono text-slate-100 whitespace-nowrap">
                        {row.category}
                      </td>
                      <td className="py-2.5 pr-4 font-mono text-right tabular-nums text-sky-300">
                        {(row.imageAuroc * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 pr-4 font-mono text-right tabular-nums text-emerald-300">
                        {row.pixelAuroc !== null ? `${(row.pixelAuroc * 100).toFixed(1)}%` : 'N/A'}
                      </td>
                      <td className="py-2.5 pr-4 font-mono text-right tabular-nums text-slate-400">
                        {(row.defectAreaFraction * 100).toFixed(2)}%
                      </td>
                      <td className="py-2.5 text-slate-300">{row.difficultyReason}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}
    </div>
  );
};
