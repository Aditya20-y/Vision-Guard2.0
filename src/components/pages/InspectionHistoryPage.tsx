import React, { useMemo, useState } from 'react';
import { CheckCircle2, Download, Search, ShieldAlert, Trash2 } from 'lucide-react';
import { DATASET_REGISTRY } from '../../ml/datasets';
import { MODEL_REGISTRY } from '../../ml/models';
import { useVisionGuard } from '../../store/useVisionGuardStore';

export const InspectionHistoryPage: React.FC = () => {
  const { history, deleteHistoryRecord, clearHistory } = useVisionGuard();
  const [query, setQuery] = useState('');
  const [predFilter, setPredFilter] = useState<'all' | 'Normal' | 'Anomalous'>('all');

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return history.filter((rec) => {
      if (predFilter !== 'all' && rec.prediction !== predFilter) return false;
      if (!q) return true;
      return (
        rec.category.toLowerCase().includes(q) ||
        rec.imageReference.toLowerCase().includes(q) ||
        rec.architecture.toLowerCase().includes(q) ||
        rec.datasetId.toLowerCase().includes(q)
      );
    });
  }, [history, query, predFilter]);

  const exportCsv = () => {
    const header = [
      'timestamp',
      'dataset',
      'category',
      'model',
      'image_reference',
      'prediction',
      'raw_anomaly_score',
      'threshold',
      'latency_ms',
      'quality_status',
    ];
    const rows = [header.join(',')];
    for (const r of filtered) {
      rows.push(
        [
          r.timestamp,
          r.datasetId,
          r.category,
          r.architecture,
          `"${r.imageReference.replace(/"/g, '""')}"`,
          r.prediction,
          r.rawAnomalyScore.toFixed(6),
          r.threshold.toFixed(6),
          r.latencyMs.toFixed(2),
          r.qualityStatus,
        ].join(',')
      );
    }
    const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'visionguard_inspection_history.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <section className="bg-[#111827] border border-slate-800 rounded-xl p-6 space-y-5">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h2 className="font-display text-base font-semibold text-slate-100">
              Local Inspection History Log
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Persisted locally in browser storage with metadata only (raw uploaded images are never
              stored unnecessarily).
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2.5">
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search file, category, model..."
                className="h-9 pl-8 pr-3 rounded-lg bg-[#0B0F17] border border-slate-700 text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500"
              />
            </div>

            <div className="flex items-center gap-1 p-1 bg-[#0B0F17] border border-slate-800 rounded-lg">
              {(['all', 'Normal', 'Anomalous'] as const).map((st) => (
                <button
                  key={st}
                  type="button"
                  onClick={() => setPredFilter(st)}
                  className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                    predFilter === st ? 'bg-slate-800 text-sky-300' : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {st === 'all' ? 'All' : st}
                </button>
              ))}
            </div>

            <button
              type="button"
              disabled={filtered.length === 0}
              onClick={exportCsv}
              className="inline-flex items-center gap-1.5 h-9 px-3 rounded-lg text-xs font-medium border border-slate-700 text-slate-200 hover:bg-slate-800 disabled:opacity-40 whitespace-nowrap"
            >
              <Download className="w-3.5 h-3.5" />
              Export CSV
            </button>

            <button
              type="button"
              disabled={history.length === 0}
              onClick={clearHistory}
              className="inline-flex items-center gap-1.5 h-9 px-3 rounded-lg text-xs font-medium border border-rose-900/70 text-rose-300 hover:bg-rose-950/50 disabled:opacity-40 whitespace-nowrap"
            >
              <Trash2 className="w-3.5 h-3.5" />
              Clear History
            </button>
          </div>
        </div>

        {filtered.length === 0 ? (
          <div className="py-14 border border-dashed border-slate-800 rounded-lg text-center text-sm text-slate-500">
            {history.length === 0
              ? 'No inspection records logged yet. Run an inspection on the Live Inspection page.'
              : 'No inspection records match your current filter.'}
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th className="py-2.5 pr-4 font-medium">Timestamp</th>
                  <th className="py-2.5 pr-4 font-medium">Dataset / Category</th>
                  <th className="py-2.5 pr-4 font-medium">Model</th>
                  <th className="py-2.5 pr-4 font-medium">Image Reference</th>
                  <th className="py-2.5 pr-4 font-medium">Prediction</th>
                  <th className="py-2.5 pr-4 font-medium text-right">Raw Score / Thresh</th>
                  <th className="py-2.5 pr-4 font-medium">Image Quality</th>
                  <th className="py-2.5 pr-4 font-medium text-right">Latency</th>
                  <th className="py-2.5 font-medium text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70">
                {filtered.map((rec) => (
                  <tr key={rec.id} className="hover:bg-slate-800/30">
                    <td className="py-2.5 pr-4 font-mono text-slate-400 tabular-nums whitespace-nowrap">
                      {new Date(rec.timestamp).toLocaleString()}
                    </td>
                    <td className="py-2.5 pr-4 text-slate-200 whitespace-nowrap">
                      {DATASET_REGISTRY[rec.datasetId].name} · {rec.category}
                    </td>
                    <td className="py-2.5 pr-4 text-slate-300 whitespace-nowrap">
                      {MODEL_REGISTRY[rec.architecture].name}
                    </td>
                    <td className="py-2.5 pr-4 font-mono text-slate-300 truncate max-w-[200px]">
                      {rec.imageReference}
                    </td>
                    <td className="py-2.5 pr-4 whitespace-nowrap">
                      {rec.prediction === 'Normal' ? (
                        <span className="inline-flex items-center gap-1.5 text-emerald-400 font-medium">
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          Normal
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1.5 text-rose-400 font-medium">
                          <ShieldAlert className="w-3.5 h-3.5" />
                          Anomalous
                        </span>
                      )}
                    </td>
                    <td className="py-2.5 pr-4 font-mono text-right tabular-nums text-slate-200 whitespace-nowrap">
                      {rec.rawAnomalyScore.toFixed(4)} / {rec.threshold.toFixed(4)}
                    </td>
                    <td className="py-2.5 pr-4 whitespace-nowrap">
                      <span
                        className={
                          rec.qualityStatus === 'acceptable'
                            ? 'text-slate-400'
                            : 'text-amber-300 font-medium'
                        }
                      >
                        {rec.qualityStatus} ({rec.qualityWarningCount} warn)
                      </span>
                    </td>
                    <td className="py-2.5 pr-4 font-mono text-right tabular-nums text-slate-400 whitespace-nowrap">
                      {rec.latencyMs.toFixed(1)} ms
                    </td>
                    <td className="py-2.5 text-right">
                      <button
                        type="button"
                        aria-label={`Delete record ${rec.imageReference}`}
                        onClick={() => deleteHistoryRecord(rec.id)}
                        className="p-1 text-slate-500 hover:text-rose-400 transition-colors"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
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
