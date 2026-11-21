import React from 'react';
import { Download, Loader2, Play } from 'lucide-react';
import { DashboardPageId } from '../types/visionguard';

const PAGE_TITLES: Record<DashboardPageId, string> = {
  overview: 'Executive Overview',
  live_inspection: 'Live Visual Inspection',
  model_lab: 'Model Laboratory',
  dataset_explorer: 'Dataset Explorer',
  experiments: 'Research Experiments',
  history: 'Inspection History',
  methodology: 'Methodology & Competition Kit',
};

interface TopBarProps {
  activePage: DashboardPageId;
  busyMessage: string | null;
  onQuickBenchmark: () => void;
  onExportEvidence: () => void;
  hasEvidence: boolean;
}

export const TopBar: React.FC<TopBarProps> = ({
  activePage,
  busyMessage,
  onQuickBenchmark,
  onExportEvidence,
  hasEvidence,
}) => {
  return (
    <header className="h-16 px-8 border-b border-slate-800 bg-[#0B0F17] flex items-center justify-between gap-8 shrink-0">
      <div className="flex items-center gap-2 text-sm min-w-0">
        <span className="text-slate-500 whitespace-nowrap">VisionGuard</span>
        <span className="text-slate-700" aria-hidden="true">
          /
        </span>
        <h1 className="font-display font-semibold text-slate-100 truncate">
          {PAGE_TITLES[activePage]}
        </h1>
      </div>

      <div className="flex items-center gap-3 shrink-0">
        {busyMessage && (
          <div className="flex items-center gap-2 text-xs text-sky-400 mr-2 max-w-md truncate">
            <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0" />
            <span className="truncate">{busyMessage}</span>
          </div>
        )}

        <button
          type="button"
          onClick={onExportEvidence}
          disabled={!hasEvidence}
          className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-medium border border-slate-700 text-slate-200 hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500"
        >
          <Download className="w-3.5 h-3.5" />
          Export JSON Results
        </button>

        <button
          type="button"
          onClick={onQuickBenchmark}
          disabled={Boolean(busyMessage)}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium bg-sky-600 text-white hover:bg-sky-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
        >
          <Play className="w-3.5 h-3.5" />
          Run Core Benchmark
        </button>
      </div>
    </header>
  );
};
