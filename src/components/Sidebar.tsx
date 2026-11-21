import React from 'react';
import {
  Activity,
  BookOpen,
  Database,
  FlaskConical,
  History,
  LayoutDashboard,
  ScanSearch,
} from 'lucide-react';
import { DashboardPageId } from '../types/visionguard';

interface SidebarProps {
  activePage: DashboardPageId;
  onSelectPage: (page: DashboardPageId) => void;
  checkpointCount: number;
  evaluationCount: number;
  historyCount: number;
}

const NAV_ITEMS: { id: DashboardPageId; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { id: 'overview', label: 'Executive Overview', icon: LayoutDashboard },
  { id: 'live_inspection', label: 'Live Inspection', icon: ScanSearch },
  { id: 'model_lab', label: 'Model Laboratory', icon: Activity },
  { id: 'dataset_explorer', label: 'Dataset Explorer', icon: Database },
  { id: 'experiments', label: 'Research Experiments', icon: FlaskConical },
  { id: 'history', label: 'Inspection History', icon: History },
  { id: 'methodology', label: 'Methodology & Demo', icon: BookOpen },
];

export const Sidebar: React.FC<SidebarProps> = ({
  activePage,
  onSelectPage,
  checkpointCount,
  evaluationCount,
  historyCount,
}) => {
  return (
    <aside className="w-64 shrink-0 bg-[#0F141F] border-r border-slate-800 flex flex-col justify-between min-h-screen">
      <div>
        <div className="h-16 px-6 flex items-center border-b border-slate-800">
          <button
            type="button"
            onClick={() => onSelectPage('overview')}
            className="text-left font-display text-lg font-bold tracking-tight text-slate-50 whitespace-nowrap focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500 rounded"
          >
            VisionGuard
          </button>
        </div>

        <nav aria-label="Primary" className="px-3 py-5 space-y-1">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = activePage === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onSelectPage(item.id)}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors whitespace-nowrap focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500 ${
                  isActive
                    ? 'bg-slate-800/90 text-sky-400'
                    : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/40'
                }`}
              >
                <Icon className="w-4 h-4 shrink-0" />
                <span className="truncate">{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      <div className="p-5 border-t border-slate-800 space-y-2.5 text-xs text-slate-400">
        <div className="flex items-center justify-between">
          <span>Trained Checkpoints</span>
          <span className="font-mono tabular-nums text-slate-200">{checkpointCount}</span>
        </div>
        <div className="flex items-center justify-between">
          <span>Verified Evaluations</span>
          <span className="font-mono tabular-nums text-slate-200">{evaluationCount}</span>
        </div>
        <div className="flex items-center justify-between">
          <span>Inspection Records</span>
          <span className="font-mono tabular-nums text-slate-200">{historyCount}</span>
        </div>
      </div>
    </aside>
  );
};
