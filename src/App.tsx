import React, { useState } from 'react';
import { AlertTriangle, X } from 'lucide-react';
import { Sidebar } from './components/Sidebar';
import { TopBar } from './components/TopBar';
import { DatasetExplorerPage } from './components/pages/DatasetExplorerPage';
import { ExecutiveOverviewPage } from './components/pages/ExecutiveOverviewPage';
import { InspectionHistoryPage } from './components/pages/InspectionHistoryPage';
import { LiveInspectionPage } from './components/pages/LiveInspectionPage';
import { MethodologyPage } from './components/pages/MethodologyPage';
import { ModelLaboratoryPage } from './components/pages/ModelLaboratoryPage';
import { ResearchExperimentsPage } from './components/pages/ResearchExperimentsPage';
import { useVisionGuard, VisionGuardProvider } from './store/useVisionGuardStore';
import { DashboardPageId } from './types/visionguard';

const VisionGuardShell: React.FC = () => {
  const [activePage, setActivePage] = useState<DashboardPageId>('overview');
  const {
    checkpoints,
    evaluations,
    history,
    dataEfficiencyResults,
    robustnessResults,
    ablationResults,
    difficultyResults,
    busyMessage,
    lastError,
    clearError,
    runBaselineComparison,
  } = useVisionGuard();

  const hasEvidence =
    evaluations.length > 0 ||
    dataEfficiencyResults.length > 0 ||
    robustnessResults.length > 0 ||
    ablationResults.length > 0 ||
    difficultyResults.length > 0;

  const handleQuickBenchmark = async () => {
    await runBaselineComparison('mvtec_ad', 'bottle', ['autoencoder', 'padim', 'patchcore'], 42);
  };

  const handleExportAllEvidence = () => {
    const bundle = {
      platform: 'VisionGuard Industrial Anomaly Detection Platform',
      exportedAt: new Date().toISOString(),
      evaluations,
      dataEfficiencyResults,
      robustnessResults,
      ablationResults,
      difficultyResults,
      inspectionHistoryCount: history.length,
    };
    const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `visionguard_verified_results_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-[#0B0F17] text-slate-100 flex">
      <Sidebar
        activePage={activePage}
        onSelectPage={setActivePage}
        checkpointCount={checkpoints.length}
        evaluationCount={evaluations.length}
        historyCount={history.length}
      />

      <div className="flex-1 flex flex-col min-w-0">
        <TopBar
          activePage={activePage}
          busyMessage={busyMessage}
          onQuickBenchmark={handleQuickBenchmark}
          onExportEvidence={handleExportAllEvidence}
          hasEvidence={hasEvidence}
        />

        <main className="flex-1 p-8 max-w-[1440px] w-full mx-auto space-y-6">
          {lastError && (
            <div
              role="alert"
              className="bg-rose-950/50 border border-rose-800 rounded-xl px-4 py-3 text-sm text-rose-200 flex items-center justify-between gap-4"
            >
              <div className="flex items-center gap-2.5">
                <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
                <span>{lastError}</span>
              </div>
              <button
                type="button"
                aria-label="Dismiss error"
                onClick={clearError}
                className="text-rose-300 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          )}

          {activePage === 'overview' && <ExecutiveOverviewPage onNavigate={setActivePage} />}
          {activePage === 'live_inspection' && <LiveInspectionPage />}
          {activePage === 'model_lab' && <ModelLaboratoryPage />}
          {activePage === 'dataset_explorer' && <DatasetExplorerPage />}
          {activePage === 'experiments' && <ResearchExperimentsPage />}
          {activePage === 'history' && <InspectionHistoryPage />}
          {activePage === 'methodology' && <MethodologyPage />}
        </main>
      </div>
    </div>
  );
};

export default function App() {
  return (
    <VisionGuardProvider>
      <VisionGuardShell />
    </VisionGuardProvider>
  );
}
