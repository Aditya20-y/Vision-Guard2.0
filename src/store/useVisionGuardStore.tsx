import React, { createContext, useCallback, useContext, useEffect, useState } from 'react';
import {
  CategoryDifficultyRow,
  DataEfficiencyPoint,
  DatasetId,
  EvaluationMetrics,
  InspectionHistoryRecord,
  ModelArchitecture,
  ModelConfig,
  PatchCoreAblationRow,
  RobustnessResultRow,
  TrainedCheckpoint,
} from '../types/visionguard';
import {
  CategoryDatasetBundle,
  DATASET_REGISTRY,
  getCategoryDatasetBundle,
  importLocalDatasetFiles,
} from '../ml/datasets';
import {
  evaluateCheckpoint,
  runCategoryDifficultyAnalysis,
  runDataEfficiencyExperiment,
  runPatchCoreAblations,
  runRobustnessExperiment,
} from '../ml/experiments';
import { MODEL_REGISTRY, trainAnomalyModel } from '../ml/models';

const HISTORY_STORAGE_KEY = 'visionguard_inspection_history_v1';

interface VisionGuardContextValue {
  checkpoints: TrainedCheckpoint[];
  evaluations: EvaluationMetrics[];
  history: InspectionHistoryRecord[];
  dataEfficiencyResults: DataEfficiencyPoint[];
  robustnessResults: RobustnessResultRow[];
  ablationResults: PatchCoreAblationRow[];
  difficultyResults: CategoryDifficultyRow[];
  importedBundles: Record<string, CategoryDatasetBundle>;
  busyMessage: string | null;
  lastError: string | null;
  clearError: () => void;

  getBundle: (datasetId: DatasetId, category: string) => CategoryDatasetBundle;
  importLocalFolder: (datasetId: DatasetId, category: string, files: FileList) => Promise<CategoryDatasetBundle>;
  trainAndEvaluateModel: (params: {
    datasetId: DatasetId;
    category: string;
    config: ModelConfig;
    trainFraction?: number;
  }) => Promise<{ checkpoint: TrainedCheckpoint; evaluation: EvaluationMetrics }>;
  runBaselineComparison: (
    datasetId: DatasetId,
    category: string,
    architectures?: ModelArchitecture[],
    seed?: number
  ) => Promise<void>;
  importCheckpointJson: (jsonText: string) => void;
  deleteCheckpoint: (checkpointId: string) => void;
  recordInspection: (record: InspectionHistoryRecord) => void;
  deleteHistoryRecord: (id: string) => void;
  clearHistory: () => void;

  executeDataEfficiency: (
    datasetId: DatasetId,
    category: string,
    architectures: ModelArchitecture[],
    seed?: number
  ) => Promise<void>;
  executeRobustness: (datasetId: DatasetId, category: string, seed?: number) => Promise<void>;
  executePatchCoreAblations: (datasetId: DatasetId, category: string, seed?: number) => Promise<void>;
  executeCategoryDifficulty: (architecture: ModelArchitecture, seed?: number) => Promise<void>;
}

const VisionGuardContext = createContext<VisionGuardContextValue | null>(null);

export const VisionGuardProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [checkpoints, setCheckpoints] = useState<TrainedCheckpoint[]>([]);
  const [evaluations, setEvaluations] = useState<EvaluationMetrics[]>([]);
  const [history, setHistory] = useState<InspectionHistoryRecord[]>(() => {
    try {
      const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
      if (raw) return JSON.parse(raw) as InspectionHistoryRecord[];
    } catch {
      // ignore storage errors
    }
    return [];
  });

  const [dataEfficiencyResults, setDataEfficiencyResults] = useState<DataEfficiencyPoint[]>([]);
  const [robustnessResults, setRobustnessResults] = useState<RobustnessResultRow[]>([]);
  const [ablationResults, setAblationResults] = useState<PatchCoreAblationRow[]>([]);
  const [difficultyResults, setDifficultyResults] = useState<CategoryDifficultyRow[]>([]);
  const [importedBundles, setImportedBundles] = useState<Record<string, CategoryDatasetBundle>>({});
  const [busyMessage, setBusyMessage] = useState<string | null>(null);
  const [lastError, setLastError] = useState<string | null>(null);

  useEffect(() => {
    try {
      localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history));
    } catch {
      // ignore storage quota errors
    }
  }, [history]);

  const clearError = useCallback(() => setLastError(null), []);

  const getBundle = useCallback(
    (datasetId: DatasetId, category: string) => {
      const key = `${datasetId}::${category}`;
      if (importedBundles[key]) return importedBundles[key];
      return getCategoryDatasetBundle(datasetId, category);
    },
    [importedBundles]
  );

  const importLocalFolder = useCallback(
    async (datasetId: DatasetId, category: string, files: FileList) => {
      setBusyMessage(`Validating local ${DATASET_REGISTRY[datasetId].name} (${category}) folder...`);
      setLastError(null);
      try {
        const bundle = await importLocalDatasetFiles({ datasetId, category, files });
        setImportedBundles((prev) => ({
          ...prev,
          [`${datasetId}::${category}`]: bundle,
        }));
        return bundle;
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Failed to import local dataset folder.';
        setLastError(msg);
        throw err;
      } finally {
        setBusyMessage(null);
      }
    },
    []
  );

  const trainAndEvaluateModel = useCallback(
    async (params: {
      datasetId: DatasetId;
      category: string;
      config: ModelConfig;
      trainFraction?: number;
    }) => {
      const { datasetId, category, config, trainFraction = 1.0 } = params;
      setLastError(null);
      setBusyMessage(
        `Training ${MODEL_REGISTRY[config.architecture].name} on ${DATASET_REGISTRY[datasetId].name} / ${category}...`
      );
      await new Promise((r) => setTimeout(r, 15));

      try {
        const bundle = getBundle(datasetId, category);
        const checkpoint = await trainAnomalyModel({
          datasetId,
          category,
          provenance: bundle.provenance,
          trainSamples: bundle.train,
          valSamples: bundle.val,
          config,
          trainFraction,
        });

        setBusyMessage(
          `Evaluating ${MODEL_REGISTRY[config.architecture].name} on ${category} test split (${bundle.test.length} samples)...`
        );
        await new Promise((r) => setTimeout(r, 10));

        const evaluation = await evaluateCheckpoint(checkpoint, bundle);

        setCheckpoints((prev) => [checkpoint, ...prev]);
        setEvaluations((prev) => [evaluation, ...prev]);
        return { checkpoint, evaluation };
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Training or evaluation failed.';
        setLastError(msg);
        throw err;
      } finally {
        setBusyMessage(null);
      }
    },
    [getBundle]
  );

  const runBaselineComparison = useCallback(
    async (
      datasetId: DatasetId,
      category: string,
      architectures: ModelArchitecture[] = ['autoencoder', 'padim', 'patchcore', 'fastflow'],
      seed = 42
    ) => {
      setLastError(null);
      try {
        const bundle = getBundle(datasetId, category);
        for (const arch of architectures) {
          setBusyMessage(
            `Benchmarking ${MODEL_REGISTRY[arch].name} on ${DATASET_REGISTRY[datasetId].name} / ${category}...`
          );
          await new Promise((r) => setTimeout(r, 15));
          const ckpt = await trainAnomalyModel({
            datasetId,
            category,
            provenance: bundle.provenance,
            trainSamples: bundle.train,
            valSamples: bundle.val,
            config: { ...MODEL_REGISTRY[arch].defaultConfig, seed },
          });
          const evalRes = await evaluateCheckpoint(ckpt, bundle);
          setCheckpoints((prev) => [ckpt, ...prev]);
          setEvaluations((prev) => [evalRes, ...prev]);
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Baseline comparison failed.';
        setLastError(msg);
      } finally {
        setBusyMessage(null);
      }
    },
    [getBundle]
  );

  const importCheckpointJson = useCallback((jsonText: string) => {
    try {
      const parsed = JSON.parse(jsonText) as TrainedCheckpoint;
      if (!parsed.id || !parsed.architecture || !parsed.artifact || !parsed.thresholdPolicy) {
        throw new Error('Invalid VisionGuard checkpoint schema.');
      }
      setCheckpoints((prev) => [parsed, ...prev.filter((c) => c.id !== parsed.id)]);
      setLastError(null);
    } catch (err) {
      setLastError(err instanceof Error ? err.message : 'Could not parse checkpoint file.');
    }
  }, []);

  const deleteCheckpoint = useCallback((checkpointId: string) => {
    setCheckpoints((prev) => prev.filter((c) => c.id !== checkpointId));
  }, []);

  const recordInspection = useCallback((record: InspectionHistoryRecord) => {
    setHistory((prev) => [record, ...prev].slice(0, 200));
  }, []);

  const deleteHistoryRecord = useCallback((id: string) => {
    setHistory((prev) => prev.filter((r) => r.id !== id));
  }, []);

  const clearHistory = useCallback(() => {
    setHistory([]);
  }, []);

  const executeDataEfficiency = useCallback(
    async (
      datasetId: DatasetId,
      category: string,
      architectures: ModelArchitecture[],
      seed = 42
    ) => {
      setLastError(null);
      try {
        const res = await runDataEfficiencyExperiment({
          datasetId,
          category,
          architectures,
          seed,
          onProgress: (m) => setBusyMessage(m),
        });
        setDataEfficiencyResults(res);
      } catch (err) {
        setLastError(err instanceof Error ? err.message : 'Data efficiency experiment failed.');
      } finally {
        setBusyMessage(null);
      }
    },
    []
  );

  const executeRobustness = useCallback(
    async (datasetId: DatasetId, category: string, seed = 42) => {
      setLastError(null);
      try {
        // Ensure we have trained checkpoints for Autoencoder, PaDiM, and PatchCore on this category
        const bundle = getBundle(datasetId, category);
        const targetArchs: ModelArchitecture[] = ['autoencoder', 'padim', 'patchcore'];
        const activeCkpts: TrainedCheckpoint[] = [];

        for (const arch of targetArchs) {
          const existing = checkpoints.find(
            (c) => c.datasetId === datasetId && c.category === category && c.architecture === arch
          );
          if (existing) {
            activeCkpts.push(existing);
          } else {
            setBusyMessage(`Training ${MODEL_REGISTRY[arch].name} for robustness baseline...`);
            await new Promise((r) => setTimeout(r, 10));
            const ckpt = await trainAnomalyModel({
              datasetId,
              category,
              provenance: bundle.provenance,
              trainSamples: bundle.train,
              valSamples: bundle.val,
              config: { ...MODEL_REGISTRY[arch].defaultConfig, seed },
            });
            const evalRes = await evaluateCheckpoint(ckpt, bundle);
            setCheckpoints((prev) => [ckpt, ...prev]);
            setEvaluations((prev) => [evalRes, ...prev]);
            activeCkpts.push(ckpt);
          }
        }

        const res = await runRobustnessExperiment({
          checkpoints: activeCkpts,
          datasetId,
          category,
          seed,
          onProgress: (m) => setBusyMessage(m),
        });
        setRobustnessResults(res);
      } catch (err) {
        setLastError(err instanceof Error ? err.message : 'Robustness experiment failed.');
      } finally {
        setBusyMessage(null);
      }
    },
    [checkpoints, getBundle]
  );

  const executePatchCoreAblations = useCallback(
    async (datasetId: DatasetId, category: string, seed = 42) => {
      setLastError(null);
      try {
        const res = await runPatchCoreAblations({
          datasetId,
          category,
          seed,
          onProgress: (m) => setBusyMessage(m),
        });
        setAblationResults(res);
      } catch (err) {
        setLastError(err instanceof Error ? err.message : 'PatchCore ablation failed.');
      } finally {
        setBusyMessage(null);
      }
    },
    []
  );

  const executeCategoryDifficulty = useCallback(
    async (architecture: ModelArchitecture, seed = 42) => {
      setLastError(null);
      try {
        const targets: { datasetId: DatasetId; category: string }[] = [
          { datasetId: 'mvtec_ad', category: 'bottle' },
          { datasetId: 'mvtec_ad', category: 'metal_nut' },
          { datasetId: 'mvtec_ad', category: 'screw' },
          { datasetId: 'mvtec_ad', category: 'capsule' },
          { datasetId: 'visa', category: 'pcb1' },
          { datasetId: 'mvtec_loco', category: 'breakfast_box' },
          { datasetId: 'mvtec_loco', category: 'screw_bag' },
          { datasetId: 'mvtec_ad2', category: 'sheet_metal' },
        ];
        const res = await runCategoryDifficultyAnalysis({
          targets,
          architecture,
          seed,
          onProgress: (m) => setBusyMessage(m),
        });
        setDifficultyResults(res);
      } catch (err) {
        setLastError(err instanceof Error ? err.message : 'Category difficulty analysis failed.');
      } finally {
        setBusyMessage(null);
      }
    },
    []
  );

  return (
    <VisionGuardContext.Provider
      value={{
        checkpoints,
        evaluations,
        history,
        dataEfficiencyResults,
        robustnessResults,
        ablationResults,
        difficultyResults,
        importedBundles,
        busyMessage,
        lastError,
        clearError,
        getBundle,
        importLocalFolder,
        trainAndEvaluateModel,
        runBaselineComparison,
        importCheckpointJson,
        deleteCheckpoint,
        recordInspection,
        deleteHistoryRecord,
        clearHistory,
        executeDataEfficiency,
        executeRobustness,
        executePatchCoreAblations,
        executeCategoryDifficulty,
      }}
    >
      {children}
    </VisionGuardContext.Provider>
  );
};

export function useVisionGuard(): VisionGuardContextValue {
  const ctx = useContext(VisionGuardContext);
  if (!ctx) throw new Error('useVisionGuard must be used within VisionGuardProvider.');
  return ctx;
}
