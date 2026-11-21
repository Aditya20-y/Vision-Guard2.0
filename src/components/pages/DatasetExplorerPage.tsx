import React, { useMemo, useRef, useState } from 'react';
import { CheckCircle2, ExternalLink, FolderUp } from 'lucide-react';
import { DATASET_REGISTRY } from '../../ml/datasets';
import { useVisionGuard } from '../../store/useVisionGuardStore';
import { DatasetId } from '../../types/visionguard';

export const DatasetExplorerPage: React.FC = () => {
  const { getBundle, importLocalFolder, busyMessage } = useVisionGuard();

  const [selectedDataset, setSelectedDataset] = useState<DatasetId>('mvtec_ad');
  const [selectedCategory, setSelectedCategory] = useState<string>('bottle');
  const [galleryTab, setGalleryTab] = useState<'train' | 'test_good' | 'test_anom'>('test_anom');
  const [importFeedback, setImportFeedback] = useState<string | null>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  const meta = DATASET_REGISTRY[selectedDataset];
  const activeCategory = meta.supportedCategories.includes(selectedCategory)
    ? selectedCategory
    : meta.supportedCategories[0];

  const bundle = useMemo(
    () => getBundle(selectedDataset, activeCategory),
    [getBundle, selectedDataset, activeCategory]
  );

  const displayedSamples = useMemo(() => {
    if (galleryTab === 'train') return bundle.train;
    if (galleryTab === 'test_good') return bundle.test.filter((s) => s.label === 0);
    return bundle.test.filter((s) => s.label === 1);
  }, [bundle, galleryTab]);

  const handleFolderSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    setImportFeedback(null);
    try {
      const imported = await importLocalFolder(selectedDataset, activeCategory, files);
      setImportFeedback(
        `Verified ${imported.validation.readableImages} local images for ${meta.name} / ${activeCategory} (${imported.validation.trainNormalCount} train, ${imported.validation.valNormalCount} val, ${imported.validation.testNormalCount} test normal, ${imported.validation.testAnomalousCount} test anomalous, ${imported.validation.masksPresent} masks).`
      );
    } catch {
      // error already shown in global banner
    }
  };

  return (
    <div className="space-y-6">
      {/* Dataset Selector Tabs */}
      <div className="flex flex-wrap items-center gap-1.5 p-1 bg-[#111827] border border-slate-800 rounded-xl w-fit">
        {(['mvtec_ad', 'visa', 'mvtec_loco', 'mvtec_ad2', 'btad'] as DatasetId[]).map((id) => (
          <button
            key={id}
            type="button"
            onClick={() => {
              setSelectedDataset(id);
              setSelectedCategory(DATASET_REGISTRY[id].supportedCategories[0]);
              setImportFeedback(null);
            }}
            className={`px-3.5 py-2 rounded-lg text-xs font-medium transition-colors whitespace-nowrap ${
              selectedDataset === id
                ? 'bg-sky-600 text-white'
                : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/60'
            }`}
          >
            {DATASET_REGISTRY[id].name}
          </button>
        ))}
      </div>

      {/* Dataset Provenance, License & Local Import Card */}
      <section className="bg-[#111827] border border-slate-800 rounded-xl p-6 space-y-5">
        <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-4">
          <div className="space-y-2 max-w-3xl">
            <div className="text-xs text-slate-400 flex flex-wrap items-center gap-2">
              <span>{meta.role}</span>
              <span aria-hidden="true">·</span>
              <span>License: {meta.license}</span>
              <span aria-hidden="true">·</span>
              <span>{meta.totalOfficialImages}</span>
            </div>
            <h2 className="font-display text-xl font-bold text-slate-50">{meta.name}</h2>
            <p className="text-sm text-slate-300 leading-relaxed">{meta.description}</p>
          </div>

          <div className="flex flex-wrap items-center gap-2.5 shrink-0">
            <a
              href={meta.officialUrl}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg text-xs font-medium border border-slate-700 text-slate-200 hover:bg-slate-800 transition-colors whitespace-nowrap"
            >
              Official Source
              <ExternalLink className="w-3.5 h-3.5" />
            </a>
            <button
              type="button"
              disabled={Boolean(busyMessage)}
              onClick={() => folderInputRef.current?.click()}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg text-xs font-medium bg-sky-600 text-white hover:bg-sky-500 disabled:opacity-50 transition-colors whitespace-nowrap"
            >
              <FolderUp className="w-3.5 h-3.5" />
              Import Real Local Folder
            </button>
            <input
              ref={folderInputRef}
              type="file"
              // @ts-expect-error webkitdirectory is standard in Chromium/WebKit browsers
              webkitdirectory=""
              multiple
              onChange={handleFolderSelect}
              className="hidden"
            />
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-4 border-t border-slate-800 text-xs">
          <div className="space-y-1">
            <div className="text-slate-400 font-medium">Official Folder Layout Adapter</div>
            <div className="font-mono text-slate-300 break-all">{meta.folderStructureNote}</div>
          </div>
          <div className="space-y-1">
            <div className="text-slate-400 font-medium">Citation & Acquisition Policy</div>
            <div className="text-slate-300">{meta.citation}</div>
            <div className="text-slate-500">{meta.acquisitionNote}</div>
          </div>
        </div>

        {importFeedback && (
          <div className="rounded-lg bg-emerald-950/40 border border-emerald-800/70 p-3 text-xs text-emerald-200 flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>{importFeedback}</span>
          </div>
        )}
      </section>

      {/* Automated Dataset Validation Report & Category Selector */}
      <section className="bg-[#111827] border border-slate-800 rounded-xl p-6 space-y-5">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h3 className="font-display text-base font-semibold text-slate-100">
              Category Validation Report
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Provenance:{' '}
              <span
                className={
                  bundle.provenance === 'local_real_dataset'
                    ? 'text-emerald-400 font-medium'
                    : 'text-amber-300 font-medium'
                }
              >
                {bundle.provenance === 'local_real_dataset'
                  ? 'Local Real Dataset (User Verified)'
                  : 'Synthetic Verification Fixture (Non-Official Sandbox Data)'}
              </span>
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-1.5">
            {meta.supportedCategories.map((cat) => (
              <button
                key={cat}
                type="button"
                onClick={() => setSelectedCategory(cat)}
                className={`px-3 py-1.5 rounded-md text-xs font-mono transition-colors ${
                  activeCategory === cat
                    ? 'bg-slate-700 text-sky-300'
                    : 'bg-[#0B0F17] text-slate-400 hover:text-slate-200 border border-slate-800'
                }`}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 text-xs">
          <div className="p-3.5 rounded-lg bg-[#0B0F17] border border-slate-800">
            <div className="text-slate-500">Train (Normal Only)</div>
            <div className="font-mono text-lg font-semibold text-slate-100 tabular-nums mt-0.5">
              {bundle.validation.trainNormalCount}
            </div>
          </div>
          <div className="p-3.5 rounded-lg bg-[#0B0F17] border border-slate-800">
            <div className="text-slate-500">Val (Normal Only)</div>
            <div className="font-mono text-lg font-semibold text-slate-100 tabular-nums mt-0.5">
              {bundle.validation.valNormalCount}
            </div>
          </div>
          <div className="p-3.5 rounded-lg bg-[#0B0F17] border border-slate-800">
            <div className="text-slate-500">Test Normal</div>
            <div className="font-mono text-lg font-semibold text-emerald-400 tabular-nums mt-0.5">
              {bundle.validation.testNormalCount}
            </div>
          </div>
          <div className="p-3.5 rounded-lg bg-[#0B0F17] border border-slate-800">
            <div className="text-slate-500">Test Anomalous</div>
            <div className="font-mono text-lg font-semibold text-rose-400 tabular-nums mt-0.5">
              {bundle.validation.testAnomalousCount}
            </div>
          </div>
          <div className="p-3.5 rounded-lg bg-[#0B0F17] border border-slate-800">
            <div className="text-slate-500">GT Pixel Masks</div>
            <div className="font-mono text-lg font-semibold text-sky-300 tabular-nums mt-0.5">
              {bundle.validation.masksPresent}
            </div>
          </div>
          <div className="p-3.5 rounded-lg bg-[#0B0F17] border border-slate-800">
            <div className="text-slate-500">Split Leakage Check</div>
            <div className="font-mono text-lg font-semibold text-emerald-400 uppercase mt-0.5">
              {bundle.validation.splitLeakageCheck}
            </div>
          </div>
        </div>

        <ul className="text-xs text-slate-400 list-disc pl-5 space-y-1">
          {bundle.validation.notes.map((n, idx) => (
            <li key={idx}>{n}</li>
          ))}
        </ul>
      </section>

      {/* Interactive Sample & Ground-Truth Mask Gallery */}
      <section className="bg-[#111827] border border-slate-800 rounded-xl p-6 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <h3 className="font-display text-base font-semibold text-slate-100">
            Sample & Mask Inspection — {activeCategory}
          </h3>
          <div className="flex items-center gap-1 p-1 bg-[#0B0F17] border border-slate-800 rounded-lg">
            {(
              [
                ['train', 'Normal Train'],
                ['test_good', 'Normal Test'],
                ['test_anom', 'Anomalous + GT Mask'],
              ] as const
            ).map(([tabId, label]) => (
              <button
                key={tabId}
                type="button"
                onClick={() => setGalleryTab(tabId)}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors whitespace-nowrap ${
                  galleryTab === tabId
                    ? 'bg-slate-800 text-sky-300'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
          {displayedSamples.slice(0, 10).map((sample) => (
            <div key={sample.id} className="space-y-1.5">
              <div className="grid grid-cols-1 gap-1.5">
                <div className="aspect-square rounded-lg overflow-hidden border border-slate-800 bg-[#0B0F17]">
                  <img
                    src={sample.dataUrl}
                    alt={sample.fileName}
                    referrerPolicy="no-referrer"
                    className="w-full h-full object-cover"
                  />
                </div>
                {sample.maskDataUrl && (
                  <div className="h-16 rounded-md overflow-hidden border border-slate-800 bg-black flex items-center justify-center">
                    <img
                      src={sample.maskDataUrl}
                      alt={`Ground truth mask for ${sample.fileName}`}
                      referrerPolicy="no-referrer"
                      className="h-full object-contain"
                    />
                  </div>
                )}
              </div>
              <div className="text-[11px] font-mono text-slate-300 truncate">{sample.fileName}</div>
              <div className="text-[11px] text-slate-500">
                {sample.defectType} · {sample.anomalyKind}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
};
