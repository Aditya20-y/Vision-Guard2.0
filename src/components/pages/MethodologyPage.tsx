import React, { useState } from 'react';
import { MODEL_REGISTRY } from '../../ml/models';
import { ModelArchitecture } from '../../types/visionguard';

const JUDGE_QA: { q: string; a: string }[] = [
  {
    q: 'Why is normal-only learning necessary in industrial inspection?',
    a: 'Real factory lines produce thousands of defect-free parts for every rare defect, and future defect modes (novel scratches, contamination, tooling burrs) cannot be exhaustively labeled ahead of time. Learning the normal visual manifold enables open-set detection of previously unseen defects.',
  },
  {
    q: 'Why do pretrained feature methods (PatchCore / PaDiM) outperform reconstruction Autoencoders?',
    a: 'Pixel-level Autoencoders frequently suffer from over-generalization (reconstructing minor scratches cleanly) or blurry reconstructions on fine textures, inflating background error. Pretrained multi-scale patch descriptors encode rich local geometry and texture invariants, allowing Mahalanobis (PaDiM) or k-NN coreset distance (PatchCore) to isolate subtle anomalies sharply.',
  },
  {
    q: 'Why are raw anomaly scores not probabilities, and how are thresholds chosen?',
    a: 'Raw outputs are unbounded reconstruction MSE, Mahalanobis distances, Euclidean nearest-neighbor distances, or negative log-likelihoods. VisionGuard never calls them probabilities and calibrates decision thresholds strictly on held-out normal validation images (μ + 2.5σ or P95) without ever leaking test labels.',
  },
  {
    q: 'Why do patch-based methods struggle on MVTec LOCO AD logical anomalies?',
    a: 'Patch-level detectors inspect local receptive fields. When a logical constraint is violated (e.g., 3 screws instead of 2 screws and a washer, or a missing orange in a breakfast box), every individual local patch still looks like a valid normal object — causing pixel-wise appearance detectors to miss global count or relational violations.',
  },
  {
    q: 'How is Image Quality Assessment kept separate from defect classification?',
    a: 'Blur, poor illumination, or severe JPEG blockiness degrade camera input rather than indicating a broken physical product. VisionGuard computes Laplacian variance, luminance, RMS contrast, and 8×8 blockiness independently and raises a reliability caution banner without conflating optical faults with manufacturing defects.',
  },
];

export const MethodologyPage: React.FC = () => {
  const [section, setSection] = useState<'methodology' | 'competition' | 'python_cli'>('methodology');

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-1.5 p-1 bg-[#111827] border border-slate-800 rounded-xl w-fit">
        {(
          [
            ['methodology', 'Scientific Methodology & Algorithms'],
            ['competition', 'Competition Demo & Judge Q&A'],
            ['python_cli', 'Python 3.11 / PyTorch / Streamlit Repo'],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            type="button"
            onClick={() => setSection(id)}
            className={`px-3.5 py-2 rounded-lg text-xs font-medium transition-colors whitespace-nowrap ${
              section === id
                ? 'bg-sky-600 text-white'
                : 'text-slate-400 hover:text-slate-100'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {section === 'methodology' && (
        <div className="space-y-6">
          <section className="bg-[#111827] border border-slate-800 rounded-xl p-6 space-y-4">
            <h2 className="font-display text-lg font-bold text-slate-50">
              Problem Statement &amp; Research Question
            </h2>
            <p className="text-sm text-slate-300 leading-relaxed max-w-4xl">
              &ldquo;How effectively do reconstruction-based and pretrained feature-based anomaly
              detection methods identify and localize unseen industrial defects across different
              product categories and datasets, and how does their performance change under limited
              normal training data and realistic image-quality degradation?&rdquo;
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-3 border-t border-slate-800 text-xs text-slate-300">
              <div className="space-y-1.5">
                <h3 className="font-display text-sm font-semibold text-slate-100">
                  Scientific Validity Guarantees
                </h3>
                <ul className="list-disc pl-5 space-y-1 text-slate-400">
                  <li>Zero defective test images ever enter training or fitting.</li>
                  <li>Thresholds are calibrated exclusively on held-out normal validation images.</li>
                  <li>Continuous raw scores are used for Image AUROC, Pixel AUROC, and AUPRO.</li>
                  <li>Synthetic verification fixtures are strictly distinguished from real official datasets.</li>
                  <li>Datasets are never merged across benchmarks; seeds and preprocessing are logged.</li>
                </ul>
              </div>
              <div className="space-y-1.5">
                <h3 className="font-display text-sm font-semibold text-slate-100">
                  Known Methodological Limitations
                </h3>
                <ul className="list-disc pl-5 space-y-1 text-slate-400">
                  <li>Local patch models exhibit degraded recall on MVTec LOCO AD global logical anomalies.</li>
                  <li>Browser Float32Array engine uses compact 64–128px tensors for real-time CPU speed; full WideResNet-50-2 256px training is provided in <code className="text-slate-200">python/</code>.</li>
                  <li>Official MVTec AD / LOCO AD / AD 2 archives require manual acceptance of CC BY-NC-SA 4.0 terms and cannot be redistributed inside public repositories.</li>
                </ul>
              </div>
            </div>
          </section>

          <section className="bg-[#111827] border border-slate-800 rounded-xl p-6 space-y-4">
            <h3 className="font-display text-base font-semibold text-slate-100">
              Implemented Anomaly Detection Architectures
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {(['autoencoder', 'padim', 'patchcore', 'fastflow'] as ModelArchitecture[]).map(
                (arch) => {
                  const m = MODEL_REGISTRY[arch];
                  return (
                    <div
                      key={arch}
                      className="p-4 rounded-lg bg-[#0B0F17] border border-slate-800 space-y-1.5"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <h4 className="font-display text-sm font-semibold text-slate-100">
                          {m.name}
                        </h4>
                        <span className="text-[11px] text-sky-400">{m.family}</span>
                      </div>
                      <p className="text-xs text-slate-400 leading-relaxed">{m.shortSummary}</p>
                      <div className="text-[11px] text-slate-500 pt-1">{m.paperReference}</div>
                    </div>
                  );
                }
              )}
            </div>
          </section>
        </div>
      )}

      {section === 'competition' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <section className="bg-[#111827] border border-slate-800 rounded-xl p-6 space-y-3">
              <h2 className="font-display text-base font-semibold text-slate-100">
                5-Minute Live Competition Demonstration Script
              </h2>
              <ol className="list-decimal pl-5 space-y-2 text-xs text-slate-300 leading-relaxed">
                <li>
                  <strong className="text-slate-100">0:00–0:45 · Problem &amp; Protocol:</strong>{' '}
                  Show Executive Overview. Explain normal-only training, zero fabricated metrics
                  (&ldquo;Not evaluated&rdquo; prior to running), and strict train/val/test isolation.
                </li>
                <li>
                  <strong className="text-slate-100">0:45–2:00 · Live Training &amp; Comparison:</strong>{' '}
                  In Model Laboratory, run &ldquo;Compare All 4 Models&rdquo; on MVTec AD bottle.
                  Highlight real measured AUROC, Pixel AUROC, AUPRO, ROC/PR curves, and latency.
                </li>
                <li>
                  <strong className="text-slate-100">2:00–3:15 · Live Visual Inspection:</strong>{' '}
                  Open Live Inspection, upload a test image, show the pixel-level heatmap, overlay,
                  Autoencoder reconstruction, and decoupled Image Quality Assessment panel. Download
                  the JSON inspection report.
                </li>
                <li>
                  <strong className="text-slate-100">3:15–4:30 · Scientific Experiments:</strong>{' '}
                  Show Experiment B (10%–100% data efficiency), Experiment C (noise/blur/JPEG
                  degradation), and MVTec LOCO AD structural vs logical anomaly breakdown.
                </li>
                <li>
                  <strong className="text-slate-100">4:30–5:00 · Reproducibility &amp; Python Repo:</strong>{' '}
                  Highlight saved JSON checkpoints, CSV/JSON evidence export, pytest suite, and
                  PyTorch/Anomalib/Streamlit production architecture.
                </li>
              </ol>
            </section>

            <section className="bg-[#111827] border border-slate-800 rounded-xl p-6 space-y-3">
              <h2 className="font-display text-base font-semibold text-slate-100">
                10-Minute Presentation Outline
              </h2>
              <ul className="space-y-1.5 text-xs text-slate-300 leading-relaxed">
                <li>1. Industrial Visual Inspection Challenge &amp; Why Supervised Classifiers Fail (1 min)</li>
                <li>2. Research Question &amp; Multi-Dataset Benchmark Design: MVTec AD, VisA, LOCO AD, AD 2 (1.5 min)</li>
                <li>3. Architecture Comparison: Autoencoder vs PaDiM vs PatchCore vs FastFlow (2 min)</li>
                <li>4. Scientific Validity: Normal-Only Validation Thresholds &amp; Pixel AUPRO (1 min)</li>
                <li>5. Live System Demo &amp; Real-Time Heatmap Localization (2 min)</li>
                <li>6. Experimental Findings: Data Efficiency, Corruption Robustness &amp; Ablations (1.5 min)</li>
                <li>7. Limitations (Logical Anomalies, Domain Shift) &amp; Future Work (1 min)</li>
              </ul>
            </section>
          </div>

          <section className="bg-[#111827] border border-slate-800 rounded-xl p-6 space-y-4">
            <h3 className="font-display text-base font-semibold text-slate-100">
              Anticipated Judge Questions &amp; Technically Accurate Answers
            </h3>
            <div className="divide-y divide-slate-800">
              {JUDGE_QA.map((item, idx) => (
                <div key={idx} className="py-3.5 space-y-1">
                  <div className="text-sm font-medium text-slate-100">{item.q}</div>
                  <p className="text-xs text-slate-400 leading-relaxed">{item.a}</p>
                </div>
              ))}
            </div>
          </section>
        </div>
      )}

      {section === 'python_cli' && (
        <section className="bg-[#111827] border border-slate-800 rounded-xl p-6 space-y-4">
          <div>
            <h2 className="font-display text-base font-semibold text-slate-100">
              Python 3.11 + PyTorch + Anomalib + Streamlit Research Suite
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              In addition to this live browser ML platform, the repository packages the complete
              modular Python research codebase under <code className="text-slate-200">python/</code>{' '}
              for GPU/CPU WideResNet-50-2 training on full-resolution MVTec AD, VisA, LOCO AD, and
              MVTec AD 2 archives.
            </p>
          </div>

          <pre className="p-4 rounded-lg bg-[#0B0F17] border border-slate-800 text-xs font-mono text-slate-300 overflow-x-auto leading-relaxed">
{`# 1. Create Python 3.11 environment and install dependencies
cd python
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt

# 2. Verify hardware (CPU/CUDA), environment, and dataset layout
python -m src.data.validation --dataset mvtec_ad --data-dir data/raw/mvtec_ad

# 3. Train Autoencoder, PaDiM, PatchCore (WideResNet-50-2), or FastFlow
python -m src.training.train --model patchcore --dataset mvtec_ad --category bottle --seed 42
python -m src.training.train --model padim --dataset mvtec_ad --category bottle --seed 42
python -m src.training.train --model autoencoder --dataset mvtec_ad --category bottle --epochs 25

# 4. Evaluate metrics (Image AUROC, Pixel AUROC, AUPRO, F1, Latency)
python -m src.evaluation.evaluate --dataset mvtec_ad --category bottle

# 5. Run Robustness, Data Efficiency, and PatchCore Ablation studies
python -m src.robustness.run --dataset mvtec_ad --category bottle

# 6. Launch Streamlit enterprise dashboard & run pytest suite
streamlit run dashboard/app.py
pytest -q`}
          </pre>
        </section>
      )}
    </div>
  );
};
