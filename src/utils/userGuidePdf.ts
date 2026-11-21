/**
 * Bulletproof VisionGuard User Guide PDF builder & downloader.
 *
 * 1. Attempts to fetch the prebuilt static PDF and strictly verifies the `%PDF-` magic header
 *    (preventing corrupted HTML error/auth pages from ever being saved as .pdf).
 * 2. Falls back seamlessly to generating an identical valid PDF 1.4 binary in memory via
 *    pure TypeScript so downloads work 100% reliably inside sandboxed iframes and offline.
 */

const PAGE_W = 595.28;
const PAGE_H = 841.89;
const MARGIN_X = 50.0;
const MARGIN_TOP = 58.0;
const MARGIN_BOTTOM = 54.0;
const CONTENT_W = PAGE_W - 2 * MARGIN_X;

function pdfEscape(raw: string): string {
  const clean = raw
    .replace(/\u2014/g, '--')
    .replace(/\u2013/g, '-')
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[\u201c\u201d]/g, '"')
    .replace(/\u2022/g, '*')
    .replace(/\u00b7/g, '|')
    .replace(/\u00d7/g, 'x')
    .replace(/\u2192/g, '->')
    .replace(/\u03bc/g, 'mean')
    .replace(/\u03c3/g, 'std')
    .replace(/[^\x20-\x7e]/g, ' ');
  return clean.replace(/\\/g, '\\\\').replace(/\(/g, '\\(').replace(/\)/g, '\\)');
}

function wrapText(text: string, maxChars: number): string[] {
  const words = text.split(/\s+/);
  const lines: string[] = [];
  let current = '';
  for (const w of words) {
    if (!current) {
      current = w;
    } else if (current.length + 1 + w.length <= maxChars) {
      current += ` ${w}`;
    } else {
      lines.push(current);
      current = w;
    }
  }
  if (current) lines.push(current);
  return lines.length > 0 ? lines : [''];
}

function estimateChars(fontSize: number, widthPt: number, isMono = false): number {
  const factor = isMono ? 0.6 : 0.5;
  return Math.max(16, Math.floor(widthPt / (fontSize * factor)));
}

class ClientPdfBuilder {
  private pages: string[] = [];
  private cmds: string[] = [];
  private y = PAGE_H - MARGIN_TOP;

  constructor() {
    this.newPage();
  }

  private newPage(): void {
    if (this.cmds.length > 0) {
      this.pages.push(this.cmds.join('\n'));
    }
    this.cmds = [];
    this.y = PAGE_H - MARGIN_TOP;
    const pageNum = this.pages.length + 1;
    if (pageNum > 1) {
      this.cmds.push('0.40 0.45 0.55 rg');
      this.cmds.push(
        `BT /F1 8.5 Tf ${MARGIN_X.toFixed(2)} ${(PAGE_H - 34).toFixed(2)} Td (${pdfEscape(
          'VisionGuard -- Industrial Visual Anomaly Detection Platform | User Guide & Judge Playbook'
        )}) Tj ET`
      );
      this.cmds.push('0.85 0.88 0.92 RG 0.6 w');
      this.cmds.push(
        `${MARGIN_X.toFixed(2)} ${(PAGE_H - 40).toFixed(2)} m ${(PAGE_W - MARGIN_X).toFixed(2)} ${(
          PAGE_H - 40
        ).toFixed(2)} l S`
      );
      this.y = PAGE_H - 64.0;
    }
  }

  private ensureSpace(neededPt: number): void {
    if (this.y - neededPt < MARGIN_BOTTOM) {
      this.newPage();
    }
  }

  drawCoverBanner(): void {
    const bannerH = 112.0;
    const top = PAGE_H - 42.0;
    const bottom = top - bannerH;
    this.cmds.push('0.06 0.09 0.16 rg');
    this.cmds.push(
      `${MARGIN_X.toFixed(2)} ${bottom.toFixed(2)} ${CONTENT_W.toFixed(2)} ${bannerH.toFixed(2)} re f`
    );
    this.cmds.push('0.05 0.65 0.91 rg');
    this.cmds.push(
      `${MARGIN_X.toFixed(2)} ${(top - 3).toFixed(2)} ${CONTENT_W.toFixed(2)} 3.0 re f`
    );

    this.cmds.push('0.56 0.84 0.98 rg');
    this.cmds.push(
      `BT /F2 9 Tf ${(MARGIN_X + 18).toFixed(2)} ${(top - 26).toFixed(2)} Td (${pdfEscape(
        'USER GUIDE & COMPETITION SHOWCASE PLAYBOOK'
      )}) Tj ET`
    );
    this.cmds.push('0.98 0.99 1.00 rg');
    this.cmds.push(
      `BT /F2 22 Tf ${(MARGIN_X + 18).toFixed(2)} ${(top - 54).toFixed(2)} Td (VisionGuard) Tj ET`
    );
    this.cmds.push('0.82 0.87 0.93 rg');
    this.cmds.push(
      `BT /F1 10.5 Tf ${(MARGIN_X + 18).toFixed(2)} ${(top - 74).toFixed(2)} Td (${pdfEscape(
        'Intelligent Industrial Defect Detection, Anomaly Localization & Visual Inspection Platform'
      )}) Tj ET`
    );
    this.cmds.push('0.58 0.66 0.76 rg');
    this.cmds.push(
      `BT /F1 8.5 Tf ${(MARGIN_X + 18).toFixed(2)} ${(top - 94).toFixed(2)} Td (${pdfEscape(
        'Models: Autoencoder (Stable) | PaDiM (Stable) | PatchCore (Optimized) | FastFlow (Experimental)'
      )}) Tj ET`
    );
    this.y = bottom - 22.0;
  }

  sectionHeading(num: string, title: string): void {
    this.ensureSpace(42);
    this.y -= 6;
    this.cmds.push('0.05 0.45 0.72 rg');
    this.cmds.push(
      `BT /F2 13 Tf ${MARGIN_X.toFixed(2)} ${this.y.toFixed(2)} Td (${pdfEscape(`${num}. ${title}`)}) Tj ET`
    );
    this.y -= 6;
    this.cmds.push('0.82 0.87 0.92 RG 0.8 w');
    this.cmds.push(
      `${MARGIN_X.toFixed(2)} ${this.y.toFixed(2)} m ${(PAGE_W - MARGIN_X).toFixed(2)} ${this.y.toFixed(2)} l S`
    );
    this.y -= 14;
  }

  subheading(title: string): void {
    this.ensureSpace(26);
    this.y -= 3;
    this.cmds.push('0.10 0.14 0.22 rg');
    this.cmds.push(
      `BT /F2 10.5 Tf ${MARGIN_X.toFixed(2)} ${this.y.toFixed(2)} Td (${pdfEscape(title)}) Tj ET`
    );
    this.y -= 14;
  }

  calloutBox(label: string, text: string): void {
    const fontSize = 9.3;
    const lines = wrapText(text, estimateChars(fontSize, CONTENT_W - 28));
    const lineH = fontSize * 1.45;
    const boxH = 22 + lines.length * lineH + 10;
    this.ensureSpace(boxH + 10);
    const boxTop = this.y + 4;
    const boxBottom = boxTop - boxH;

    this.cmds.push('0.95 0.97 0.99 rg');
    this.cmds.push(
      `${MARGIN_X.toFixed(2)} ${boxBottom.toFixed(2)} ${CONTENT_W.toFixed(2)} ${boxH.toFixed(2)} re f`
    );
    this.cmds.push('0.78 0.85 0.92 RG 0.8 w');
    this.cmds.push(
      `${MARGIN_X.toFixed(2)} ${boxBottom.toFixed(2)} ${CONTENT_W.toFixed(2)} ${boxH.toFixed(2)} re S`
    );

    let cy = boxTop - 15;
    this.cmds.push('0.04 0.39 0.65 rg');
    this.cmds.push(
      `BT /F2 9.2 Tf ${(MARGIN_X + 12).toFixed(2)} ${cy.toFixed(2)} Td (${pdfEscape(label)}) Tj ET`
    );
    cy -= 14;
    this.cmds.push('0.14 0.19 0.26 rg');
    for (const ln of lines) {
      this.cmds.push(
        `BT /F1 ${fontSize.toFixed(1)} Tf ${(MARGIN_X + 12).toFixed(2)} ${cy.toFixed(2)} Td (${pdfEscape(ln)}) Tj ET`
      );
      cy -= lineH;
    }
    this.y = boxBottom - 12;
  }

  bullet(lead: string, body: string): void {
    const fontSize = 9.4;
    const full = lead ? `${lead}: ${body}` : body;
    const lines = wrapText(full, estimateChars(fontSize, CONTENT_W - 18));
    const lineH = fontSize * 1.42;
    this.ensureSpace(lines.length * lineH + 4);
    this.cmds.push('0.05 0.45 0.72 rg');
    this.cmds.push(
      `BT /F2 9.5 Tf ${(MARGIN_X + 4).toFixed(2)} ${this.y.toFixed(2)} Td (-) Tj ET`
    );
    this.cmds.push('0.16 0.20 0.28 rg');
    for (const ln of lines) {
      this.cmds.push(
        `BT /F1 ${fontSize.toFixed(1)} Tf ${(MARGIN_X + 14).toFixed(2)} ${this.y.toFixed(2)} Td (${pdfEscape(ln)}) Tj ET`
      );
      this.y -= lineH;
    }
    this.y -= 2.5;
  }

  table(headers: string[], rows: string[][], colWidths: number[]): void {
    const fontSize = 8.6;
    const lineH = 11.8;
    this.ensureSpace(40);
    const hTop = this.y + 4;
    const hH = 20.0;
    this.cmds.push('0.10 0.15 0.24 rg');
    this.cmds.push(
      `${MARGIN_X.toFixed(2)} ${(hTop - hH).toFixed(2)} ${CONTENT_W.toFixed(2)} ${hH.toFixed(2)} re f`
    );
    let cx = MARGIN_X;
    this.cmds.push('0.97 0.98 1.00 rg');
    for (let i = 0; i < headers.length; i++) {
      this.cmds.push(
        `BT /F2 8.5 Tf ${(cx + 6).toFixed(2)} ${(hTop - 13.5).toFixed(2)} Td (${pdfEscape(headers[i])}) Tj ET`
      );
      cx += colWidths[i];
    }
    this.y = hTop - hH;

    for (let rIdx = 0; rIdx < rows.length; rIdx++) {
      const row = rows[rIdx];
      const wrappedCols = row.map((cell, ci) =>
        wrapText(cell, Math.max(10, estimateChars(fontSize, colWidths[ci] - 12)))
      );
      const maxLines = Math.max(...wrappedCols.map((c) => c.length));
      const rowH = maxLines * lineH + 10;
      if (this.y - rowH < MARGIN_BOTTOM) this.newPage();

      const rTop = this.y;
      const rBottom = rTop - rowH;
      if (rIdx % 2 === 1) {
        this.cmds.push('0.96 0.97 0.99 rg');
        this.cmds.push(
          `${MARGIN_X.toFixed(2)} ${rBottom.toFixed(2)} ${CONTENT_W.toFixed(2)} ${rowH.toFixed(2)} re f`
        );
      }
      this.cmds.push('0.84 0.87 0.91 RG 0.5 w');
      this.cmds.push(
        `${MARGIN_X.toFixed(2)} ${rBottom.toFixed(2)} m ${(MARGIN_X + CONTENT_W).toFixed(2)} ${rBottom.toFixed(2)} l S`
      );

      let colX = MARGIN_X;
      for (let ci = 0; ci < wrappedCols.length; ci++) {
        let cy = rTop - 12;
        this.cmds.push('0.14 0.19 0.27 rg');
        const font = ci === 0 ? '/F2' : '/F1';
        for (const ln of wrappedCols[ci]) {
          this.cmds.push(
            `BT ${font} ${fontSize.toFixed(1)} Tf ${(colX + 6).toFixed(2)} ${cy.toFixed(2)} Td (${pdfEscape(ln)}) Tj ET`
          );
          cy -= lineH;
        }
        colX += colWidths[ci];
      }
      this.y = rBottom;
    }
    this.y -= 10;
  }

  codeBlock(codeLines: string[]): void {
    const fontSize = 8.5;
    const lineH = 12.2;
    const boxH = codeLines.length * lineH + 16;
    this.ensureSpace(boxH + 8);
    const boxTop = this.y + 4;
    const boxBottom = boxTop - boxH;
    this.cmds.push('0.08 0.11 0.17 rg');
    this.cmds.push(
      `${MARGIN_X.toFixed(2)} ${boxBottom.toFixed(2)} ${CONTENT_W.toFixed(2)} ${boxH.toFixed(2)} re f`
    );
    let cy = boxTop - 14;
    this.cmds.push('0.88 0.93 0.98 rg');
    for (const ln of codeLines) {
      this.cmds.push(
        `BT /F3 ${fontSize.toFixed(1)} Tf ${(MARGIN_X + 12).toFixed(2)} ${cy.toFixed(2)} Td (${pdfEscape(ln)}) Tj ET`
      );
      cy -= lineH;
    }
    this.y = boxBottom - 12;
  }

  buildBytes(): Uint8Array<ArrayBuffer> {
    if (this.cmds.length > 0) {
      this.pages.push(this.cmds.join('\n'));
      this.cmds = [];
    }

    const totalPages = this.pages.length;
    const finalPages = this.pages.map((body, idx) => {
      const footer = [
        '0.85 0.88 0.92 RG 0.5 w',
        `${MARGIN_X.toFixed(2)} 38.00 m ${(PAGE_W - MARGIN_X).toFixed(2)} 38.00 l S`,
        '0.45 0.50 0.58 rg',
        `BT /F1 8 Tf ${MARGIN_X.toFixed(2)} 26.00 Td (${pdfEscape('VisionGuard User Guide & Competition Playbook')}) Tj ET`,
        `BT /F1 8 Tf ${(PAGE_W - MARGIN_X - 52).toFixed(2)} 26.00 Td (Page ${idx + 1} of ${totalPages}) Tj ET`,
      ];
      return `${body}\n${footer.join('\n')}`;
    });

    const objects: string[] = [];
    const addObj = (content: string) => {
      objects.push(content);
      return objects.length;
    };

    const catalogId = addObj('');
    const pagesId = addObj('');
    const f1Id = addObj('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>');
    const f2Id = addObj('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>');
    const f3Id = addObj('<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>');

    const pageIds: number[] = [];
    for (const streamText of finalPages) {
      const contentId = addObj(`<< /Length ${streamText.length} >>\nstream\n${streamText}\nendstream`);
      const pageId = addObj(
        `<< /Type /Page /Parent ${pagesId} 0 R /MediaBox [0 0 ${PAGE_W.toFixed(2)} ${PAGE_H.toFixed(2)}] ` +
          `/Resources << /Font << /F1 ${f1Id} 0 R /F2 ${f2Id} 0 R /F3 ${f3Id} 0 R >> >> ` +
          `/Contents ${contentId} 0 R >>`
      );
      pageIds.push(pageId);
    }

    objects[catalogId - 1] = `<< /Type /Catalog /Pages ${pagesId} 0 R >>`;
    objects[pagesId - 1] = `<< /Type /Pages /Kids [${pageIds.map((p) => `${p} 0 R`).join(' ')}] /Count ${pageIds.length} >>`;

    const chunks: string[] = ['%PDF-1.4\n'];
    let byteCursor = chunks[0].length;
    const offsets: number[] = [0];

    for (let i = 0; i < objects.length; i++) {
      offsets.push(byteCursor);
      const block = `${i + 1} 0 obj\n${objects[i]}\nendobj\n`;
      chunks.push(block);
      byteCursor += block.length;
    }

    const xrefPos = byteCursor;
    chunks.push(`xref\n0 ${objects.length + 1}\n`);
    chunks.push('0000000000 65535 f \n');
    for (let i = 1; i <= objects.length; i++) {
      chunks.push(`${String(offsets[i]).padStart(10, '0')} 00000 n \n`);
    }
    chunks.push(
      `trailer\n<< /Size ${objects.length + 1} /Root ${catalogId} 0 R >>\nstartxref\n${xrefPos}\n%%EOF\n`
    );

    const fullStr = chunks.join('');
    const bytes = new Uint8Array(new ArrayBuffer(fullStr.length));
    for (let i = 0; i < fullStr.length; i++) {
      bytes[i] = fullStr.charCodeAt(i) & 0xff;
    }
    return bytes;
  }
}

export function buildUserGuidePdfBytes(): Uint8Array<ArrayBuffer> {
  const pdf = new ClientPdfBuilder();
  pdf.drawCoverBanner();

  pdf.sectionHeading('1', 'How to Explain VisionGuard Simply (30-Second Pitch)');
  pdf.calloutBox(
    'For General Audiences, Friends & Non-Technical Stakeholders',
    "Imagine a factory producing thousands of glass bottles, metal nuts, or circuit boards every hour. 99.9% of parts are good, and defects are unpredictable -- a crack today, an oil stain tomorrow, a bent pin next week. Standard AI needs thousands of broken examples to learn, which factories do not have. VisionGuard flips the problem: we train the AI ONLY on good, defect-free products. Once it learns what 'normal' looks like, it inspects any new product, flags whether it is Normal or Defective, and paints a glowing heatmap directly onto the exact scratch, crack, or stain."
  );
  pdf.calloutBox(
    'One-Line Technical Summary (For Competition Judges & Faculty)',
    'VisionGuard is an unsupervised industrial visual anomaly detection and pixel-level localization platform that systematically benchmarks reconstruction-based (Autoencoder) and feature-based (PaDiM, PatchCore, FastFlow) engines across MVTec AD, VisA, MVTec LOCO AD, and MVTec AD 2 under strict zero-leakage scientific evaluation protocols.'
  );

  pdf.sectionHeading('2', 'Step-by-Step User Guide (All 7 Dashboard Pages)');
  pdf.subheading('Page 1: Executive Overview');
  pdf.bullet('Zero Fabricated Metrics', "When opened fresh, metrics honestly display 'Not evaluated' until real models are trained and evaluated.");
  pdf.bullet('One-Click Baseline', "Click 'Train & Evaluate 4 Models' (or 'Run Core Benchmark' in the top bar) to train Autoencoder, PaDiM, PatchCore, and FastFlow live on MVTec AD bottle and populate real Image AUROC, Pixel AUROC, and F1 charts.");

  pdf.subheading('Page 2: Model Laboratory (with Engine Maturity Badges)');
  pdf.bullet('Maturity Catalog', 'Inspect and select from four engines tagged by maturity: PatchCore (Optimized -- coreset memory bank tuned for speed & recall), PaDiM (Stable -- per-patch Mahalanobis Gaussian), Convolutional Autoencoder (Stable -- reconstruction baseline), and FastFlow (Experimental -- 2D normalizing flow density estimator).');
  pdf.bullet('Train & Compare', "Choose Dataset, Category, Input Resolution (64-128px), and Random Seed, then click 'Train & Evaluate' or 'Compare All 4 Models'. Filter benchmark rows by Maturity Tier or Dataset, click any row to inspect live ROC, Precision-Recall, Confusion Matrix, and Training Loss plots, or export checkpoints (JSON) and benchmark tables (CSV).");

  pdf.subheading('Page 3: Live Inspection (Main Demonstration Screen)');
  pdf.bullet('Select or Upload', "Choose a dataset, category, and trained model (click 'Train Model First' if untrained). Pick any unseen test image ('OK' or 'DEF') or click 'Upload PNG / JPG' to inspect a custom image.");
  pdf.bullet('4 Visual Outputs', "Click 'Run Live Inspection' to inspect (1) Preprocessed Input, (2) Pixel-Level Anomaly Heatmap, (3) Blended Localization Overlay with live opacity slider, and (4) Autoencoder Reconstruction or Ground-Truth Mask.");
  pdf.bullet('Decoupled Image Quality Audit', "Laplacian blur variance, luminance, RMS contrast, resolution, and 8x8 JPEG blockiness are evaluated independently so a blurry or dark camera feed raises a reliability warning instead of a false defect verdict. Click 'Download Inspection Report' to save a JSON audit.");

  pdf.subheading('Page 4: Dataset Explorer & Local Real Dataset Import');
  pdf.bullet('Independent Multi-Dataset Registry', 'Explore official sources, licenses (CC BY-NC-SA 4.0 / CC BY 4.0), citations, and sample/mask galleries for MVTec AD, VisA, MVTec LOCO AD, MVTec AD 2, and optional BTAD.');
  pdf.bullet('Importing Official Datasets', "Click 'Import Real Local Folder' and select any downloaded MVTec AD / VisA / LOCO AD / AD 2 category folder from disk. VisionGuard automatically checks image readability, mask alignment, and split leakage.");

  pdf.subheading('Page 5: Research Experiments (Controlled Scientific Studies)');
  pdf.bullet('Exp B -- Data Efficiency', 'Subsamples 10%, 25%, 50%, and 100% of normal training images deterministically to quantify accuracy under scarce factory data.');
  pdf.bullet('Exp C -- Robustness', 'Applies 6 industrial degradations (Gaussian noise, blur, brightness shift, contrast shift, JPEG compression, resize/resample) at severities 1, 3, and 5 without corrupting ground-truth masks.');
  pdf.bullet('Exp D -- PatchCore Ablations', 'Varies Coreset Ratio (10%-100%), Feature Layers (L1, L2, L1+L2, L1+L2+L3), and Resolution to show memory-bank vs. latency vs. AUROC trade-offs.');
  pdf.bullet('Exp E/F -- Cross-Dataset & Category Difficulty', 'Evaluates 8 categories across 4 datasets independently and demonstrates why MVTec LOCO AD logical anomalies (missing/extra parts) are harder for patch detectors than surface defects.');

  pdf.subheading('Pages 6 & 7: Inspection History & Methodology / Competition Kit');
  pdf.bullet('Inspection History', 'Persists metadata-only inspection logs locally (no unnecessary raw image retention) with search, Normal/Anomalous filtering, CSV export, and delete/clear controls.');
  pdf.bullet('Methodology & Demo', 'Includes full algorithm documentation, scientific validity guarantees, 5-minute demo script, 10-minute presentation outline, Judge Q&A, and Python 3.11 / PyTorch CLI instructions.');

  pdf.sectionHeading('3', 'Winning 5-Minute Live Demo Flow for Competition Judges');
  pdf.table(
    ['Time', 'Dashboard Page', 'Live Actions', 'Key Talking Points for Judges'],
    [
      [
        '0:00-0:45',
        'Executive Overview',
        "Point out 'Not evaluated' before clicking anything.",
        'Emphasize scientific honesty: zero hardcoded or fabricated metrics. Every score is measured live on unseen test splits.',
      ],
      [
        '0:45-1:45',
        'Model Laboratory',
        "Click 'Compare All 4 Models' on MVTec AD bottle; click PatchCore row.",
        'Highlight maturity badges (Optimized, Stable, Experimental), live ROC/PR curves, Pixel AUROC, AUPRO, and normal-validation-only threshold (mean + 2.5*std).',
      ],
      [
        '1:45-3:00',
        'Live Inspection',
        'Inspect a DEF sample; adjust Overlay Intensity; switch to Autoencoder; download JSON report.',
        'Show real-time pixel localization, Autoencoder reconstruction, and decoupled Image Quality Assessment preventing camera blur from causing false defect alarms.',
      ],
      [
        '3:00-4:15',
        'Research Experiments',
        'Run Exp B (Data Efficiency) or Exp C (Robustness), or show Exp E/F LOCO AD breakdown.',
        'Answer the core research question quantitatively: compare scarce normal data (10%-100%), image corruptions, and structural vs. logical anomaly failure modes.',
      ],
      [
        '4:15-5:00',
        'Dataset Explorer & Methodology',
        "Show 'Import Real Local Folder' and Python 3.11 / PyTorch CLI tab.",
        'Explain verification fixtures vs. official datasets, strict CC BY-NC-SA 4.0 license compliance, and full PyTorch WideResNet-50-2 / Streamlit / pytest codebase.',
      ],
    ],
    [58, 92, 145, CONTENT_W - 295]
  );

  pdf.sectionHeading('4', 'Real-World Industrial Use Cases to Showcase');
  pdf.bullet('Electronics & PCB Quality (VisA pcb1)', 'Detects missing SMD components, trace scratches, and solder anomalies without enumerating every possible board fault.');
  pdf.bullet('Precision Metal & Automotive (MVTec AD metal_nut, screw, sheet_metal)', 'Localizes micro-cracks, surface dents, and oil contamination on machined brushed steel.');
  pdf.bullet('Pharmaceutical & Bottling Lines (capsule, bottle)', 'Flags chipped bottle rims, capsule discoloration, and packaging contamination at inspection speed.');
  pdf.bullet('Kit & Assembly Completeness (MVTec LOCO AD)', 'Illustrates where local appearance detectors must be combined with global count and layout reasoning.');

  pdf.sectionHeading('5', 'Top 5 Tough Judge Questions & Crisp Answers');
  pdf.bullet('Q1: Are these real ML models or pre-saved demo outputs?', "100% real. Before training, metrics read 'Not evaluated'. Clicking Train executes real Adam backpropagation (Autoencoder), Mahalanobis covariance inversion (PaDiM), greedy k-center coreset sampling + k-NN (PatchCore), and invertible affine coupling flow fitting (FastFlow).");
  pdf.bullet('Q2: Why not just train a standard defect classifier (CNN/YOLO)?', 'Real factories have abundant healthy parts and rare, unpredictable defect types. Supervised classifiers fail on unseen defects; normal-only anomaly detection flags any deviation from healthy appearance.');
  pdf.bullet('Q3: Is the anomaly score a probability like 85%?', 'No. Raw scores are unbounded reconstruction errors, Mahalanobis distances, or k-NN distances. VisionGuard never calls them probabilities and calibrates thresholds strictly on held-out normal validation images.');
  pdf.bullet('Q4: What if the factory camera blurs or lighting dims?', 'Our decoupled Image Quality Assessment checks blur, illumination, contrast, resolution, and JPEG blockiness independently, warning the operator without misclassifying camera faults as broken parts.');
  pdf.bullet('Q5: What is the biggest scientific limitation of your system?', 'Local patch-based methods excel on structural surface defects but degrade on MVTec LOCO AD logical anomalies (e.g., missing item in a box where every local patch looks normal). Measuring and reporting this honestly is a core research contribution.');

  pdf.sectionHeading('6', 'Launching Locally & Pushing to GitHub');
  pdf.codeBlock([
    '# 1. Run Live Interactive Web Platform (Port 3000)',
    'npm install && npm run dev',
    '',
    '# 2. Run Python 3.11 + PyTorch WideResNet-50-2 + Streamlit Suite',
    'cd python && python -m venv .venv && pip install -r requirements.txt',
    'python -m src.training.train --model patchcore --dataset mvtec_ad --category bottle',
    'pytest -q && streamlit run dashboard/app.py',
    '',
    '# 3. Push Repository to GitHub (from authenticated machine)',
    'git push -u origin main',
  ]);

  return pdf.buildBytes();
}

export async function downloadUserGuidePdf(): Promise<{ source: 'static' | 'generated'; sizeBytes: number }> {
  let pdfBytes: Uint8Array<ArrayBuffer> | null = null;
  let source: 'static' | 'generated' = 'generated';

  try {
    const res = await fetch('./VisionGuard_User_Guide.pdf', { credentials: 'same-origin' });
    if (res.ok) {
      const ab = await res.arrayBuffer();
      const buf = new Uint8Array(ab);
      // Verify %PDF- header bytes (0x25 0x50 0x44 0x46 0x2D) so HTML proxy pages are never saved as PDF
      if (
        buf.length > 512 &&
        buf[0] === 0x25 &&
        buf[1] === 0x50 &&
        buf[2] === 0x44 &&
        buf[3] === 0x46 &&
        buf[4] === 0x2d
      ) {
        pdfBytes = buf;
        source = 'static';
      }
    }
  } catch {
    // Fall back cleanly to client-side binary generation
  }

  if (!pdfBytes) {
    pdfBytes = buildUserGuidePdfBytes();
    source = 'generated';
  }

  const blob = new Blob([pdfBytes], { type: 'application/pdf' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = 'VisionGuard_User_Guide.pdf';
  anchor.rel = 'noopener';
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  setTimeout(() => URL.revokeObjectURL(url), 5000);

  return { source, sizeBytes: pdfBytes.byteLength };
}
