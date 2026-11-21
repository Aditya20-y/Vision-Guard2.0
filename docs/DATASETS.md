# VisionGuard — Dataset Acquisition & Licensing Guide

All industrial datasets remain independently identifiable, independently configurable, and independently evaluated. Never merge datasets during benchmark training or metric reporting.

## 1. MVTec AD (Primary Benchmark)
- **Official Source**: https://www.mvtec.com/research-teaching/datasets/mvtec-ad
- **License**: CC BY-NC-SA 4.0 (Non-Commercial, ShareAlike)
- **Scale**: 5,354 high-resolution images across 15 categories (~4.9 GB archive)
- **Priority Categories**: `bottle`, `screw`, `metal_nut`, `capsule`, `cable`, `hazelnut`
- **Local Placement**: `python/data/raw/mvtec_ad/<category>/{train,test,ground_truth}`
- **Web App Import**: Use **Dataset Explorer → Import Real Local Folder** and select any `<category>` directory.

## 2. VisA (Independent Multi-Category Benchmark)
- **Official Source**: https://registry.opendata.aws/visa/
- **Official Documentation**: https://github.com/amazon-research/spot-diff
- **License**: CC BY 4.0
- **Scale**: 10,821 images across 12 object categories (~1.8 GB archive)
- **Dedicated Adapter**: Preserves VisA `Data/Images/{Normal,Anomaly}`, `Data/Masks/Anomaly`, and official `split_csv/1cls.csv` layout independently of MVTec AD.

## 3. MVTec LOCO AD (Structural vs. Logical Anomalies)
- **Official Source**: https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad
- **License**: CC BY-NC-SA 4.0
- **Scale**: 3,644 images across 5 categories (`breakfast_box`, `juice_bottle`, `pushpins`, `screw_bag`, `splicing_connectors`)
- **Scientific Note**: Evaluates structural and logical anomalies separately to expose the inherent limitations of local patch-level appearance detectors on global constraint violations.

## 4. MVTec AD 2 (Challenging Conditions & Distribution Shift)
- **Official Source**: https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2
- **License**: CC BY-NC-SA 4.0
- **Protocol**: Strictly uses `test_public/` official public evaluation data. Private evaluation splits are never accessed or inferred from.

## 5. BTAD (Optional)
- **Official Source**: http://avires.dimi.uniud.it/papers/btad/btad.zip
- **Status**: Optional adapter implemented (`BTADAdapter`). Does not block core platform operation if unavailable.
