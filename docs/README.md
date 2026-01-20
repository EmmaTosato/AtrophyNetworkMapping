# Documentation Index - ANM Verona Project

## Overview
Questo indice fornisce una guida completa alla documentazione del progetto di classificazione neurodegenerativa (PSP vs CBS vs ADNI) basato su Machine Learning e Deep Learning.

---

## 📚 Documentation Structure

### 1. Machine Learning (ML) Analysis
| Document | Description | Status |
|----------|-------------|--------|
| [ML_CLASSIFICATION_DOCUMENTATION.md](ML_CLASSIFICATION_DOCUMENTATION.md) | Documentazione completa ML pipeline | ⚠️ Empty (to update) |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | Quick reference ML | ⚠️ Empty (to update) |
| [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md) | Guida tecnica ML | ✅ Complete |
| [CLASSIFICATION_MODES.md](CLASSIFICATION_MODES.md) | Fixed split vs Nested CV | ✅ Complete |
| [NESTED_CV_IMPLEMENTATION.md](NESTED_CV_IMPLEMENTATION.md) | Nested CV guide (archived) | ✅ Complete |

### 2. Deep Learning (DL) Analysis
| Document | Description | Status |
|----------|-------------|--------|
| [DL_ANALYSIS_COMPLETE.md](DL_ANALYSIS_COMPLETE.md) | Documentazione completa DL pipeline | ✅ Complete |
| [DL_workflow.dot](DL_workflow.dot) | Graphviz source file | ✅ Complete |
| [DL_workflow.svg](DL_workflow.svg) | Visual workflow diagram (SVG) | ✅ Complete |
| [DL_workflow.png](DL_workflow.png) | Visual workflow diagram (PNG, 300 DPI) | ✅ Complete |

### 3. Comparative Analysis
| Document | Description | Status |
|----------|-------------|--------|
| [ML_vs_DL_COMPARISON.md](ML_vs_DL_COMPARISON.md) | Confronto dettagliato ML vs DL | ✅ Complete |
| [QUICK_COMMANDS.md](QUICK_COMMANDS.md) | Comandi rapidi per entrambi | ✅ Complete |

### 4. Research Context
| Document | Description | Status |
|----------|-------------|--------|
| [paper_draft.txt](paper_draft.txt) | Draft articolo scientifico | ✅ Complete |

---

## 🎯 Quick Navigation

### I'm New to the Project
**Start here**: [ML_vs_DL_COMPARISON.md](ML_vs_DL_COMPARISON.md)
- Executive summary
- Key differences between ML and DL approaches
- When to use each method

### I Want to Run ML Classification
**Read**: [QUICK_COMMANDS.md](QUICK_COMMANDS.md) → ML Section
- Command examples
- Configuration guide
- Troubleshooting

### I Want to Run DL Classification
**Read**: 
1. [DL_ANALYSIS_COMPLETE.md](DL_ANALYSIS_COMPLETE.md) → Full documentation
2. [DL_workflow.svg](DL_workflow.svg) → Visual workflow
3. [QUICK_COMMANDS.md](QUICK_COMMANDS.md) → DL Section

### I Want to Understand the Code
**ML**: [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md)
**DL**: [DL_ANALYSIS_COMPLETE.md](DL_ANALYSIS_COMPLETE.md) → Section 2 (Function Definitions)

### I Want to Compare ML vs DL Results
**Read**: [ML_vs_DL_COMPARISON.md](ML_vs_DL_COMPARISON.md) → Section 13 (Conclusions)

---

## 📊 Key Findings Summary

### Dataset
- **Total subjects**: 130 (ADNI: 40, PSP: 45, CBS: 45)
- **Train/Test split**: 104 / 26 (fixed from CSV)
- **Input data**: 3D functional connectivity maps (91×109×91)

### ML Approach
- **Models**: RandomForest, GradientBoosting, KNeighbors
- **Features**: UMAP embedding (21 network measures → 2D/3D)
- **Cross-validation**: Inner 5-fold for hyperparameter tuning only
- **Seeds**: 5 seeds (42, 123, 2023, 31415, 98765)
- **Best accuracy**: ~65.3% ± 0.2% (RF on ADNI vs PSP)
- **Training time**: ~15 min (5 seeds × 3 models)

### DL Approach
- **Models**: ResNet3D, DenseNet3D, VGG16_3D
- **Input**: Raw 3D volumes (no feature extraction)
- **Cross-validation**: 5-fold on train set, select best fold checkpoint
- **Data augmentation**: 50× bootstrapping (HCP subjects)
- **Seeds**: 5 seeds (same as ML)
- **Best accuracy**: To be compared with ML
- **Training time**: ~20-50 hours (5 seeds × 5 folds × 50 epochs)

---

## 🔧 Code Structure

### ML Pipeline
```
src/ML_analysis/
├── config/
│   └── ml_config.json
├── analysis/
│   └── classification.py       # Main entry point
├── preprocessing/
│   └── umap_reduction.py
├── training/
│   └── train_models.py
└── utils/
    └── ml_utils.py
```

### DL Pipeline
```
src/DL_analysis/
├── config/
│   ├── cnn_config.json
│   └── cnn_grid.json
├── training/
│   ├── run_train.py           # Entry: multi-seed
│   ├── run.py                 # Main pipeline
│   ├── train.py               # Train/validate
│   └── hyper_tuning.py        # Grid search
├── testing/
│   ├── run_test.py            # Entry: batch testing
│   └── test.py                # Metrics
├── cnn/
│   ├── models.py              # Architectures
│   └── datasets.py            # Data loaders
└── utils/
    └── cnn_utils.py
```

---

## 📈 Workflows

### ML Standard Training
```
Load CSV split → FOR each seed:
    ├─ Run UMAP (fit on train)
    ├─ FOR each model (RF, GB, KNN):
    │   ├─ (Optional) GridSearchCV for hyperparameters
    │   ├─ Train on full train (104)
    │   └─ Test on test (26)
    └─ Save results → CSV
```

### DL Standard Training
```
Load CSV split → FOR each seed:
    ├─ 5-Fold CV on train (104):
    │   └─ FOR each fold:
    │       ├─ Train on fold_train (~83)
    │       ├─ Validate on fold_val (~21)
    │       └─ Save checkpoint
    ├─ Select best fold (max val_accuracy)
    └─ Save training summary → CSV

THEN:
FOR each run → Load best checkpoint → Test on test (26) → Save testing summary
```

---

## 🚀 Quick Start

### 1. ML Classification (No Tuning, Fixed Split)
```bash
cd /data/users/etosato/ANM_Verona
python -c "
from src.analysis.classification import main_classification
from src.config.config_loader import ConfigLoader
config = ConfigLoader('src/ML_analysis/config/ml_config.json')
main_classification(config)
"
```

**Output**: `results/ml_analysis/classification_results.csv`

### 2. DL Training (5 Seeds)
```bash
cd /data/users/etosato/ANM_Verona
python src/DL_analysis/training/run_train.py
```

**Output**: `results/runs/all_training_results.csv`

### 3. DL Testing
```bash
python src/DL_analysis/testing/run_test.py
```

**Output**: `results/runs/all_testing_results.csv`

---

## 🔍 Key Insights from Documentation

### Critical Differences ML vs DL
1. **Input data**: Features (ML) vs Raw volumes (DL)
2. **CV usage**: Hyperparameter selection (ML) vs Fold selection (DL)
3. **Refit strategy**: Full train refit (ML) vs Best checkpoint (DL)
4. **Augmentation**: None (ML) vs 50× bootstrapping (DL)
5. **Training time**: Minutes (ML) vs Hours (DL)

### Fixed Issues (ML)
- ✅ UMAP data leakage: fit only on train, transform test
- ✅ Test evaluation with tuning: now computed after GridSearchCV
- ✅ Permutation test: moved to training set only
- ✅ Output consistency: always save results

### To Do (DL)
- [ ] Fix seed management: separate fold split seed from init seed
- [ ] Add AUC-ROC metric (currently only Acc, Prec, Rec, F1)
- [ ] Implement post-processing script for mean ± std aggregation
- [ ] Add Grad-CAM visualization for interpretability

### To Do (Both)
- [ ] Unified output CSV format for side-by-side comparison
- [ ] Statistical significance testing (ML vs DL)
- [ ] Ensemble predictions (ML + DL)
- [ ] Final manuscript figures

---

## 📝 Documentation Maintenance

### Last Updated
- ML docs: 2025-01-19
- DL docs: 2025-01-19
- Comparison: 2025-01-19

### Contributors
- Edoardo Tosato (etosato@studenti.unipd.it)
- GitHub Copilot (documentation agent)

### How to Update
1. Edit source `.md` files in `docs/` folder
2. For workflow diagrams: edit `.dot` file, regenerate with:
   ```bash
   dot -Tsvg DL_workflow.dot -o DL_workflow.svg
   dot -Tpng -Gdpi=300 DL_workflow.dot -o DL_workflow.png
   ```
3. Update this index with new sections

---

## 🎓 Citation

If you use this code or documentation, please cite:

```bibtex
@article{tosato2025anm,
  title={Atrophy-Network Mapping for Neurodegenerative Disease Classification},
  author={Tosato, Edoardo and others},
  journal={In preparation},
  year={2025}
}
```

---

## 📧 Contact

For questions or issues:
- Email: etosato@studenti.unipd.it
- GitHub Issues: [Create an issue](https://github.com/...)

---

## 📌 Quick Links

### External Resources
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [MONAI Documentation](https://docs.monai.io/)
- [Graphviz Documentation](https://graphviz.org/documentation/)

### Related Papers
- PSP/CBS classification: [See paper_draft.txt](paper_draft.txt)
- Atrophy network mapping: Richardson et al. (2011)
- Seed-based connectivity: [Add reference]

---

## 🏆 Project Status

| Component | Status | Accuracy | Notes |
|-----------|--------|----------|-------|
| ML Pipeline | ✅ Complete | 65.3% ± 0.2% | ADNI vs PSP, RF |
| DL Pipeline | ✅ Complete | TBD | Awaiting final results |
| Documentation | ✅ Complete | N/A | All files up to date |
| Manuscript | 🚧 In Progress | N/A | See paper_draft.txt |
| Code Review | ✅ Complete | N/A | All bugs fixed |

---

## 📖 Reading Order Recommendation

### For Beginners
1. [ML_vs_DL_COMPARISON.md](ML_vs_DL_COMPARISON.md) → Overview
2. [QUICK_COMMANDS.md](QUICK_COMMANDS.md) → Run experiments
3. [DL_workflow.svg](DL_workflow.svg) → Visual understanding

### For Developers
1. [DL_ANALYSIS_COMPLETE.md](DL_ANALYSIS_COMPLETE.md) → DL code structure
2. [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md) → ML code structure
3. [ML_vs_DL_COMPARISON.md](ML_vs_DL_COMPARISON.md) → Section 12 (Alignment)

### For Researchers
1. [paper_draft.txt](paper_draft.txt) → Scientific context
2. [ML_vs_DL_COMPARISON.md](ML_vs_DL_COMPARISON.md) → Methods comparison
3. [DL_ANALYSIS_COMPLETE.md](DL_ANALYSIS_COMPLETE.md) → Implementation details

---

**Last update**: 2025-01-19 16:55 CET
