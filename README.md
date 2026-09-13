# Lung Cancer Digital Twin Project

A federated neural–mechanistic digital-twin framework for externally validated survival prediction and safety-constrained clinical decision support in lung cancer.

## Overview

This repository contains the complete analysis code for a research prototype that integrates deep learning survival models with mechanistic tumor growth modeling, federated learning, and a safety-constrained clinical decision support system (CDSS) for non-small cell lung cancer (NSCLC).

**⚠️ Important:** This is a research prototype and is **NOT for clinical use**. It has not undergone prospective or regulatory validation.

## Project Structure

```
Lung_Cancer_Digital_Twin_Project/
├── 00_CODE/                    # Main analysis scripts for all phases
├── 01_RAW_DATA/                # Data download scripts and accession IDs
├── 02_PROCESSED_DATA/          # Processed datasets (not included in repo - regenerated from raw data)
├── 03_ANALYSIS_READY_DATA/     # Analysis-ready datasets (not included in repo - regenerated from processed data)
├── 04_DOCUMENTATION/           # Manuscript and documentation
├── PHASE1C_GEO_HARMONIZATION/  # Phase 1C: GEO harmonization
├── PHASE1D_GEO_SCALE_ALIGNMENT/# Phase 1D: Scale alignment
├── PHASE2_EXTERNAL_VALIDATION/ # Phase 2: External validation
├── PHASE3_DEEP_LEARNING/       # Phase 3: DeepSurv model
├── PHASE4_MECHANISTIC_MODELING/# Phase 4: Mechanistic modeling
├── PHASE5_NEURAL_MECHANISTIC_INTEGRATION/ # Phase 5: Integration
├── PHASE6_FEDERATED_LEARNING/  # Phase 6: Federated learning
├── PHASE7_DIGITAL_TWIN/        # Phase 7: Digital twin
└── PHASE8_CDSS/                # Phase 8: Clinical decision support system
```

## Data Access

This project uses publicly available datasets:
- **TCGA-LUAD** and **TCGA-LUSC** (The Cancer Genome Atlas)
- **GEO cohorts**: GSE30219, GSE31210, GSE50081, GSE72094

Due to size and licensing restrictions, raw data files are not included in this repository. Download scripts and accession IDs are provided in `01_RAW_DATA/`.

## Installation

### Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies

### Setup

```bash
# Clone repository
git clone https://github.com/BRomeo777/Lung-Ca.git
cd Lung-Ca

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running Phase Analyses

Each phase can be run independently from the `00_CODE/` directory:

```bash
# Phase 1: Data harmonization
python 00_CODE/run_phase1c_geo.py
python 00_CODE/run_phase1d_scale.py

# Phase 2: External validation
python 00_CODE/run_phase2_analysis.py

# Phase 3: DeepSurv model
python 00_CODE/run_phase3_deepsurv.py

# Phase 3.1: Statistical rigor
python 00_CODE/run_phase3_rigor.py

# Phase 4: Mechanistic modeling
python 00_CODE/run_phase4_mechanistic.py

# Phase 5: Neural-mechanistic integration
python 00_CODE/run_phase5_integration.py

# Phase 6: Federated learning
python 00_CODE/run_phase6_federated.py

# Phase 7: Digital twin
python 00_CODE/run_phase7_digital_twin.py
```

### Running CDSS Tests

```bash
cd PHASE8_CDSS
pytest tests/test_phase8_cdss.py -v
```

## Key Results

- **External C-index**: 0.659 (DeepSurv) vs 0.616 (Cox baseline)
- **Calibration slopes**: 0.61–0.99 after recalibration
- **Federated learning**: 0.632 pooled C-index (between centralized 0.677 and no-collaboration 0.543)
- **CDSS safety**: 87/87 tests passed (37 functional + 50 boundary)

## Reproducibility

All analyses use fixed seeds (42) and versioned artifacts. The Phase 3.1 rigor and calibration analysis is reproduced with a single command:

```bash
python 00_CODE/run_phase3_rigor.py
```

## Citation

If you use this code or data, please cite the corresponding manuscript. Citation details will be added upon publication.

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Ethics

This study uses de-identified public datasets (TCGA, GEO). It is a computational research prototype and does not constitute medical advice or a medical device.

## Contact

For questions about this analysis, please contact bananeza777@gmail.com

## Acknowledgments

This work represents part of an ongoing research program in computational oncology. The framework demonstrates methodological rigor in translational oncology machine learning.
