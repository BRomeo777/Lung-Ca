# Data Access Instructions

This folder contains information for downloading the public datasets used in this project. Raw data files are not included in the repository due to size and licensing restrictions.

## Datasets Used

### TCGA (The Cancer Genome Atlas)

**TCGA-LUAD (Lung Adenocarcinoma)**
- Project ID: TCGA-LUAD
- Accession: https://portal.gdc.cancer.gov/projects/TCGA-LUAD
- Data types: RNA-seq counts, clinical data, survival data

**TCGA-LUSC (Lung Squamous Cell Carcinoma)**
- Project ID: TCGA-LUSC
- Accession: https://portal.gdc.cancer.gov/projects/TCGA-LUSC
- Data types: RNA-seq counts, clinical data, survival data

### GEO (Gene Expression Omnibus)

**GSE30219**
- Accession: GSE30219
- Platform: Affymetrix HuGene 1.0
- URL: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE30219

**GSE31210**
- Accession: GSE31210
- Platform: Affymetrix HG-U133 Plus 2.0
- URL: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE31210

**GSE50081**
- Accession: GSE50081
- Platform: Affymetrix HG-U133 Plus 2.0
- URL: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE50081

**GSE72094**
- Accession: GSE72094
- Platform: Affymetrix HG-U133 Plus 2.0
- URL: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE72094

## Download Instructions

### Using TCGA GDC Data Transfer Tool

```bash
# Install GDC Data Transfer Tool
# Download from: https://gdc.cancer.gov/access-data/gdc-data-transfer-tool

# Download TCGA-LUAD data
gdc-client download -m TCGA-LUAD_manifest.txt

# Download TCGA-LUSC data
gdc-client download -m TCGA-LUSC_manifest.txt
```

### Using GEOquery (R)

```r
library(GEOquery)

# Download GEO datasets
getGEO("GSE30219", GSEMatrix = TRUE)
getGEO("GSE31210", GSEMatrix = TRUE)
getGEO("GSE50081", GSEMatrix = TRUE)
getGEO("GSE72094", GSEMatrix = TRUE)
```

### Using Python (geopandas)

```python
from GEOparse import get_GEO

# Download GEO datasets
geo1 = get_GEO("GSE30219", destdir="./data/GEO")
geo2 = get_GEO("GSE31210", destdir="./data/GEO")
geo3 = get_GEO("GSE50081", destdir="./data/GEO")
geo4 = get_GEO("GSE72094", destdir="./data/GEO")
```

## Data Processing

After downloading, place the data in the following structure:

```
01_RAW_DATA/
├── TCGA_LUAD/
│   ├── TCGA-LUAD_STAR_counts_matrix.tsv.gz
│   ├── TCGA-LUAD_clinical.tsv
│   └── TCGA-LUAD_survival.tsv
├── TCGA_LUSC/
│   ├── TCGA-LUSC_STAR_counts_matrix.tsv.gz
│   ├── TCGA-LUSC_clinical.tsv
│   └── TCGA-LUSC_survival.tsv
└── GEO/
    ├── GSE30219/
    ├── GSE31210/
    ├── GSE50081/
    └── GSE72094/
```

Then run the harmonization scripts in `00_CODE/run_phase1c_geo.py` and `00_CODE/run_phase1d_scale.py`.

## Data Licensing

- TCGA data: Open access under TCGA data use agreement
- GEO data: Public domain or as specified in individual dataset licenses

## Citations

When using these datasets, please cite:

- TCGA: https://www.cancer.gov/about-nci/organization/ccg/research/genomic-characterization/tcga
- GEO: Barrett T et al. Nucleic Acids Res. 2013;41(Database issue):D991-5.
