"""Export all Phase 1B ML data to a single Excel workbook with multiple sheets."""
import pandas as pd
import numpy as np
from pathlib import Path

READY_DIR = Path(__file__).resolve().parent.parent / "03_ANALYSIS_READY_DATA"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "ML_RESULTS" / "reports"
OUT_PATH = Path(__file__).resolve().parent.parent / "ML_RESULTS" / "Phase1B_ML_Data.xlsx"

def main():
    sheets = {}

    # 1. Training clinical data
    train = pd.read_csv(READY_DIR / "TCGA_train.csv")
    for c in train.columns:
        if train[c].dtype == object:
            train[c] = train[c].replace("not_available", np.nan)
    if "Age" in train.columns:
        train["Age"] = pd.to_numeric(train["Age"], errors="coerce")
    sheets["Train_Clinical"] = train

    # 2. Validation clinical data
    val = pd.read_csv(READY_DIR / "TCGA_internal_validation.csv")
    for c in val.columns:
        if val[c].dtype == object:
            val[c] = val[c].replace("not_available", np.nan)
    if "Age" in val.columns:
        val["Age"] = pd.to_numeric(val["Age"], errors="coerce")
    sheets["Validation_Clinical"] = val

    # 3. Training expression (top selected genes only — full matrix is too large)
    expr_train = pd.read_csv(READY_DIR / "TCGA_train_expression.tsv.gz", sep="\t", index_col=0)
    # Get selected genes from feature importance
    imp = pd.read_csv(REPORTS_DIR / "feature_importance.csv")
    selected_genes = sorted(imp["Feature"].unique())
    # Filter to genes that are in expression index
    gene_list = [g for g in selected_genes if g in expr_train.index]
    sheets["Train_Expression_Selected"] = expr_train.loc[gene_list].T

    # 4. Validation expression (same selected genes)
    expr_val = pd.read_csv(READY_DIR / "TCGA_internal_validation_expression.tsv.gz", sep="\t", index_col=0)
    gene_list_v = [g for g in gene_list if g in expr_val.index]
    sheets["Validation_Expression_Selected"] = expr_val.loc[gene_list_v].T

    # 5. Model performance
    sheets["Model_Performance"] = pd.read_csv(REPORTS_DIR / "model_performance.csv")

    # 6. Hyperparameter tuning
    sheets["Hyperparameter_Tuning"] = pd.read_csv(REPORTS_DIR / "hyperparameter_tuning_report.csv")

    # 7. Feature importance
    sheets["Feature_Importance"] = pd.read_csv(REPORTS_DIR / "feature_importance.csv")

    # 8. Missing data report (parse the text file into a table)
    missing_rows = []
    with open(REPORTS_DIR / "missing_data_report.txt", "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line and not line.startswith("=") and not line.startswith("-") and not line.startswith("MISSING") and not line.startswith("──"):
            parts = line.split()
            if len(parts) >= 4 and parts[0] not in ("Applied", "Dropped", "Kept"):
                try:
                    missing_rows.append({"Feature": parts[0], "Train_Missing": parts[1], "Train_Pct": parts[2], "Decision": " ".join(parts[3:])})
                except:
                    pass
    if missing_rows:
        sheets["Missing_Data"] = pd.DataFrame(missing_rows)

    # 9. Selected genes summary
    sheets["Selected_Genes"] = pd.DataFrame({"Gene_ID": gene_list})

    # 10. Data audit summary (key stats)
    audit_stats = pd.DataFrame([
        {"Metric": "Training patients (raw)", "Value": 814},
        {"Metric": "Training patients (after dropping missing OS)", "Value": 738},
        {"Metric": "Validation patients (raw)", "Value": 204},
        {"Metric": "Validation patients (after dropping missing OS)", "Value": 181},
        {"Metric": "Total features before preprocessing", "Value": 60664},
        {"Metric": "Clinical features (kept)", "Value": "Age, Cancer_Type, Stage, Smoking_Status"},
        {"Metric": "Clinical features (dropped)", "Value": "Sex, Treatment (>50% missing)"},
        {"Metric": "Expression genes (raw)", "Value": 60660},
        {"Metric": "Genes after variance filter (top)", "Value": 1000},
        {"Metric": "Genes after univariate Cox (p<0.05)", "Value": 127},
        {"Metric": "Total features in final model", "Value": 143},
        {"Metric": "Train events (deaths)", "Value": 316},
        {"Metric": "Train event rate", "Value": "42.8%"},
        {"Metric": "Validation events (deaths)", "Value": 80},
        {"Metric": "Validation event rate", "Value": "44.2%"},
        {"Metric": "Random seed (split + models)", "Value": 42},
        {"Metric": "CV folds", "Value": 5},
        {"Metric": "Models trained", "Value": "CoxPH, Coxnet, RSF, XGBoost"},
        {"Metric": "Best validation C-index", "Value": "Coxnet (0.5998)"},
        {"Metric": "CV winner", "Value": "CoxPH (0.6486)"},
    ])
    sheets["Audit_Summary"] = audit_stats

    # Write to Excel
    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=(sheet_name.endswith("_Selected") or sheet_name == "Selected_Genes"))
            # Auto-adjust column widths
            ws = writer.sheets[sheet_name]
            for col in ws.columns:
                max_len = max(len(str(cell.value)) if cell.value else 0 for cell in col)
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 50)

    print(f"Excel saved to: {OUT_PATH}")
    print(f"Sheets: {list(sheets.keys())}")
    for name, df in sheets.items():
        print(f"  {name}: {df.shape}")

if __name__ == "__main__":
    main()
