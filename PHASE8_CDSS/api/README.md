# Phase 8 CDSS — API Documentation

## Base URL
```
http://localhost:8000
```

## Endpoints

### GET /
System information and available endpoints.

### GET /health
Health check.

### POST /predict
Get risk prediction for a patient.
```json
{
  "patient_id": "TCGA-05-4249",  // optional
  "age": 65,
  "cancer_type": "LUAD",
  "stage": "IIA",
  "smoking_status": "Former",
  "gene_expression": {}  // optional
}
```

### POST /simulate
Run Digital Twin treatment simulation.
```json
{
  "patient_id": "TCGA-05-4249",  // optional
  "age": 65,
  "cancer_type": "LUAD",
  "stage": "IIA",
  "smoking_status": "Former",
  "treatment": "chemo"
}
```

### POST /explain
Get explainability outputs.
```json
{
  "age": 65,
  "cancer_type": "LUAD",
  "stage": "IIA",
  "smoking_status": "Former"
}
```

### GET /evidence
Get evidence panel data from Phase 7.1 deliverables.

### GET /statements
Get all permitted and forbidden clinical interpretation statements.

### GET /patients?limit=20
List patients from the state matrix.

## Safety
All endpoints enforce the Phase 7.1 constraint contract. Stage IIIB/IV patients are hard-blocked.
