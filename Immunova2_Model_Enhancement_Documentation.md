
# Immunova 2.0 Model Enhancement – Internal Documentation

## Overview

This document outlines the proposed enhancements to the four core models in the Immunova 1.0 framework. These enhancements reflect a transition toward a more structured, analysis-granularity-based architecture, separating cell-level from patient-level tasks, while integrating advanced AI strategies and immunological reasoning.

---

## 1. TIL Classification Model (Cell-Level)

### Purpose:
Detect TIL presence and, in the future, subtype at the patch or cell level from whole slide images (WSIs).

### Enhancements:
- **Multi-class TIL Subtyping**: CD8+, CD4+, Treg, NK detection via morphology or spatial transcriptomics.
- **Graph-based Spatial Modeling**: Use GNNs to understand spatial immune-tumor interactions.
- **Segmentation Integration**: Contextualize TIL location relative to stroma, tumor, etc.
- **Explainability**: Upgrade Grad-CAM to Grad-CAM++ or Score-CAM.

---

## 2. Treatment Response Prediction (Patient-Level)

### Purpose:
Predict binary outcome: will the patient respond to immunotherapy?

### Enhancements:
- **TIL Subtype Features**: Incorporate output from TIL classification model.
- **Multi-omics Fusion**: Add proteomics, TCR-seq, epigenomics.
- **Transfer Learning**: Train on multi-cancer, fine-tune for specific types.
- **Immune Evasion Profiling**: Include tumor escape mechanism pathways.

---

## 3. Survival Analysis Model (Patient-Level, Temporal)

### Purpose:
Estimate time-to-event (e.g., overall survival, progression-free survival).

### Enhancements:
- **Deep Survival Models**: Replace Cox with DeepSurv or Survival Transformer.
- **TIL Evolution**: Model how immune context changes over time.
- **Cause-specific Analysis**: Separate death causes (tumor, immune, toxic).
- **Dynamic KM Curves**: Tie to biomarker thresholds dynamically.

---

## 4. Drug Optimization Model (Patient-Level, Policy Learning)

### Purpose:
Optimize treatment sequences using RL + GNN based on immune-tumor state.

### Enhancements:
- **TIL-aware State Encoding**: Policy is informed by TIL subtype context.
- **Biological Graphs**: Encode drug-mechanism links into treatment network.
- **Trial Simulation**: Predict outcomes of trial arms per immune profile.
- **Explainable Decisions**: Justify sequence recommendation.

---

## Unified Workflow Diagram

[WSI + RNA-seq] → 
  TIL Classification →
  Subtype Quantification →
      ┌──────────────────────┐
      │                      │
      ↓                      ↓
Treatment Response      Survival Estimation
Prediction (binary)     (time-to-event)  
          ↓
    Drug Path Optimization

---

## Note for Developers
- Each model's output should be modularized for downstream chaining.
- All enhancements should be tied to clinically interpretable logic.
- Refer to `genomics_model.py`, `fusion_model.py`, `survival_transformer.py`, and `rl_optimizer.py` for implementation anchors.

---

## Next Step
See `Immunova2_Beginner_Guide.md` for ML/AI accessible explanations of this architecture.
