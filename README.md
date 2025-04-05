# 🧬 Immunova

**Immunova** is an open-source framework for multi-modal, AI-powered treatment response prediction and survival modeling in pancreatic cancer. We bring together histopathology, genomics, and clinical data into one modular ecosystem — designed for researchers, data scientists, and clinicians.

---

## 🚀 Features

- 🔬 **TIL Classification** with CNNs (ResNet, EfficientNet)
- 🧠 **Fusion Networks** combining imaging + genomics
- 📈 **Survival Prediction** using Transformer + Cox models
- 💊 **Drug Optimization** via GNNs + RL for combo therapies
- 📊 **Validation Tools** for KM curves & external datasets
- 🌐 **Frontend Dashboard** for visualization & deployment
- ☁️ **Azure Integration** for scalable model serving

---
## 📁 Project Structure

immunova/
├── til_classification/         # TIL CNNs + Grad-CAM
│   ├── train.py
│   └── ...
├── treatment_response/         # Imaging/genomics/fusion models
│   ├── fusion_model.py
│   └── ...
├── survival_analysis/          # Transformer survival models
│   ├── km_validation.py
│   └── ...
├── drug_optimization/          # GNN + RL drug strategies
│   └── ...
├── validation/                 # Clinical/external dataset validators
│   └── ...
├── frontend/                   # Web frontend (HTML/CSS/JS)
│   └── ...
├── deployment/                 # Azure deployment configs
│   └── ...
├── tests/                      # Unit tests and model tests
├── requirements.txt
├── CONTRIBUTING.md
├── LICENSE
└── README.md


## 🛠 Quickstart

```bash
# Train a TIL classifier
python til_classification/train.py

# Run inference on genomics + imaging
python treatment_response/fusion_model.py

# Validate survival predictions
python survival_analysis/km_validation.py
