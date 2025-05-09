# Immunova Project

This repository contains a collection of machine learning models for various tasks, including image classification, graph neural networks, survival analysis, and multimodal learning. The project is structured to facilitate model training, evaluation, and deployment via FastAPI.

---

## 📂 Directory Structure

```
/immunova/

├── api.py
├── model.py
├── requirements.txt
├── til_classification/
│   ├── grad_cam.py
│   ├── model.py
│   └── train.py
├── drug_optimization/
│   ├── gnn_model.py
│   └── rl_optimizer.py
├── survival_analysis/
│   ├── km_validation.py
│   └── survival_transformer.py
├── treatment_response/
│   ├── fusion_model.py
│   ├── genomics_model.py
│   └── imaging_model.py
├── validation/
│   ├── clinical_validation.py
│   └── external_datasets_loader.py
|── preprocessing/
    |── WSIs/
        |── WSIprocessor.py 
└── datasets/
    ├── TIL/
    │   └── exampl.jpg
    ├── Non-TIL/
    │   └── ex3.jpg
    ├── graphs/
    │   └── ex.pt
    ├── csv/
    │   └── sample.csv
    ├── multimodal_images/
    │   └── ex.jpg
    └── multimodal_csv/
        └── ex2.csv
```
#The datasets folder hadn't been uploaded since there are no datasets currently
---

## 📁 Dataset Preparation

All datasets should be uploaded under the `/immunova/datasets/` directory. Below are the specific folders required for each model:

### 1. Image Classification (TIL Classifier)
- Path: `/immunova/datasets/`
- Folder Structure:
```
/datasets/
├── TIL/
├── Non-TIL/
```

### 2. Graph Neural Network (GNN Model)
- Path: `/immunova/datasets/graphs/`
- File Format: `.pt` or `.pickle`
- Upload graph data files to this folder.

### 3. Survival Analysis (Survival Transformer)
- Path: `/immunova/datasets/csv/`
- File Format: `.csv`
- Example Filename: `survival_data.csv`

### 4. Multimodal Model (Fusion Model)
- Image Path: `/immunova/datasets/multimodal_images/`
- Genomics Data Path: `/immunova/datasets/multimodal_csv/`

---

## 📌 Training Instructions

### Image Classification (TIL Classifier)
```
%cd /immunova/til_classification
!python3 train.py
```

### Graph Neural Network (GNN Model)
```
%cd /immunova/drug_optimization
!python3 gnn_model.py
```

### Survival Analysis (Survival Transformer)
```
%cd /immunova/survival_analysis
!python3 survival_transformer.py
```

### Multimodal Model (Fusion Model)
```
%cd /immunova/treatment_response
!python3 fusion_model.py
```

---

## 📌 Inference (Model Usage)

Once training is complete, you can load the models for inference using the following approach:

```python
from til_classification.model import TILClassifier
import torch

model = TILClassifier()
model.load_state_dict(torch.load('/immunova/til_classification/til_classifier.pth'))
model.eval()
```

Each model will have its own method for inference. Refer to the corresponding `.py` files for details.

---

## 📌 FastAPI Deployment

To deploy the models via FastAPI, use the following command:

```
%cd /immunova
!uvicorn api:app --reload --port 8000
```

This will launch a FastAPI server where you can send requests for inference.

---

## 📌 Requirements Installation

Install all required packages by running:

```
!pip install -r /immunova/requirements.txt
```

Or manually install necessary packages:

```
!pip install torch torchvision torch-geometric lifelines fastapi uvicorn
```

---

## ✅ Important Notes

- Ensure all datasets are uploaded to the correct directories before training.
- Check that your file paths are correct according to this `README.md` file.
- Follow the training commands precisely to avoid errors.

---
