
# Immunova 2.0 – Beginner-Friendly Guide

This guide explains the purpose and function of the four core models in the Immunova AI system in plain, accessible language. It is intended for those without a background in machine learning (ML) or artificial intelligence (AI), such as medical researchers, clinicians, or junior data scientists.

---

## Key Concepts

### What is TIL?
**TIL (Tumor-Infiltrating Lymphocyte)**: A type of immune cell (like T cells) that has entered a tumor. TILs are important because their presence often means the immune system is trying to fight the cancer.

---

### What is RNA-seq?
**RNA-seq (RNA sequencing)**: A method used to measure which genes are active in a sample (gene expression). It tells us what the tumor and surrounding immune cells are "doing" at a molecular level.

---

### What is WSI?
**WSI (Whole Slide Image)**: A high-resolution digital scan of a tissue slide from a biopsy. AI can analyze these images to find patterns (like the presence of TILs).

---

## Model 1: TIL Classification Model (Cell-Level)

**What it does:**  
This model looks at images of tumor tissue and decides whether immune cells (TILs) are present in small regions of the image.

**Why it matters:**  
Knowing where the immune cells are helps us understand the immune response to the tumor.

**Future enhancement:**  
Eventually, we want this model to not just detect immune cells, but to identify *what type* they are (e.g. CD8+ killer T cells, helper T cells, suppressive Tregs).

---

## Model 2: Treatment Response Prediction Model (Patient-Level)

**What it does:**  
Combines image data and RNA-seq data to predict whether a patient will respond to immunotherapy.

**Why it matters:**  
Immunotherapy only works for some patients. This model helps doctors know who will benefit most, and who might need another treatment.

**Future enhancement:**  
Add more types of molecular data (like protein levels or T-cell receptor patterns) to make the prediction more accurate.

---

## Model 3: Survival Analysis Model (Patient-Level with Time)

**What it does:**  
Estimates how long a patient will live (or stay progression-free) after treatment based on their tumor and immune profile.

**Why it matters:**  
It helps predict prognosis and guide treatment plans.

**Future enhancement:**  
Model how the immune system changes over time and how that impacts survival.

---

## Model 4: Drug Optimization Model (Patient-Level, Decision Simulation)

**What it does:**  
Simulates different treatment sequences (e.g. surgery → chemo → immunotherapy) and recommends the one with the best outcome for the patient.

**Why it matters:**  
Some treatment orders work better than others. This model helps personalize treatment paths.

**Future enhancement:**  
Make the model smarter by feeding in immune cell information (like TIL types) so it can recommend drugs that best match the patient’s immune profile.

---

## Putting It All Together

Immunova is like a layered brain:
- **Layer 1:** Finds immune cells in images.
- **Layer 2:** Predicts if immunotherapy will work.
- **Layer 3:** Estimates survival time.
- **Layer 4:** Plans the best treatment path.

Each model builds on the previous one. Together, they help bring personalized cancer treatment closer to reality.

---

If you’re a beginner and want to contribute:
- Start by exploring how the TIL classification model works.
- Learn how gene expression data is stored and interpreted.
- Join the team that matches your interests: images, genomics, outcomes, or optimization.

Welcome to the Immunova project!
