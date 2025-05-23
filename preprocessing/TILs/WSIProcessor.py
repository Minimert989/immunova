import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

# ——— CONFIGURATION ———

METADATA_CSV = r"C:\TILs\TCGA-TILs\images-tcga-tils-metadata.csv"
IMAGE_ROOT   = r"C:\TILs\TCGA-TILs"
OUTPUT_DIR   = r"C:\TILs\til_preprocessed_by_study"

IMAGE_SIZE = (224, 224)
LABEL_MAP  = {"til-negative": 0, "til-positive": 1}

# ImageNet statistics for normalization
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# ——— SETUP ———
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Loading metadata...")
df = pd.read_csv(METADATA_CSV)
df["label_bin"] = df["label"].map(LABEL_MAP)

# ——— GROUP BY STUDY (CANCER LINE), THEN BY BARCODE (PATIENT) ———
study_groups = df.groupby("study")
overall_mapping = {}
counter = 0

print("Processing each study and patient...")
for study, study_df in tqdm(study_groups, desc="Studies"):
    study_dir = os.path.join(OUTPUT_DIR, study)
    os.makedirs(study_dir, exist_ok=True)

    barcode_groups = study_df.groupby("barcode")
    study_mapping = {}

    for barcode, group in barcode_groups:
        imgs, labs = [], []
        label_distribution = {"til-negative": 0, "til-positive": 0}

        for _, row in group.iterrows():
            img_path = os.path.join(IMAGE_ROOT, row["path"])
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(IMAGE_SIZE, Image.BILINEAR)
                arr = np.array(img, dtype=np.uint8)
                if arr.shape != (224, 224, 3):
                    print(f" Skipped malformed: {row['path']}")
                    continue

                # Convert to tensor, float32, normalize
                arr_tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
                arr_tensor = (arr_tensor - IMAGENET_MEAN) / IMAGENET_STD
                imgs.append(arr_tensor)

                # Update label distribution
                label = row["label"]
                label_distribution[label] += 1

                labs.append(row["label_bin"])
            except Exception as e:
                print(f" Error loading {row['path']}: {e}")
                continue

            if counter >= 7:
                break
            else:
                counter += 1

        if not imgs:
            continue

        # Handle label types (binary)
        label_type = "binary"  # We're only using binary labels here

        imgs_tensor = torch.stack(imgs)  # (N, 3, 224, 224)
        labs_tensor = torch.tensor(labs, dtype=torch.long)  # (N,)

        patient_data = {
            "images": imgs_tensor,
            "labels": labs_tensor,
            "study": study,
            "barcode": barcode,
            "partition": row["partition"],  # Using partition from the CSV
            "label_type": label_type,
            "label_distribution": label_distribution,
            "num_patches": len(imgs),
        }

        out_file = os.path.join(study_dir, f"{barcode}.pt")
        torch.save(patient_data, out_file)

        study_mapping[barcode] = {
            "num_patches": len(imgs),
            "file": f"{barcode}.pt",
            "label_type": label_type,
            "label_distribution": label_distribution
        }

    overall_mapping[study] = study_mapping
    if counter >= 7:
        break

# ——— WRITE TRACEABILITY MAP ———
with open(os.path.join(OUTPUT_DIR, "study_patient_mapping.json"), "w") as f:
    json.dump(overall_mapping, f, indent=2)

print("✅ Done.")
print(f" Preprocessed data in: {OUTPUT_DIR}")
print("• One .pt file per patient per study")
print("• study_patient_mapping.json for metadata")

# ——— DEFINE DATASET CLASS ———

class TILPatientDataset(Dataset):
    def __init__(self, data_dir, mapping_json):
        self.data_dir = data_dir
        with open(mapping_json, "r") as f:
            self.mapping = json.load(f)
        self.study_barcode_pairs = [
            (study, barcode)
            for study in self.mapping
            for barcode in self.mapping[study]
        ]

    def __len__(self):
        return len(self.study_barcode_pairs)

    def __getitem__(self, idx):
        study, barcode = self.study_barcode_pairs[idx]
        pt_path = os.path.join(self.data_dir, study, f"{barcode}.pt")
        data = torch.load(pt_path)
        return data["images"], data["labels"]
