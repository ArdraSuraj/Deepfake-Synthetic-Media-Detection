# 🧠 Deepfake and Synthetic Media Detection - Data Analysis

## 📌 Project Overview

This project focuses on analyzing a dataset for detecting deepfake and synthetic media. The goal is to clean the dataset, extract meaningful insights, and compute accurate statistical values that help distinguish between real and fake media.

---

## 📂 Dataset Information

* **Dataset Name:** Deepfake and Synthetic Media Detection Dataset
* **Source:** Kaggle
* **File Used:** `deepfake_detection_metadata_dataset.csv`

### 📊 Features in Dataset:

* `media_id` – Unique identifier
* `media_type` – Type of media (image/video/audio)
* `content_category` – Category of content
* `face_count` – Number of faces detected
* `audio_present` – Whether audio exists
* `lip_sync_score` – Lip synchronization accuracy
* `visual_artifacts_score` – Visual distortion level
* `compression_level` – Compression applied
* `lighting_inconsistency_score` – Lighting irregularities
* `source_platform` – Platform source
* `generation_method` – Method used to generate media
* `label` – Target (Real / Fake)

---

## ⚙️ Steps Performed

### 1️⃣ Data Loading

* Loaded dataset using Pandas from Kaggle input path.

### 2️⃣ Data Cleaning

* Removed duplicate records
* Handled missing values
* Standardized column names

### 3️⃣ Exploratory Data Analysis (EDA)

* Checked dataset shape and structure
* Identified data types and null values

### 4️⃣ Accurate Value Extraction

* Computed total number of records
* Calculated label distribution (Real vs Fake)
* Derived percentage distribution
* Computed mean values of key features

---

## 📈 Key Results

### ✅ Label Distribution

* Exact count of real vs fake media
* Percentage split of dataset

### ✅ Feature Insights

* Fake media shows:

  * Higher `visual_artifacts_score`
  * Higher `lighting_inconsistency_score`
* Real media shows:

  * Better `lip_sync_score`

---

## 🧪 Sample Code

```python
import pandas as pd

# Load dataset
df = pd.read_csv("deepfake_detection_metadata_dataset.csv")

# Clean data
df = df.drop_duplicates()
df = df.dropna()

# Label distribution
print(df['label'].value_counts())
print(df['label'].value_counts(normalize=True) * 100)

# Feature analysis
print(df.groupby('label')[[
    'lip_sync_score',
    'visual_artifacts_score',
    'lighting_inconsistency_score'
]].mean())
```

---

## 🎯 Conclusion

The dataset provides strong indicators for detecting deepfake media. Features like visual artifacts and lighting inconsistencies play a crucial role in distinguishing synthetic content from real media.

---

## 🚀 Future Work

* Train machine learning models for classification
* Apply deep learning (CNN) for image/video detection
* Improve detection accuracy using feature engineering

---

## 🛠️ Technologies Used

* Python 🐍
* Pandas
* NumPy
* KaggleHub

---

## 📎 Author

Ardra Suraj

---
