[![Status](https://img.shields.io/badge/Status-ongoing-orange.svg)](https://github.com/zakizulham/telco-churn-analysis-crispml/graphs/commit-activity)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-blue.svg)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Used-orange.svg)](https://scikit-learn.org/stable/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Used-darkgreen.svg)](https://xgboost.ai/)
[![SHAP](https://img.shields.io/badge/SHAP-Used-900c3f.svg)](https://shap.readthedocs.io/en/latest/)

# Analisis Prediksi Churn Pelanggan Telco (CRISP-ML)

Repositori ini mendokumentasikan analisis komprehensif dari dataset "Telco Customer Churn" Kaggle, mengikuti metodologi CRISP-ML(Q) yang terstruktur.

Tujuan utama dari proyek ini adalah untuk memprediksi **apakah** seorang pelanggan akan *churn* (berhenti berlangganan) dan **mengapa**. Analisis ini diperluas untuk mencakup pemodelan nilai pelanggan (CLV) dan segmentasi pasar (clustering), dengan fokus pada evaluasi teknis (AUC) dan evaluasi bisnis (analisis biaya retensi).

## Metodologi

Proyek ini secara ketat mengikuti alur kerja **CRISP-ML(Q) (Cross-Industry Standard Process for Machine Learning with Quality Assurance)**. Analisis dibagi menjadi beberapa fase yang logis, di mana setiap *notebook* mewakili satu atau lebih dari fase-fase tersebut.

## Dataset

* **Sumber:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Target Utama:** `Churn` (Yes/No).
* **Fitur:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`.

## Struktur Repositori

Struktur direktori dirancang untuk mencerminkan alur kerja CRISP-ML(Q).

```
telco-churn-analysis-crispml/
├── .gitignore
├── LICENSE
├── README.md
│
├── data/
│   ├── raw/            <-- (Data asli tidak dimodifikasi, diabaikan oleh Git)
│   └── prepared/       <-- (Data bersih hasil Notebook 01, diabaikan oleh Git)
│
├── models/             <-- (Artefak model yang dilatih, diabaikan oleh Git)
│
├── notebooks/
│   ├── 00_Business_Data_Understanding.ipynb
│   ├── 01_Data_Preparation.ipynb
│   ├── 02_Modeling_Churn_Classification_XAI.ipynb
│   ├── 03_Modeling_CLV_Regression.ipynb
│   └── 04_Modeling_Customer_Segmentation.ipynb
│
└── requirements.txt
```

## Alur Kerja Analisis (Per Notebook)

Setiap *notebook* dibangun di atas *notebook* sebelumnya, mengikuti alur kerja yang logis.

### 00_Business_Data_Understanding.ipynb
* **Fase CRISP:** 1. Business Understanding, 2. Data Understanding.
* **Tujuan:** Mendefinisikan pertanyaan bisnis (prediksi churn, retensi, CLV) dan melakukan Exploratory Data Analysis (EDA) lengkap. Ini termasuk menangani data `TotalCharges` yang salah (tipe data `object`) dan menganalisis distribusi kelas `Churn` yang tidak seimbang (*imbalanced*).

### 01_Data_Preparation.ipynb
* **Fase CRISP:** 3. Data Preparation.
* **Tujuan:** Mem-prototipe dan memvalidasi "resep" *preprocessing* (mengubah `TotalCharges` ke numerik, *scaling* `tenure` dan `MonthlyCharges`, *one-hot encoding* puluhan fitur kategorikal).
* **Output:** `data/prepared/telco_features.csv` (Hanya untuk inspeksi manual).

### 02_Modeling_Churn_Classification_XAI.ipynb
* **Fase CRISP:** 4. Modeling, 5. Evaluation, 6. Deployment (Interpretasi).
* **Tujuan:** Memprediksi `Churn` (Klasifikasi Biner).
* **Model:** Membandingkan Logistic Regression (baseline) dengan XGBoost Classifier (challenger) menggunakan `Pipeline` untuk mencegah *data leakage*.
* **Evaluasi (Teknis):** Menggunakan **ROC-AUC** dan **Precision-Recall Curve**, karena dataset tidak seimbang.
* **Interpretasi (XAI):** Menerapkan **SHAP** pada model XGBoost pemenang untuk mengidentifikasi faktor pendorong *churn* (misal: `Contract`, `tenure`, `InternetService`).
* **Evaluasi (Bisnis):** Mengimplementasikan **analisis biaya (cost-sensitive)** untuk mengevaluasi model berdasarkan profitabilitas retensi (Biaya FN vs. Biaya FP), bukan hanya akurasi.
* **Output:** `models/churn_classifier_pipeline.joblib`.

### 03_Modeling_CLV_Regression.ipynb
* **Fase CRISP:** 4. Modeling, 5. Evaluation.
* **Tujuan:** Memprediksi `TotalCharges` (sebagai *proxy* Customer Lifetime Value/CLV).
* **Model:** GLM vs. XGBoost Regressor.
* **Evaluasi:** Menggunakan metrik **RMSE** dan **R-squared (R²)**.
* **Analisis Tambahan:** Menggabungkan prediksi CLV dengan prediksi *churn* untuk mengidentifikasi segmen "Nasabah Bernilai Tinggi, Berisiko Tinggi".

### 04_Modeling_Customer_Segmentation.ipynb
* **Fase CRISP:** 4. Modeling (Unsupervised), 5. Evaluation.
* **Tujuan:** Menggunakan K-Means Clustering untuk mengidentifikasi segmen nasabah (persona) alami tanpa menggunakan data `Churn` atau `TotalCharges`.
* **Evaluasi:** Menggunakan **Elbow Method** dan **Silhouette Score** untuk memilih K optimal. Memvalidasi secara bisnis dengan menganalisis `Churn Rate` dan `MonthlyCharges` rata-rata per cluster yang ditemukan.

## Replikasi

Untuk mereplikasi analisis ini secara lokal:

1.  Clone repositori ini: `git clone [URL_ANDA]`
2.  Buat dan aktifkan *virtual environment*: `python -m venv venv && source venv/bin/activate`
3.  Install dependensi: `pip install -r requirements.txt`
4.  Unduh dataset dari [link Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dan letakkan `WA_Fn-UseC_-Telco-Customer-Churn.csv` di dalam folder `data/raw/` (ganti nama menjadi `telco_churn.csv` agar lebih mudah).
5.  Jalankan *notebook* secara berurutan (00 sampai 04).

## Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).