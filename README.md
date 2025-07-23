# 📊 Customer Churn Prediction & Segmentation

This project aims to predict customer churn and segment customers into meaningful groups using machine learning and clustering algorithms. It helps businesses identify at-risk customers and improve retention strategies.

---

## ⚙️ Pipeline Overview

### 1. 🧼 Data Cleaning (`1_data_cleaning.ipynb`)
- Handles missing values and incorrect data types
- Drops or fixes inconsistent entries
- Converts categorical values and scales numerical ones

### 2. 📊 Exploratory Analysis (`2_exploratory_analysis.ipynb`)
- Visualizes churn distribution and trends
- Identifies patterns using customer demographics, usage behavior, and service types
- Uses heatmaps, histograms, bar plots, and correlation analysis

### 3. 🤖 Classification Models (`3_classification_models.ipynb`)
- Predicts churn using supervised learning
- Models used:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine
- Evaluates models using accuracy, precision, recall, F1 score

### 4. 🔍 Clustering Analysis (`4_clustering_analysis.ipynb`)
- Groups customers into similar segments using:
  - K-Means clustering
  - PCA for dimensionality reduction
- Profiles customer segments based on behavior and services

### 5. 📉 Model Evaluation (`5_model_evaluation.ipynb`)
- Compares model performance with confusion matrices and classification reports
- Visualizes:
  - ROC curves
  - Precision-Recall curves
- Interprets best-performing models for business insights

---

## 🚀 How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/customer-churn-project.git
   cd customer-churn-project

2. Install dependencies:
    pip install -r requirements.txt

3. Run the full pipeline:
    python main.py

4. Or use individual notebooks:
    cd notebooks
    jupyter lab


### 📁 Project Structure

    customer_churn_project/
        ├── data/                    # Raw dataset
        ├── notebooks/               # Jupyter analysis notebooks
        ├── outputs/                 # Model files and visualizations
        ├── src/                     # Python scripts (preprocessing, models, utils)
        ├── main.py                  # Main execution script
        ├── requirements.txt         # Required Python packages
        └── README.md                # Project documentation


### 📄 License
    
---

Would you like this `README.md` saved into your project as a file, or exported as downloadable text?



