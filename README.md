# Data Science Internship Project

## Week 1 — Data Acquisition, Cleaning and Preprocessing

This project contains the Week 1 deliverable for a data science internship. It demonstrates a reproducible Python workflow for acquiring, profiling, cleaning, validating, and preprocessing a public dataset.

### Dataset
- Iris dataset
- Original source: UCI Machine Learning Repository
- DOI: 10.24432/C56C76
- Acquisition in Python: `sklearn.datasets.load_iris(as_frame=True)`

### Methodological note
The original UCI Iris dataset is documented as having no missing values. To demonstrate the required handling of missing values, inconsistent labels, erroneous values, and outliers without misrepresenting the source, the project creates a separate controlled working copy (`data/iris_dirty.csv`). The original dataset remains unchanged.

### Run

```bash
pip install -r requirements.txt
python week1_data_cleaning.py
python create_report.py
```

### Main outputs
- `data/iris_original.csv`
- `data/iris_dirty.csv`
- `data/iris_cleaned.csv`
- `data/iris_preprocessed_scaled.csv`
- `outputs/missing_values.png`
- `outputs/boxplot_before.png`
- `outputs/boxplot_after.png`
- `outputs/class_distribution.png`
- `Week_1_Data_Acquisition_Cleaning_Preprocessing_Report.docx`
