from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

NUMERIC = ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm']
VALID_SPECIES = {'setosa', 'versicolor', 'virginica'}


def acquire_dataset():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = NUMERIC + ['species']
    df['species'] = df['target'].map(dict(enumerate(iris.target_names)))
    return df.drop(columns='target')


def introduce_quality_issues(df):
    dirty = df.copy()
    dirty.loc[5, 'sepal_length_cm'] = np.nan
    dirty.loc[42, 'petal_width_cm'] = np.nan
    dirty.loc[100, 'petal_length_cm'] = np.nan
    dirty.loc[123, 'sepal_width_cm'] = np.nan
    dirty.loc[10, 'species'] = ' Setosa '
    dirty.loc[60, 'species'] = 'VERSICOLOR'
    dirty.loc[110, 'species'] = 'Virginica '
    dirty.loc[20, 'sepal_length_cm'] = 15.0
    dirty.loc[80, 'petal_length_cm'] = 20.0
    dirty.loc[130, 'petal_width_cm'] = -1.5
    dirty.to_csv(DATA_DIR / 'iris_dirty.csv', index=False)
    return dirty


def iqr_bounds(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def main():
    original = acquire_dataset()
    original.to_csv(DATA_DIR / 'iris_original.csv', index=False)
    dirty = introduce_quality_issues(original)

    cleaned = dirty.copy()
    cleaned['species'] = cleaned['species'].astype('string').str.strip().str.lower()
    cleaned.loc[~cleaned['species'].isin(VALID_SPECIES), 'species'] = pd.NA

    for col in NUMERIC:
        cleaned.loc[cleaned[col] < 0, col] = np.nan

    for col in NUMERIC:
        cleaned[col] = cleaned.groupby('species')[col].transform(
            lambda s: s.fillna(s.median())
        )
        cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    for col in NUMERIC:
        lo, hi = iqr_bounds(cleaned[col].dropna())
        cleaned[col] = cleaned[col].clip(lo, hi)

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    cleaned.to_csv(DATA_DIR / 'iris_cleaned.csv', index=False)

    scaler = StandardScaler()
    preprocessed = cleaned.copy()
    preprocessed[NUMERIC] = scaler.fit_transform(preprocessed[NUMERIC])
    preprocessed.to_csv(DATA_DIR / 'iris_preprocessed_scaled.csv', index=False)

    print('Original shape:', original.shape)
    print('Dirty shape:', dirty.shape)
    print('Missing before:', int(dirty.isna().sum().sum()))
    print('Missing after:', int(preprocessed.isna().sum().sum()))
    print('Duplicates after:', int(preprocessed.duplicated().sum()))


if __name__ == '__main__':
    main()
