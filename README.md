# Data Analytics & NLP Pipeline (HW2)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c.svg)

A multi-part data analytics workspace implementing custom Natural Language Processing (NLP) algorithms, financial credit risk profiling, and automated panel data cleaning routines.

---

## Overview & Tasks

This repository contains the complete codebase, datasets, and execution outputs for the three primary analytical tasks in Homework 2:

1. **Task 1: Custom NLP & TF-IDF Engine (`tfidf.py`)**  
   A pure Python implementation of text preprocessing and TF-IDF calculation built without relying on high-level NLP frameworks like NLTK or scikit-learn.
2. **Task 2: German Credit Risk Assessment**  
   Exploratory data analysis (EDA) and cross-tabulations on the German Credit Dataset (1,000 records) evaluating creditworthiness across applicant demographics.
3. **Task 3: Panel Data Cleaning Pipeline (Grunfeld Dataset)**  
   A multi-step data cleaning pipeline dealing with panel investment data, including custom date parsing, missing value imputation, and outlier capping.

---

## Technical Details

### 1. Custom Text Processing & TF-IDF Engine
The script `tfidf.py` processes raw text documents through sequential preprocessing stages:
* **Cleaning**: Strips non-alphanumeric characters, converts text to lower case, and removes URLs (`http/https`).
* **Stop Word Removal & Stemming**: Filters out common stop words using `stopwords.txt` and applies custom rule-based suffix truncation (`-ing`, `-ly`, `-ment`).
* **Mathematical Weighting**: Calculates term frequency and inverse document frequency using log smoothing:

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \left( \ln\left(\frac{N}{\text{DF}(t)}\right) + 1 \right)$$

* **Output Files**: Generates sanitized text files (`preproc_*.txt`) and top keyword ranking lists (`tfidf_*.txt`).

---

### 2. German Credit Analysis
* Analyzes credit risk patterns across checking balance tiers, employment duration, and personal status.
* Aggregates credit amounts to identify demographic segments with high financial risk exposure.

---

### 3. Panel Data Cleaning (Grunfeld Dataset)
* **Custom Date Normalization**: Standardizes mixed date formats (`YYYY-MM-DD`, standard years) into unified numerical years.
* **Outlier Capping**: Detects and caps extreme investment values using Interquartile Range (IQR) limits.
* **Group-Wise Analysis**: Evaluates firm market value against gross investment trends over time.

---

## Repository Files

* `tfidf.py`: Python script containing the full NLP text processing and TF-IDF pipeline.
* `tfidf_docs.txt`: List of target documents to process.
* `stopwords.txt`: Document containing words to exclude during NLP filtering.
* `hw2.ipynb` / notebooks: Jupyter Notebooks containing the data processing logic and visual output plots for Tasks 2 and 3.
* Data files: Raw input datasets (`german.data`, Grunfeld investment data).

---

## Setup & Running

### Dependencies
Install required runtime libraries:
```bash
pip install pandas numpy matplotlib seaborn
