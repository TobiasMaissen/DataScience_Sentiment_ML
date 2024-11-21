# Machine Learning Sentiment Analysis for Alpine Tour Reports

This repository contains Jupyter Notebooks and resources related to a machine learning project focused on sentiment analysis of alpine tour reports. The primary goal of this project is to classify critical sentences regarding route exposure (exposition) into three categories:
- **Neutral**: Irrelevant for the assessment of exposure.
- **Positive**: Light or well-secured exposure.
- **Negative**: Strong exposure with minimal or no security.

By leveraging modern NLP techniques and pre-trained models, this project aims to enhance the safety and decision-making process for alpine enthusiasts by analyzing subjective route descriptions.

---

## Project Structure

### 1. **Notebooks**
- `01_DataEngineering.ipynb`
  - Demonstrates the process of data collection and preparation using web scraping and manual labeling.
  - Includes exploratory data analysis (EDA) and preparation of a labeled dataset for sentiment analysis.
  - Establishes a baseline accuracy of 46% using minimal data and no advanced techniques.

- `02_ML_Evaluation.ipynb`
  - Illustrates the consequences of skipping essential ML practices such as cross-validation, hyperparameter tuning, and adequate data preparation.
  - Serves as an example of what **not to do** in a machine learning workflow.
  - Focused on showing how insufficient data and missing concepts degrade performance.

- `03_ML_SentimentAnalyse_Bergsport.ipynb`
  - Finalized implementation incorporating best practices:
    - Cross-validation
    - Hyperparameter optimization using Optuna
    - Use of multiple pre-trained transformer models for comparative analysis
  - Achieves a model accuracy of 97.8% with a robust evaluation pipeline.

---

## Problem Definition

In alpine sports, selecting the right route is critical for safety and success. Current classification systems like the **SAC Trekking Scale (T1-T6)** and **High Alpine Scale (L-AS)** often overgeneralize route exposure. This project aims to address the gap by leveraging machine learning to classify sentences describing route exposure.

**Objective**:
- Achieve a classification accuracy >85% and an F1-score >80% on evaluation data.
- Create a foundation for an end-to-end application that rates the exposure level of alpine routes.

---

## Dataset

### Data Sources
- Tour reports from [hikr.org](https://www.hikr.org/) were collected via web scraping.
- Reports include subjective route descriptions varying in length, detail, and structure.

### Data Pipeline
1. **Web Scraping**:
   - Extracted route descriptions, titles, and classifications by region using Python.
2. **Data Transformation**:
   - Processed text using `spaCy` for tokenization and lemmatization.
   - Extracted sentences relevant to exposure using a custom keyword list.
3. **Labeling**:
   - Sentences were manually labeled into three categories: neutral, positive, or negative.
   - Final dataset contains around 1,000 labeled sentences.

---

## Machine Learning Workflow

### 1. **Model Selection**
A variety of transformer models were tested, with a focus on those optimized for German NLP tasks:
- `deepset/gbert-base` (selected as the best model)
- `aari1995/German_Sentiment`
- `oliverguhr/german-sentiment-bert`
- `xlm-roberta-base`
- Other multilingual or distilled models

### 2. **Baseline Evaluation**
- Initial tests in `02_ML_Evaluation.ipynb` were conducted without advanced techniques, achieving a low accuracy of ~46%.
- Highlights the impact of insufficient data and missing practices like cross-validation.

### 3. **Advanced Techniques**
- **Cross-Validation**: Implemented K-Fold (5 folds) for robust evaluation.
- **Hyperparameter Tuning**: Used Optuna to optimize learning rate, batch size, and other parameters.
- **Data Augmentation**: Adjusted keyword lists to balance label distributions, focusing on increasing positive and negative samples.

### 4. **Final Results**
- **Model**: `deepset/gbert-base`
- **Accuracy**: 97.8%
- **F1-Score**: 0.892
- Achieved through rigorous optimization and sufficient data preparation.

---

## Key Insights

### Lessons Learned
1. **Data Quality Matters**: Proper labeling and balanced datasets are essential for effective ML models.
2. **Advanced Practices Pay Off**: Techniques like cross-validation and hyperparameter tuning significantly improve performance.
3. **Transformer Models Shine**: Models like `gbert-base` excel in understanding German text nuances, thanks to features like Whole Word Masking (WWM).

### Challenges
- Manual labeling was time-consuming but crucial for training effective models.
- Resource-intensive hyperparameter tuning required efficient use of Google Colab and cloud GPUs.

---

## Usage

### Prerequisites
- Python 3.8 or later
- Required libraries: `spaCy`, `transformers`, `Optuna`, `pandas`, `scikit-learn`

### Running the Notebooks
1. Clone this repository.
2. Install dependencies using:
   ```bash
   pip install -r requirements.txt
