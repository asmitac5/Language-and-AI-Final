# Political Leaning Prediction from Textual Data

This project predicts political leanings (Left, Center, Right) based on social media posts. The project utilizes multiple natural language processing (NLP) techniques and machine learning algorithms to classify political leanings. The key features include sentiment analysis, aggression detection, emotion recognition, readability scoring, and topic modeling.

## Project Structure

### 1. **EDA.ipynb** - Exploratory Data Analysis
This notebook performs **Exploratory Data Analysis (EDA)** on the dataset. It explores:
- Distribution of political leanings.
- Length and word count of posts.
- Vocabulary size and word frequency analysis.

### 2. **Emotional_Analysis.ipynb** - Emotion Detection
This notebook analyzes the **emotions** present in the posts using the `j-hartmann/emotion-english-distilroberta-base` model. It predicts emotional states such as joy, sadness, anger, fear, etc. The results are added to the dataset as emotion probabilities.

### 3. **Polarity.ipynb** - Sentiment Polarity Analysis
Here, **sentiment polarity** (positive, negative, neutral) is determined for each post using the `sarkerlab/SocBERT-base` model. This helps to capture the overall sentiment of posts, which is an important feature for political leaning prediction.

### 4. **ModelExperimentation.ipynb** - Sentiment and Aggression Analysis
This notebook focuses on **sentiment and aggression detection**:
- **Sentiment Analysis**: The sentiment (positive, neutral, or negative) is detected using the `cardiffnlp/twitter-roberta-base-sentiment` model.
- **Aggression Detection**: Aggression or toxicity in posts is measured using the `unitary/toxic-bert` model.

The results of these analyses (sentiment and aggression scores) are appended to the dataset, which are then used for building predictive models.

### 5. **Heuristic.ipynb** - Heuristic Baseline Model
This notebook implements a **heuristic baseline model** that classifies political leanings based on the occurrence of keywords specific to liberal, conservative, and centrist ideologies. This provides a baseline for evaluating the performance of more advanced models.

### 6. **Pollution.ipynb** - Data Pollution Removal
This notebook addresses the issue of **data pollution**, which arises when training and testing datasets have similar posts. It calculates cosine similarity between the training and testing datasets, identifies polluted rows, and removes them to improve model performance.

### 7. **Preprocessing.ipynb** - Text Preprocessing
This notebook focuses on **preprocessing the dataset** for emotion and polarity analysis. It applies **SpaCy** for lemmatization, tokenization, and stopword removal, ensuring that the dataset is clean and ready.

### 8. **Qual_Analysis.ipynb** - Qualitative Analysis
This notebook performs **qualitative analysis** to investigate the relationships between linguistic features (such as sentiment, aggression, readability, etc.) and political leanings. It uses **Spearman’s rank correlation** to measure these relationships and provides insights into the importance of these features.

### 9. **Regression.ipynb** - Regression Data Preparation
This notebook prepares the dataset for **regression modeling**:
- It merges the sentiment, aggression, emotion, readability, and topic features into a single dataset.
- Normalizes the data and applies label encoding to the `political_leaning` column.

### 10. **RegressionRunning.ipynb** - Model Training and Evaluation
This notebook trains and evaluates various machine learning models, including **XGBoost**:
- The models are trained using the features from sentiment, aggression, emotion, and readability analysis.
- It evaluates the performance using **classification metrics** like accuracy, precision, recall, F1-score, and ROC-AUC, and visualizes the results using confusion matrices and ROC curves.

### 11. **Topic+Readability.ipynb** - Topic Modeling and Readability Analysis
This notebook performs **topic modeling** using **BERTopic** and analyzes the **readability** of posts:
- **BERTopic** is used to extract topics from the posts.
- **Readability scores** (such as Flesch Reading Ease and Gunning Fog Index) are calculated to measure the complexity of posts.

## How to Run the Notebooks
To ensure that the analyses are conducted correctly, run the notebooks in the following order. Each analysis file creates its own dataset with specific columns, and these datasets are joined together in Regression.ipynb for machine learning model training.

### 1. EDA.ipynb: Exploratory Data Analysis
Start by running the EDA.ipynb notebook. This step performs exploratory data analysis on the political_leaning.csv dataset, providing insights into the distribution of political leanings, post lengths, and word counts. This is a crucial first step to understand the data before any preprocessing.
Purpose: To explore and analyze the data distribution.
Output: Insights into data distribution, vocabulary, and post statistics.

## 2. Preprocessing.ipynb: Preprocessing for Emotion and Polarity Analysis
Run Preprocessing.ipynb next. This notebook prepares the text data for emotion analysis and polarity detection. The text is cleaned and preprocessed, which includes tasks like tokenization and lemmatization.
Purpose: To preprocess the text data for emotion and sentiment analysis.
Output: A dataset with preprocessed text ready for analysis.

## 3. Emotion_Analysis.ipynb: Emotion Detection
After preprocessing, run Emotion_Analysis.ipynb. This notebook uses a pre-trained model to detect emotions in the text, such as anger, joy, sadness, etc. These emotion labels are added as new features in the dataset.
Purpose: To detect emotions in the posts and add them as features.
Output: A dataset with emotion-related features (e.g., anger, joy, sadness).

## 4. Polarity.ipynb: Sentiment Polarity Detection
Run Polarity.ipynb after emotion detection. This notebook performs sentiment analysis and categorizes posts as positive, negative, or neutral. The sentiment label is added as a feature to the dataset.
Purpose: To classify sentiment polarity (positive, negative, or neutral) of each post.
Output: A dataset with sentiment polarity features.

## 5. Topic+Readability.ipynb: Topic Modeling and Readability Analysis
Now run Topic+Readability.ipynb. This notebook applies BERTopic for topic modeling and calculates readability scores like Flesch Reading Ease and Gunning Fog Index. Both the topics and readability scores are added as columns to the dataset.
Purpose: To extract topics and calculate readability metrics.
Output: A dataset with topic and readability features.

## 6. ModelExperimentation.ipynb: Sentiment and Aggression Analysis
Next, run ModelExperimentation.ipynb. This notebook uses pre-trained models to analyze sentiment and aggression in the text. The sentiment model detects positive, negative, or neutral sentiment, while the aggression model detects the level of toxicity in the posts. These two features are then added to the dataset.
Purpose: To run sentiment and aggression models, adding the results as new features.
Output: A dataset with sentiment and aggression features.

## 7. Pollution.ipynb: Handling Polluted Data
Run Pollution.ipynb to clean the data by identifying and removing polluted rows. Polluted data refers to posts that are too similar between the training and testing datasets, identified using cosine similarity.
Purpose: To remove polluted rows from the training and testing datasets.
Output: A cleaned dataset with polluted rows removed.
Note: Once you have cleaned the data using Pollution.ipynb, you should re-run all the previous models (such as Emotion_Analysis.ipynb, Polarity.ipynb, Topic+Readability.ipynb, ModelExperimentation.ipynb) on the cleaned data to regenerate the relevant features. After that, proceed to the next steps to merge these features into a final dataset for regression.

## 8. Regression.ipynb: Merging All Features
After collecting all features from previous notebooks (emotion, sentiment, aggression, topic, readability), run Regression.ipynb. This notebook merges all the datasets from the previous analyses, creating a combined dataset. This merged dataset will be used to train machine learning models.
Purpose: To merge the emotion, sentiment, aggression, topic, and readability features into a single dataset.
Output: A merged dataset with all features added for machine learning.

## 9. RegressionRunning.ipynb: Train and Evaluate Models
With the merged dataset ready, run RegressionRunning.ipynb. This notebook trains machine learning models (like XGBoost) using the combined dataset. It evaluates the performance of the models and provides insights into accuracy, precision, recall, and F1-score.
Purpose: To train and evaluate machine learning models on the merged dataset.
Output: Model evaluation metrics, including accuracy, precision, recall, and F1-score.

## 10. Qual_Analysis.ipynb: Qualitative Analysis
After running machine learning models, run Qual_Analysis.ipynb. This notebook performs a qualitative analysis of the results, exploring the relationships between the features (like sentiment, aggression, readability) and political leaning. You need the full dataset with all features for this analysis.
Purpose: To interpret and analyze the results of the models qualitatively.
Output: Insights into how different features affect political leanings.

## 11. Heuristic.ipynb: Heuristic Baseline Model
Finally, run Heuristic.ipynb. This notebook builds a heuristic baseline model using predefined political keywords to classify posts. It serves as a simple benchmark for comparison with the machine learning models.
Purpose: To create a simple heuristic baseline model for comparison.
Output: A heuristic baseline model with basic predictions.

## Dataset

The main dataset used in this project is `political_leaning.csv`. This dataset contains:
- `post`: The text of the post.
- `political_leaning`: The labeled political leaning (Left, Center, Right).
- Other features derived from sentiment, aggression, emotion, readability, and topics.
We also have added an extra dataset from Kaggle to check our robustness.

## How to Clone the Repository

To get started with this project, you can clone the repository to your local machine. Follow these steps:

1. Open a terminal on your machine.
2. Run the following command to clone the repository:

```bash
git clone https://github.com/asmitac5/Language-and-AI-Final.git

To get the requirements, run:

'''bash
pip install -r requirements.txt
