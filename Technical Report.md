# Table of Contents
- [Project Background](#project-background)
- [Executive Summary](#executive-summary)
- [Data](#data)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Sentiment Prediction Modeling](#sentiment-prediction-modeling)
- [Data Source and License](#data-source-and-license)
---

# Project Background
This project focuses on predicting sentiment in text from financial news using natural language processing (NLP) and logistic regression.

## Links
- [Jupyter Notebook](./Financial%20News%20Sentiment%20Analysis.ipynb)
- [Data](./Data/Sentences_AllAgree.txt)

## Tools and Technologies
- **Pandas**: data manipulation and cleaning
- **Matplotlib / Seaborn / Wordcloud**: EDA and visualizations
- **Scikit-learn**: data preprocessing and machine learning modeling
- **NLTK**: text specific data cleaning
- **SHAP**: computing Shapley values and visualizing word/phrase importance

## Approach
- Used TF-IDF vectorization and logistic regression for sentiment classification.
- Preprocessed text with tokenization, stopword removal, and lemmatization.
- Explored patterns with word clouds, sentiment distribution, and word count box plots.
- Built 4 models based on the word count of the phrases used for training.
- Evaluated models via confusion matrices, classification reports, and SHAP plots for interpretability.

# Executive Summary
## Insights
- In this dataset, unigrams contributed the most to the best performing models.
- The data may contain noise that makes the model biased towards Neutral sentiment. Temporal anchors are suspected to be responsible.
- Training the model on N-grams, does not affect model performance much for Negative and Neutral sentiment, but does have an observable increase in recall for Positive sentiment.
- Overall model performance declines when the model is trained on longer phrases, with each additional term driving larger performance drops. 
- The N-gram model retains roughly the same level of performance for negative and neutral sentiment in the unigram baseline, while providing a slight boost in Positive recall and clearer SHAP interpretability at the expense of Positive precision.

## Recommendations
- For maximum raw performance or neutral sentiment prediction focus, the Unigram model is the best choice of the 4 models for predicting sentiment in financial news.
- For clarity, context, and a slight increase in Postive sentiment recall, the N-Gram model may be preferred at the cost of negligible differences in performance across other metrics.

## Next Steps
- Explore potential sentence length bias across sentiment classes by normalizing TF-IDF by document length or including word count as a feature.
- Balance the data by resampling before modeling.
- Adjust cleaning to exclude certain financial phrases we would want to keep in the data, such as year-on-year.
- Remove temporal anchors and check for any changes in model performance as those terms may be noise.
- Fit different models using other N-gram ranges such as 1-3 words or 2-5 words and compare the models' performance to the Unigram model.
- Use different random seeds or increase proportion of training data when splitting, changing the 70/30 split to an 80/20 split. Since the dataset is small, model performance may change by using a different random seed or increasing the sample size of the training data.
-  Account for overlapping terms when modeling N-grams. This would also require steps to excluded terms like operating profit or net profit from the process, since those are specific financial terms we would want to keep rather than actual overlapping terms we would want to remove.

# Data
- The datasets contains sentences and labels from various financial news sources. 
- Labels take on values of (0, 1, 2) which represent (negative, neutral, positive) and are labeled are annotated by 8 people. 
- The specific dataset used is the all_agree dataset, which includes only sentences and labels where all 8 annotators agree on the same sentiment that the sentence takes.

## Data Cleaning
- Duplicates are identified and removed.
- Punctuation, whitespace, and stop words are removed from the text.
- Text is converted to lowercase and lemmatized, grouping words into groups based on their roots (ex. increase, increased, and increasing are one group).
- Cleaned text is added as a column to the dataset.

## Exploratory Data Analysis
### Sentiment Distribution
![Sentiment Distribution](<Visuals/sentiment_distribution.png>)

- Over half of the sentiment in the data is Neutral. 
- Positive sentiment is around double the Negative sentiment. 
- The data is imbalanced with Neutral sentiment being a majority class, while Positive and Negative sentiment are minority classes, especially Negative sentiment.
- Sentiment prediction models are suceptible to biases toward Neutral sentiment.
---

### Raw Word Count Distribution
![Raw Word Count Distribution](<Visuals/word_count_dist_uncleaned.png>)

- Word count is slightly higher when sentiment is Positive than when it is Negative, but the difference is small.
- When sentiment is Neutral, word count tends to be the lowest.
- Neutral sentiment also has the most outliers in word count.
- Positive sentiment seems to have a larger box than Negative sentiment, indicating larger variance in word count.
- In the context of the data, financial news tends to have longer sentences when sentiment has a clear direction.
- The data seems to be skewed towards higher word counts since the line representing the median is below the center of each box.
---

### Cleaned Word Count Distribution
![Cleaned Word Count Distribution](<Visuals/word_count_dist_cleaned.png>)
- After cleaning the text and removing stopwords, the boxes for each sentiment are now lower. 
- The previous pattern of positive > negative > neutral sentiment in word count is observed again.
- Skew is more obvious for positive and negative sentiment (the median line is closer to the bottom of the box).
- For neutral sentiment, the median line approaches the middle of the box indicating reduced skew.
- The box for positive sentiment appears larger indicating wider spread in word count.

### Word Clouds
Multiple Word Clouds were generated based on N-Gram, representing word count within phrases. Unigrams contain only a single word, while Bigrams are phrases containing 2 words.

#### Unigram Word Cloud
![Unigram Word Cloud](<Visuals/unigram_wordcloud.png>)
- Financial currencies and business terms appear frequently.
- The data is dominantly based on European companies/news since euro appears most frequently as a currency.
- From the size of the words, sales, profits, and million are mentioned most frequently.
- With single words it can be difficult to identify sentiment. Words like 'increase' or 'fell' could be used describe to either profits or losses, but it's uncertain which is the case here.

---

### Bigram Word Cloud
![Bigram Word Cloud](<Visuals/bigram_wordcloud.png>)
- With Bigrams, the word cloud contains 2 word phrases instead.
- The biggest change in this word cloud are the temporal anchors and directional signals such as more specific time periods and whether profit increased or decreased.
- Other financial terms like cash flow and voting rights can now be captured.

### N-Gram Word Cloud
![N-Gram Word Cloud](<Visuals/n-gram_wordcloud.png>)
- This word cloud seems more sparse compared to the Bigram Word Cloud
- Captures much richer context such as 'today net profit fell' and longer phrases like 'recent market analysis'.

---
# Sentiment Prediction Modeling
Multiple models were built using Term Frequency Inverse Document Frequency (TF-IDF) vectorization combined with a logistic regression classifier. This is a hybrid vectorization method using Term Frequency (TF) and Inverse Document Frequency (IDF). TF measures the number of occurances of a term, while IDF penalizes terms that appear across many documents. TF-IDF combines both methods, emphasizing terms that are frequent but uniquely distributed across different observations.

The TF-IDF vectorizer converts text to numerical features and inputs those into a logistic regression classification algorithm, which finds the probabilities for each sentiment class and chooses the sentiment with the highest probability as its prediction.

Models built include:
- Unigram
- Bigram
- Trigram
- N-Gram (1-5 words)

## Model Performance Metrics
- A classification report is generated for each model, showing model performance metrics such as Precision, Recall, Accuracy, and F1-Score.
- A confusion matrix is generated for each model to show how its predictions compare to the actual sentiment labels.
- SHAP values are computed and visualized to highlight which words or phrases most influence the model’s predictions.

<details>
<summary><b>
Click to see individual performance evaluations for each model
</b></summary>

## Unigram Model
### Unigram Classification Report Heatmap
![Unigram Classification Report](<Visuals/unigram_classification_heatmap.png>)
- The unigram model has excellent performance for Neutral sentiment, but performs poorly with other sentiment.
- Overall, the model has good accuracy, but it is biased towards Neutral sentiment.

---
### Unigram Confusion Matrix
![Unigram Confusion Matrix](<Visuals/unigram_confusion_matrix.png>)
- The model is strong when identifying neutral sentiment.
- Misclassifications between Positive and Negative classes indicate a need for richer context modeling.
- Recall for Negative and Positive sentiments is limited, reflecting the challenge of capturing polarity with single-word features.

---
### Unigram SHAP Plot
![Unigram SHAP Plot](<Visuals/unigram_shap.png>)
- Words like rose, fell, increased, and decreased, are directional indicators, but lack context.
- Profit, sales, and eps are financial metrics frequent used to assess business performance and are strongly tied to sentiment.
- Words like 2008, quarter, and year often lean towards Neutral or Negative. Their influence may reflect historical comparisons that tends to skew sentiment downwards. Perhaps the news documented poorer financials than those time periods.


## Bigram Model
### Bigram Classification Report Heatmap
![Bigram Classification Report](<Visuals/bigram_classification_heatmap.png>)
- The Bigram model has poorer performance than the Unigram model.
- While Bigrams may capture more context, the precision, recall, and f1-score all decreased for Positive and Negative sentiment.
- The model makes fewer correct predictions and has poorer balance between precision and recall.
- Metrics for Neutral sentiment remain about the same as the Unigram model.

---
### Bigram Confusion Matrix
![Bigram Confusion Matrix](<Visuals/bigram_confusion_matrix.png>)
- The number of Neutral predictions is about the same, but it seems the model has more predictions for Neutral labels when actual sentiment is Positive or Negative.
- The number of correct predictions for Positive and Negative sentiment has decreased compared to the Unigram model.

---
### Bigram SHAP Plot
![Bigram SHAP Plot](<Visuals/bigram_shap.png>)
- Financial metrics appear to be the greatest contributors towards this model's predictions. Time anchors are now more specific, narrowing down time periods to certain months of certain years.
- Directional indicators can now be combined with financial metrics, clearly indicating how movements of those metrics may contribute to sentiment. Sales rose and profit fell contribute most to Positive and Negative sentiment respective, which is to be expected.
- Using bigrams also introduces certain financial metrics that consist of 2 words like net profit or operating profit. The bigram year year looks strange, but should refer to year-on-year, which measures changes in a metric from one year to another. This phrase is incorrectly placed here, likely due to stopword cleaning removing the word 'on' in between the phrase.

## Trigram Model
### Trigram Classification Report Heatmap
![Trigram Classification Report](<Visuals/trigram_classification_heatmap.png>)
- Overall accuracy with this model is even lower than the Bigram model.
- The model shows increased precision for Positive sentiment and increased recall for Neutral sentiment compared to the Bigram model.
- Performance across other metrics has dropped. This is especially the case for Negative sentiment, where recall dropped by over 50% from the Bigram model.

---
### Trigram Confusion Matrix
![Trigram Confusion Matrix](<Visuals/trigram_confusion_matrix.png>)
- There are greater frequencies of Neutral predictions from the model.
- The model exhibits more extreme bias towards Neutral sentiment than the previous models.
- The model also has around 50% fewer correct predictions for Positive and Negative sentiment.

---
### Trigram SHAP Plot
![Trigram SHAP Plot](<Visuals/trigram_shap.png>)
- With trigrams, certain bigram financial metrics can now be combined with directional indicators.
- The top contributors are directionally indicated financials and time periods.
- Time period trigrams appear rather frequently and mostly contribute to neutral sentiment.
- Model performance may improve if those temporal anchors were removed from the model.

## N-gram Model
### N-gram Classification Report Heatmap
![N-gram Classification Report](<Visuals/ngram_classification_heatmap.png>)
- Changes of 3 to 5% are observed for positive sentiment, with a decrease in precision and an increase in recall compared to the Unigram model.
- For other sentiment and metrics, there is little to no difference in performance from the Unigram model.
- Overall accuracy is the same as the Unigram model.

---
### N-gram Confusion Matrix
![N-Gram Confusion Matrix](<Visuals/ngram_confusion_matrix.png>)
- The number of predictions for each class and the number of true positives are roughly the same as the Unigram model.

---
### N-gram SHAP Plot
![N-gram SHAP Plot](<Visuals/ngram_shap.png>)
- Unigrams contribute the most towards model predictions.
- Corresponding and corresponding period are both in the plot. The word corresponding overlaps and is rather redundant.
- Profit and sales are the highest contributing financial metrics to the model's sentiment predictions.

</details>

## Model Metrics Overall Comparison Table
| Model     | Sentiment | Precision | Recall | F1-Score |
|-----------|-----------|-----------|--------|----------|
| Unigram   | Negative  | 0.62      | 0.65   | 0.64     |
|           | Neutral   | 0.88      | 0.96   | 0.92     |
|           | Positive  | 0.81      | 0.61   | 0.70     |
| Bigram    | Negative  | 0.57      | 0.48   | 0.52     |
|           | Neutral   | 0.85      | 0.96   | 0.90     |
|           | Positive  | 0.73      | 0.56   | 0.63     |
| Trigram   | Negative  | 0.53      | 0.23   | 0.32     |
|           | Neutral   | 0.71      | 1.00   | 0.83     |
|           | Positive  | 0.83      | 0.29   | 0.43     |
| N-Gram    | Negative  | 0.63      | 0.63   | 0.63     |
|           | Neutral   | 0.89      | 0.95   | 0.92     |
|           | Positive  | 0.76      | 0.64   | 0.70     |

- The Unigram and the N-Gram models have the strongest performance, with minimal variation between them.
- Performance declines when the model is trained on longer phrases like bigrams and trigrams.

---

## Accuracy and Average Metrics Overall Comparison Table
| Model   | Precision | Recall | F1-Score | Accuracy |
|---------|-----------|--------|----------|----------|
| Unigram | 0.77      | 0.74   | 0.75     | 0.83     |
| Bigram  | 0.71      | 0.67   | 0.68     | 0.79     |
| Trigram | 0.69      | 0.51   | 0.53     | 0.71     |
| N-Gram  | 0.76      | 0.74   | 0.75     | 0.83     |
> Note: This table uses macro-averages for precision, recall, and f1-score for each model.

- Results from the previous table are reflected here; the accuracy scores are the same for the Unigram and N-Gram models and macro-averages for precision, recall, and f1-score are about the same.
- The performance decline from unigram to bigram is lower than the decline from bigram to trigram.

## Data Source and License
- Dataset: Financial Phrasebank
- Authors: Pekka Malo, Ankur Sinha, Panu Korhonen, Jyrki Wallenius, and Peter Takala  
- Source: [Hugging Face Dataset Repository](https://huggingface.co/datasets/takala/financial_phrasebank)  
- License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported (CC BY-NC-SA 3.0)  
- Reference: Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). *Good debt or bad debt: Detecting semantic orientations in economic texts*. Journal of the Association for Information Science and Technology, 65(4), 782–796.