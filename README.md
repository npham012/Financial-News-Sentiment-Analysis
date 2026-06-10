# Project Background
This project focuses on predicting sentiment in text from financial news using natural language processing (NLP) and logistic regression. 

## Tools and Technologies
- **Pandas**: data manipulation and cleaning
- **Matplotlib / Seaborn / Wordcloud**: EDA and visualizations
- **Scikit-learn**: data preprocessing and machine learning modeling
- **NLTK**: text specific data cleaning
- **SHAP**: computing Shapley values and visualizing word/phrase importance

# Executive Summary
## Insights
- In this dataset, one word phrases (Unigrams) contributed the most to the best performing models.
- The data may contain noise that makes the model biased towards Neutral sentiment. Temporal anchors are suspected to be responsible.
- Training the model on phrases from 1 to 5 words (N-grams) does not affect model performance much for Negative and Neutral sentiment, but does have an observable increase in recall for Positive sentiment.
- Overall model performance declines when the model is trained on longer phrases, with each additional term driving larger performance drops. 
- The N-gram model retains roughly the same level of performance as the unigram model, but is more interpretable.

## Recommendations
- For maximum raw performance, the Unigram model is the best choice of the 4 models for predicting sentiment in financial news.
- For clarity and context, the N-Gram model may be preferred at the cost of negligible differences in performance.

## Next Steps
- Explore potential sentence length bias across sentiment classes by normalizing with document length or including word count as a feature.
- Balance the data by resampling before modeling.
- Adjust cleaning to exclude certain financial phrases we would want to keep in the data, such as year-on-year.
- Remove temporal anchors and check for any changes in model performance as those terms may be noise.
- Fit different models using other word count ranges for the N-gram model.
- Use other random seeds or increase adjust data splitting proportions. Since the dataset is small, model performance may change by using a different random seed or increasing the sample size of the training data.
- Account for overlapping terms when modeling. This would also require steps to excluded terms like operating profit or net profit from the process, since those are specific financial terms we would want to keep rather than actual overlapping terms we would want to remove.

# Approach
- Used vectorization and logistic regression for sentiment classification.
- Preprocessed text with tokenization, stopword removal, and lemmatization.
- Explored patterns with word clouds, sentiment distribution, and word count box plots.
- Built 4 models based on the word count of the phrases used for training.
- Evaluated models via performance metrics and visualizations such as confusion matrices, classification reports, and SHAP plots.

# Links
- [Project Jupyter Notebook](./Financial%20News%20Sentiment%20Analysis.ipynb)
- [Full Technical Report](./Technical%20Report.md)
- [Data](./Data/Sentences_AllAgree.txt)

# Data
- The datasets contains sentences and labels from various financial news sources. 
- Labels take on values of (0, 1, 2) which represent (negative, neutral, positive) and are labeled are annotated by 8 people. 
- The specific dataset used is the all_agree dataset, which includes only sentences and labels where all 8 annotators agree on the same sentiment that the sentence takes.

## Data Source and License
- Dataset: Financial Phrasebank
- Authors: Pekka Malo, Ankur Sinha, Panu Korhonen, Jyrki Wallenius, and Peter Takala  
- Source: [Hugging Face Dataset Repository](https://huggingface.co/datasets/takala/financial_phrasebank)  
- License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported (CC BY-NC-SA 3.0)  
- Reference: Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). *Good debt or bad debt: Detecting semantic orientations in economic texts*. Journal of the Association for Information Science and Technology, 65(4), 782–796.
