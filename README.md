# Fake News Challenge - News Stance Classification

In context of news, a claim is made in a news headline, as well as in the piece of text in an article body. Quite often, the headline of a news article is created so that it is attractive to the readers, even though the body of the article may be about a different subject/may have another claim than the headline.

Stance Detection involves estimating the relative perspective (or stance), of two pieces of text relative, i.e. do the two pieces agree, disagree, discuss or are unrelated to one another. Your task in this project is to estimate the stance of a body text from a news article relative to a headline.

The goal in stance detection is to detect whether the headline and the body of an article have the same claim. The stance can be categorized as one of the four labels: “agree”, “disagree”, “discuss” and “unrelated”. Formal definitions of the four stances are as:

- “agree” – the body text agrees with the headline;
- “disagree” – the body text disagrees with the headline;
- “discuss” – the body text discusses the same claim as the headline, but does not take a
position;
- “unrelated" – the body text discusses a different claim but not that in the headline.

## Dataset
We will be using the publicly available FNC-1 dataset (https://github.com/FakeNewsChallenge/fnc-1/). This dataset is divided into a training set and a testing set. The ratio of training data over testing data is about 2:1. Every data sample is a pair of a headline and a body. There are 49972 pairs in the training set, with 49972 unique headlines and 1683 unique bodies. This means that an article body can be seen in more than one pair. “unrelated” data takes the majority (over 70%) in both sets while the percentage of “disagree” is less than 3%. The percentage of “agree” and “discuss” are less than 20% and 10%, respectively. Severe class imbalance exists in the FNC-1 dataset.

## Tasks
This project involves several subtasks that are required to be solved:
- Split the training set into a training subset and a validation subset with the data number proportion about 9:1.The training subset and the validation subset should have similar ratios of the four classes. Statistics of the ratios should be presented.
Extract vector representation of headlines and bodies in the all the datasets, and compute
the cosine similarity between these two vectors. You can use representations based on bag-ofwords
or other methods like Word2Vec for vector based representations. You are encouraged
to explore alternative representations as well.
- Establish language model based representations of the headlines and the article bodies in all the datasets and calculate the KL-divergence for each pair of headlines and article bodies. Explore different smoothing techniques for language model based representations.
- Propose and implement alternative features/distances that might be helpful for the stance detection task.
- Choose representative distances/features that are most important for stance detection and plot the distance distribution for the four stances.
- Using the features created, implement a linear regression and a logistic regression model using gradient descent for stance classification. Note these are implemented from scratch.
- Results analysis with carefully chosen evaluation metric. Compare and contrast the performance of the two models implemented. Analyse the effect of learning rate on both models.
- Explore features that are the most important for the stance detection task by analysing their importance for the machine learning models built.
