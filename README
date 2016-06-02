# Submission for Kaggle San Fransisco Crime Classification

## Background

[Kaggle's San Fransisco Crime Classification](https://www.kaggle.com/c/sf-crime) competition asks user to predict what
type of crime was committed based on the following features: date, day of week, district, address, longitude and latitude.
The crime has 39 potential outcomes.

## Overview

I decided to use python, specifically sklearn and pandas because I have been learning those lately and the dataset was
small enough to use in memory. The three different classification I considered were logistic regression, K nearest
neighbors and random forest. After many experimentation, random forest had the best combination of accuracy and
generalization.

I also attempted to use the three different classifiers in combination with a voting classifier. The
voting classifier will train all three classifier and use the combined predictions to vote for the correct choice.
Ultimately the voting classifier didn't improve the accuracy and restricted the ability to compute predicted probabilities,
which is important since the scoring function produces best results with predicted probability compared to hard membership.

After settling on random forest, I used grid search to find the best hyper parameters for random forest. One contradiction
I found to literature is that the model performed best when including all features in the split decision. The literature
suggest only using the square root of the features in decision making, however I think all were required since there
were so few features.

The features had to be transformed to produce the best results. I transformed the dates feature into five different
features: year, month, day, hour and minute. The other transformation I did was one hot encoding for DayOfTheWeek.

## Result

After picking random forest and tuning the parameters I was able to get a multiclass log loss score of 2.38866 which
ranked the submission at 466 of 2251 (submitted by user ckipers). I am fairly happy with my result being this is my first
Kaggle competition. If I were to invest more time in this competition there were be a couple of things I would like to try:

1. Use a machine with more memory and increase the n_estimators for random forest
2. Combine features to create new features
3. Use a deep neural network.