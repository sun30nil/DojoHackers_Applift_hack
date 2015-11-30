# Predicting Conversion Probability and Bid Value

This is project is aimed at predicting the probability that a customer will either:

1. Just Clicked an ad - 'c'

2. Ad was shown (representing an ad auction win) - 'w'

3. Ad was not shown - '0'

Out of 28 attributes around 18 where found be good to start with Feature Engineering.

Ensemble Model - Adaboost was used to classify with an accuracy of 96% on 12 gb dataset.

Similarly the bid value was also predicted using Gradiest Descent Regressor Model, which gave a MSE of 0.1 which was 60% of the avg bid value.

Also future values of bid for a customer were forecasted using Exponential Smooting Averages Method.

If you need the dataset just test mail me on sun30nil@gmail.com
