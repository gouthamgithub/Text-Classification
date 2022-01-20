**Data Preparation**
1. Read 50k XML Files and extract required fields into dataframe.
2. Identy and extract classes for each newspaper(row).
3. NLTK for data preprocessing newspaper text.
4. Feature Extraction using tfidf for vector representation, and SelectKBest chi2 for filtering unwnated features.

**Training ML models**
Since a large data set, using k-fold cross validation with a small k value(3 to 5) is the best way to divide train and test sets. Doing normal train set-test set split on imbalanced class distribution data set may lead to uneven distribution where a complete n-labeled cases may fall in test set. And since our data set size is high, using a large number for 'k' will be computationally expensive. Hence small k value. We can mitigate overfitting as well since every data row is used in train set more than once.

The best evaluater to use when we have imbalanced class distribution is F1-score. Even though Accuracy is most widely used, this metric treats all classes equally, where as f1 score keeps a balance between recall and precision and gives a better measure of incorrectly classified inputs. Similarly AUC is the complete area under ROC curse and averages all thesholds, where as F1 score points to particular theshold which is more accurate for imbalanced data.

Trained six models:
1. RandomForest Classifier
2. Logistic Regression
3. MLP Classifier
4. Decision Tree
5. Support Vector Classifier
6. SGD Classifier

Linear Support Vector Classifier gave the best results, hence SVM is the best classifier for this Text dataset. However, given high computational power, MLP classifier(Neural Network) can be trained by tuning the hyperparameters(number of hidden layers) for much better results. Since Linear regression is best suited for continuos output, logistic regression is a more suitable for classification problems. Random forest considers multiple bootstrapped sets and trains in an iteration instead of training full dataset at once like decision tree. Hence, Random forest gave better results than decision tree.

**Data Clustering**
Performed K-means clustering the whole dataset. Advantage of using silhouette score for finding the best number of clusters is to use it for un-labelled data set. This is usually the case when running k-means. Since I need to use a different classifer for each cluster, we checked score for 4-8 number of clusters and found the best value to be 6. Hence used 6 different classifiers(RF, Decision tree, MLP, SVM, SGDClassifier, LogisticRegression) for 6 clusters.

**Feature Engineering**
I used research paper 'Text feature extraction based on deep learning: a review by Hong Liang, Xiao Sun' as reference for feature extraction. For the initial classifiers we used word frequency and TF IDF methods for feature extractions. Later we used convolutional neural networks to convole text matrices with the help of dropouts to reduce overfitting. Finally LSTM is used to encode the original text, using deep neural network.

**Evaluation**
After implementing feature extraction using CNN, the accuracy have increased for each cluster. In comparision with the fisrt assignment, where chi2 function was used with the help of kBest features technique, usage of word embedding and convoluting layers has made the text more interpretable by the neural network with minimal loss.
