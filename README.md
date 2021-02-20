# Machine Learning by Stanford University
Machine learning problem sets from Stanford University's Machine Learning course on Coursera.

## Problem Set #1: Linear Regression and Gradient Descent

### Linear Regression with One Variable

This involves implementing a linear regression with one variable, using profits and populations of cities, in which food trucks have been in operations, to predict profits for a new food truck given the population in its operating city, using the gradient descent algorithm. 

Training data with linear regression fitted by gradient descent:

![alt text](https://kurtcms.org/git/standford-machine-learning/ex1_profit-vs-population-size-linear-regression.png)

Surface and contour plots of the cost function J(&theta;):

![alt text](https://kurtcms.org/git/standford-machine-learning/ex1_cost-function-surface-plot.png)![alt text](https://kurtcms.org/git/standford-machine-learning/ex1_cost-function-contour-plot.png)

### Linear Regression with Multiple Variables

This involves implementing a linear regression with two variables, using the house size in square feet and number of bedrooms from historical house sales data, to predict the price for a new house given its size and number of bedrooms, using the gradient descent algorithm.

With a learning rate (&alpha;) of 0.01, the cost function J(&theta;) converges after a number of iteration using gradient descent:

![alt text](https://kurtcms.org/git/standford-machine-learning/ex1_multi_gradient-decent-convergence.png)

## Problem Set #2: Logistic Regression

### Logistic Regression

This involves implementing a logistic regression with two variables, using historical students' exams scores in two different exams and their university admission results, to predict whether a student will be admitted to university given exam scores, using Octave/MATLAB's fminunc function.

Training data with decision boundary fitted by the fminunc function:

![alt text](https://kurtcms.org/git/standford-machine-learning/ex2_exam-scores-vs-university-admission-decision-boundary.png)

Training accuracy is 89%.

### Logistic Regression Regularised

This involves implementing a regularised logistic regression with two variables, using historical microchip test results, in two different tests and their quality assurance test results from a fabrication plant, to predict whether a microchip will pass quality assurance given test results, using Octave/MATLAB's fminunc function.

Training data with decision boundary fitted by the fminunc function:

![alt text](https://kurtcms.org/git/standford-machine-learning/ex2_reg-microchip-test-results-vs-qa-decision-boundary.png)

Training accuracy is 83%

Training data with decision boundary fitted by the fminunc function using a regularisation parameter (&lambda;) of 0 (no regularisation/overfitting) and of 100 (underfitting):

![alt text](https://kurtcms.org/git/standford-machine-learning/ex2_reg-microchip-test-results-vs-qa-decision-boundary-lambda-0.png)
![alt text](https://kurtcms.org/git/standford-machine-learning/ex2_reg-microchip-test-results-vs-qa-decision-boundary-lambda-100.png)

## Problem Set #3: Multi-class Classification and Neural Networks

This involves implementing one-vs-all logistic regression, and neural networks of one hidden layer, to recognise handwritten digits from the [MNIST handwritten digit dataset](http://yann.lecun.com/exdb/mnist/), using Octave/MATLAB's fmincg function, and to compare the performance between the two algorithms.

Handwritten digits visualised:

![alt text](https://kurtcms.org/git/standford-machine-learning/ex3_hand-written-digits.png)

Training accuracy is 95% with one-vs-all logistic regression whereas training accuracy is 97.52% with one-vs-all neural networks.

## Problem Set #4: Neural Network Learning

This involves implementing the backpropagation algorithm for neural networks of one hidden layer, and applying it to recognise handwritten digits, learning from a subset of the [MNIST handwritten digit dataset](http://yann.lecun.com/exdb/mnist/) of 5,000 samples on 50 iterations, using Octave/MATLAB's fmincg function.

Handwritten digits and the representation captured by the hidden layer visualised:

![alt text](https://kurtcms.org/git/standford-machine-learning/ex4_hand-written-digits.png)
![alt text](https://kurtcms.org/git/standford-machine-learning/ex4_hand-written-digits-representation-hidden-layer.png)

Training accuracy is 95.06%.

## Problem Set #5: Regularised Linear Regression and Bias v.s. Variance

This involves implementing regularised linear regression, using historical record of change of water level in a reservoir and the amount of water flowing out of a dam, to predict water outflow given the reservoir's water level change, using Octave/MATLAB's fmincg function.

The dataset will be randomly divided into 3 parts.

* Training set: for regression learning.
* Cross-validation set: for determining the regularisation parameter.
* Test set: for evaluating the regression performance.

### Regularised Linear Regression

Training data with regularised linear regression fitted by the fmincg function (high bias) and the training and cross-validation errors as a function of training set size:

![alt text](https://kurtcms.org/git/standford-machine-learning/ex5_reservoir-water-level-change-vs-water-outflow-linear-regression.png)![alt text](https://kurtcms.org/git/standford-machine-learning/ex5_training-cross-validation-error-vs-training-set-size-linear-regression.png)

### Regularised Polynomial Regression

Training data with regularised polynomial regression fitted by the fmincg function using a regularisation parameter (&lambda;) of 1 and the training and cross-validation errors as a function of training set size

![alt text](https://kurtcms.org/git/standford-machine-learning/ex5_reservoir-water-level-change-vs-water-outflow-polynomial-regression-lambda-1.png)![alt text](https://kurtcms.org/git/standford-machine-learning/ex5_training-cross-validation-error-vs-training-set-size-polynomial-regression-lambda-1.png)

Training data with regularised polynomial regression fitted by the fmincg function using a regularisation parameter (&lambda;) of 0 (no regularisation/overfitting/high variance) and the training and cross-validation errors as a function of training set size

![alt text](https://kurtcms.org/git/standford-machine-learning/ex5_reservoir-water-level-change-vs-water-outflow-polynomial-regression-lambda-0.png)![alt text](https://kurtcms.org/git/standford-machine-learning/ex5_training-cross-validation-error-vs-training-set-size-polynomial-regression-lambda-0.png)

Training data with regularised polynomial regression fitted by the fmincg function using a regularisation parameter (&lambda;) of 100 (underfitting/high bias) and the training and cross-validation error as a function of training set size

![alt text](https://kurtcms.org/git/standford-machine-learning/ex5_reservoir-water-level-change-vs-water-outflow-polynomial-regression-lambda-100.png)![alt text](https://kurtcms.org/git/standford-machine-learning/ex5_training-cross-validation-error-vs-training-set-size-polynomial-regression-lambda-100.png)

Training and cross-validation error as a function of the regularisation parameter (&lambda;).

![alt text](https://kurtcms.org/git/standford-machine-learning/ex5_training-cross-validation-error-vs-lambda.png)

## Problem Set #6: Support Vector Machines

### Support Vector Machines (SVMs) with a C Parameter and a Gaussian Kernel

This involves implementing Support Vector Machines (SVMs) with a C parameter and a Gaussian kernel, to draw decision boundary, and using the cross-validation dataset to determine the optimal C parameter, and the bandwidth parameter (&sigma;) for the Gaussian kernel.

SVM linear decision boundary with the C parameter of 1 and of 100 on example dataset #1:

![alt text](https://kurtcms.org/git/standford-machine-learning/ex6_example-dataset-1-linear-decison-boundary-c-1.png)![alt text](https://kurtcms.org/git/standford-machine-learning/ex6_example-dataset-1-linear-decison-boundary-c-100.png)

SVM non-linear decision boundary with a Gaussian kernel on example dataset #2:

![alt text](https://kurtcms.org/git/standford-machine-learning/ex6_example-dataset-2-non-linear-decison-boundary-gaussian-kernel.png)

SVM non-linear decision boundary with a C parameter, and a bandwidth parameter (&sigma;) for the Gaussian kernel, that minimise prediction error on example dataset #3:

![alt text](https://kurtcms.org/git/standford-machine-learning/ex6_example-dataset-3-non-linear-decison-boundary-gaussian-kernel-c-min-error.png)

### Support Vector Machines (SVMs) Spam Classifier

This involves implementing Support Vector Machines (SVMs) with a C parameter and a Gaussian kernel, to build a spam classifier, learning from a subset of spam emails made available in [SpamAssassin Public Corpus](http://spamassassin.apache.org/).

SVM spam predictor, using a vocabulary list of 1899 words that occur at least 100 times in the spam corpus, has a training accuracy of 99.85% and a test accuracy of 98.80%.

Top predictor words for spam are:

 Word            | Weight
 ---             | ---
 our             |(0.491561)
 click           |(0.467062)
 remov           |(0.421572)
 guarante        |(0.387703)
 visit           |(0.366002)
 basenumb        |(0.345912)
 dollar          |(0.323080)
 will            |(0.263241)
 price           |(0.262449)
 pleas           |(0.259879)
 nbsp            |(0.254624)
 most            |(0.253783)
 lo              |(0.251302)
 ga              |(0.248725)
 hour            |(0.241374)

## Problem Set #7: K-means Clustering and Principal Component Analysis

### K-means Clustering

This involves implementing the K-means clustering algorithm and applying it to compress an image

Moving paths of centroids using K-means clustering with iteration steps of 10 and a number of cluster of 3 on an example dataset:

![alt text](https://kurtcms.org/git/standford-machine-learning/ex7_example-dataset-k-means-iteration-1.png)![alt text](https://kurtcms.org/git/standford-machine-learning/ex7_example-dataset-k-means-iteration-10.png)

Bird image of 128×128 resolution in 24-bit colour requires 24 bits for each pixel, and results in a size of 128 × 128 × 24 = 393,216 bits.

If however, using K-means clustering to identify 16 principal colours, and represent the image using only the 16 principal colours, it requires an extra overhead colour dictionary of 24 bits, for each of the 16 principal colours, yet each pixel requires only 4 bits.

The final number of bits used is therefore 16 × 24 + 128 × 128 × 4 = 65,920 bits, which corresponds to a compression factor of 6:

![alt text](https://kurtcms.org/git/standford-machine-learning/ex7_bird-image-compression-k-means-16-colours.png)

Each pixel of the bird image, plotted with RGB values on a different axis each, grouped in 16 principal colour clusters:

![alt text](https://kurtcms.org/git/standford-machine-learning/ex7_bird-image-compression-k-means-16-colours-pixel-plot-3d.png)![alt text](https://kurtcms.org/git/standford-machine-learning/ex7_bird-image-compression-k-means-16-colours-pixel-plot-2d.png)

### Principal Component Analysis

This involves using principal component analysis to find a low-dimensional representation of face images.

Dimensionality reduction with principal component analysis on an example dataset:

![alt text](https://kurtcms.org/git/standford-machine-learning/ex7_pca-example-dataset-eigenvectors.png)![alt text](https://kurtcms.org/git/standford-machine-learning/ex7_pca-example-dataset-dimensionality-reduction-principal-component-analysis.png)

Face image of 32x32 resolution, in grayscale, have 32 x 32 = 1,024 pixels or dimensions. Using principal component analysis to reduce the dimensions from 1,024 to 100, reduces the dataset size by a factor of 10, while maintaining the general structure and appearance of the faces, despite forgoing the fine details:

![alt text](https://kurtcms.org/git/standford-machine-learning/ex7_pca-face-image-compression-dimensionality-reduction-principal-component-analysis.png)

## Problem Set #8: Anomaly Detection and Recommender Systems

### Anomaly Detection with Gaussian Distribution and F-score

This involves implementing an anomaly detection algorithm, with Gaussian distribution and a threshold (&epsilon;) optimised by F-score, on a cross-validation set, and applying it to detect failing servers on a network using their throughput (mb/s) and response latency (ms).

Anomalies with a probability of occurrence lower than the threshold (&epsilon;), which is set to maximise F-score on a cross-validation set, fitted against a Gaussian distribution:

![alt text](https://kurtcms.org/git/standford-machine-learning/ex8_anomalies-detection-gaussian-distribution.png)

### Recommender Systems using Collaborative Filtering

This involves using collaborative filtering to build a movie recommender system from a subset of the [MovieLens 100k](https://grouplens.org/datasets/movielens/) dataset from GroupLens Research of 943 users and 1682 movies.

Movie rating dataset of 943 users and 1682 movies visualised:

![alt text](https://kurtcms.org/git/standford-machine-learning/ex8_movielens-100k-movies-vs-users.png)

Provides a user with movie ratings of:
|---|
|Rated 4 for Toy Story (1995)|
Rated 3 for Twelve Monkeys (1995)
Rated 5 for Usual Suspects, The (1995)
Rated 4 for Outbreak (1995)
Rated 5 for Shawshank Redemption, The (1994)
Rated 3 for While You Were Sleeping (1995)
Rated 5 for Forrest Gump (1994)
Rated 2 for Silence of the Lambs, The (1991)
Rated 4 for Alien (1979)
Rated 5 for Die Hard 2 (1990)
Rated 5 for Sphere (1998)

Collaborative filtering trained on 100 iterations recommends:
|---|
Predicting rating 5.0 for movie Marlene Dietrich: Shadow and Light (1996)
Predicting rating 5.0 for movie Great Day in Harlem, A (1994)
Predicting rating 5.0 for movie Star Kid (1997)
Predicting rating 5.0 for movie They Made Me a Criminal (1939)
Predicting rating 5.0 for movie Saint of Fort Washington, The (1993)
Predicting rating 5.0 for movie Entertaining Angels: The Dorothy Day Story (1996)
Predicting rating 5.0 for movie Aiqing wansui (1994)
Predicting rating 5.0 for movie Santa with Muscles (1996)
Predicting rating 5.0 for movie Prefontaine (1997)
Predicting rating 5.0 for movie Someone Else's America (1995)
