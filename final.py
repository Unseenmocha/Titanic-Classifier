import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

datapath = './'

# uses sklearn StandardScaler to standard normalize certain columns of a 2D dataset
def standard_normalize(data, cols):
    scaler = StandardScaler()
    data[:, cols] = scaler.fit_transform(data[:, cols])

# fills in missing/nan values with the average value of the given column
def fill_nan(col):
    non_nan_values = col[~np.isnan(col)]
    mean = np.mean(non_nan_values)
    col[np.isnan(col)] = mean
    return col

# fills in missing/nan values with a random value from a specified options array
def fill_nan_rand(col, options):
    empty_indices = np.isnan(col)
    replacements = np.random.choice(options, size=np.sum(empty_indices))
    col[empty_indices] = replacements
    return col

# shuffles and splits dataset given x and y into training and validation datasets
def split_training_validation(x, y):
    np.random.seed(1)
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    split = int(len(y)* 0.9)
    x = x[indices, :]
    y = y[indices]

    x_valid = x[split:, :]
    x_train = x[:split, :]

    y_valid = y[split:]
    y_train = y[:split]

    return x_train, x_valid, y_train, y_valid

# given passenger ids, a parallel array of survival predictions, and a filename, writes predictions in the format expected by kaggle
def kaggle_submit_format(ids, predict, filename):
    data = list(zip(ids, predict))
    header = ["PassengerId", "Survived"]
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

# rbf kernalized model
def kernel_approx(x_train, x_valid, y_train, y_valid, test_x):
    # hyper parameter tuning originally done using the below for loop implementation of grid search
    # better testing accuracy than validation accuracy was found manually however, leading to the exclusion
    # of the below code
    # Cs = [1e-2, 1, 1e2, 1e3, 1e4]
    # gammas = [1e-3, 1e-2, 1e-1, 1, 1e1]
    # classifiers = []
    # scores = []
    # for C in Cs:
    #     for gamma in gammas:
    #         clf = SVC(C=C, gamma=gamma)
    #         clf.fit(x_train, y_train)
    #         classifiers.append((C, gamma, clf))
    #         score = clf.score(x_valid, y_valid, sample_weight=None)
    #         scores.append(score)
    # bestIndex = np.argmax(scores)
    # bestScore = scores[bestIndex]
    # bestClassifier = classifiers[bestIndex]
    # print("C=%s, gamma=%s, score=%s" % (bestClassifier[0], bestClassifier[1], bestScore))

    # initialize model with optimal parameters
    clf = SVC(C=1, gamma='scale', verbose=1, kernel='rbf')
    # fit data
    clf.fit(x_train, y_train)
    # get validation accuracy
    score = clf.score(x_valid, y_valid, sample_weight=None)

    print("kernalized validation score: "+str(score))

    # get predictions using model on testing dataset
    predictions = np.array(clf.predict(test_x))
    predictions = predictions.astype(int)
    return predictions

# sklearn relu neural net with 
# tries various learning rates and L2 regularization penalties
# cross validation via early stopping
def neural_approx(x_train, x_valid, y_train, y_valid, test_x):
    alphas = [1e-2, 1e-3, 1e-4, 1e-5]
    learning_rates = [0.0001, 0.001, 0.01]
    scores = []
    classifiers = []
    for alpha in alphas: # grid search for hyper parameter tuning
        for learning_rate in learning_rates:
            # initialize model
            clf = MLPClassifier(
                early_stopping=True,
                random_state=1, 
                max_iter=200,
                alpha=alpha, 
                learning_rate_init=learning_rate, 
                # verbose=1, 
                activation='relu', 
                solver='adam').fit(x_train, y_train) # train on training set with given parameters
            # cross validate
            scores.append(clf.score(x_valid, y_valid))
            classifiers.append((alpha, learning_rate, clf))

    # get best hyperparameters based on validation scores
    bestIndex = np.argmax(scores)
    bestScore = scores[bestIndex]
    bestClassifier = classifiers[bestIndex]
    bestCLF = bestClassifier[2]

    print("neural net approach: alpha=%s, learning rate=%s, score=%s" % (bestClassifier[0], bestClassifier[1], bestScore))
    # get predictions based on model on testing dataset
    predictions = np.array(clf.predict(test_x))
    predictions = predictions.astype(int)
    return predictions

# sklearn random forest classifier model
# uses grid search to find optimal parameters
def tree_approx(x_train, x_valid, y_train, y_valid, test_x):
    # hyperparameters to tune
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 3],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    # initialize model
    estimator = RandomForestClassifier()
    # intial training before gridsearch
    estimator.fit(x_train, y_train)
    # grid search to find best hyperparameters based on k-fold cross validation accuracy
    clf = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5)
    clf.fit(x_train, y_train)
    best_params = clf.best_params_
    print("best_params = "+str(best_params))
    print("tree based score: " + str(clf.score(x_valid, y_valid)))

    # get predictions using model on testing dataset
    predictions = np.array(clf.predict(test_x))
    predictions = predictions.astype(int)
    return predictions

def run_tasks():
    # import test.csv
    test_csvname = datapath + 'test.csv'
    test_data = pd.read_csv(test_csvname)

    # import train.csv
    train_csvname = datapath + 'train.csv'
    train_data = pd.read_csv(train_csvname)

    train_data = np.array(train_data)
    test_data = np.array(test_data)

    print("train data shape: "+str(np.shape(train_data)))
    print(train_data[0])

    # select PClass, Sex, Age, SibSp, Parch, Fare and Embarked columns for training
    x = np.hstack((train_data[:, 2:3], train_data[:, 4:8], train_data[:, 9:10], train_data[:, 11:12]))
    # select survived column as ground truth
    y = train_data[:, 1:2]
    # select PClass, Sex, Age, SibSp, Parch, Fare and Embarked columns for testing
    test_x = np.hstack((test_data[:, 1:2], test_data[:, 3:7], test_data[:, 8:9], test_data[:, 10:11]))
    # get passenger ids of testing data for submission to Kaggle
    test_ids = test_data[:, 0]

    print("x shape: "+str(np.shape(x)))
    print("y shape: "+str(np.shape(y)))

    # encode gender as 0 and 1 in training and testing set
    x[x[:, 1] == "male", 1] = 1
    x[x[:, 1] == "female", 1] = 0

    test_x[test_x[:, 1] == "male", 1] = 1
    test_x[test_x[:, 1] == "female", 1] = 0

    # encode embarked column values as 0, 1 and 2 in training and testing set
    x[x[:, 6] == 'Q', 6] = 0
    x[x[:, 6] == 'S', 6] = 1
    x[x[:, 6] == 'C', 6] = 2

    test_x[test_x[:, 6] == "Q", 6] = 0
    test_x[test_x[:, 6] == "S", 6] = 1
    test_x[test_x[:, 6] == "C", 6] = 2

    # convert training and testing datatypes to float
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    test_x = np.asarray(test_x, dtype=np.float64)

    # replace missing values in the Embarked column in the training and testing set to 'S', 'Q' or 'C' randomly
    x[:, 6] = fill_nan_rand(x[:, 6], [0, 1, 2])
    test_x[:, 6] = fill_nan_rand(test_x[:, 6], [0, 1, 2])

    # randomly replace missing gender values in the training and testing set
    x[:, 1] = fill_nan_rand(x[:, 1], [0, 1])
    test_x[:, 1] = fill_nan_rand(test_x[:, 1], [0, 1])

    # fill in remaining empty values with mean value of their respective columns
    x = np.apply_along_axis(fill_nan, axis=0, arr=x)
    test_x = np.apply_along_axis(fill_nan, axis=0, arr=test_x)

    # change dimensions of y to be compatible with sklearn
    y = y[:, 0]

    # create training/validation split on training set
    x_train, x_valid, y_train, y_valid = split_training_validation(x, y)

    # standard normalize training, validation and testing data
    standard_normalize(x_train, [2,3,4,5])
    standard_normalize(x_valid, [2,3,4,5])
    standard_normalize(test_x, [2,3,4,5])

    print("train x data shape: " + str(np.shape(x_train)))
    print(x_train[0])
    print("validation x data shape: " + str(np.shape(x_valid)))
    print(x_valid[0])
    print("train y data shape: " + str(np.shape(y_train)))
    print(y_train[0])
    print("validation y data shape: " + str(np.shape(y_valid)))
    print(y_valid[0])

    # train models using three methods, make predictions on the testing dataset and format for submission to kaggle

    kernel_predict = kernel_approx(x_train, x_valid, y_train, y_valid, test_x)
    kaggle_submit_format(test_ids, kernel_predict, "kernel_approx.csv")

    neural_predict = neural_approx(x_train, x_valid, y_train, y_valid, test_x)
    kaggle_submit_format(test_ids, neural_predict, "neural_approx.csv")
    
    tree_predict = tree_approx(x_train, x_valid, y_train, y_valid, test_x)
    kaggle_submit_format(test_ids, tree_predict, "tree_approx.csv")


run_tasks()