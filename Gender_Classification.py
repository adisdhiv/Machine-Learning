# Building a gender classifier with input data of height, weight and shoe size

from sklearn import tree, svm, ensemble, naive_bayes, neighbors, neural_network
from sklearn.metrics import accuracy_score

# Defining different classifiers
clf = [tree.DecisionTreeClassifier(),svm.SVC(), ensemble.RandomForestClassifier(),
       naive_bayes.GaussianNB(), neighbors.KNeighborsClassifier(),
       neural_network.MLPClassifier(), ensemble.AdaBoostClassifier()]

#Training Data (Height, Weight and Shoe size and their labels)

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# Creating List for storing scores
ClassifiersList = ['Decision Tree', 'SVC', 'Random Forest', 'Naive Bayes', 'KNN', 'MLP', 'Ada Boost']
scores = []

# Training and Testing

for eachClassifier in clf:
    eachClassifier = eachClassifier.fit(X, Y)
    prediction = eachClassifier.predict(X)
    score = accuracy_score(Y, prediction) * 100
    #print("Accuracy for {} is {}".format(eachClassifier, score))
    scores.append(score)

# Finding the best classifier
print("Best Classifier is {} with {}.".format(ClassifiersList[np.argmax(scores)],np.max(scores)))
