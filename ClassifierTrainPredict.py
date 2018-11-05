# Training a classifier and checking its accuracy for a simple dataset


from sklearn import tree, svm, ensemble, naive_bayes
from sklearn.metrics import accuracy_score

# Defining different classifiers
clf = [tree.DecisionTreeClassifier(),svm.SVC(), ensemble.RandomForestClassifier(), naive_bayes.GaussianNB()]

#Training Data (Height, Weight and Shoe size and their labels)

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

Y_true = ['male']

# Training and prediction

for eachClassifier in clf:
    eachClassifier = eachClassifier.fit(X, Y)
    prediction = eachClassifier.predict([[190,70,43]])
    score = accuracy_score(Y_true, prediction)
    print(prediction, score)
