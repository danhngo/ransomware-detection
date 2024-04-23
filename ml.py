from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix

def knn_evaluate(x, y):
    # Splitting the dataset into the Training set and Test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)

    # Feature Scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # Fitting kNN to the Training set
    kNN = KNeighborsClassifier()
    kNN.fit(x_train, y_train)

    # Predicting the Test set results
    y_pred = kNN.predict(x_test)

    # Model evaluation
    accuracy = accuracy_score(y_pred, y_test)
    report = classification_report(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=1)
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

    return accuracy, report, f1, conf_matrix

def nb_evaluate(x, y):
    # Splitting the dataset into the Training set and Test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)

    # Feature Scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    #Train the Gaussian Naive Bayes model
    nb_classifier = GaussianNB()
    nb_classifier.fit(x_train, y_train)

    # Predicting the Test set results
    y_pred = nb_classifier.predict(x_test)

    # Model evaluation
    accuracy = accuracy_score(y_pred, y_test)
    report = classification_report(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=1)
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

    return accuracy, report, f1, conf_matrix

def rf_evaluate(x, y):
    # Splitting the dataset into the Training set and Test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)

    # Feature Scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

     #Train the Random Forest model
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(x_train, y_train)

    # Predict ransomware status for test data
    y_pred = rf_classifier.predict(x_test)

    # Model evaluation
    accuracy = accuracy_score(y_pred, y_test)
    report = classification_report(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=1)
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

    return accuracy, report, f1, conf_matrix

def knn_predict(x_train, y_train, new_data):
    # Feature Scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    
    # Fitting kNN to the Training set
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)

    new_data = sc.transform(new_data)
    # Predicting the Test set results
    knn_predictions = knn.predict(new_data)
    
    return knn_predictions

def nb_predict(x_train, y_train, new_data):
     # Feature Scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    
    # Train the Gaussian Naive Bayes model
    nb_classifier = GaussianNB()
    nb_classifier.fit(x_train, y_train)

    new_data = sc.transform(new_data)
    
    # Predicting the Test set results
    nb_predictions = nb_classifier.predict(new_data)
    
    return nb_predictions

def rf_predict(x_train, y_train, new_data):
    # Feature Scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    
    # Train the Random Forest model
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(x_train, y_train)
    
    new_data = sc.transform(new_data)
    # Predicting the Test set results
    rf_predictions = rf_classifier.predict(new_data)
    
    return rf_predictions
    





