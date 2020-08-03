from sklearn import svm
from sklearn.metrics import accuracy_score

from classifier import Classifier

class Classifier_SVM(Classifier):
    def __init__(self, kernel='linear'):
        self.kernel = kernel

    def run(self, Train_X_Tfidf, Train_Y, Test_X_Tfidf, Test_Y):
        print("Running SVM Classifier")

        # Classifier - Algorithm - SVM
        # fit the training dataset on the classifier
        SVM = svm.SVC(C=1.0, kernel=self.kernel, degree=3, gamma='auto')
        SVM.fit(Train_X_Tfidf,Train_Y)
        # predict the labels on validation dataset
        predictions_SVM = SVM.predict(Test_X_Tfidf)
        # Use accuracy_score function to get the accuracy
        print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
