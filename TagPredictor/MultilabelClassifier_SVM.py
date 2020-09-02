'''
@file       MultilabelClassifier_SVM.py
@date       2020/08/25
@brief      Multilabel classifier that uses SVM
'''

import numpy as np

from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC

from classifier import Classifier


'''
Multilabel classifier that uses SVM
'''
class MultilabelClassifier_SVM(Classifier):
    kernel = None       # SVM kernel type
    model = None        # SVC object


    '''
    @brief      Class constructor
    @param      kernel      SVM kernel type
    @return     None
    '''
    def __init__(self, kernel='linear'):
        # Set SVM kernel type
        self.kernel = kernel


    '''
    @brief      Trains the model using given X and Y matrices
    @param      Train_X_Tfidf   Scikit-learn compatible matrix of TF-IDF embeddings for each topic text
    @param      Train_Y         Binary indicator matrix for the Y labels (tags) of the topics
    @return     self.model
    '''
    def train(self, Train_X_Tfidf, Train_Y, Test_X_Tfidf=None, Test_Y=None):
        # Fit the training dataset on the classifier
        self.model = MultiOutputClassifier(SVC(C=1.0, kernel=self.kernel, degree=3, gamma='auto', probability=True))
        self.model.fit(Train_X_Tfidf, Train_Y)
        
        return self.model


    '''
    @brief      Predicts an indicator matrix and confidence level for each topic
    @param      df                  Pandas dataframe of topic text
    @return     predictionMatrix    NumPy indicator matrix for the predicted tags
    @return     confidenceList      NumPy array of prediction confidence scores for each topic
    '''    
    def predict(self, df):
        # Raw list of category probabilities (Shape n_categories x n_samples x n_outputs)
        rawList = self.model.predict_proba(df)
        
        # Convert to NumPy array
        # Extract relevant output for indicator matrix generation and transpose
        probabilityMatrix = np.array(rawList)[:, :, 1].T
        
        # Round probabilities to create an indicator matrix
        predictionMatrix = np.round_(probabilityMatrix)
        
        # If no tags are predicted for a topic, assign one tag based on highest probability
        for i in range(predictionMatrix.shape[0]):
            if (np.sum(predictionMatrix[i]) == 0):
                maxProbabilityIndex = np.argmax(predictionMatrix[i])
                predictionMatrix[i][maxProbabilityIndex] = 1
        
        # Extract relevant probability output for confidence calculation and transpose
        confidenceMatrix = np.amax(np.array(rawList), axis=2).T
        
        # Average confidences across all categories for all samples
        #print('Using Average confidence calculation')
        #confidenceList = np.average(confidenceMatrix, axis=1)
        
        # Average confidences for bottom 5 categories for all samples
        print('Using Low-5 confidence calculation')
        confidenceList = np.average(np.sort(confidenceMatrix)[:,0:5], axis=1)
        
        return predictionMatrix, confidenceList