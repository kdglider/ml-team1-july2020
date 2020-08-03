from abc import ABC, abstractmethod

class Classifier(ABC):
    @abstractmethod
    def run(self, Train_X_Tfidf, Train_Y, Test_X_Tfidf, Test_Y):
        pass