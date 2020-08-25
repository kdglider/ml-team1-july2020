from classifier import Classifier
from TagPredictor.classifier_SVM import Classifier_SVM
from TagPredictor.TagPredictor import TagPredictor
from ClassHumanTagger import MannualTagger

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import ast

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

'''
@file       Annotator.ipynb
@date       2020/08/03
@brief      Top level class that defines the annotation tool and active learning algorithm
'''


'''
@brief  NLP classification annotation tool
'''
class Annotator:
    groundTruthDB = None            # Pandas dataframe of all data with ground truth labels
    labeledDB = None                # Pandas dataframe of labeled data
    unlabeledDB = None                # Pandas dataframe of unlabeled data

    tagPredictor = None             # TagPredictor object
    manualTagger = None             # ManualTagger object

    confidenceThreshold = 0.8       # Prediction confidence threshold to determine if a topic should be passed to ManualTagger


    def __init__(self, datafile):
        # Create databases
        self. groundTruthDB = pd.read_csv(datafile)

        self.labeledDB, self.unlabeledDB = self.createDatabases(datafile)

        # Set up ManualTagger
        manualTagger = MannualTagger()
    

    '''
    @brief      Performs preprocessing and cleaning on a sentence
    @param      text    String that contains the raw sentence
    @return     text    String that contains the cleaned sentence
    '''
    def cleanText(self, text):
        def is_ascii(s):
            return all(ord(c) < 128 for c in s)
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text, flags=re.MULTILINE)

        # Replace newline and tab characters with spaces
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')

        # Convert all letters to lowercase
        text = text.lower()
        
        # Strip all punctuation
        #table = str.maketrans('', '', string.punctuation)
        #text = text.translate(table)

        # Remove all non-ASCII characters
        #text = text.encode(encoding='ascii', errors='ignore').decode('ascii')

        # Split feature string into a list to perform processing on each word
        wordList = text.split()

        # Remove all stop words
        #stop_words = set(stopwords.words('english'))
        #wordList = [word for word in wordList if not word in stop_words]

        # Remove all words to contain non-ASCII characters
        wordList = [word for word in wordList if is_ascii(word)]

        # Remove all leading/training punctuation, except for '$'
        punctuation = '!"#%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        wordList = [word.strip(punctuation) for word in wordList]

        # Replace all numbers with ######## identifier
        # Replace all costs with $$$$$$$$ identifier
        wordList = ['########' if (word.replace('.','').isdigit()) \
                    else '$$$$$$$$' if (word.replace('.','').replace('$','').isdigit()) \
                    else word \
                    for word in wordList]
        #wordList = ['########' if (word.replace('.','').isdigit()) else word for word in wordList]
        #wordList = ['########' if (word.translate(table).isdigit()) else word for word in wordList]

        # Reconstruct text
        # If it is empty, do not add this sample to the final output
        text = ' '.join(wordList)

        return text


    '''
    @brief      Loads data from CSV files into Pandas dataframes and performs cleanText() on all columns
    @param      datafile        CSV file with all data
    @return     groundTruthDB   Pandas dataframe of all data with ground truth labels
    @return     labeledDB       Pandas dataframe of the labeled data
    @return     unlabeledDB     Pandas dataframe of the unlabeled data
    '''
    def createDatabases(self, datafile):
        # Load CSV file as ground truth database
        groundTruthDB = self.groundTruthDB

        # Combine topic title and leading comment columns
        groundTruthDB['Bag_of_Words'] = groundTruthDB['Topic Title'] + groundTruthDB['Leading Comment']
        groundTruthDB['Bag_of_Words'] = groundTruthDB['Bag_of_Words'].str.strip().str.replace('   ', ' ').str.replace('  ', ' ')

        groundTruthDB = groundTruthDB.drop(columns=['Topic Title', 'Leading Comment', 'Unnamed: 0'])
        
        # Apply cleanText() to all columns with this:
        groundTruthDB['Bag_of_Words'] = groundTruthDB['Bag_of_Words'].apply(lambda x: self.cleanText(x))


        #create an offset value
        offset = 0
        #the total number of unique comments
        total = len(groundTruthDB)
        for index, entry in enumerate(groundTruthDB['Bag_of_Words']):
            #create a duplicate if post has multiple tags
            tag_list = ast.literal_eval(groundTruthDB.loc[index, 'Tags'])
            text = groundTruthDB.loc[index,'Bag_of_Words']
            while (isinstance(tag_list, list) and len(tag_list) > 1):
                #print(index)
                #sets the tag for the duplicate to a string
                groundTruthDB.loc[total+offset, 'Tags'] = tag_list.pop()
                #Adds the duplicate to the end of the pandas dataframe
                groundTruthDB.loc[total+offset, 'Bag_of_Words'] = text
                offset = offset + 1
            #Changes the first tag to a string
            if (len(tag_list) == 1):
                groundTruthDB.loc[index, 'Tags'] = tag_list.pop()
            #Changes empty tags from lists to strings
            if (isinstance(groundTruthDB.loc[index, 'Tags'], list)):
                groundTruthDB.loc[index, 'Tags'] = ''
                # Not sure why this element is stored as '[]' instead of ''

        # Filter out topics with no tags
        groundTruthDB = groundTruthDB[groundTruthDB['Tags'].map(len) > 2]

        # Split ground truth database into labeled and unlabelled databases
        #mask = np.random.rand(len(groundTruthDB)) < 0.8
        #labeledDB = groundTruthDB[~mask]
        #unlabeledDB = groundTruthDB[mask]['Bag_of_Words']

        unlabeledDB, labeledDB = train_test_split(groundTruthDB, test_size=0.2)
        unlabeledDB = unlabeledDB['Bag_of_Words']

        return labeledDB, unlabeledDB


    '''
    @brief      Demonstration function to run the entire annotator application
    @param      
    @return     None
    '''
    def runApplication(self, classifier):
        # Set up TagPredictor object
        tagPredictor = TagPredictor(classifier, self.labeledDB)

        # Train tagPredictor
        tagPredictor.train()

        # Predict tags for all unlabeled topics
        tagList, confidenceList = tagPredictor.predict(self.unlabeledDB)

        # Continue running the active learning loop as long as there are still low-confidence topics
        while (any(p < self.confidenceThreshold for p in confidenceList) == True):
            # Log tagging statistics
            
            # Get low-confidence topic indices
            lowConfIndices = [i for i in range(len(L)) if confidenceList[i] < self.confidenceThreshold]

            # Pass low-confidence topics to the manual tagger
            lowConfTopics = self.unlabelDB.iloc(lowConfIndices)
            labeledTopics = self.manualTagger.run(lowConfTopics)

            # Add manually tagged topics to the labeled database
            self.labeledDB = pd.concat([self.labeledDB, labeledTopics], join='inner')

            # Remove tagged topics from unlabeled database
            self.unlabeledDB = self.unlabeledDB.drop(lowConfTopics)

            # Train tagPredictor with updated database
            tagPredictor = TagPredictor(classifier, self.labeledDB)
            tagPredictor.train()

            # Predict tags for all unlabeled topics
            tagList, confidenceList = tagPredictor.predict(self.unlabeledDB)

if __name__ == '__main__':

    # Path to CSV datafile
    datafile = '/Users/kittyguz/Desktop/AnnotatorPythonVersion/StackOverflow_new_tags.csv'
    df = pd.read_csv(datafile)
    manualTagger = MannualTagger()
    result = manualTagger.run(df)

    #print(annotator.groundTruthDB)

    #text = annotator.groundTruthDB.iloc[96]['Bag_of_Words']
    #print(text)
    #print(annotator.cleanText(text))