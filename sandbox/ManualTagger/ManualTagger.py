import pandas as pd

#User Interface
#Save low confidence data into a file for later analysis
#Print outs the low confidence data
class ManualTagger:
    groundTruthDF = None
    
    def __init__(self, data_frame):
        self.groundTruthDF = data_frame
        
        
    def tag_this_topic(self, bagOfWords):
        filt = (self.groundTruthDF['Bag_of_Words'] == bagOfWords)
        tag_in_df = self.groundTruthDF.loc[filt, 'Tags'] #filt gives us the row we want, and 'Tag' gives us the column that we want
        return  tag_in_df