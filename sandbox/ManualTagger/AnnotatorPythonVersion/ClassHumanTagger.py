import streamlit as st
import pandas as pd
import numpy as np
from openpyxl import Workbook, load_workbook
import SessionState

class MannualTagger:
    df_posts = None
    data_dic = None
    iterate_key = None
    curr_key = None
    session = None
    df_tags = None
    tags_list = None
    df_labeled = None
    
    def __init__(self):
        self.df_posts
        #self.df_posts.to_csv("Initial_Data.csv")

        self.data_dic
        self.iterate_key
        self.curr_key
        self.session
        #importing tags list
        self.df_tags = pd.read_csv("Tags.csv")
        self.tags_list = self.df_tags.Tags.tolist()

        self.df_labeled = pd.read_csv('LabeledData.csv')
    """
    This function takes in a dataframe and returns a dictionary with the topic title as key and the leading comment as value
    """
    #@st.cache(suppress_st_warning=True, allow_output_mutation=True) # This function will be cashed and won't be run again
    def load_data(self, unlabeled_data):

        def get_data_from_file(df):
            dic = {}
            for row in range(len(df)):
                dic[df['Topic Title'].iloc[row]] = df['Leading Comment'].iloc[row]
            return dic

        """
        These two statements read in our desired csv files. The first one reads in the file that contains all redefined TagNames, which we have to make ourselves. The second contains scrapped data.
        """
        #new_labeled_topics: is the dataframe where we store labeled data
        #new_labeled_topics = self.df_posts.copy()
        #data_dic: contains the the scrapped data: with the topic title as key and the leading comment as value
        self.data_dic = get_data_from_file(self.df_posts)
        #iterate_key: an iterater that goes to the next topic when called next(iterate_key)
        self.iterate_key = iter(self.data_dic.keys())
        #curr_key: the tracker that keeps track of the current topic title that we are on
        self.curr_key = next(self.iterate_key)

    def run(self, dataframe):
        self.df_posts = dataframe
        if self.df_posts.empty:
            st.write("""
                        ** ML July Team1 Manual Tagging App**
                        All taggings are complete! Thank you for your help!
                    """)
            return self.df_labeled
        else:
            """
            Below is mainly code for StreamLit display.
            """
            st.write("""
            ** ML July Team1 Manual Tagging App**
            """)

            #the line below is for the drop down menu for tag selection. We will switch df_posts with df_all_tag.
            self.load_data(self.df_posts)

            #remove the tagged post from the dataset and reset
            if st.button("Reset"):
                self.df_posts = self.df_posts.iloc[1:]

            #displays next topic
            self.session = SessionState.get(run_id=0)
            if st.button("Next Topic"):
                self.session.run_id += 1

            options = st.multiselect('Please select suitable tags for the following topic.', self.tags_list, key=self.session.run_id)
            st.write('You selected:', options)

            st.write("Topic Title:")
            st.write(self.curr_key)

            st.write("Leading Comment:")
            st.write(self.data_dic[self.curr_key])

            #writes the tagged post to a excel file
            if st.button("Submit"):
                row_to_append = pd.DataFrame({self.df_labeled.columns[0]: [self.curr_key], self.df_labeled.columns[1]: [self.data_dic[self.curr_key]], self.df_labeled.columns[2]: [options]})
                self.df_labeled = self.df_labeled.append(row_to_append)
         
        


