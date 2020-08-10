import streamlit as st
import pandas as pd
import numpy as np
#df = pd.read_csv("FileWithAllTagName.csv")
#df_topics = pd.read_csv("FileWithUnlabeledTopics.csv")
def get_data_from_file(df):
	#creating a dictionary with the topic title as key and the leading comment as value
	dic = {}
	for row in df.itertuples():
		dic[row._2] = row._4
	return dic

df_topics = pd.read_csv("StackOverflow_new_tags.csv")
new_labeled_topics = df_topics.copy()
data_dic = get_data_from_file(df_topics)
iterate_key = iter(data_dic.keys())
curr_key = next(iterate_key)

st.write("""
** ML July Team1 Manual Tagging App**
""")
options = st.multiselect('Please select suitable tags for the following topic.', df_topics['Tags'].unique())
st.write('You selected:', options)

if st.button("Next Topic"):
	curr_key = next(iterate_key)

st.write("""
	**Topic Title**
	""")
st.write(curr_key)

st.write("""
	**Leading Comment**
	""")
st.write(data_dic[curr_key])
topi.new_labeled_topics.ix[new_labeled_topics._2 == topic.curr_key, 'Tags'] = options