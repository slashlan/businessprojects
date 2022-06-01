import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import altair as alt
from scipy import stats

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')

# DATA CLEANING AND MANIPULATION # streamlit run Case_study_dashboard.py

# Load clicks dataset
df_c = pd.read_csv('jobs_ad_clicks.csv')

# Load job ads details dataset
df_j = pd.read_csv('dfmeta_jobs_filtered_v2.csv', encoding='unicode_escape')

# Merge two datasets
df = df_c.merge(df_j, how='inner', on='jobId')

# Assign correct datatypes
df['dateYM'] = pd.to_datetime(df['dateYM'])
df['pub_datum'] = pd.to_datetime(df['pub_datum'])
df['place'] = df['place'].astype('category')
df['sector'] = df['sector'].astype('category')

# Drop observations missing job ads details (5% of total sample)
df = df.dropna(thresh=df.shape[1]*0.66).copy()

# Fill missing values by mapping work region-place columns
mapping = df[df['place'].notnull()].drop_duplicates('workregion').set_index('workregion').place

# Fill missing values in "place" by mapping "workregion"
df['place'] = df['workregion'].map(mapping).fillna(np.nan)

# Encode string features
df['loggedIn_num'] = pd.get_dummies(df['loggedIn'], drop_first=True)
df['place_num'] = df['place'].cat.codes
df['sector_num'] = df['sector'].cat.codes

# Adjust date format publication date
df['pub_date'] = pd.to_datetime(df['pub_datum']).dt.date
df['pub_date'] = pd.to_datetime(df['pub_date'])

# Create new features
# Days passed between publication and visit
df['pub_visit_days'] = (df['dateYM'] - df['pub_date']).dt.days

# EXPLORATORY DATA ANALYSIS

# Group By UserID
user_g = df.groupby('userId').agg({'jobId': ['count', 'nunique'],  # Number of clicks per user, Number of clicks on unique ads per user
                                   'nsession': 'sum',  # Total number of sessions
                                   'place': 'nunique',  # Number of unique regions
                                   'sector': 'nunique',
                                   'pub_visit_days': 'mean'
                                   })

# Filter for outliers
user_filt = user_g[(np.abs(stats.zscore(user_g)) < 3).all(axis=1)]
non_outliers = list(user_filt.index)


# START STREAMLIT APP
class Toc:

    def __init__(self):
        self._items = []
        self._placeholder = None

    def title(self, text):
        self._markdown(text, "h1")

    def header(self, text):
        self._markdown(text, "h2", " " * 2)

    def subheader(self, text):
        self._markdown(text, "h3", " " * 4)

    def placeholder(self, sidebar=False):
        self._placeholder = st.sidebar.empty() if sidebar else st.empty()

    def generate(self):
        if self._placeholder:
            self._placeholder.markdown("\n".join(self._items), unsafe_allow_html=True)

    def _markdown(self, text, level, space=""):
        key = "".join(filter(str.isalnum, text)).lower()

        st.markdown(f"<{level} id='{key}'>{text}</{level}>", unsafe_allow_html=True)
        self._items.append(f"{space}* <a href='#{key}'>{text}</a>")


toc = Toc()

# Notify the reader that the data was successfully loaded.
data_load_state.text("Done!")

st.title('Jobcloud data challenge - Data Analyst (Seeker)')
toc.placeholder()

toc.header('Overview')
st.text(" ")
st.write(""" 
- The goal of this challenge is to **analyze and draw useful insights** from the provided data consisting of two CSVs,
one containing the clicked job ads per user and one containing the details of each job ad.

- After merging these CSVs files, cleaning and formatting the data, the sample is composed by:
    - **2.5 million** clicks
    - **751k** unique users
    - **59k** unique job ads
    - Publication dates ranging between **April 1st, 2021** and **December 14, 2021** 
    - Click dates spanning between **October 28th, 2021** and **December 15, 2021**

    
- Notice that around **30k users** responsible for ** additional 2.5 million clicks** have been **removed** as they were
considered **outliers**.
These users had a disproportionate number of clicks and sessions compare to the rest of the sample as they are probably 
constituted by search engines and spider bots crawling through the website.
         """
         )

# 1) MOST CLICKED JOBS, SECTORS AND REGIONS Most clicked job titles, sectors and regions
toc.header('1) Which are the most clicked jobs, sectors and work regions?')
st.text(" ")
# Jobs
most_job1, most_job2 = st.columns([2, 1])

most_job1.subheader("Top 10 most clicked jobs")
jobs = df[df['userId'].isin(non_outliers)]['title'].value_counts().head(10).to_frame().reset_index()
jobs.rename(columns={'index': 'Job Title', 'title': 'Clicks'}, inplace=True)

c = alt.Chart(jobs).mark_bar().encode(x='Clicks', y=alt.Y('Job Title', sort='-x'), color=alt.Color('Clicks', legend=None)
                                      ).configure_range(category={'scheme': 'tableau20'})

most_job1.altair_chart(c, use_container_width=True)
most_job2.write(jobs)
st.text(" ")

# Sectors
most_sect1, most_sect2 = st.columns([2, 1])

most_sect1.subheader("Top 10 most clicked sectors")
sectors = df[df['userId'].isin(non_outliers)]['sector'].value_counts().head(10).to_frame().reset_index()
sectors.rename(columns={'index': 'Sector', 'sector': 'Clicks'}, inplace=True)

c = alt.Chart(sectors).mark_bar().encode(x='Clicks', y=alt.Y('Sector', sort='-x'), color=alt.Color('Clicks', legend=None)
                                         ).configure_range(category={'scheme': 'plasma'})
most_sect1.altair_chart(c, use_container_width=True)
most_sect2.write(sectors)
st.text(" ")

# Work regions
most_reg1, most_reg2 = st.columns([2, 1])

most_reg1.subheader("Top 10 most clicked region/city")
work_regions = df[df['userId'].isin(non_outliers)]['place'].value_counts().head(10).to_frame().reset_index()
work_regions.rename(columns={'index': 'Region', 'place': 'Clicks'}, inplace=True)

c = alt.Chart(work_regions).mark_bar().encode(x='Clicks', y=alt.Y('Region', sort='-x'), color=alt.Color('Clicks', legend=None)
                                              ).configure_range(category={'scheme': 'orangered'})

most_reg1.altair_chart(c, use_container_width=True)
most_reg2.write(work_regions)
st.text(" ")


# 2) HOW MANY CLICKS DOES A USER MAKE? HOW MANY REGIONS ARE USERS CLICKING?
toc.header('2) How many clicks does a user make? In how many different regions?')
st.text(" ")

# Facts
col1, col2, col3 = st.columns(3)
col1.metric("Average clicks per user", round(user_filt.describe().loc['mean', ('jobId', 'count')], 2))
col2.metric("Average sessions per user", round(user_filt.describe().loc['mean', ('nsession', 'sum')], 2))
col3.metric("Average number of work region/cities", round(user_filt.describe().loc['mean', ('place', 'nunique')], 2))

col4, col5 = st.columns(2)
col4.metric("Average number of different sectors explored", round(user_filt.describe().loc['mean', ('sector', 'nunique')], 2))
col5.metric("Average number days publication date-click", round(user_filt.describe().loc['mean', ('pub_visit_days', 'mean')], 2))
st.text(" ")


# 3) ARE THERE DIFFERENCES BETWEEN LOGGED-IN USERS AND ANONYMOUS SESSIONS
toc.header('3) Are there differences between logged-in users and anonymous sessions?')
st.text(" ")

# Logged-in vs anonymous

# Groupby loggedIn
logged_g = df[df['userId'].isin(non_outliers)].groupby(['loggedIn']).agg({'loggedIn': 'count',
                                                                          'userId': 'nunique',
                                                                          'jobId': 'nunique',
                                                                          'nsession': 'mean',
                                                                          'title': 'nunique',
                                                                          'sector': 'nunique',
                                                                          'pub_visit_days': 'mean'}
                                                                         ).sort_values('nsession', ascending=False)

logged_g['avg_click_per_user'] = logged_g['loggedIn']/logged_g['userId']
logged_g['avg_unique_job_per_user'] = logged_g['jobId']/logged_g['userId']

logged_g.rename(columns={'loggedIn': 'Clicks', 'userId': 'Unique_users'}, inplace=True)
logged_ch = logged_g.reset_index()[['loggedIn', 'Clicks', 'Unique_users', 'nsession', 'pub_visit_days', 'avg_click_per_user']]

# Charts level 1
login1, login2, login3 = st.columns([1, 1, 1])

# Total clicks
login1.subheader("Total clicks")
c = alt.Chart(logged_ch).mark_bar().encode(x='loggedIn', y=alt.Y('Clicks'), color=alt.Color('loggedIn')
                                           ).configure_range(category={'scheme': 'dark2'})
login1.altair_chart(c, use_container_width=True)

# Unique users
login2.subheader("Unique users")
c = alt.Chart(logged_ch).mark_bar().encode(x='loggedIn', y=alt.Y('Unique_users'), color=alt.Color('loggedIn')
                                           ).configure_range(category={'scheme': 'dark2'})
login2.altair_chart(c, use_container_width=True)

# Average number sessions
login3.subheader("Avg sessions")
c = alt.Chart(logged_ch).mark_bar().encode(x='loggedIn', y=alt.Y('nsession'), color=alt.Color('loggedIn')
                                           ).configure_range(category={'scheme': 'dark2'})
login3.altair_chart(c, use_container_width=True)
st.text(" ")

# Charts level 2
login4, login5 = st.columns([1, 1])

# Avg clicks user
login4.subheader("Avg clicks per user")
c = alt.Chart(logged_ch).mark_bar().encode(x='loggedIn', y=alt.Y('avg_click_per_user'), color=alt.Color('loggedIn')
                                           ).configure_range(category={'scheme': 'dark2'})
login4.altair_chart(c, use_container_width=True)

# Avg clicks user
login5.subheader("Avg days publication-click")
c = alt.Chart(logged_ch).mark_bar().encode(x='loggedIn', y=alt.Y('pub_visit_days'), color=alt.Color('loggedIn')
                                           ).configure_range(category={'scheme': 'dark2'})
login5.altair_chart(c, use_container_width=True)

st.markdown('''All differences between logged-in sessions and anonymous ones are **statistically significant at 99% level**. 
As the distributions of these variables were not normal, differences between groups have been tested using
the **Mann-Whitney U Test** for ordinal/continuous, not normally distributed data.''')
st.text(" ")

# FOOD FOR THOUGHT
toc.header('Takeaways & food for thought')
st.text(" ")
toc.subheader('Findings')
st.text(" ")
st.write("""
- Most clicked ads involve clerkship & middle-skilled technical jobs
- Most clicked job sectors are 2, 12, and 10
- Jobs based in Swiss-German cantons/cities are by far the most clicked
- On average, a user clicks on 3 different job ads and explores 2 work regions
- Average age of the job ad when clicked is 12 days
- Logged-in users are only 7% of the total but:
    - they account for 25% of total clicks
    - they click 4 times more often than anonymous users (12 vs 3 clicks on average)
    - they are more reactive in clicking new job ads (6 days after publication on average)
""")
st.text(" ")

toc.subheader('Action points')
st.write("""
- Incentivizing users to register would definitely increase clicks
- There is space to grow traffic for highly-skilled job seekers
- There is space to increase job ads based in Swiss-French cantons
- Should something be done to prevent/limit search engines & bot from crawling the website?

""")

# Generate Table of contents
toc.generate()
