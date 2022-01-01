import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn import linear_model
from sklearn.model_selection import train_test_split

import pandas as pd
import plotly.express as px
import sqlalchemy as s

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

st.set_page_config(page_title='Marathon Predictor', page_icon=':run:')

def load_data():
    '''
    Loads previously saved data from the sqlite db
    '''
    engine = s.create_engine('sqlite:///data/marathon.db')
    cnx = engine.connect()

    dataframe = pd.read_sql('akron_marathon', cnx)
    
    dataframe = dataframe[dataframe['Age'] != 0].reset_index(drop=True)
    dataframe.loc[1344,'Age'] = 10
    dataframe['AgeDiv'] = pd.cut(
        dataframe['Age'], 
        bins=[9,19,24,29,34,39,44,49,54,59,64,69,82], # 9 is used as divisor so that 10 is included
        precision=0
    )
    return dataframe

    
def to_minutes(time):
    '''
    Helper function to convert dt.datetime.time() to minutes
    '''
    seconds = (time.hour * 60 + time.minute) * 60 + time.second
    minutes = seconds/60
    return minutes


def feature_engineer(dataframe):
    '''
    Converts time to minutes, create placement scores, and calculates the pace of each runner
    '''

    placement_df = pd.DataFrame()
    for col in ['Overall', 'SexPl', 'DivPl']:
        placement = dataframe[col].str.replace(' ','').str.split('/',expand=True)
        num = placement[0].astype(float)
        denom = placement[1].astype(float)

        percentile = 100 * (1-(num/denom))

        placement_df[f"{col}_num"] = num
        placement_df[f"{col}_denom"] = denom
        placement_df[f"{col}_percentile"] = percentile

        dataframe[f'{col}Percentile'] = percentile

    dataframe['minutes'] = dataframe['Time'].apply(to_minutes)
    dataframe['pace'] = dataframe['minutes']/26.2
    
    return dataframe

def plot_data(dataframe):
    g = sns.catplot(x='Sex', data=dataframe, kind='count')
    person_stats = dataframe.groupby('State').count()['PersonId'].reset_index()
    fig = px.choropleth(person_stats, 
                    color='PersonId',
                    locations=person_stats['State'], 
                    locationmode="USA-states", 
                    scope="usa",
                    title='Origin of Akron Marathon runners',
                    hover_data=['PersonId'],
                    labels={'PersonId':'Participants'}
                    )
    st.plotly_chart(fig)
    st.write("The vast, vast majority of Akron Marathon runners between 2017-2021 were from Ohio.")
    col1, col2 = st.columns(2)

    plt.title('Gender distribution')
    with col1:
        st.pyplot(g)
        st.write('More males than females participated.')
    
    g = sns.catplot(x='AgeDiv', data=dataframe, kind='count', hue='Sex')
    plt.xticks(rotation=45, horizontalalignment='right')
    plt.title('Age distribution')
    with col2:
        st.pyplot(g)
        st.write('And the age distribution follows a normal curve, with the bulk of runners between 19 and 54.')
    
    g = sns.catplot(x='AgeDiv', y='pace', data=dataframe, kind='box', hue='Sex')
    plt.xticks(rotation=45, horizontalalignment='right')
    plt.title('Pace distribution by age and sex')
    plt.ylabel('Minutes per mile')
    st.pyplot(g)
    st.write("__Example__: 32 year old males ran at a pace of around 9.8 minutes/mile on average. Around 75% of them ran between 7.4 and 11.4 minutes/mile.")
    

def user_data(dataframe):
    
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    with col1:
        sex = st.selectbox("Gender", options=['Female', 'Male'])
        pace = st.number_input('Most recent minutes per mile', value=15.3)
    with col2:
        age = st.number_input('Age', min_value=10.0)
    
    state_cols = list(dataframe['State'].unique())
    df_states = state_cols.copy()
    df_states.sort()
    state_cols = ['State_' + state_var for state_var in state_cols]
    
    with col2:
        state = st.selectbox('State of Origin', options=df_states, index=1)


    user_df = dict()
    user_df['Age'] = age
    user_df['pace'] = pace
    
    if sex == 'M':
        user_df['Sex_M'] = [1]
        user_df['Sex_F'] = [0]
    else:
        user_df['Sex_M'] = [0]
        user_df['Sex_F'] = [1]
    
    for col in state_cols:
        if col == f"State_{state}":
            user_df[col] = [1]
        else:
            user_df[col] = [0]
    
    user_df = pd.DataFrame(user_df)

    return user_df

def user_predict(user_df, model):
    prediction = model.predict(user_df)
    statement = f"We predict that you will run __faster than {round(prediction[0], 2)}%__ of people your age and gender!"
    if prediction[0] >= 75:
        st.success(statement)
    elif prediction[0]<= 30:
        st.warning(statement)
    else:
        st.info(statement)
    return prediction[0]

def model(dataframe):
    dataframe = pd.get_dummies(dataframe, columns=['Sex', 'State'])

    misc_cols = ['Overall', 'SexPl', 'DivPl','PersonId',
          'Bib','RaceId','Year','AgeDiv','OverallPercentile',
          'SexPlPercentile','DivPlPercentile', 'Time', 'minutes', 'City']
    feature_cols = [col for col in dataframe.columns if col not in misc_cols]

    X = dataframe[feature_cols]
    y = dataframe['DivPlPercentile']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

    rf = RandomForestRegressor(random_state=42)
    rf = rf.fit(X_train, y_train)

    return rf, X_test, y_test

def plot_user(dataframe, user_df):
    if user_df['Sex_M'][0] == 1:
        sex = 'M'
    else:
        sex = 'F'
    user_percentile = user_df['predicted'][0]
    user_pace = user_df['pace'][0]
    female = dataframe[dataframe['Sex']=='F']
    male = dataframe[dataframe['Sex']=='M']
    
    col1, col2 = st.columns(2)
    fem_plot = sns.relplot(x='pace', y='DivPlPercentile', data=female)
    plt.title('Pace and percentile rank of female runners')
    if sex == 'F':
        plt.plot(user_pace, user_percentile, 'r+', markersize=20)
    mal_plot = sns.relplot(x='pace', y='DivPlPercentile', data=male)
    plt.title('Pace and percentile rank of male runners')
    if sex == 'M':
        plt.plot(user_pace, user_percentile, 'r+', markersize=20)
    with col1:
        st.pyplot(fem_plot)
    with col2:
        st.pyplot(mal_plot)
    st.info('The relationship between minutes per mile and Division Placement in Akron Marathons, the `+` represents your pace and predicted placement')

    
def plot_result(y_predict, y_test, X_test, title, score):
    graph_data = X_test.copy()
    graph_data['Sex'] = graph_data['Sex_F'].apply(lambda x: 'Female' if x == 0 else 'Male')
    graph_data['y'] = y_test
    graph_data['predict'] = y_predict

    g = sns.relplot(x='y', y='predict', hue='Sex', data=graph_data)
    plt.xlabel('Predicted Division Percentile Placement')
    plt.ylabel('Actual Division Percentile Placement')
    plt.title(f"Performance of RandomForest Model (R2={round(score,2)})")
    st.pyplot(g)
    st.info("This plot demonstrates the scores our model got right")

def app():
    '''
    Coordinates the above functions
    '''

    st.title('Marathon Predictor')

    st.write("A fun little tool to predict the percentile placement within your age/sex division in the Akron Marathon. Not very accurate, but _very_ cool.")
    
    with st.expander('Why not accurate?'):
        st.write('''Oh boy are we glad you asked! This was just a fun little project to practice scraping, cleaning, exploring, and modeling data. The major reasons to NOT take it seriously is:
        
* While gender, state, and age are all taken in account by the model, the sample size at that level is very small. 
    * There are only 7 people from North Carolina, and more than 3000 from Ohio. 
    * If all 7 finish the marathon at an average of 9.81 minutes/mile, and you finished at 12 minute/mile, your placement would be artifically low.
    * There was only 1 nine year old in the data, they were place in the 10-19 division.

So why not just use your gender and pace to predict placement? Because that's no fun!

__Maybe the best predictor of your placement:__

* Go to the EDA page, scroll down to the boxplot. There you can see how people in each age division placed.
        ''')
    df = load_data()
    df = feature_engineer(df)

    show_plots = st.sidebar.radio('Choose a page', ['Predict my placement', 'Show EDA'])
    if show_plots == 'Show EDA':
        plot_data(df)

    else:
        user_df = user_data(df)
        random_forest_model, X_test, y_test = model(df)
        score = random_forest_model.score(X_test, y_test)
        user_prediction = user_predict(user_df, random_forest_model)
        user_df['predicted'] = user_prediction
        rf_predict = random_forest_model.predict(X_test)

        plot_user(df, user_df)
        plot_result(rf_predict, y_test, X_test, 'RandomForest', score)

app()