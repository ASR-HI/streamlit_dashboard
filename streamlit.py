import streamlit as st
import pandas as pd
import requests
import base64
import json
from dotenv import load_dotenv
import os
from pymongo import MongoClient
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from wordcloud import WordCloud
from pyspark.sql.functions import udf, when, size, col, regexp_replace, concat_ws , regexp_extract, count, split, lower , trim, year, month

from pyspark.sql.types import DateType
spark = SparkSession.builder.appName("PandasToPySpark").getOrCreate()

# Load environment variables
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    raise ValueError("No MONGO_URI found in environment variables")

def convert_date_columns(df):
    """Convertit les colonnes de type datetime.date en datetime.datetime"""
    for col in df.select_dtypes(include=["object", "datetime"]):
        if df[col].dtype == "object" or isinstance(df[col].iloc[0], (pd.Timestamp, datetime.date)):
            try:
                df[col] = pd.to_datetime(df[col])  # Conversion en datetime.datetime
            except Exception:
                continue
    return df

def safe_get(data, index):
    try:
        if isinstance(data, list) and len(data) > index:
            return data[index]
        return None
    except Exception as e:
        print(f"Erreur lors de l'extraction des données: {e}")
        return None

def fetch_and_process_data():

    client = MongoClient(mongo_uri)
    db = client["scraping_data_cleaned_v1"]

    # Define collections
    authors_labs_collection = db["authors_labs"]
    journaux_collection = db["journaux"]
    articles_collection = db["articles"]

    try:
        # Fetch data from MongoDB and convert to Pandas DataFrame
        authors_labs_data = authors_labs_collection.find()
        journaux_data = journaux_collection.find()
        articles_data = articles_collection.find()

        # Convert data to Pandas DataFrame
        authors_labs_pd = pd.DataFrame(list(authors_labs_data))
        journaux_pd = pd.DataFrame(list(journaux_data))
        articles_pd = pd.DataFrame(list(articles_data))

        # Drop '_id' column for better readability
        articles_pd = articles_pd.drop(columns=['_id'], errors='ignore')
        journaux_pd = journaux_pd.drop(columns=['_id'], errors='ignore')
        authors_labs_pd = authors_labs_pd.drop(columns=['_id'], errors='ignore')

        # Convert dates before using DataFrames
        authors_labs_pd = convert_date_columns(authors_labs_pd)
        journaux_pd = convert_date_columns(journaux_pd)
        articles_pd = convert_date_columns(articles_pd)

        # Check data after conversion
        print("Authors labs DataFrame:", authors_labs_pd.head())
        print("Journaux DataFrame:", journaux_pd.head())
        print("Articles DataFrame:", articles_pd.head())

        # Flatten the 'Data' column (explode the list of dictionaries) for journaux
        if isinstance(journaux_pd['Data'].iloc[0], list):
            exploded_df = journaux_pd.explode('Data').reset_index(drop=True)

            # Extract fields from each dictionary in the exploded Data column
            exploded_df['Category'] = exploded_df['Data'].apply(lambda x: safe_get(x, 0))
            exploded_df['Quartile'] = exploded_df['Data'].apply(lambda x: safe_get(x, 1))
            exploded_df['Year'] = exploded_df['Data'].apply(lambda x: safe_get(x, 2))
            
            
            ########################
            
            journaux_df_exploded = journaux_pd.explode('Data').reset_index(drop=True)

            # Extraire les champs des dictionnaires
            journaux_df_exploded['Category'] = journaux_df_exploded['Data'].apply(lambda x: safe_get(x, 0))
            journaux_df_exploded['Year'] = journaux_df_exploded['Data'].apply(lambda x: safe_get(x, 2))
            journaux_df_exploded['Quartile'] = journaux_df_exploded['Data'].apply(lambda x: safe_get(x, 1))

            # Supprimer la colonne 'Data' si elle n'est plus nécessaire
            journaux_df_exploded = journaux_df_exploded.drop(columns=['Data'])
            journaux_df_exploded = journaux_df_exploded.dropna(subset=['Quartile'])
            new_data = {}
            for country in journaux_df_exploded['Country'].unique():
                # Initialize counts for each quartile to zero
                new_data[country] = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}
                # Iterate over the dataframe rows for the current country
                for index, row in journaux_df_exploded[journaux_df_exploded['Country'] == country].iterrows():
                    # Increment the count for the corresponding quartile
                    new_data[country][row['Quartile']] += 1

        
            # Create the new DataFrame
            new_journaux_df = pd.DataFrame.from_dict(new_data, orient='index')                       
            new_journaux_df['Country'] = new_journaux_df.index
            new_journaux_df = new_journaux_df[['Country', 'Q1', 'Q2', 'Q3', 'Q4']]

        else:
            st.error("The 'Data' column is not in the expected format (list of dictionaries).")

        
        return authors_labs_pd, journaux_pd, articles_pd, exploded_df, new_journaux_df

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None, None, None, None, None
    finally:
        client.close()

def create_seaborn_plot(exploded_df):
    plt.figure(figsize=(20, 6))
    sns.countplot(data=exploded_df, x='Category', hue='Quartile', palette='Set1')
    plt.title('Distribution of Quartiles per Category')
    plt.xticks(rotation=90)
    plt.xlabel('Category')
    plt.ylabel('Count')
    st.pyplot(plt)

def Distribution_Quartiles_per_Category(exploded_df):
    fig = px.histogram(exploded_df, x='Category', color='Quartile',
                       title='Distribution of Quartiles per Category',
                       labels={'Category': 'Category', 'Quartile': 'Quartile'},
                       category_orders={'Quartile': ['Q1', 'Q2', 'Q3', 'Q4']})
    fig.update_layout(
        barmode='stack',
        xaxis_tickangle=-90,
        width=2000,
        height=800,
    )
    st.plotly_chart(fig)

def plot_quartile_distribution(df):
    """Creates and displays a time series plot of quartile distribution over the years"""
    quartile_counts = df.groupby(['Year', 'Quartile']).size().reset_index(name='Count')
    fig = px.line(
        quartile_counts,
        x='Year',
        y='Count',
        color='Quartile',
        markers=True,
        title='Time Series of Quartile Distribution Over the Years',
        labels={'Year': 'Year', 'Count': 'Count of Publications'},
        line_shape='linear',
        template='plotly_white'
    )
    fig.update_layout(
        width=2000,
        height=800,
        title_font=dict(size=18, family="Arial, sans-serif", weight='bold'),
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
        xaxis=dict(tickangle=45, tickfont=dict(size=12)),
        yaxis=dict(tickfont=dict(size=12)),
        plot_bgcolor='white',
        legend_title=dict(text='Quartile', font=dict(size=14)),
        legend=dict(font=dict(size=12))
    )
    st.plotly_chart(fig)

def word_cloud_distribution(df, column):
    """Creates and displays a word cloud of categories distribution"""
    category_counts = df[column].value_counts()
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(category_counts)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud of {column} Distribution')
    st.pyplot(plt)

def Word_Cloud_Categories_Distribution(df):
    """Creates and displays a word cloud of categories distribution"""
    category_counts = df['Category'].value_counts()
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(category_counts)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Categories Distribution')
    st.pyplot(plt)

def plot_country_quartile_distribution(df):
    sns.set_theme(style="whitegrid")
    df_long = df.melt(id_vars='Country', var_name='Quarter', value_name='Count')
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_long, x='Count', hue='Quarter', kde=True, bins=20, palette='Set2', edgecolor='black', multiple="stack")
    plt.title('Distribution of Contributions by Quarter (Q1, Q2, Q3, Q4)', fontsize=16, weight='bold')
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(title='Quarter', fontsize=10, title_fontsize=12, loc='upper right')
    st.pyplot(plt)

def plot_country_Contributions(new_journaux_df):
    sns.set_theme(style="whitegrid")

    df_long = new_journaux_df.melt(id_vars='Country', var_name='Quarter', value_name='Count')
    countries = df_long['Country'].unique()
    n_columns =  4
    n_rows = (len(countries) // n_columns) + (len(countries) % n_columns > 0)  # Calculate the number of rows

    fig, axes = plt.subplots(n_rows, n_columns, figsize=(5 * n_columns, 5 * n_rows))
    axes = axes.flatten()

    for i, country in enumerate(countries):
        country_data = df_long[df_long['Country'] == country]
        sns.barplot(
            data=country_data,
            x='Quarter',
            y='Count',
            palette='Set2',
            ax=axes[i],
            edgecolor='black'
        )

        axes[i].set_title(f'{country} Contributions by Quarter', fontsize=12, weight='bold')
        axes[i].set_xlabel('Quarter', fontsize=10)
        axes[i].set_ylabel('Count', fontsize=10)

    for i in range(len(countries), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    st.pyplot(plt)

def process_keywords(df_with_dates):
    # Split keywords by comma, trim spaces, convert to lowercase, and explode into individual keywords
    df_with_dates['keywords'] = df_with_dates['keywords'].str.split(',')
    df_keywords = df_with_dates.explode('keywords')
    
    # Clean up each keyword: trim spaces and convert to lowercase
    df_keywords['keywords'] = df_keywords['keywords'].str.strip().str.lower()
    
    # Remove duplicates in the exploded keywords (across all rows)
    df_keywords_unique = df_keywords.drop_duplicates(subset=['keywords'])
    
    # Count the frequency of each unique keyword
    df_keywords_grouped = df_keywords.groupby('keywords').size().reset_index(name='keyword_count')
    
    # Sort the keyword counts in descending order
    df_keywords_grouped = df_keywords_grouped.sort_values(by='keyword_count', ascending=False)
    
    # Create a bar plot for the most common keywords
    fig = px.bar(
        df_keywords_grouped.head(10),
        x='keywords',
        y='keyword_count',
        title='Top 10 Most Common Keywords',
        labels={'keywords': 'Keyword', 'keyword_count': 'Count'}
    )
    
    # Customize plot appearance
    fig.update_traces(marker=dict(color='#8da0cb'))
    
    st.plotly_chart(fig)


def NumberArticlesperYear(articles_pd):
    # Drop 'authors_data' column if exists
    df = articles_pd.copy()
    df = df.drop(columns=['authors_data'], errors='ignore')
    
    # Convert 'publication_date' column to datetime format
    df['publication_date'] = pd.to_datetime(df['publication_date'])
    
    # Group by 'publication_date' and count the number of articles per date
    df_grouped = df.groupby('publication_date').size().reset_index(name='article_count')
    
    # Sort by 'publication_date'
    df_grouped = df_grouped.sort_values(by=['publication_date'], ascending=[True])
    
    # Create an interactive time series plot using Plotly
    fig = px.line(df_grouped,
                  x='publication_date',
                  y='article_count',
                  title='Number of Articles by Date',
                  labels={'publication_date': 'Date', 'article_count': 'Number of Articles'},
                  markers=True)
    
    # Customize the plot's appearance
    fig.update_traces(
        line=dict(color='#6B5B95', width=4, dash='solid'),
        marker=dict(size=8, color='#FFB6C1', symbol='circle', line=dict(color='#6B5B95', width=2))
    )

    fig.update_layout(
        title_font_size=24,
        title_font_color='rgba(94, 45, 120, 1)',
        title_x=0.5,
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        xaxis_title_font_color='rgba(94, 45, 120, 1)',
        yaxis_title_font_color='rgba(94, 45, 120, 1)',
        xaxis=dict(tickangle=45, tickfont=dict(size=12, color='rgba(94, 45, 120, 0.7)'), showgrid=True),
        yaxis=dict(tickfont=dict(size=12, color='rgba(94, 45, 120, 0.7)'), showgrid=True),
        template='plotly_white',
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
    )
    
    # Add animation frames for smooth transitions
    frames = [dict(
        data=[dict(
            type='scatter',
            x=df_grouped['publication_date'][:i+1],
            y=df_grouped['article_count'][:i+1],
            mode='lines+markers',
            marker=dict(size=12, color='#98FB98', symbol='circle', line=dict(color='#FFB6C1', width=4))
        )],
        name=str(i)
    ) for i in range(len(df_grouped))]
    
    # Add animation buttons
    fig.update_layout(
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[dict(label='Play',
                        method='animate',
                        args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)])]
        )]
    )
    
    fig.frames = frames
    
    # Show the interactive plot
    st.plotly_chart(fig)


# Main function to run the Streamlit app
def main():
    st.set_page_config(
        page_title="Dashboard",
        page_icon="📜",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Data Visualization Application")

    st.sidebar.title("Navigation")
    data_type = st.sidebar.selectbox("Select data type:", ["Articles", "Authors", "Journals", "Collaborations"])

    authors_labs_pd, journaux_pd, articles_pd, exploded_df, new_journaux_df = fetch_and_process_data()

    if data_type == "Articles":
        #st.write(articles_pd.head())
        NumberArticlesperYear(articles_pd)
        process_keywords(articles_pd)   
    elif data_type == "Authors":
        st.write(authors_labs_pd.head())
    elif data_type == "Journals":
        st.write(journaux_pd.head())

        if 'Data' in journaux_pd.columns:
            col1, col2= st.columns(2)


            Distribution_Quartiles_per_Category(exploded_df)
            plot_quartile_distribution(exploded_df)

            with col1:
                #plot_country_quartile_distribution(new_journaux_df)
                word_cloud_distribution(exploded_df, 'Category')

            with col2:
                #word_cloud_distribution(exploded_df, 'Category')
                plot_country_quartile_distribution(new_journaux_df)
            plot_country_Contributions(new_journaux_df)

            
        else:
            st.write("No 'Data' column available for plotting.")
    elif data_type == "Collaborations":
        st.write("Collaborations data not available yet.")

if __name__ == "__main__":
    main()
