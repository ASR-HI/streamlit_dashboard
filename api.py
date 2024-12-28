import matplotlib
matplotlib.use('Agg')  # Utiliser le backend sans interface graphique
from wordcloud import WordCloud
from pyspark.sql.functions import udf, when, size, col, regexp_replace, concat_ws , regexp_extract, count, split, lower , trim, year, month, array_sort, desc, array, explode

from pyspark.sql.types import DateType
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import io
import os
import base64
from flask import Flask, jsonify
from pymongo import MongoClient
from pyspark.sql import SparkSession
import json
from pyspark.sql import SparkSession
import seaborn as sns
import networkx as nx
from dotenv import load_dotenv
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")

app = Flask(__name__)

# MongoDB client setup
client = MongoClient(mongo_uri)
db = client["scraping_data_final"]


# Set the path to your Python executable
os.environ['PYSPARK_PYTHON'] = 'C:\\Users\\pc\\AppData\\Local\\Programs\\Python\\Python311\\python.exe'

# Créer une session Spark
spark = SparkSession.builder \
    .appName("Data Cleaning") \
    .config("spark.worker.timeout", "600") \
    .config("spark.python.worker.reuse", "true") \
    .getOrCreate()

# Fonction pour exporter une collection MongoDB en fichier JSON
def export_collection_to_json(collection_name, file_name):
    collection = db[collection_name]
    documents = collection.find()
    docs = list(documents)
    
    with open(file_name, 'w') as file:
        json.dump([doc for doc in docs], file, default=str)

# Fonction pour charger un fichier JSON avec Spark
def load_json_to_spark(file_name):
    return spark.read.json(file_name)

# Fonction pour nettoyer les données
def clean_data():
    # Exporter les collections MongoDB en fichiers JSON
    export_collection_to_json("authors_labs", "authors_labs.json")
    export_collection_to_json("journaux", "journaux.json")
    export_collection_to_json("articles", "articles.json")

    # Lire les fichiers JSON avec Spark
    authors_labs_df = load_json_to_spark("authors_labs.json")
    journaux_df = load_json_to_spark("journaux.json")
    articles_df = load_json_to_spark("articles.json")

    # Supprimer la colonne '_id'
    authors_labs_df = authors_labs_df.drop("_id")
    journaux_df = journaux_df.drop("_id")
    articles_df = articles_df.drop("_id")

    # Compter les lignes avant suppression des doublons
    before_counts = {
        "Collection": ["Authors Labs", "Journaux", "Articles"],
        "Before Dropping Duplicates": [
            authors_labs_df.count(),
            journaux_df.count(),
            articles_df.count()
        ]
    }

    # Supprimer les doublons
    authors_labs_df = authors_labs_df.dropDuplicates()
    journaux_df = journaux_df.dropDuplicates()
    articles_df = articles_df.dropDuplicates()

    # Compter les lignes après suppression des doublons
    after_counts = {
        "Collection": ["Authors Labs", "Journaux", "Articles"],
        "After Dropping Duplicates": [
            authors_labs_df.count(),
            journaux_df.count(),
            articles_df.count()
        ]
    }

    # Convertir les comptages en DataFrame Pandas
    before_counts_df = pd.DataFrame(before_counts)
    after_counts_df = pd.DataFrame(after_counts)

    # Fusionner les comptages avant et après dans un seul DataFrame
    final_df = pd.merge(before_counts_df, after_counts_df, on="Collection")

    # Convertir le DataFrame final en liste de dictionnaires pour le JSON
    final_df_records = final_df.to_dict(orient="records")

    return final_df_records ,final_df, journaux_df , articles_df , authors_labs_df 

final_df = clean_data()[1]
journaux_df = clean_data()[2]
articles_df = clean_data()[3]
authors_labs_df = clean_data()[4]
date_pattern_1 = r"\d{1,2} \w+ \d{4}"  # e.g., 3 December 2024
date_pattern_2 = r"\d{2} \w+ \d{4}"  # e.g., 02 April 2024
date_pattern_3 = r"\d{1,2} \w+ \d{4}"  # e.g., 16 March 2025
date_pattern_4 = r"Volume \d+,\s?\w+ \d{4}"  # e.g., Volume 139, January 2025
date_pattern_5 = r"\d{4}"  # Matches a year like "2024"
df_with_dates = articles_df.withColumn(
    "extracted_date",
    when(col("publication_date").rlike(date_pattern_1), regexp_extract(col("publication_date"), date_pattern_1, 0))
    .when(col("publication_date").rlike(date_pattern_2), regexp_extract(col("publication_date"), date_pattern_2, 0))
    .when(col("publication_date").rlike(date_pattern_3), regexp_extract(col("publication_date"), date_pattern_3, 0))
    .when(col("publication_date").rlike(date_pattern_4), regexp_extract(col("publication_date"), r"\w+ \d{4}", 0))
    .when(col("publication_date").rlike(date_pattern_5), regexp_extract(col("publication_date"), date_pattern_5, 0))
    .otherwise(None)  # If no match is found
)
# Define the parse_date function to ensure None is handled
def parse_date(date_str):
    if not date_str:  # If None or empty string
        return None

    # Try multiple formats
    for fmt in ('%d %B %Y', '%B %Y', '%Y'):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None  # Return None if no format matches

# Register the UDF
parse_date_udf = udf(parse_date, DateType())

# Apply the UDF to clean and create the 'publication_date'
df_with_dates = df_with_dates.withColumn('publication_date', parse_date_udf(df_with_dates['extracted_date']))

# Split keywords by comma, trim spaces, convert to lowercase, and explode into individual keywords
df_keywords = (
    df_with_dates.withColumn(
        "keyword",
        explode(split(col('keywords'), ','))
    ).withColumn(
        "keyword", trim(lower(col("keyword")))  # Trim spaces and convert to lowercase
    )
)

# Remove duplicates in the exploded keywords (across all rows)
df_keywords_unique = df_keywords.distinct()

# Count the frequency of each unique keyword
df_keywords_grouped = (
    df_keywords_unique.groupBy('keyword')
    .agg(count('*').alias('keyword_count'))
    .orderBy('keyword_count', ascending=False)
)

articles = df_with_dates

# 1. Exploser les auteurs et leurs laboratoires dans articles_df
exploded_authors_df = articles.withColumn(
    "author", explode("authors_data")
).select(
    "article_title",
    "doi",
    "author.name",
    "author.labs"
)

# 2. Exploser la liste des laboratoires (car un auteur peut avoir plusieurs laboratoires)
exploded_labs_df = exploded_authors_df.withColumn(
    "lab", explode("labs")
).select(
    "article_title",
    "doi",
    "name",
    "lab"
)

# 3. Renommer la colonne 'name' de authors_labs_df pour éviter l'ambiguïté
authors_labs_df_renamed = authors_labs_df.withColumnRenamed("name", "lab_name")

# 4. Joindre avec authors_labs_df pour obtenir les pays associés aux laboratoires
authors_with_country_df = exploded_labs_df.join(
    authors_labs_df_renamed,
    exploded_labs_df["lab"] == authors_labs_df_renamed["lab"],
    "left"
).select(
    "article_title",
    "doi",
    "name",  # Auteur
    "lab_name",  # Nouveau nom de la colonne 'name'
    "Country"
)

# 5. Créer des paires de pays collaborant ensemble dans un même article, sans double comptage
collaboration_df = authors_with_country_df.alias("a").join(
    authors_with_country_df.alias("b"),
    (col("a.doi") == col("b.doi")) & (col("a.Country") != col("b.Country")),
    "inner"
).select(
    array_sort(array("a.Country", "b.Country")).alias("Country"),
    "a.doi"  # Ajouter 'doi' pour garantir que la collaboration est pour le même article
)

# 6. Supprimer les doublons de pays pour chaque article
distinct_collaboration_df = collaboration_df.distinct()

# 7. Grouper par paire de pays et compter le nombre de collaborations distinctes
country_collaboration_df = distinct_collaboration_df.groupBy("Country").agg(
    count("doi").alias("collaboration_count")
).orderBy(desc("collaboration_count"))

# Afficher les résultats
# country_collaboration_df.show(truncate=False)








print("-------------------------traitement des données terminé-------------------------")

# Fonction pour générer le graphique
def generate_plot(final_df):
    plt.figure(figsize=(10, 6))

    # Tracer les barres pour "Before" et "After" counts
    bar_width = 0.35
    index = range(len(final_df))

    plt.bar(index, [x['Before Dropping Duplicates'] for x in final_df], bar_width, label='Before')
    plt.bar([i + bar_width for i in index], [x['After Dropping Duplicates'] for x in final_df], bar_width, label='After')

    # Personnaliser le graphique
    plt.xlabel('Collection')
    plt.ylabel('Row Count')
    plt.xticks([i + bar_width / 2 for i in index], [x['Collection'] for x in final_df], rotation=45)
    plt.legend()

    # Enregistrer le graphique dans un buffer
    img = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)

    # Convertir l'image en base64 pour l'envoyer dans la réponse
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    return img_base64

def traiter_journaux(journaux_df):
   # Initialize Spark session
    spark = SparkSession.builder.appName("JsonSchemaChange").getOrCreate()
    # Flatten the 'Data' array
    flattened_journaux_df = journaux_df.select(
        "Journal Name",
        "ISSN",
        "Country",
        explode(col("Data")).alias("data")
    )

    # Select the nested fields inside 'data'
    final_flattened_journaux_df = flattened_journaux_df.select(
        "Journal Name",
        "ISSN",
        "Country",
        col("data.Category"),
        col("data.Year"),
        col("data.Quartile")
    )

    return final_flattened_journaux_df

# Votre fonction pour générer le graphique
def generate_plot_Q_par_Catego():
    # Vous devez d'abord traiter `journaux_df` et le convertir en DataFrame pandas
    pandas_journaux_df = traiter_journaux(journaux_df).toPandas()  # Assurez-vous que cette fonction existe et que `journaux_df` est bien défini
    plt.figure(figsize=(30,20))  # Augmenter la hauteur de la figure pour mieux voir les catégories
    sns.countplot(data=pandas_journaux_df, x='Category', hue='Quartile', palette='Set1')
    plt.xticks(rotation=90, ha='right')  # Rotation des labels et alignement à droite
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.tight_layout()  # Assurer que tout s'affiche correctement
    # Enregistrer le graphique dans un buffer
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    # Convertir l'image en base64
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return img_base64

def generate_heatmap_Q_par_Catego_Year():
    # Convertir journaux_df en DataFrame pandas
    pandas_journaux_df = traiter_journaux(journaux_df).toPandas()

    # Create pivot table
    heatmap_data = pandas_journaux_df.pivot_table(
        index='Category',
        columns='Year',
        values='Quartile',
        aggfunc=lambda x: ' '.join(x)  # Join multiple Quartile values if present
    )

    # Replace NaN with empty strings and map quartiles
    def map_quartile(cell):
        if 'Q1' in cell:
            return 1
        elif 'Q2' in cell:
            return 2
        elif 'Q3' in cell:
            return 3
        elif 'Q4' in cell:
            return 4
        else:
            return 0  # Use 0 or leave empty for cells with no quartile

    heatmap_data = heatmap_data.fillna('').applymap(map_quartile)

    # Plot the heatmap using Plotly
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Year", y="Category", color="Quartile"),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale='YlGnBu'
    )

    # Customize the layout
    fig.update_layout(
        xaxis=dict(
            title="Year",
            tickangle=45,  # Rotate x-axis labels for better readability
            tickfont=dict(size=10)  # Adjust font size
        ),
        yaxis=dict(
            title="Category",
            tickfont=dict(size=10)  # Adjust font size
        ),
        width=1400,  # Increase figure width
        height=800,  # Adjust height for better proportions
        margin=dict(l=300, r=50, t=50, b=150),  # Adjust margins
    )

    return fig

def generate_plot_Q_par_Year_TS():
    # Group data by Year and Quartile and calculate counts
    pandas_journaux_df = traiter_journaux(journaux_df).toPandas()  # Assurez-vous que cette fonction existe et que `journaux_df` est bien défini
    quartile_counts = pandas_journaux_df.groupby(['Year', 'Quartile']).size().reset_index(name='Count')

    # Create a line plot to show the distribution of Quartiles over the years
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=quartile_counts, x='Year', y='Count', hue='Quartile', marker='o', palette='Set2')

    # Add plot title and labels
    plt.xlabel('Year')
    plt.ylabel('Count of Publications')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
    plt.grid(axis='y')  # Add horizontal grid lines for better readability
    plt.legend(title='Quartile')

    img = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    # Convertir l'image en base64
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return img_base64

def generate_wordcloud_categories_journaux():
    # Group data by Year and Quartile and calculate counts
    pandas_journaux_df = traiter_journaux(journaux_df).toPandas()  # Assurez-vous que cette fonction existe et que `journaux_df` est bien défini
    # Join all categories into a single string
    categories_text = ' '.join(pandas_journaux_df['Category'].dropna())

    # Create the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(categories_text)

    # Plot the word cloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Turn off axis

    img = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    # Convertir l'image en base64
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return img_base64



def generate_plot_Q_par_Year():
    pandas_journaux_df = traiter_journaux(journaux_df).toPandas()  # Assurez-vous que cette fonction existe et que `journaux_df` est bien défini
    plt.figure(figsize=(12, 6))
    sns.countplot(data=pandas_journaux_df, x='Year', hue='Quartile', palette='Set2')

    # Add plot title and labels
    plt.xlabel('Year')
    plt.ylabel('Count of Publications')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
    img = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    # Convertir l'image en base64
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return img_base64
def generate_plot_Q_par_Year_TS2():
    pandas_journaux_df = traiter_journaux(journaux_df).toPandas()  # Assurez-vous que cette fonction existe et que `journaux_df` est bien défini
    # Group data by Year and Quartile and calculate counts
    quartile_counts = pandas_journaux_df.groupby(['Year', 'Quartile']).size().reset_index(name='Count')

    # Set theme for improved aesthetics
    sns.set_theme(style="whitegrid", context="talk")

    # Create the time series plot
    plt.figure(figsize=(12,6))
    sns.lineplot(
        data=quartile_counts,
        x='Year',
        y='Count',
        hue='Quartile',
        marker='o',
        palette="husl",
        linewidth=2
    )

    # Add plot enhancements
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Count of Publications', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Quartile', fontsize=12, title_fontsize=14, loc='upper left')
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    # Annotate data points
    for _, row in quartile_counts.iterrows():
        plt.text(
            x=row['Year'],
            y=row['Count'] + 0.5,  # Offset to avoid overlap
            s=f"{row['Count']}",
            fontsize=10,
            color="black",
            ha='center'
        )

    # Display the plot
    plt.tight_layout()
    img = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    # Convertir l'image en base64
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return img_base64
def generate_Q_distribution():
    pandas_journaux_df = traiter_journaux(journaux_df).toPandas()  # Assurez-vous que cette fonction existe et que `journaux_df` est bien défini
    # Group by 'Country' and 'Quartile' to count the number of occurrences of each quartile per country
    quartile_distribution = pandas_journaux_df.groupby(['Country', 'Quartile']).size().reset_index(name='Count')
    quartile_distribution.groupby('Quartile').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
    plt.gca().spines[['top', 'right',]].set_visible(False)
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    # Convertir l'image en base64
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return img_base64

def generate_contributions_by_Q():
    pandas_journaux_df = traiter_journaux(journaux_df).toPandas()  # Assurez-vous que cette fonction existe et que `journaux_df` est bien défini
    new_data = {}
    # Iterate over unique countries
    for country in pandas_journaux_df['Country'].unique():
        # Initialize counts for each quartile to zero
        new_data[country] = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}
        # Iterate over the dataframe rows for the current country
        for index, row in pandas_journaux_df[pandas_journaux_df['Country'] == country].iterrows():
            # Increment the count for the corresponding quartile
            new_data[country][row['Quartile']] += 1

    new_journaux_df = pd.DataFrame.from_dict(new_data, orient='index')

    # Add 'Country' column as part of the DataFrame
    new_journaux_df['Country'] = new_journaux_df.index

    # Reorder columns to have 'Country' first
    new_journaux_df = new_journaux_df[['Country', 'Q1', 'Q2', 'Q3', 'Q4']]
    sns.set_theme(style="whitegrid")
    df_long = new_journaux_df.melt(id_vars='Country', var_name='Quarter', value_name='Count')
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_long, x='Count', hue='Quarter', kde=True, bins=20, palette='Set2', edgecolor='black', multiple="stack")
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(title='Quarter', fontsize=10, title_fontsize=12, loc='upper right')

    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    # Convertir l'image en base64
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return img_base64
def generate_contributions_by_Q2():
    pandas_journaux_df = traiter_journaux(journaux_df).toPandas()  # Assurez-vous que cette fonction existe et que `journaux_df` est bien défini
    new_data = {}
    # Iterate over unique countries
    for country in pandas_journaux_df['Country'].unique():
        # Initialize counts for each quartile to zero
        new_data[country] = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}
        # Iterate over the dataframe rows for the current country
        for index, row in pandas_journaux_df[pandas_journaux_df['Country'] == country].iterrows():
            # Increment the count for the corresponding quartile
            new_data[country][row['Quartile']] += 1

    new_journaux_df = pd.DataFrame.from_dict(new_data, orient='index')

    # Add 'Country' column as part of the DataFrame
    new_journaux_df['Country'] = new_journaux_df.index

    # Reorder columns to have 'Country' first
    new_journaux_df = new_journaux_df[['Country', 'Q1', 'Q2', 'Q3', 'Q4']]
    sns.set_theme(style="whitegrid")

    # Convert the DataFrame to long format for the histogram
    df_long = new_journaux_df.melt(id_vars='Country', var_name='Quarter', value_name='Count')

    # Create the plot
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(
        data=df_long,
        x='Count',
        hue='Quarter',
        kde=True,
        bins=20,
        palette='Set2',
        edgecolor='black',
        multiple="stack"
    )

    # Add labels to the bars
    for p in ax.patches:
        if p.get_height() > 0:  # Only label bars with a positive height
            x = p.get_x() + p.get_width() / 2  # Center the label horizontally
            y = p.get_height()  # Get the height of the bar
            ax.annotate(
                int(y),  # Convert count to integer
                (x, y),  # Position the label at the top of the bar
                ha='center',  # Align the label horizontally at the center
                va='bottom',  # Align the label vertically above the bar
                fontsize=10,  # Font size for the label
                color='black'  # Color of the label
            )

    # Force the legend to show correct mapping
    handles, labels = ax.get_legend_handles_labels()
    unique_quarters = df_long['Quarter'].unique()  # Ensure correct ordering of quarters
    ax.legend(
        title='Quarter',
        labels=unique_quarters,  # Use unique quarter names directly
        handles=handles,
        fontsize=10,
        title_fontsize=12,
        loc='upper right'
    )

    # Add titles and labels
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    # Adjust layout
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    # Convertir l'image en base64
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return img_base64

def generate_contributions_by_Q_by_Country():
    pandas_journaux_df = traiter_journaux(journaux_df).toPandas()  # Assurez-vous que cette fonction existe et que `journaux_df` est bien défini
    new_data = {}
    # Iterate over unique countries
    for country in pandas_journaux_df['Country'].unique():
        # Initialize counts for each quartile to zero
        new_data[country] = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}
        # Iterate over the dataframe rows for the current country
        for index, row in pandas_journaux_df[pandas_journaux_df['Country'] == country].iterrows():
            # Increment the count for the corresponding quartile
            new_data[country][row['Quartile']] += 1

    new_journaux_df = pd.DataFrame.from_dict(new_data, orient='index')

    # Add 'Country' column as part of the DataFrame
    new_journaux_df['Country'] = new_journaux_df.index

    # Reorder columns to have 'Country' first
    new_journaux_df = new_journaux_df[['Country', 'Q1', 'Q2', 'Q3', 'Q4']]
    sns.set_theme(style="whitegrid")
    df_long = new_journaux_df.melt(id_vars='Country', var_name='Quarter', value_name='Count')
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=df_long,
        x='Country',
        y='Count',
        hue='Quarter',
        palette='Set2',
        edgecolor='black'
    )
    plt.xlabel('Country', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Quarter', fontsize=10, title_fontsize=12, loc='upper right')
    plt.xticks(rotation=45, fontsize=10)

    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    # Convertir l'image en base64
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return img_base64


def generate_country_contribution_by_Q():
    pandas_journaux_df = traiter_journaux(journaux_df).toPandas()  # Assurez-vous que cette fonction existe et que `journaux_df` est bien défini
    new_data = {}
    # Iterate over unique countries
    for country in pandas_journaux_df['Country'].unique():
        # Initialize counts for each quartile to zero
        new_data[country] = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}
        # Iterate over the dataframe rows for the current country
        for index, row in pandas_journaux_df[pandas_journaux_df['Country'] == country].iterrows():
            # Increment the count for the corresponding quartile
            new_data[country][row['Quartile']] += 1

    new_journaux_df = pd.DataFrame.from_dict(new_data, orient='index')

    # Add 'Country' column as part of the DataFrame
    new_journaux_df['Country'] = new_journaux_df.index

    # Reorder columns to have 'Country' first
    new_journaux_df = new_journaux_df[['Country', 'Q1', 'Q2', 'Q3', 'Q4']]
    sns.set_theme(style="whitegrid")

    df_long = new_journaux_df.melt(id_vars='Country', var_name='Quarter', value_name='Count')
    countries = df_long['Country'].unique()
    n_columns =  2
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
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    # Convertir l'image en base64
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return img_base64

def generate_bubble_chart_Q_par_Country():
    pandas_journaux_df = traiter_journaux(journaux_df).toPandas()  # Assurez-vous que cette fonction existe et que `journaux_df` est bien défini
    # Group by 'Country' and 'Quartile' to count the number of occurrences of each quartile per country
    quartile_distribution = pandas_journaux_df.groupby(['Country', 'Quartile']).size().reset_index(name='Count')
    # Create a pivot table to aggregate the quartile distribution per country
    # Pivot the data so that each quartile has its own column
    pivot_df = quartile_distribution.pivot_table(index='Country', columns='Quartile', values='Count', aggfunc='sum', fill_value=0)

    # Normalize the data to get the relative distribution of quartiles per country
    pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0)  # Normalize by row (country)
    melted_df = pivot_df.reset_index().melt(id_vars="Country", var_name="Quartile", value_name="Count")
    # Filter out rows where Quartile is "Total"
    melted_df = melted_df[melted_df['Quartile'] != 'Total']

    # Plot a bubble chart
    fig = px.scatter(
        melted_df,
        x="Country",
        y="Quartile",
        size="Count",
        color="Quartile",
        hover_name="Country",
        hover_data={"Count": ":.1%"},
        color_continuous_scale="Viridis",
        size_max=75
    )

    # Update layout: center the title and add axis labels
    fig.update_layout(

        xaxis_title="Country",
        yaxis_title="Quartile"
    )
    return fig
    
def generate_map_Q_par_Country():
    pandas_journaux_df = traiter_journaux(journaux_df).toPandas()  # Assurez-vous que cette fonction existe et que `journaux_df` est bien défini
    new_data = {}
    # Iterate over unique countries
    for country in pandas_journaux_df['Country'].unique():
        # Initialize counts for each quartile to zero
        new_data[country] = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}
        # Iterate over the dataframe rows for the current country
        for index, row in pandas_journaux_df[pandas_journaux_df['Country'] == country].iterrows():
            # Increment the count for the corresponding quartile
            new_data[country][row['Quartile']] += 1

    new_journaux_df = pd.DataFrame.from_dict(new_data, orient='index')

    # Add 'Country' column as part of the DataFrame
    new_journaux_df['Country'] = new_journaux_df.index

    # Reorder columns to have 'Country' first
    new_journaux_df = new_journaux_df[['Country', 'Q1', 'Q2', 'Q3', 'Q4']]
    # Quartiles et leurs noms respectifs
    quartiles = ['Q1', 'Q2', 'Q3', 'Q4']

    figures = []
    # Créer un graphique pour chaque quartile
    for quartile in quartiles:
        fig = go.Figure()

        # Ajouter une carte choroplèthe
        fig.add_trace(
            go.Choropleth(
                locations=new_journaux_df['Country'],  # Noms des pays
                locationmode='country names',  # Utiliser les noms des pays
                z=new_journaux_df[quartile],  # Valeurs associées au quartile
                hoverinfo='location+z',  # Afficher le pays et la valeur au survol
                colorscale='Viridis',  # Palette de couleurs
                colorbar=dict(
                    title='',  # Pas de titre pour la barre colorée
                    titleside='right',  # Placer le titre sur le côté droit
                    orientation='v',  # Orientation verticale
                    x=1.05,  # Position horizontale (légèrement à droite du graphique)
                    y=0.5,  # Centrer la barre verticalement
                    len=0.75,  # Longueur de la barre colorée
                    thickness=15,  # Épaisseur de la barre colorée
                ),
            )
        )

        # Mettre à jour la disposition pour chaque graphique
        fig.update_layout(
            title_text=f"Distribution of {quartile} by Country",
            title_x=0.5,  # Centrer le titre
            height=600,  # Hauteur de la figure
            width=800,  # Largeur de la figure
            geo=dict(
                showcoastlines=True,
                coastlinecolor='Black',
                projection_type='natural earth'  # Type de projection pour les cartes
            )
        )
    
        figures.append(fig)
    
    return figures

def genrate_plot_authors_par_pays():
    authors_labs_pd = authors_labs_df.toPandas()
    country_counts = authors_labs_pd['country'].value_counts().head(20)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=country_counts.values, y=country_counts.index, palette='viridis')
    plt.xlabel("Nombre d'auteurs")  
    plt.ylabel("Pays")
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return img_base64

def genrate_plot_top_20_labs(): 
    authors_labs_pd = authors_labs_df.toPandas()
    # 2. Top 20 laboratoires les plus représentés
    top_labs = authors_labs_pd['lab'].value_counts().head(20)

    plt.figure(figsize=(12, 8))  # Increase the figure size
    sns.barplot(x=top_labs.values, y=top_labs.index, palette='coolwarm')
    plt.xlabel("Nombre d'auteurs")
    plt.ylabel("Laboratoire")
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return img_base64

def genrate_plot_top_20_labs2(): 
    authors_labs_pd = authors_labs_df.toPandas()
    top_labs_df = authors_labs_pd['lab'].value_counts().head(20).reset_index()
    top_labs_df.columns = ['lab', 'count']

    fig = px.bar(
        top_labs_df,
        x='count',
        y='lab',
        orientation='h',
        labels={'count': 'Nombre d’auteurs', 'lab': 'Laboratoire'},
        color='count',
        color_continuous_scale='Aggrnyl'
    )
    return fig


def generate_map_authors():
    authors_labs_pd = authors_labs_df.toPandas()
    country_counts = authors_labs_pd['country'].value_counts().head(20)
    country_counts_df = country_counts.reset_index()
    country_counts_df.columns = ['country', 'count']

    fig = px.choropleth(country_counts_df,
                        locations="country",
                        locationmode="country names",
                        color="count")
    return fig




def generate_plot_number_of_articles_par_year():
    # Extract the year from the extracted_date column
    df_with_year = df_with_dates.withColumn(
        "year",
        regexp_extract(col("extracted_date"), r"\d{4}", 0)
    )

    # Group by year and count the number of articles
    articles_by_year = df_with_year.groupBy("year").agg(count("*").alias("article_count"))

    # Filter out rows without valid years (if needed)
    articles_by_year = articles_by_year.filter(col("year") != "")
    # Convert the Spark DataFrame to a Pandas DataFrame
    articles_by_year_pd = articles_by_year.toPandas()

    # Ensure the year is in datetime format and sort by year
    articles_by_year_pd['year'] = pd.to_datetime(articles_by_year_pd['year'], format='%Y')

    # Sort the DataFrame by the year column
    articles_by_year_pd = articles_by_year_pd.sort_values('year')

    # Plotting the time series
    plt.figure(figsize=(10, 6))
    plt.plot(articles_by_year_pd['year'], articles_by_year_pd['article_count'], marker='o', linestyle='-', color='b')
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Article Count", fontsize=12)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return img_base64



def generate_plot_number_of_articles_par_date():
    # Group by 'publication_date' and count the number of articles
    df_grouped = df_with_dates.groupBy('publication_date').agg(count('*').alias('article_count'))

    # Sort by date
    df_grouped = df_grouped.orderBy('publication_date')

    # Convert the result to a Pandas DataFrame
    df_pandas = df_grouped.toPandas()

    # Plot the time series
    plt.figure(figsize=(10, 6))
    plt.plot(df_pandas['publication_date'], df_pandas['article_count'], marker='o', linestyle='-', color='b')

    # Add titles and labels
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45)  # Rotate date labels for better readability
    plt.grid(True)
    # Show the plot
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return img_base64

def generate_plot_number_of_articles_par_date2():
    # Group by 'publication_date' and count the number of articles
    df_grouped = df_with_dates.groupBy('publication_date').agg(count('*').alias('article_count'))

    # Sort by date
    df_grouped = df_grouped.orderBy('publication_date')

    # Convert the result to a Pandas DataFrame
    df_pandas = df_grouped.toPandas()

    # Create an interactive time series plot using Plotly
    fig = px.line(df_pandas,
                x='publication_date',
                y='article_count',
                labels={'publication_date': 'Date', 'article_count': 'Number of Articles'},
                markers=True)

    # Customize the plot for aesthetics
    fig.update_traces(line=dict(color='blue', width=2),  # Line color and width
                    marker=dict(size=6, color='red', symbol='circle', line=dict(color='black', width=2)))  # Markers
    fig.update_layout(
        xaxis_title_font_size=14,  # X-axis label font size
        yaxis_title_font_size=14,  # Y-axis label font size
        xaxis=dict(tickangle=45),  # Rotate x-axis labels for better readability
        template='plotly_dark',  # Dark theme
        hovermode='x unified'  # Hover over the x-axis to show all data at that point
    )
    return fig


def generate_plot_number_of_articles_par_date3():
    # Group by 'publication_date' and count the number of articles
    df_grouped = df_with_dates.groupBy('publication_date').agg(count('*').alias('article_count'))

    # Sort by date
    df_grouped = df_grouped.orderBy('publication_date')

    # Convert the result to a Pandas DataFrame
    df_pandas = df_grouped.toPandas()

    # Create an interactive time series plot using Plotly with cute pastel colors
    fig = px.line(df_pandas,
                x='publication_date',
                y='article_count',
                labels={'publication_date': 'Date', 'article_count': 'Number of Articles'},
                markers=True)

    # Customize the plot for aesthetics
    fig.update_traces(line=dict(color='#FF6F61', width=3),  # Soft Coral color for line
                    marker=dict(size=8, color='#FFB6C1', symbol='circle', line=dict(color='#FF6F61', width=2)))  # Pastel pink markers with soft coral border

    # Update layout with white background and cute pastel theme
    fig.update_layout(
        
        xaxis_title_font_size=14,  # X-axis label font size
        yaxis_title_font_size=14,  # Y-axis label font size
        xaxis=dict(tickangle=45),  # Rotate x-axis labels for better readability
        template='plotly_white',  # White background
        hovermode='x unified',  # Hover over the x-axis to show all data at that point
        plot_bgcolor='white',  # Set the plot background to white
        paper_bgcolor='white',  # Set the entire paper background to white
    )
    return fig

def generate_plot_number_of_articles_par_date4():
    # Group by 'publication_date' and count the number of articles
    df_grouped = df_with_dates.groupBy('publication_date').agg(count('*').alias('article_count'))

    # Sort by date
    df_grouped = df_grouped.orderBy('publication_date')

    # Convert the result to a Pandas DataFrame
    df_pandas = df_grouped.toPandas()

    # Create an interactive time series plot with soft pastel tones and rounded markers
    fig = px.line(df_pandas,
                x='publication_date',
                y='article_count',
                labels={'publication_date': 'Date', 'article_count': 'Number of Articles'},
                markers=True)

    # Customize the plot to be more cute and playful
    fig.update_traces(
        line=dict(color='#FFB6C1', width=6),  # Soft pastel pink line
        marker=dict(size=12, color='#98FB98', symbol='circle', line=dict(color='#FFB6C1', width=4))  # Light green markers with a pink border
    )

    # Create frames for animation (each frame represents a new date)
    frames = [dict(
        data=[dict(
            type='scatter',
            x=df_pandas['publication_date'][:i+1],  # Slice the data up to the i-th date
            y=df_pandas['article_count'][:i+1],  # Slice the article count
            mode='lines+markers',
            marker=dict(size=12, color='#98FB98', symbol='circle', line=dict(color='#FFB6C1', width=4))
        )],
        name=str(i)
    ) for i in range(len(df_pandas))]

    # Update layout with a light, pastel-colored background, and playful font styles
    fig.update_layout(
        xaxis_title_font_size=18,  # X-axis label font size
        yaxis_title_font_size=18,  # Y-axis label font size
        xaxis_title_font_color='rgba(255, 182, 193, 1)',  # Pastel pink axis title color
        yaxis_title_font_color='rgba(255, 182, 193, 1)',  # Pastel pink axis title color
        xaxis=dict(tickangle=45, tickfont=dict(size=14, color='rgba(255, 182, 193, 0.7)'), showgrid=True),  # Soft pink tick labels with grid
        yaxis=dict(tickfont=dict(size=14, color='rgba(255, 182, 193, 0.7)'), showgrid=True),  # Y-axis font color and grid
        template='plotly_white',  # Light background with soft accents
        hovermode='x unified',  # Hover over the x-axis to show all data at that point
        plot_bgcolor='rgba(255, 250, 250, 1)',  # Soft white background
        paper_bgcolor='rgba(255, 250, 250, 1)',  # Same soft white for the paper background
        showlegend=False,  # No legend needed
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[dict(label='Play',
                        method='animate',
                        args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)])]
        )]
    )

    # Add frames to the layout
    fig.frames = frames
    return fig




def generate_top_10_keywords():
    # Convert to Pandas DataFrame
    df_keywords_pandas = df_keywords_grouped.toPandas()

    # Create a bar plot for the most common keywords
    fig = px.bar(
        df_keywords_pandas.head(10),
        x='keyword',
        y='keyword_count',
        labels={'keyword': 'Keyword', 'keyword_count': 'Count'}
    )

    # Customize plot appearance
    fig.update_traces(marker=dict(color='#8da0cb'))
    return fig

def generate_wordcloud_keywords():
    df_keywords_pandas = df_keywords_grouped.toPandas()
    # Collect all keywords and combine them into a single string
    all_keywords = " ".join(df_keywords_pandas['keyword'].tolist())

    # Create a word cloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_keywords)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return img_base64


def generate_top_10_authors_with_most_articles():
    # Extract author names from authors_data
    df_authors = df_with_dates.withColumn("author", explode(col("authors_data"))).select("author.name").distinct()

    # Count the number of articles per author
    df_authors_count = df_with_dates.withColumn("author", explode(col("authors_data"))).groupBy("author.name").agg(count('*').alias('article_count')).orderBy('article_count', ascending=False)

    # Convert to Pandas DataFrame
    df_authors_pandas = df_authors_count.toPandas()

    # Create a bar plot for the number of articles per author
    fig = px.bar(df_authors_pandas.head(10),
                x='name',
                y='article_count',
                labels={'name': 'Author', 'article_count': 'Article Count'})

    # Customize plot appearance
    fig.update_traces(marker=dict(color='#a6d854'))
    return fig

def generate_articles_by_journal():
    # Group by journal name and count the number of articles per journal
    df_journals = df_with_dates.groupBy('journal_name').agg(count('*').alias('article_count'))

    # Convert to Pandas DataFrame
    df_journals_pandas = df_journals.toPandas()

    # Create a pie chart for the distribution of articles by journal name
    fig = px.pie(df_journals_pandas,
                names='journal_name',
                values='article_count')
    # Customize plot appearance
    fig.update_traces(marker=dict(colors=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']))
    return fig

def generate_3d_articles_by_journal():
    # Group by journal name and count the number of articles per journal
    df_journals = df_with_dates.groupBy('journal_name').agg(count('*').alias('article_count'))

    # Convert to Pandas DataFrame
    df_journals_pandas = df_journals.toPandas()

    # Create a 3D scatter plot with bubbles for the distribution of articles by journal name
    fig = go.Figure(data=[go.Scatter3d(
        x=df_journals_pandas['journal_name'],  # Journal names on x-axis
        y=df_journals_pandas['article_count'],  # Article count on y-axis
        z=[0] * len(df_journals_pandas),  # Set z to 0 to place all bubbles on the same plane
        mode='markers',
        marker=dict(
            size=df_journals_pandas['article_count'],  # Size of the bubbles based on article count
            color=df_journals_pandas['article_count'],  # Color of the bubbles based on article count
            colorscale='Blues',  # Colorscale for better distinction
            opacity=0.6,
            line=dict(width=0.5, color='black')  # Outline for bubbles
        ),
        text=df_journals_pandas['journal_name'],  # Display journal name when hovering
        hoverinfo='x+y+text',  # Show journal name, article count, and bubble size on hover
    )])

    # Update layout for 3D appearance
    fig.update_layout(
        scene=dict(
            xaxis_title='Journal Name',
            yaxis_title='Article Count',
            zaxis_title='',  # No z-axis label
            camera=dict(
                eye=dict(x=1.25, y=1.25, z=0.75)  # Adjust the camera position for better 3D view
            )
        ),
        font=dict(size=14),
    )
    return fig



def heatmap_articles_by_year_and_month():
    # Extract year and month from publication_date
    df_heatmap = df_with_dates.withColumn('year', year(col('publication_date'))).withColumn('month', month(col('publication_date')))

    # Group by year and month, and count the number of articles
    df_heatmap_grouped = df_heatmap.groupBy('year', 'month').agg(count('*').alias('article_count'))

    # Convert to Pandas DataFrame for visualization
    df_heatmap_pandas = df_heatmap_grouped.toPandas()

    # Pivot the data for the heatmap
    df_heatmap_pivot = df_heatmap_pandas.pivot(index='year', columns='month', values='article_count').fillna(0)

    # Convert the values to integers
    df_heatmap_pivot = df_heatmap_pivot.astype(int)

    # Create a heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_heatmap_pivot, annot=True, cmap="YlGnBu", fmt="d", linewidths=.5)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Year', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return img_base64

def generate_articles_by_country():
    articles_pd = df_with_dates.toPandas()
    journaux_pd = journaux_df.toPandas()
    # Fusionner les DataFrames sur la colonne 'journal_name'
    merged_df = pd.merge(articles_pd, journaux_pd, left_on='journal_name', right_on='Journal Name', how='inner')

    # Vérifier les premières lignes pour s'assurer que la fusion a bien fonctionné
    # print(merged_df.head())

    # Compter le nombre d'articles par pays
    articles_by_country = merged_df.groupby('Country')['doi'].count().reset_index()

    # Renommer la colonne 'doi' en 'article_count' pour plus de clarté
    articles_by_country.rename(columns={'doi': 'article_count'}, inplace=True)

    # Trier les résultats par nombre d'articles
    articles_by_country.sort_values(by='article_count', ascending=False, inplace=True)

    # Visualiser les résultats
    plt.figure(figsize=(12, 8))
    plt.bar(articles_by_country['Country'], articles_by_country['article_count'], color='skyblue')
    plt.xlabel('Country')
    plt.ylabel('Number of Articles')

    # Rotation des labels de l'axe x pour éviter les chevauchements
    plt.xticks(rotation=45, ha='right')

    # Ajuster l'espace pour que tout s'affiche correctement
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return img_base64


def generate_map_articles_by_country():
    articles_pd = df_with_dates.toPandas()
    journaux_pd = journaux_df.toPandas()
    # Fusionner les DataFrames sur la colonne 'journal_name'
    merged_df = pd.merge(articles_pd, journaux_pd, left_on='journal_name', right_on='Journal Name', how='inner')

    # Vérifier les premières lignes pour s'assurer que la fusion a bien fonctionné
    # print(merged_df.head())

    # Compter le nombre d'articles par pays
    articles_by_country = merged_df.groupby('Country')['doi'].count().reset_index()

    # Renommer la colonne 'doi' en 'article_count' pour plus de clarté
    articles_by_country.rename(columns={'doi': 'article_count'}, inplace=True)

    # Trier les résultats par nombre d'articles
    articles_by_country.sort_values(by='article_count', ascending=False, inplace=True)
        # Créer la visualisation choroplèthe avec plotly et utiliser une palette de couleurs personnalisée
    fig = go.Figure(go.Choropleth(
        locations=articles_by_country['Country'],   # Noms des pays
        z=articles_by_country['article_count'],     # Nombre d'articles
        locationmode='country names',               # Mode pour faire correspondre les noms des pays
        text=articles_by_country['article_count'],  # Afficher le nombre d'articles au survol
        colorscale='Viridis',                       # Couleurs variées, mais vous pouvez utiliser d'autres échelles
        colorbar_title='Number of Articles',        # Titre de la légende
    ))

    # Ajouter un titre et des informations de mise en page
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular',     # Projection de la carte
        )
    )
    return fig


def generate_tableau_journaux():
    final_flattened_journaux_df = traiter_journaux(journaux_df).toPandas()
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(final_flattened_journaux_df.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[final_flattened_journaux_df[col] for col in final_flattened_journaux_df.columns],
                   fill_color='lavender',
                   align='left'))
    ])

    return fig




def generate_graph_collaborations():
    # Convertir les résultats en Pandas DataFrame
    collaboration_pd = country_collaboration_df.toPandas()

    # Créer un graphe vide
    G = nx.Graph()

    # Ajouter des arêtes (collaborations) dans le graphe
    for _, row in collaboration_pd.iterrows():
        countries = row['Country']
        collaboration_count = row['collaboration_count']

        # Ajouter une arête entre les pays, avec le poids de l'arête (nombre de collaborations)
        G.add_edge(countries[0], countries[1], weight=collaboration_count)

    # Obtenir les positions des nœuds
    pos = nx.spring_layout(G, k=0.5, seed=42, iterations=50)

    # Extraire les données pour Plotly
    edge_x = []
    edge_y = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2),
        text=[node for node in G.nodes()])

    # Ajouter les poids des arêtes comme annotations
    annotations = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        annotations.append(
            dict(
                x=(x0 + x1) / 2,
                y=(y0 + y1) / 2,
                text=str(edge[2]['weight']),
                showarrow=False,
                font=dict(color='red', size=10)
            )
        )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Collaborations Entre Pays (Interactions)',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=annotations,
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False))
                    )

    return fig

def afficher_tableau(df):
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[df[col] for col in df.columns],
                   fill_color='lavender',
                   align='left'))
    ])
    return fig


def collaboration3():
    collaboration_df = authors_with_country_df.alias("a").join(
    authors_with_country_df.alias("b"),
    (col("a.doi") == col("b.doi")) & (col("a.Country") < col("b.Country")),
    "inner"
).join(
    authors_with_country_df.alias("c"),
    (col("a.doi") == col("c.doi")) & 
    (col("b.doi") == col("c.doi")) & 
    (col("a.Country") < col("c.Country")) & 
    (col("b.Country") < col("c.Country")),
    "inner"
).select(
    array_sort(array("a.Country", "b.Country", "c.Country")).alias("countries"),
    "a.doi"
)

    distinct_collaboration_df = collaboration_df.distinct()

    country_collaboration_df = distinct_collaboration_df.groupBy("countries").agg(
    count("doi").alias("collaboration_count")
).orderBy(desc("collaboration_count"))
    return country_collaboration_df



@app.route('/plot_tableau_count_quartile_par_pays', methods=['GET'])
def plot_tableau_count_quartile_par_pays():
    try:
        pandas_journaux_df = traiter_journaux(journaux_df).toPandas()  # Assurez-vous que cette fonction existe et que `journaux_df` est bien défini
        new_data = {}
        # Iterate over unique countries
        for country in pandas_journaux_df['Country'].unique():
            # Initialize counts for each quartile to zero
            new_data[country] = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}
            # Iterate over the dataframe rows for the current country
            for index, row in pandas_journaux_df[pandas_journaux_df['Country'] == country].iterrows():
                # Increment the count for the corresponding quartile
                new_data[country][row['Quartile']] += 1

        new_journaux_df = pd.DataFrame.from_dict(new_data, orient='index')
        new_journaux_df['Country'] = new_journaux_df.index  # Add 'Country' column
        fig = afficher_tableau(new_journaux_df)
        fig_json = fig.to_json()
        return jsonify({
            "plotly_fig": fig_json
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/plot_tableau_article_par_an', methods=['GET'])
def plot_tableau_article_par_an():
    try:
        # Extract the year from the extracted_date column
        df_with_year = df_with_dates.withColumn(
            "year",
            regexp_extract(col("extracted_date"), r"\d{4}", 0)
        )

        # Group by year and count the number of articles
        articles_by_year = df_with_year.groupBy("year").agg(count("*").alias("article_count"))

        # Filter out rows without valid years (if needed)
        articles_by_year = articles_by_year.filter(col("year") != "")
        # Convert the Spark DataFrame to a Pandas DataFrame
        articles_by_year_pd = articles_by_year.toPandas()
        fig = afficher_tableau(articles_by_year_pd)
        fig_json = fig.to_json()
        return jsonify({
            "plotly_fig": fig_json
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/plot_tableau_keywords', methods=['GET'])
def plot_tableau_keywords():
    try:
        df_keywords_pandas = df_keywords_grouped.toPandas()
        fig = afficher_tableau(df_keywords_pandas)
        fig_json = fig.to_json()
        return jsonify({
            "plotly_fig": fig_json
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/plot_tableau_collaborations3', methods=['GET'])
def plot_tableau_collaborations3():
    try:
        fig = afficher_tableau(collaboration3().toPandas())
        fig_json = fig.to_json()
        return jsonify({
            "plotly_fig": fig_json
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/plot_tableau_collaborations2', methods=['GET'])
def plot_tableau_collaborations2():
    try:
        fig = afficher_tableau(country_collaboration_df.toPandas())
        fig_json = fig.to_json()
        return jsonify({
            "plotly_fig": fig_json
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/plot_tableau_articles', methods=['GET'])
def plot_tableau_articles():
    try:
        fig = afficher_tableau(df_with_dates.toPandas())
        fig_json = fig.to_json()
        return jsonify({
            "plotly_fig": fig_json
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/plot_tableau_labs', methods=['GET'])
def plot_tableau_labs():
    try:
        fig = afficher_tableau(authors_labs_df.toPandas())
        fig_json = fig.to_json()
        return jsonify({
            "plotly_fig": fig_json
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/plot_graph_collaborations', methods=['GET'])
def plot_graph_collaborations():
    try:
        fig = generate_graph_collaborations()
        fig_json = fig.to_json()
        return jsonify({
            "plotly_fig": fig_json
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route('/plot_tableau_journaux', methods=['GET'])
def plot_tableau_journaux():
    try:
        fig = generate_tableau_journaux()
        fig_json = fig.to_json()
        return jsonify({
            "plotly_fig": fig_json
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/plot_map_articles_by_country', methods=['GET'])
def plot_map_articles_by_country():
    try:
        fig = generate_map_articles_by_country()
        fig_json = fig.to_json()
        return jsonify({
            "plotly_fig": fig_json
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/plot_articles_by_country', methods=['GET'])
def plot_articles_by_country():
    try:
        img_base64 = generate_articles_by_country()
        return jsonify({
            "plot_base64": img_base64
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/plot_heatmap_articles_by_year_and_month', methods=['GET'])
def plot_heatmap_articles_by_year_and_month():
    try:
        img_base64 = heatmap_articles_by_year_and_month()
        return jsonify({
            "plot_base64": img_base64
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/plot_3d_articles_by_journal', methods=['GET'])
def plot_3d_articles_by_journal():
    try:
        fig = generate_3d_articles_by_journal()
        fig_json = fig.to_json()
        return jsonify({
            "plotly_fig": fig_json
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/plot_articles_by_journal', methods=['GET'])
def plot_articles_by_journal():
    try:
        fig = generate_articles_by_journal()
        fig_json = fig.to_json()
        return jsonify({
            "plotly_fig": fig_json
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/plot_top_10_authors_with_most_articles', methods=['GET'])
def plot_top_10_authors_with_most_articles():
    try:
        fig = generate_top_10_authors_with_most_articles()
        fig_json = fig.to_json()
        return jsonify({
            "plotly_fig": fig_json
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/plot_wordcloud_keywords', methods=['GET'])
def plot_wordcloud_keywords():
    try:
        img_base64 = generate_wordcloud_keywords()
        return jsonify({
            "plot_base64": img_base64
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/plot_top_10_keywords', methods=['GET'])
def plot_top_10_keywords():
    try:
        fig = generate_top_10_keywords()
        fig_json = fig.to_json()
        return jsonify({
            "plotly_fig": fig_json
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/plot_number_of_articles_par_date4', methods=['GET'])
def plot_number_of_articles_par_date4():
    try:
        fig = generate_plot_number_of_articles_par_date4()
        fig_json = fig.to_json()
        return jsonify({
            "plotly_fig": fig_json
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/plot_number_of_articles_par_date3', methods=['GET'])
def plot_number_of_articles_par_date3():
    try:
        fig = generate_plot_number_of_articles_par_date3()
        fig_json = fig.to_json()
        return jsonify({
            "plotly_fig": fig_json
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/plot_number_of_articles_par_date2', methods=['GET'])
def plot_number_of_articles_par_date2():
    try:
        fig = generate_plot_number_of_articles_par_date2()
        fig_json = fig.to_json()
        return jsonify({
            "plotly_fig": fig_json
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/plot_number_of_articles_par_date', methods=['GET'])
def plot_number_of_articles_par_date():
    try:
        img_base64 = generate_plot_number_of_articles_par_date()
        return jsonify({
            "plot_base64": img_base64
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500





@app.route('/plot_number_of_articles_par_year', methods=['GET'])
def plot_number_of_articles_par_year():
    try:
        img_base64 = generate_plot_number_of_articles_par_year()
        return jsonify({
            "plot_base64": img_base64
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/plot_map_authors', methods=['GET'])
def plot_map_authors():
    try:
        fig = generate_map_authors()
        fig_json = fig.to_json()
        return jsonify({
            "plotly_fig": fig_json
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/plot_top_20_labs2', methods=['GET'])
def plot_top_20_labs2():
    try:
        fig = genrate_plot_top_20_labs2()
        fig_json = fig.to_json()
        return jsonify({
            "plotly_fig": fig_json
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/plot_top_20_labs', methods=['GET'])
def plot_top_20_labs():
    try:
        img_base64 = genrate_plot_top_20_labs()
        return jsonify({
            "plot_base64": img_base64
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/plot_authors_par_pays', methods=['GET'])
def plot_authors_par_pays():
    try:
        img_base64 = genrate_plot_authors_par_pays()
        return jsonify({
            "plot_base64": img_base64
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/plot_map_Q_par_Country', methods=['GET'])
def plot_map_Q_par_Country():
    try:
        figures = generate_map_Q_par_Country()
        figures_json = [fig.to_json() for fig in figures]
        return jsonify({
            "plotly_figs": figures_json
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/plot_bubble_chart_Q_par_Country', methods=['GET'])
#pip install -U kaleido
def plot_bubble_chart_Q_par_Country():
    try:
        fig = generate_bubble_chart_Q_par_Country()
        fig_json = fig.to_json()
        return jsonify({
            "plotly_fig": fig_json
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/plot_country_contribution_by_Q', methods=['GET'])
def plot_country_contribution_by_Q():
    try:
        img_base64 = generate_country_contribution_by_Q()
        return jsonify({
            "plot_base64": img_base64
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/plot_contributions_by_Q_by_Country', methods=['GET'])
def plot_contributions_by_Q_by_Country():
    try:
        img_base64 = generate_contributions_by_Q_by_Country()
        return jsonify({
            "plot_base64": img_base64
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/plot_contributions_by_Q2', methods=['GET'])
def plot_contributions_by_Q2():
    try:
        img_base64 = generate_contributions_by_Q2()
        return jsonify({
            "plot_base64": img_base64
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/plot_contributions_by_Q', methods=['GET'])
def plot_contributions_by_Q():
    try:
        img_base64 = generate_contributions_by_Q()
        return jsonify({
            "plot_base64": img_base64
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/plot_Q_distribution', methods=['GET'])
def plot_Q_distribution():
    try:
        img_base64 = generate_Q_distribution()
        return jsonify({
            "plot_base64": img_base64
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/plot_wordcloud_categories_journaux', methods=['GET'])
def plot_wordcloud_categories_journaux():
    try:
        img_base64 = generate_wordcloud_categories_journaux()
        return jsonify({
            "plot_base64": img_base64
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/plot_heatmap_Q_par_Catego_Year', methods=['GET'])
def plot_heatmap_Q_par_Catego_Year():
    try:
        fig = generate_heatmap_Q_par_Catego_Year()
        fig_json = fig.to_json()
        return jsonify({
            "plotly_fig": fig_json
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/plot_Q_par_Year_TS2', methods=['GET'])
def plot_Q_par_Year_TS2():
    try:
        img_base64 = generate_plot_Q_par_Year_TS2()
        return jsonify({
            "plot_base64": img_base64
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/plot_Q_par_Year_TS', methods=['GET'])
def plot_Q_par_Year_TS():
    try:
        img_base64 = generate_plot_Q_par_Year_TS()
        return jsonify({
            "plot_base64": img_base64
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/plot_Q_par_Year', methods=['GET'])
def plot_Q_par_Year():
    try:
        img_base64 = generate_plot_Q_par_Year()
        return jsonify({
            "plot_base64": img_base64
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/plot_Q_par_Catego', methods=['GET'])
def plot_Q_par_Catego():
    try:
        img_base64 = generate_plot_Q_par_Catego()
        return jsonify({
            "plot_base64": img_base64
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint pour nettoyer les données et générer un graphique
@app.route('/clean_data', methods=['GET'])
def clean_and_plot():
    try:
        # Nettoyer les données
        final_df_records, _, _, _, _ = clean_data()

        # Générer le graphique
        img_base64 = generate_plot(final_df_records)

        # Renvoyer le graphique et les comptages sous forme JSON
        return jsonify({
            "data_before_after": final_df_records,  # Assurez-vous que final_df est une liste de dictionnaires
            "plot_base64": img_base64
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(use_reloader=False, debug=True)