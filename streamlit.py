import streamlit as st
import requests
import plotly.io as pio
from PIL import Image
from io import BytesIO
import base64

# Helper function to fetch and display Plotly figures
def display_plotly_figure(api_url, x=False, title="", description=""):
    try:
        response = requests.get(api_url, timeout=20)  # Set a timeout for the request
        response.raise_for_status()  # Raise an error for bad status codes
        st.markdown(f"### {title}")
        st.markdown(description)
        if x:
            fig_json = response.json()["plotly_figs"]
            cols = st.columns(2)  # Create two columns
            for i, fig in enumerate(fig_json):
                fig = pio.from_json(fig)
                with cols[i % 2]:  # Alternate between columns
                    st.plotly_chart(fig)
        else:
            fig_json = response.json()["plotly_fig"]
            fig = pio.from_json(fig_json)
            st.plotly_chart(fig)
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la récupération de la figure depuis {api_url}: {e}")

# Helper function to fetch and display images
def display_image(api_url, title="", description=""):
    try:
        response = requests.get(api_url, timeout=20)  # Set a timeout for the request
        response.raise_for_status()  # Raise an error for bad status codes
        st.markdown(f"### {title}")
        st.markdown(description)
        img_base64 = response.json()["plot_base64"]
        img_bytes = base64.b64decode(img_base64)
        img = Image.open(BytesIO(img_bytes))
        st.image(img)
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la récupération de l'image depuis {api_url}: {e}")

# Function to create a colored title
def colored_title(text, color):
    st.markdown(f'<h1 style="color:{color};">{text}</h1>', unsafe_allow_html=True)

# Sidebar for category selection
st.sidebar.title("Sélectionnez une catégorie")
category = st.sidebar.selectbox(
    "Catégories",
    ["", "📚 Journaux", "📄 Articles", "👤 Auteurs", "📊 Quartiles", "🏢 Labs", "🔑 Keywords", "🙏 Collaborations"]
)

# Display description only if no category is selected
if not category:
    # Streamlit app layout
    colored_title("Analyse Recherche Scientifique", "black")
    st.markdown("Cette application web permet d'analyser les données de recherche scientifique.")
    
    # Extended description with images of the technologies used
    st.markdown("### Technologies et Ressources Utilisées")
    st.markdown(
        """
        <style>
        .centered {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .centered img {
            margin: 0 10px;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class="centered">
            <div><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRAcu7QsNvfrM_f2jISwR6_TvAqjQQivrCIHQ&s" width="100"></div>
            <div><img src="https://upload.wikimedia.org/wikipedia/commons/f/f3/Apache_Spark_logo.svg" width="100"></div>
            <div><img src="https://upload.wikimedia.org/wikipedia/commons/3/3c/Flask_logo.svg" width="100"></div>
            <div><img src="https://cdn.worldvectorlogo.com/logos/ieee.svg" width="100"></div>
            <div><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQL3yUdubjq8G7wxQy19_szQzJDs0WwAy9SjQ&s" width="100"></div>
        </div>
        """, 
        unsafe_allow_html=True
    )
        
    st.markdown("""
    - **Scraping des Données**: Utilisation de Scrapy pour collecter les données des sources scientifiques comme IEEE et ScienceDirect.
    - **Stockage des Données**: Utilisation de MongoDB et Hadoop pour stocker des volumes importants de données.
    - **Analyse des Données**: Utilisation d'Apache Spark pour l'analyse des grandes données.
    - **Visualisation des Résultats**: Création de visualisations interactives dans une interface web utilisant Python et Flask.
    """)

    # Display a local image
    st.image("image.jpg", caption="Image locale", use_container_width=True)

# Display options for selected category
if category:
    option = st.selectbox(
        f"Sélectionnez l'option pour {category}",
        ["Tableaux", "Visualisation"]
    )

    if option == "Tableaux":
        st.header(f"📊 {category} - Tableaux")
        if category == "📚 Journaux":
            display_plotly_figure("http://localhost:5000/plot_tableau_journaux", title="Tableau des Journaux", description="Ce tableau présente les différents journaux scientifiques analysés.")
        if category == "📄 Articles":
            display_plotly_figure("http://localhost:5000/plot_tableau_articles", title="Tableau des Articles", description="Ce tableau présente les articles scientifiques collectés.")
            display_plotly_figure("http://localhost:5000/plot_tableau_article_par_an", title="Tableau des Articles par Année", description="Ce tableau montre la distribution des articles par année.")
        if category == "🏢 Labs":
            display_plotly_figure("http://localhost:5000/plot_tableau_labs", title="Tableau des Labs et Auteurs", description="Ce tableau présente les laboratoires et leurs auteurs associés.")
        if category == "🔑 Keywords":
            display_plotly_figure("http://localhost:5000/plot_tableau_keywords", title="Tableau des Keywords", description="Ce tableau liste les mots-clés utilisés dans les articles scientifiques.")
        if category == "👤 Auteurs":
            display_plotly_figure("http://localhost:5000/plot_tableau_labs", title="Tableau des Auteurs", description="Ce tableau présente les auteurs impliqués dans les publications.")
        if category == "📊 Quartiles":
            display_plotly_figure("http://localhost:5000/plot_tableau_count_quartile_par_pays", title="Le nombre des quartiles par pays", description="Ce tableau montre le nombre de quartiles par pays.")
        if category == "🙏 Collaborations":
            display_plotly_figure("http://localhost:5000/plot_tableau_collaborations2", title="Paires de pays collaborant ensemble dans un même article", description="Ce tableau présente les collaborations entre paires de pays.")
            display_plotly_figure("http://localhost:5000/plot_tableau_collaborations3", title="Combinaisons de 3 pays collaborant ensemble dans un même article", description="Ce tableau présente les collaborations entre trois pays.")

    elif option == "Visualisation":
        st.header(f"📈 {category} - Visualisation")

        # Journaux
        if category == "📚 Journaux":
            display_plotly_figure("http://localhost:5000/plot_heatmap_Q_par_Catego_Year", title="Heatmap des Quartiles par Catégorie et Année", description="Cette heatmap montre la distribution des quartiles par catégorie et par année.")
            display_plotly_figure("http://localhost:5000/plot_3d_articles_by_journal", title="Plot 3D des Articles par Journal", description="Ce plot 3D présente la répartition des articles par journal.")
            display_plotly_figure("http://localhost:5000/plot_articles_by_journal", title="Plot des Articles par Journal", description="Ce plot montre la répartition des articles par journal.")
            display_plotly_figure("http://localhost:5000/plot_map_Q_par_Country", True, title="Map des Quartiles par Pays", description="Cette carte montre la répartition des quartiles par pays.")
            display_plotly_figure("http://localhost:5000/plot_bubble_chart_Q_par_Country", title="Bubble Chart des Quartiles par Pays", description="Ce bubble chart présente la répartition des quartiles par pays.")
            display_image("http://localhost:5000/plot_Q_distribution", title="Distribution des Quartiles", description="Cette image montre la distribution des quartiles.")
            display_image("http://localhost:5000/plot_wordcloud_categories_journaux", title="Wordcloud des Catégories des Journaux", description="Ce wordcloud montre les différentes catégories de journaux.")

        # Articles
        elif category == "📄 Articles":
            display_image("http://localhost:5000/plot_articles_by_country", title="Articles par Pays", description="Ce plot montre le nombre d'articles par pays.")
            display_plotly_figure("http://localhost:5000/plot_map_articles_by_country", title="Map des Articles par Pays", description="Cette carte montre la répartition des articles par pays.")
            display_image("http://localhost:5000/plot_heatmap_articles_by_year_and_month", title="Heatmap des Articles par Année et Mois", description="Cette heatmap montre la répartition des articles par année et mois.")
            display_plotly_figure("http://localhost:5000/plot_number_of_articles_par_date2", title="Nombre d'Articles par Date", description="Ce plot montre le nombre d'articles par date.")
            display_image("http://localhost:5000/plot_number_of_articles_par_date", title="Nombre d'Articles par Année", description="Ce plot montre le nombre d'articles par année.")
            display_image("http://localhost:5000/plot_number_of_articles_par_year", title="Nombre d'Articles par Année", description="Ce plot montre le nombre d'articles par année.")

        # Keywords
        elif category == "🔑 Keywords":
            display_plotly_figure("http://localhost:5000/plot_top_10_keywords", title="Top 10 Keywords", description="Ce plot montre les 10 mots-clés les plus utilisés.")
            display_image("http://localhost:5000/plot_wordcloud_keywords", title="Wordcloud des Keywords", description="Ce wordcloud montre les mots-clés utilisés dans les articles.")

        # Auteurs
        elif category == "👤 Auteurs":
            display_plotly_figure("http://localhost:5000/plot_map_authors", title="Map des Auteurs", description="Cette carte montre la répartition des auteurs par pays.")
            display_plotly_figure("http://localhost:5000/plot_top_10_authors_with_most_articles", title="Top 10 Auteurs avec le plus d'Articles", description="Ce plot montre les 10 auteurs ayant publié le plus d'articles.")
            display_image("http://localhost:5000/plot_authors_par_pays", title="Auteurs par Pays", description="Ce plot montre la répartition des auteurs par pays.")

        # Quartiles
        elif category == "📊 Quartiles":
            display_image("http://localhost:5000/plot_country_contribution_by_Q", title="Contributions des Pays par Quartile", description="Ce plot montre les contributions des pays par quartile.")
            display_image("http://localhost:5000/plot_contributions_by_Q_by_Country", title="Contributions par Quartile et Pays", description="Ce plot montre les contributions par quartile et par pays.")
            # display_image("http://localhost:5000/plot_contributions_by_Q2", title="Contributions par Quartile", description="Ce plot montre les contributions par quartile.")
            # display_image("http://localhost:5000/plot_contributions_by_Q", title="Contributions par Quartile", description="Ce plot montre les contributions par quartile.")
            display_image("http://localhost:5000/plot_Q_par_Year_TS2", title="Time Series des Quartiles par Année", description="Cette série temporelle montre l'évolution des quartiles par année.")
            # display_image("http://localhost:5000/plot_Q_par_Year_TS", title="Time Series des Quartiles par Année", description="Cette série temporelle montre l'évolution des quartiles par année.")
            # display_image("http://localhost:5000/plot_Q_par_Year", title="Plot des Quartiles par Année", description="Ce plot montre la répartition des quartiles par année.")
            display_image("http://localhost:5000/plot_Q_par_Catego", title="Plot des Quartiles par Catégorie", description="Ce plot montre la répartition des quartiles par catégorie.")

        # Labs
        elif category == "🏢 Labs":
            display_plotly_figure("http://localhost:5000/plot_top_20_labs2", title="Top 20 Labs", description="Ce plot montre les 20 meilleurs laboratoires.")
            display_image("http://localhost:5000/plot_top_20_labs", title="Top 20 Labs", description="Ce plot montre les 20 meilleurs laboratoires.")

        elif category == "🙏 Collaborations":
            display_plotly_figure("http://localhost:5000/plot_graph_collaborations", title="Graph des Collaborations", description="Ce graph montre les collaborations entre les différents pays.")

        else:
            st.error("Catégorie non reconnue. Veuillez sélectionner une autre catégorie.")