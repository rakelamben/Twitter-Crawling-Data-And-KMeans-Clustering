import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk

st.set_page_config(
    page_title="Dashboard", 
    page_icon=":material/dashboard:", 
    layout="centered",
    initial_sidebar_state="expanded"
)

nltk.download('stopwords', quiet=True)

st.title("Proses Klasterisasi Data Twitter")

#konstan data
STOPWORDS = set(stopwords.words('indonesian'))
# Kata yang muncul di hampir semua tweet (hasil keyword) tidak membantu pemisahan klaster
DOMAIN_STOPWORDS = {
    'mbg', 'makan', 'bergizi', 'gratis', 'makanan', 'rt', 'via', 'amp',
    'nih', 'sih', 'deh', 'dong', 'lah', 'kok', 'yaa', 'ya', 'wkwk', 'wkwkwk', 'haha', 'gizi'
}
STOPWORDS |= DOMAIN_STOPWORDS



st.divider()

#crawling data
st.subheader(
    """
        1. Crawling data dari Twitter menggunakan tweeter-harvest.
    """
)
code_crawl = '''
    import os
    import pandas as pd
    from google.colab import drive

    drive.mount('/content/drive')

    # install chromium
    !npx -y playwright install --with-deps chromium

    # Input auth token
    twitter_auth_token = '0a4d47fffd4ddebf8e0fa336bb1350695cbbddfc'

    filename = 'mbg.csv'
    search_keyword = 'mbg makanan bergizi gratis'
    limit = 5000  # 5000 data

    !npx -y tweet-harvest@latest -o "{filename}" -s "{search_keyword}" --tab "LATEST" -l {limit} --token {twitter_auth_token} # Fokus di tab latest

    # Cek hasil dan simpan ke Drive
    path = f'/content/tweets-data/{filename}'

    #Cek kalau apakah file kesimpan
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(df.shape)
        display(df.head())
        !cp "{path}" /content/drive/MyDrive/
    else:
        print('File tidak terbentuk, cek log di atas')
    '''
st.code(code_crawl, language="python")

# Hasil crawling data
st.caption("Hasil crawling data dari Twitter menggunakan tweeter-harvest") 
df_crawl = pd.read_csv('dataset/mbg.csv')
st.dataframe(df_crawl)

# End crawling data

st.divider()

# Preprocessing data teks
st.subheader(
    """
        2. Preprocessing data teks.
    """
)
st.caption("""
    Hapus stopwords, konversi slang, smua teks huruf kecil, hapus url, hapus mention, hapus penanda retweet, hapus symbol hastag, hapus karakter selain huruf, hapus huruf berulang, stemming, merapikan spasi, menghapus tweet duplikat, dan ambil teks dengan 3 kata atau lebih
""")

df_clean = pd.read_csv('dataset/df_clean.csv')
st.dataframe(df_clean[['full_text', 'clean_text']])

# End preprocessing data teks

st.divider()

# Word cloud

st.subheader(
    """
        3. Visualisasi data teks menggunakan word cloud dan most frequent words.
    """
)

st.image('images/wordcloud.png', caption='Word Cloud dari data teks yang sudah di-preprocessing')
st.image('images/mfq.png', caption='Most Frequent Words dari data teks yang sudah di-preprocessing')

# End word cloud

st.divider()

# Clusterisasi Kelbow

st.subheader(
    """
        4. Menentukan jumlah cluster menggunakan metode Kelbow.
    """
)

code_kelbow = '''
    # TF-IDF
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, ngram_range=(1, 2), max_features=5000)
    X = vectorizer.fit_transform(df_clean['clean_text']) #TF-IDF
    print('Matriks TF-IDF:', X)

    # Menentukan k
    model = KMeans(random_state=42)

    visualizer = KElbowVisualizer(model,
                        k=(2,10),
                        metric='silhouette',
                        timings=False)

    # Jalankan (fit) visualizer pada data
    visualizer.fit(X)

    # Tampilkan plot
    visualizer.show()
'''

st.code(code_kelbow, language="python")

st.image('images/kelbow.png', caption='Hasil visualisasi Kelbow untuk menentukan jumlah cluster')

# end kelbow

# clusterisasi KMeans dan visualisasi scatterplot

st.subheader(
    """
        5. Clusterisasi menggunakan KMeans dan visualisasi scatterplot.
    """
)

code_clusterisasi = '''
    # klasifikasi menggunakan k=3
    k=3

    model = KMeans(n_clusters=k, random_state=42)

    # Latih (fit) model dengan data Anda (df)
    labels = model.fit_predict(X)

    df_clean['cluster'] = labels

    # PCA tidak menerima matriks sparse, jadi ubah X menjadi array padat
    X_dense = X.toarray()

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_dense)

    df_pca = pd.DataFrame(coords, columns=['Principal Component 1', 'Principal Component 2'])
    df_pca['Cluster'] = labels           
    print(df_pca)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='Principal Component 1',
        y='Principal Component 2',
        hue='Cluster',
        palette=sns.color_palette('viridis', n_colors=k),   # n_colors disamakan dengan k = 3
        data=df_pca,
        legend='full',
        alpha=0.8
    )

    # Centroid diproyeksikan dengan PCA yang sama
    centers = pca.transform(model.cluster_centers_)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.7, marker='X', label='Centroid')

    plt.title('Visualisasi Cluster dalam 2D (menggunakan PCA)', fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.legend()
    plt.show()

    print('Variansi dijelaskan 2 komponen:', pca.explained_variance_ratio_.sum().round(3))
'''

st.code(code_clusterisasi, language="python")
st.image('images/scatterplot_klaster.png', caption='Hasil visualisasi scatterplot dari cluster yang terbentuk menggunakan PCA')

# end clusterisasi KMeans dan visualisasi scatterplot

# distribusi cluster dan table data

st.subheader(
    """
        6. Distribusi cluster dan table data cluster.
    """
)

data_clean_cluster = pd.read_csv('dataset/df_clean_cluster.csv')
st.dataframe(data_clean_cluster[['full_text', 'cluster']])

st.image('images/distribusi_klaster.png', caption='Distribusi cluster yang terbentuk dari data teks yang sudah di-preprocessing')

#end distribusi cluster dan table data

# sentimen analisis data tweet

st.subheader(
    """
        7. Analisis sentimen data tweet hasil crawling.
    """
)

data_sentimen = pd.read_csv('dataset/df_clean_sentiment.csv')
st.dataframe(data_sentimen[['full_text', 'sentiment']])

st.image('images/distribusi_sentimen.png', caption='Distribusi sentimen dari data teks yang sudah di-preprocessing')

# end sentimen analisis data tweet




