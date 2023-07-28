#############################
# Content Based Recommendation (İçerik Temelli Tavsiye)
#############################

#############################
# Film Overview'larına Göre Tavsiye Geliştirme
#############################

# 1. TF-IDF Matrisinin Oluşturulması
# 2. Cosine Similarity Matrisinin Oluşturulması
# 3. Benzerliklere Göre Önerilerin Yapılması
# 4. Çalışma Scriptinin Hazırlanması

######################################################################################################################
# 1. TF-IDF Matrisinin Oluşturulması
######################################################################################################################
# pip install --upgrade scikit-learn
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None) # tüm sütunları göster
pd.set_option('display.width', 500) # yan yana göster
pd.set_option('display.expand_frame_repr', False) # tek satırda göster
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# https://www.kaggle.com/rounakbanik/the-movies-dataset
df = pd.read_csv("Datasets/movies_metadata.csv", low_memory=False)  # DtypeWarning kapamak icin
df.head()
df.shape

# overview açıklama kısmı biz bu alan ile igileneceğiz
df["overview"].head()

# in, on, an, the gibi kelimeler önemli bir durum ifade etmiyor
# stop_words : yaygınca kullanılan ama bir ölçüm-anlam ifade etmeyen kelimeler
tfidf = TfidfVectorizer(stop_words="english")

# overview i na olanlar
# df[df['overview'].isnull()]

# overview i nan olanları  boşluklar ile değiştir
df['overview'] = df['overview'].fillna('')

# buradaki bilgileri kullanarak fit et ve dönüştür- matrisi oluştur
# fit, fit eder yani işlemi uygular
# transform, fit edilen değerleri eski değerler ile değiştirir
tfidf_matrix = tfidf.fit_transform(df['overview'])

tfidf_matrix.shape
# satırlardakiler filmlerdir
# sütunlardakiler eşsiz kelimeler

# title lar film isimleri
df['title'].shape

# tfidf skorları kesişimde
# sütunlardaki tüm kelimelerin isimleri
tfidf.get_feature_names_out()

# Filtreleme
df = df[~df["title"].duplicated(keep='last')]
df = df[~df["title"].isna()]
df = df[~df["overview"].isna()]

#TFIDF Oluşturma
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])

# Matrisin hafızadaki boyutunu yarıya indirme
tfidf_matrix = tfidf_matrix.astype(np.float32)

# Cosine Similarity hesaplaması
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
#tfidf in dökümanlar ile terimelerin kesişimindeki skorları
tfidf_matrix.toarray()


###########################################################################
# 2. Cosine Similarity Matrisinin Oluşturulması
###########################################################################

# cosine sim e benzerliğini hesaplamak istediğimiz matrisi gireriz
# olası tüm döküman çiftleri için cosine similarity hesabı yapıldı
cosine_sim = cosine_similarity(tfidf_matrix,
                               tfidf_matrix)
# s
cosine_sim.shape

# 
cosine_sim[1]


#############################################################################################
# 3. Benzerliklere Göre Önerilerin Yapılması
#############################################################################################

# filmlerin isimleri için pandas serisi oluşturup içine filmlerin title larını yerşeltiricez
# index lerde filmin ismi ve karşılığında filmin isminin index i var
# bu isme sahip filmin hangi index te olduğunun nümerik bilgisini verdik
indices = pd.Series(df.index, index=df['title'])

# aynı filmden kaçar tane old saydırdık
# çoklama problemi, title larda çoklama var
indices.index.value_counts()

# çoklamalardan birini tutup diğerlerini silicez
# en sondaki ismi alıcaz yani en son çekileni
# duplica olan var mı, default değeri true dur, keep in ön tanımlı değeri first.
# Ama biz sonuncuyu istediğimiz için last dedik
# tilda nın sebebi duplica olmayanları istiyoruz o yüzden tilda var
indices = indices[~indices.index.duplicated(keep='last')]

# title tek mi kontrol edelim
indices["Cinderella"]

indices["Sherlock Holmes"]

# bu filmin index ini tutalım
movie_index = indices["Sherlock Holmes"]

# cosine_sim scor larını hesaplayalım
cosine_sim[movie_index]

# görüntüyü düzeletelim ve benzerlik scorlarını versin
similarity_scores = pd.DataFrame(cosine_sim[movie_index],
                                 columns=["score"])

# scor a göre azalan sıralasın ki en çok benzeyenleri görelim
# [1:11] dememizin sebebi 0 da filmin kesndisi var o yüzden 10 tanesini getirmek için [1:11] alıyoruz
# benzeyen 10 filmin de index lerini istiyoruz
movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index

# sherlock kolmes filmi ile benzerlik gösteren 10 film
df['title'].iloc[movie_indices]

##########################################################################################
# 4. Çalışma Scriptinin Hazırlanması
###########################################################################################

# fonks oluşturalım:
def content_based_recommender(title, cosine_sim, dataframe):
    # index'leri olusturma
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    # title'ın index'ini yakalama
    movie_index = indices[title]
    # title'a gore benzerlik skorlarını hesapalama
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # kendisi haric ilk 10 filmi getirme
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe['title'].iloc[movie_indices]

# sherlock holmes için benzer filmleri listeleyelim 10 tanesini
content_based_recommender("Sherlock Holmes", cosine_sim, df)
# matrix için benzer filmleri listeleyelim 10 tanesini
content_based_recommender("The Matrix", cosine_sim, df)

# godfather için için benzer filmleri listeleyelim 10 tanesini
content_based_recommender("The Godfather", cosine_sim, df)

# the dark knight rises filmi için benzer filmleri listeleyelim 10 tanesini
content_based_recommender('The Dark Knight Rises', cosine_sim, df)

# cosine sim için de fonks yazdık
def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['overview'] = dataframe['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = calculate_cosine_sim(df)

content_based_recommender('The Dark Knight Rises', cosine_sim, df)
# 1 [90, 12, 23, 45, 67]
# 2 [90, 12, 23, 45, 67]
# 3
