
##################################################################################################################
# PROJE: Hybrid Recommender System
##################################################################################################################

# ID'si verilen kullanıcı için item-based ve user-based recomennder
# yöntemlerini kullanarak tahmin yapınız.
# 5 öneri user-based modelden 5 öneri de item-based modelden ele alınız
# ve nihai olarak 10 öneriyi 2 modelden yapınız.

###################################################################################################################
# Görev 1: Verinin Hazırlanması
#######################################################################################################################
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

#######################################################################################################################
# Adım 1: Movie ve Rating veri setlerini okutunuz.
# movieId, film adı ve filmin tür bilgilerini içeren veri seti
# UserID, film adı, filme verilen oy ve zaman bilgisini içeren veri seti
#######################################################################################################################

movie = pd.read_csv('Datasets/recommendation_systems/item_based_datasets/movie.csv')
rating = pd.read_csv('Datasets/recommendation_systems/item_based_datasets/rating.csv')

movie.head()
rating.head()

#######################################################################################################################
# Adım 2: Rating veri setine filmlerin isimlerini ve türünü movie film setini kullanrak ekleyiniz.
# Ratingdeki kullanıcıların oy kullandıkları filmlerin sadece id'si var.
# Idlere ait film isimlerini ve türünü movie veri setinden ekliyoruz.
#######################################################################################################################
df = movie.merge(rating, how="left", on="movieId")
df.head()
df.shape # yorum sayısı

######################################################################################################################
# Adım 3: Herbir film için toplam kaç kişinin oy kullandığını hesaplayınız.
# Toplam oy kullanılma sayısı 1000'un altında olan filmleri veri setinden çıkarınız.
# Herbir film için toplam kaç kişinin oy kullanıldığını hesaplıyoruz.
# Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini rare_movies de tutuyoruz.
# Ve veri setinden çıkartıyoruz
######################################################################################################################

# benzersiz title sayısı
df["title"].nunique() # eşsiz film sayısı

df["title"].value_counts().head() # hangi filme kaçar rate gelmiş

# value counts u dataframe e çevirdik, her filmin nekadar yoruma sahip old
# kaç yorum-rate ya da değerlendirme
comment_counts = pd.DataFrame(df["title"].value_counts())

# 1000den az değerlendirme alanlar
rare_movies = comment_counts[comment_counts["title"] <= 1000].index

# dataframe in içinde yukarıdaki title a sahip olmayanları getir
# az rate alanlardan kurtuluyoruz
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape # kalan filmlerin değerlendirme sayısı

# 3159 benzersiz film var
common_movies["title"].nunique()

# 27262 tane vardı önceden elemeden sonra 3159 kaldı
df["title"].nunique()

#####################################################################################################################
# Adım 4: # index'te userID'lerin sutunlarda film isimlerinin
# ve değer olarakta ratinglerin bulunduğu
# dataframe için pivot table oluşturunuz.
#####################################################################################################################

# satırlarda kullanıcılar sütunlarda ise film isimleri değerlendirmeler de rating (kesişimde rating)
# pivot table kullanıyoruz
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

user_movie_df.shape
user_movie_df.columns

######################################################################################################################
# Adım 5: Yukarıda yapılan tüm işlemleri fonksiyonlaştıralım
######################################################################################################################

def create_user_movie_df():
    movie = pd.read_csv('Datasets/recommendation_systems/item_based_datasets/movie.csv')
    rating = pd.read_csv('Datasets/recommendation_systems/item_based_datasets/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

###################################################################################################################
# Görev 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
###################################################################################################################

# Adım 1: Rastgele bir kullanıcı id'si seçiniz.
###################################################################################################################
# rastgele bir user seçilecek random
# pandas series oluşturuyoruz user movie index inden
# random state 45 yapıyoruz aynı kullanıcıyı seçmek için values değeri str idi int yaptık
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)


##################################################################################################################
# Adım 2: Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturunuz.

random_user
user_movie_df
random_user_df = user_movie_df[user_movie_df.index == random_user] # kullanıcıyı seçtik ve tüm filmleri



#############################################################################################################
# Adım 3: Seçilen kullanıcının oy kullandığı filmleri movies_watched adında bir listeye atayınız.

# seçilen kullanıcının izlediği filmler
# notna yani boş olmayanlar liste formunda getir
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

# seçtiğimiz kullanıcının seçtiğimiz filme kaç puan verdiğini öğrenmek için
user_movie_df.loc[user_movie_df.index == random_user,
                  user_movie_df.columns == "Silence of the Lambs, The (1991)"]

# seçilen kullanıcının kaç film izlediğini bulmak için
len(movies_watched)
###################################################################################################################
# Görev 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
###################################################################################################################

# Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz
# ve movies_watched_df adında yeni bir dataframe oluşturuyoruz.

# izlenen filmler
movies_watched_df = user_movie_df[movies_watched]

###################################################################################################################
# Adım 2: Herbir kullancının seçili user'in izlediği filmlerin kaçını izlediği
# bilgisini taşıyan user_movie_count adında yeni bir dataframe oluşturunuz.
# Ve yeni bir df oluşturuyoruz.
####################################################################################################################

# her bir kullanıcının kaçar tane film izlediği
user_movie_count = movies_watched_df.T.notnull().sum()

# userid yi değişkene çeviriyoruz
user_movie_count = user_movie_count.reset_index()

# değişkenleri isimlendirelim
user_movie_count.columns = ["userId", "movie_count"]


####################################################################################################################
# Adım 3: Seçilen kullanıcının oy verdiği filmlerin yüzde 60
# ve üstünü izleyenleri benzer kullanıcılar olarak görüyoruz.
# Bu kullanıcıların id’lerinden users_same_movies adında bir liste oluşturunuz.
####################################################################################################################

perc = len(movies_watched) * 60 / 100
user_movie_count[user_movie_count["movie_count"] > perc].sort_values("movie_count", ascending=False)

# kullanıcımız ile aynı sayıda film isleyen kaç kişi var
user_movie_count[user_movie_count["movie_count"] == len(movies_watched)].count()
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]


####################################################################################################################
# Görev 4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
####################################################################################################################

# Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren
# kullanıcıların id’lerinin bulunacağı şekilde movies_watched_df dataframe’ini filtreleyiniz.


# Bunun için 3 adım gerçekleştireceğiz:
# 1. Sinan ve diğer kullanıcıların verilerini bir araya getireceğiz.
# 2. Korelasyon df'ini oluşturacağız.
# 3. En benzer bullanıcıları (Top Users) bulacağız

# izlenen filmler dataframe inde kullanıcımız ile aynı filmleri izleyenleri ara
# aynı filmleri izleyenler ile indexlerdeki user id leri kesiştir
# concat et belirlenen kullanıcı ve izlenen filmleri
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])

#######################################################################################################################
# Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz.
#corr_df[corr_df["user_id_1"] == random_user]
#######################################################################################################################

# sütunlara kullanıcıları al, korelasyonunu hesapla, pivot unu al değerleri sırala , duplica kayıtları çıkar
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

# corr yazalım değişkene
corr_df = pd.DataFrame(corr_df, columns=["corr"])

# index isimlerini düzenle
corr_df.index.names = ['user_id_1', 'user_id_2']

# index isimlerini değişken ismine çevirelim
corr_df = corr_df.reset_index()


########################################################################################################################
# Adım 3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan)
# kullanıcıları filtreleyerek top_users adında yeni bir dataframe oluşturunuz.
#######################################################################################################################

# istediğimiz kullanıcı ile diğer kullanıcılar arasında korelasyonu hesapla %65 korelasyon üzerindekileri getir
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

# azalan şekilde hesapla
top_users = top_users.sort_values(by='corr', ascending=False)

# düzenleme yapalım değişken ismi düzeltme
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

######################################################################################################################
# Adım 4:  top_users dataframe’ine rating veri seti ile merge ediniz
######################################################################################################################


# dosyamızı top users ile birleştielim
rating = pd.read_csv('Datasets/recommendation_systems/item_based_datasets/rating.csv')
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

# seçtiğimiz kullanıyı çıkaralım
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]



##############################################################################################################
# Görev 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
##############################################################################################################

# Adım 1: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan
# weighted_rating adında yeni bir değişken oluşturunuz.

# hem korelasyon hem de rating değerlerine göre değerlendirme yapıyoruz
# ['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.loc[:, 'weighted_rating'] = top_users_ratings.loc[:, 'corr'] * top_users_ratings.loc[:, 'rating']

top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})


##############################################################################################################
# Adım 2: Film id’sive her bir filme ait tüm kullanıcıların weighted rating’lerinin
# ortalama değerini içeren recommendation_df adında yeni bir dataframe oluşturunuz.
##############################################################################################################

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})


recommendation_df = recommendation_df.reset_index()


##############################################################################################################
# Adım 3: Adım3: recommendation_df içerisinde weighted rating'i 3.5'ten büyük
# olan filmleri seçiniz ve weighted rating’e göre sıralayınız.
# İlk 5 gözlemi movies_to_be_recommend olarak kaydediniz.
###############################################################################################################


recommendation_df[recommendation_df["weighted_rating"] > 3.5]

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)

movie = pd.read_csv('Datasets/recommendation_systems/item_based_datasets/movie.csv')

##############################################################################################################
# Adım 4:  Tavsiye edilen 5 filmin isimlerini getiriniz.
##############################################################################################################


# kullanıcımızın beğenebileceği filmler ve isimleri
movies_to_be_recommend.merge(movie[["movieId", "title"]]).head(7)

###############################################################################################################
# Adım 6: Item-Based Recommendation
###############################################################################################################

# Kullanıcının en son izlediği ve en yüksek puan verdiği filmin adına göre item-based öneri yapınız.
user = 108170
################################################################################################################
# Adım 1: movie,rating veri setlerini okutunuz.
################################################################################################################
import pandas as pd
pd.set_option('display.max_columns', 500)
movie = pd.read_csv('Datasets/recommendation_systems/item_based_datasets/movie.csv')
rating = pd.read_csv('Datasets/recommendation_systems/item_based_datasets/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()

################################################################################################################
# Adım 2: Öneri yapılacak kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınız.
#################################################################################################################

df.head()
df.shape # yorum sayısı

# benzersiz title sayısı
df["title"].nunique() # eşsiz film sayısı

df["title"].value_counts().head() # hangi filme kaçar rate gelmiş


# value counts u dataframe e çevirdik, her filmin nekadar yoruma sahip old
# kaç yorum-rate ya da değerlendirme
comment_counts = pd.DataFrame(df["title"].value_counts())

# 1000den az değerlendirme alanlar
rare_movies = comment_counts[comment_counts["title"] <= 1000].index

# dataframe in içinde yukarıdaki title a sahip olmayanları getir
# az rate alanlardan kurtuluyoruz
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape # kalan filmlerin değerlendirme sayısı

# 3159 benzersiz film var
common_movies["title"].nunique()

# 27262 tane vardı önceden elemeden sonra 3159 kaldı
df["title"].nunique()

# satırlarda kullanıcılar sütunlarda ise film isimleri değerlendirmeler de rating (kesişimde rating)
# pivot table kullanıyoruz
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

user_movie_df.shape
user_movie_df.columns


rating["timestamp"] = pd.to_datetime(rating["timestamp"])
five_points_rated = rating[(rating["rating"] == 5) & (rating["userId"] == user)].sort_values("timestamp", ascending=False)
five_points_rated.reset_index()
################################################################################################################
# Adım 3 :User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini
# seçilen film id’sine göre filtreleyiniz.
################################################################################################################


movie_name = user_movie_df[five_points_rated]
movie_name.shape
movie_name.head()


##################################################################################################################
# Adım 4: Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız.
##################################################################################################################
# korelasyona bak matrix ile büyükten küçüğe sırala 10 tane getir
# işbirlikçi filtreleme yöntemi budur.
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

##############################################################################################################
# Adım 5: Seçili film’in kendisi haricinde ilk 5 film’I öneri olarak veriniz.
##############################################################################################################
import pandas as pd
def create_user_movie_df():
    movie = pd.read_csv('Datasets/recommendation_systems/item_based_datasets/movie.csv')
    rating = pd.read_csv('Datasets/recommendation_systems/item_based_datasets/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()


def item_based_recommender(movie_name, user_movie_df):
    movie_name_column = user_movie_df[movie_name]
    correlations = user_movie_df.corrwith(movie_name_column)
    correlations = correlations.drop(movie_name)
    return user_movie_df.corrwith(movie_name_column).sort_values(ascending=False)[1:6]

item_based_recommender("Heartbreakers (2001)", user_movie_df)

movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]
print(movie_name)
item_based_recommender(movie_name, user_movie_df)

def check_id(dataframe, id):
    movie_name = dataframe[dataframe["movieId"] == id][["title"]].values[0].tolist()
    print(movie_name)

check_id(movie, 7044)

#first_five_movies_1[first_five_movies_1['title'] != 'Wild at Heart (1990)']['title'].head()



