############################################
# User-Based Collaborative Filtering
#############################################

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
# Adım 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması
# Adım 6: Çalışmanın Fonksiyonlaştırılması

#############################################
# Adım 1: Veri Setinin Hazırlanması
#############################################
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('Datasets/recommendation_systems/item_based_datasets/movie.csv')
    rating = pd.read_csv('Datasets/recommendation_systems/item_based_datasets/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()


# rastgele bir user seçilecek random
# pandas series oluşturuyoruz user movie index inden
# random state 45 yapıyoruz aynı kullanıcıyı seçmek için values değeri str idi int yaptık
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)


####################################################################################################################
# Adım 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
####################################################################################################################
random_user
user_movie_df
random_user_df = user_movie_df[user_movie_df.index == random_user] # kullanıcıyı seçtik ve tüm filmleri

# seçilen kullanıcının izlediği filmler
# notna yani boş olmayanlar liste formunda getir
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

# seçtiğimiz kullanıcının seçtiğimiz filme kaç puan verdiğini öğrenmek için
user_movie_df.loc[user_movie_df.index == random_user,
                  user_movie_df.columns == "Silence of the Lambs, The (1991)"]

# seçilen kullanıcının kaç film izlediğini bulmak için
len(movies_watched)



#############################################
# Adım 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################
# izlenen filmler
movies_watched_df = user_movie_df[movies_watched]

# her bir kullanıcının kaçar tane film izlediği
user_movie_count = movies_watched_df.T.notnull().sum()

# userid yi değişkene çeviriyoruz
user_movie_count = user_movie_count.reset_index()

# değişkenleri isimlendirelim
user_movie_count.columns = ["userId", "movie_count"]

# istediğimiz kullanıcı ile 20den fazla film izleyenleri getir azalan şekilde sırala
user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False)

# kullanıcımız ile aynı sayıda film isleyen kaç kişi var
user_movie_count[user_movie_count["movie_count"] == 33].count()

# kullanıcımız ile 20den fazla ortak film izleyen kişiler
users_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"]

# kullanıcımız ile % 60 ortak film izleyen kişiler
# users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
# perc = len(movies_watched) * 60 / 100

#############################################
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
#############################################

# Bunun için 3 adım gerçekleştireceğiz:
# 1. Sinan ve diğer kullanıcıların verilerini bir araya getireceğiz.
# 2. Korelasyon df'ini oluşturacağız.
# 3. En benzer bullanıcıları (Top Users) bulacağız

# izlenen filmler dataframe inde kullanıcımız ile aynı filmleri izleyenleri ara
# aynı filmleri izleyenler ile indexlerdeki user id leri kesiştir
# concat et belirlenen kullanıcı ve izlenen filmleri
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])
# sütunlara kullanıcıları al, korelasyonunu hesapla, pivot unu al değerleri sırala , duplica kayıtları çıkar
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

# corr yazalım değişkene
corr_df = pd.DataFrame(corr_df, columns=["corr"])

# index isimlerini düzenle
corr_df.index.names = ['user_id_1', 'user_id_2']

# index isimlerini değişken ismine çevirelim
corr_df = corr_df.reset_index()

# istediğimiz kullanıcı ile diğer kullanıcılar arasında korelasyonu hesapla %65 korelasyon üzerindekileri getir
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

# azalan şekilde hesapla
top_users = top_users.sort_values(by='corr', ascending=False)

# düzenleme yapalım değişken ismi düzeltme
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

# dosyamızı top users ile birleştielim
rating = pd.read_csv('Datasets/recommendation_systems/item_based_datasets/rating.csv')
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

# seçtiğimiz kullanıyı çıkaralım
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]


#################################################################################################
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması
#################################################################################################

# hem korelasyon hem de rating değerlerine göre değerlendirme yapıyoruz
# ['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.loc[:, 'weighted_rating'] = top_users_ratings.loc[:, 'corr'] * top_users_ratings.loc[:, 'rating']

#
top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

#
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

#
recommendation_df = recommendation_df.reset_index()

recommendation_df[recommendation_df["weighted_rating"] > 3.5]

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)

movie = pd.read_csv('Datasets/recommendation_systems/item_based_datasets/movie.csv')

# kullanıcımızın beğenebileceği filmler ve isimleri
movies_to_be_recommend.merge(movie[["movieId", "title"]])



#############################################
# Adım 6: Çalışmanın Fonksiyonlaştırılması
#############################################
import pandas as pd
def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('Datasets/recommendation_systems/item_based_datasets/movie.csv')
    rating = pd.read_csv('Datasets/recommendation_systems/item_based_datasets/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

# perc = len(movies_watched) * 60 / 100
# users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

# rasgele kullanıcı seç, %60 ortak film izleme (ratio), kullanıcımız ile benzer beğenme davranışını %65 göstersin
# score de 3.5 üstü korelasyon ve rating in çarpımından gelen weigthted değeri
def user_based_recommender(random_user, user_movie_df, ratio=60, cor_th=0.65, score=3.5):
    random_user_df = user_movie_df[user_movie_df.index == random_user]
    movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
    movies_watched_df = user_movie_df[movies_watched]
    user_movie_count = movies_watched_df.T.notnull().sum()
    user_movie_count = user_movie_count.reset_index()
    user_movie_count.columns = ["userId", "movie_count"]
    perc = len(movies_watched) * ratio / 100
    users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

    final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                          random_user_df[movies_watched]])

    corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
    corr_df = pd.DataFrame(corr_df, columns=["corr"])
    corr_df.index.names = ['user_id_1', 'user_id_2']
    corr_df = corr_df.reset_index()

    top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= cor_th)][
        ["user_id_2", "corr"]].reset_index(drop=True)

    top_users = top_users.sort_values(by='corr', ascending=False)
    top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
    rating = pd.read_csv('Datasets/recommendation_systems/item_based_datasets/rating.csv')
    top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
    top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']


    recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
    recommendation_df = recommendation_df.reset_index()

    movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > score].sort_values("weighted_rating", ascending=False)
    movie = pd.read_csv('Datasets/recommendation_systems/item_based_datasets/movie.csv')
    return movies_to_be_recommend.merge(movie[["movieId", "title"]])



random_user = int(pd.Series(user_movie_df.index).sample(1).values)
user_based_recommender(random_user, user_movie_df, cor_th=0.70, score=4)


