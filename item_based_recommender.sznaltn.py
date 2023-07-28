###########################################
# Item-Based Collaborative Filtering
###########################################

# Veri seti: https://grouplens.org/datasets/movielens/

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: User Movie Df'inin Oluşturulması
# Adım 3: Item-Based Film Önerilerinin Yapılması
# Adım 4: Çalışma Scriptinin Hazırlanması

######################################
# Adım 1: Veri Setinin Hazırlanması
######################################
import pandas as pd
pd.set_option('display.max_columns', 500)
movie = pd.read_csv('Datasets/recommendation_systems/item_based_datasets/movie.csv')
rating = pd.read_csv('Datasets/recommendation_systems/item_based_datasets/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()


######################################
# Adım 2: User Movie Df'inin Oluşturulması
######################################

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


########################################################################
# Adım 3: Item-Based Film Önerilerinin Yapılması
########################################################################

movie_name = "Matrix, The (1999)"
movie_name = "Ocean's Twelve (2004)"
movie_name = user_movie_df[movie_name]

# korelasyona bak matrix ile büyükten küçüğe sırala 10 tane getir
# işbirlikçi filtreleme yöntemi budur.
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

# sample örneklem alıyorum 0.indextekini ver
movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

# istenilen keyword giriliyor dataframe de geziyor girilen keyword ü barındıran filmleri listeliyor
def check_film(keyword, user_movie_df):
    return [col for col in user_movie_df.columns if keyword in col]

check_film("Insomnia", user_movie_df)


########################################################################################
# Adım 4: Çalışma Scriptinin Hazırlanması
#########################################################################################
# item based yöntemi ile film tavsiyesi fonksiyon yazma

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

# ismi verilen filmi beğenilme değerlerine bakıp ona uygun 10 filmi getirir
def item_based_recommender(movie_name, user_movie_df):
    movie_name = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

item_based_recommender("Matrix, The (1999)", user_movie_df)

movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]

item_based_recommender(movie_name, user_movie_df)





