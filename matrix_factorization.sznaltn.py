#############################
# Model-Based Collaborative Filtering: Matrix Factorization
#############################

# !pip install surprise
import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
pd.set_option('display.max_columns', None)

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Modelleme
# Adım 3: Model Tuning
# Adım 4: Final Model ve Tahmin

#############################
# Adım 1: Veri Setinin Hazırlanması
#############################

movie = pd.read_csv('Datasets/recommendation_systems/item_based_datasets/movie.csv')
rating = pd.read_csv('Datasets/recommendation_systems/item_based_datasets/rating.csv')
# iki veri setini birleştiriyoruz
df = movie.merge(rating, how="left", on="movieId")
df.head()

# örnek 4 id ye bakalım ve isimlerine bakalım
movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]
# üstteki bu id ve isimlerden bir sample df oluşturuyoruz
sample_df = df[df.movieId.isin(movie_ids)]
sample_df.head()

sample_df.shape

# kullanıcı özelince satırlara user lara sütunlara isimleri kesişimlere de rating i yaz
user_movie_df = sample_df.pivot_table(index=["userId"],
                                      columns=["title"],
                                      values="rating")

user_movie_df.shape

#sklam 1 ile 5 arasında
reader = Reader(rating_scale=(1, 5))
# dataframe i python ın istediği metodu kullanarak düzenliyoruz
data = Dataset.load_from_df(sample_df[['userId',
                                       'movieId',
                                       'rating']], reader)

############################################################################
# Adım 2: Modelleme
############################################################################

# makine öğrenmesi modelleri oluştururken test etmek için kullanılır
# data yı rain ve test olarak 2 ye ayırıyoruz
# % 75i  train% 25 i test olacak şekilde böldük

trainset, testset = train_test_split(data, test_size=.25)
svd_model = SVD()
svd_model.fit(trainset) # p ve q yu bulduk burada

# tahminleme yapalım, p ve q yu buldum test setinin değerlerini tahmin edicez
# ve bu tahminler sayesinde tahmin performansını değerlendiricez

predictions = svd_model.test(testset)

# ortalama nekadar hata yaptık tahminleme yaparken
accuracy.rmse(predictions)

# user id 1 , itemid 541 için tahminleme yapalım
svd_model.predict(uid=1.0, iid=541, verbose=True)
# user id 1 , itemid 356 için tahminleme yapalım
svd_model.predict(uid=1.0, iid=356, verbose=True)

# sample df in içinde 1 numaralı kullanıcı için gerçek değerlerine bakalım
# üstte tahmin etmiştik kontrolü sağlayabiliriz
sample_df[sample_df["userId"] == 1]

##############################
# Adım 3: Model Tuning
##############################

# modeli optimize etmeye çalışmak tahmin performansını arttırmaya çalışmaktır.
# modelin dışsal, hiper, kullanıcı tarafından ayarlanabilen parametreler
# n_epochs sayısı iterasyon sayısı kaç defa ağırlıklar güncellenecek
# lr_all: öğrenme oranı
param_grid = {'n_epochs': [5, 10, 20],
              'lr_all': [0.002, 0.005, 0.007]}


# svd method u kullan, param grid dekileri dene,
# hatamı gerçek değerler ile tahmin edilenler arasındaki farkın karelerinin ortalamasını ya da
# o ortalamanın karekökünü al (measures=['rmse', 'mae'])
# 3 katlı çarpraz doğrulama yap verisetini 3 e böl 2 çarpası ile model kur 1 parçası ile test et
# 3 e böldüğün verisetinin kombinasyonlarını yap yani 2li 2li grupla üstekini yap
# test işlemlerinin ortalamasını yap işlemcileri full performansla kullan ( n_jobs=-1,)
# işlemler gerçekleşirken raporlama yap joblib_verbose=True
gs = GridSearchCV(SVD,
                  param_grid,
                  measures=['rmse', 'mae'],
                  cv=3,
                  n_jobs=-1,
                  joblib_verbose=True)
# gs ye verimizi verip fit ediyoruz
gs.fit(data)

# en iyi score u bulalım
gs.best_score['rmse']
gs.best_params['rmse'] # bu sonucu veren en iyi değerler


#############################################################################################
# Adım 4: Final Model ve Tahmin
#############################################################################################
# svd model i dökümanını okuyalım en iyi değerleri bulmuştuk onu değiştirelim
dir(svd_model)
# n_epocs değerini çağıralım, ön tanımlı değeri 20 idi
svd_model.n_epochs
# modele best params dan gelen en iyi argümanları girelim, modeli düzenlemiş olduk
svd_model = SVD(**gs.best_params['rmse'])
# tekrar modelleme yapmadan önce modelin tamamını traintest yapalım verimizi test ettik
# yeni modelimizi tüm veriye uygulayalım  test etmeye gerek kalmadı
data = data.build_full_trainset()
svd_model.fit(data) # modeli fit edelim

# user id si 1 olan item id si 541 için tahmin sonucu isteyelim
svd_model.predict(uid=1.0, iid=541, verbose=True)
# est değeri 4.20 tahmini bu

# gerçek değeri ise: rating i 4 fena bir tahminleme olmamış
sample_df[sample_df["userId"] == 1]





