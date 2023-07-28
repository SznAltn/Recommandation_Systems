# Recommandation_Systems
Association Rule Learning - Content Based Recommendation - Item Based Collaborative Filtering - User Based Collaborative Filtering - Model Based Matrix Factorization
- **Association Rule Learning**
- **Content Based Recommendation**
- **Item Based Collaborative Filtering**
- **User Based Collaborative Filtering**
- **Model Based Matrix Factorization**

**Amacımız;** bol olan olası içerik, ürün, hizmet, film ve benzeri durumları kullanıcının tercih edebileceği ve seçimleri ile örtüşebilecek şekilde yapabilmektir. 

**Recommendation Systems:** Kullanıcıları bazı teknikleri kullanarak ürün ya da hizmet önermek, tavsiye etmektir.

## **Recommendation Systems:**

**Simple Recommender Systems (Basit Tavsiye Sistemleri):** 

- İş bilgisi ya da basit tekniklerle yapılan genel öneriler
- Kategorinin en yüksek puanlıları, trend olanlar, efsaneler …

**Association Rule Learning (Birliktelik Kuralı Öğrenimi) :** Birliktelik analizi ile öğrenilen kurallara göre ürün önerileri. Çok sık bir şekilde birlikte satın alınan ürünlerin olasılıklarını çıkarır ve bunlara göre belirli öneriler yapma imkanı sağlar.

**Content Based Recommendation (İçerik Temelli Filtreleme):** Ürün benzerliğine göre öneriler yapılan uzaklık temelli yöntemler 

**Collaborative Filtering (İşbirlikçi Filtreleme):** Topluluğun kullanıcı ya da ürün bazında ortak kanaatlerini yansıtan yöntemler.

- **User Based**
- **Item Based**
- **Model Based (Matrix Factorization)**

# 1. Association  Ruled Recommendation System (Birliktelik Kurallı Tavsiye sistemi)

Veri içerisindeki örüntüleri (pattern, ilişki, yapı) bulmak için kullanılan kural tabanlı bir makine öğrenmesi tekniğidir.

### Apriori Algorithm

Sepet analizi yöntemidir, ürün birlikteliklerini ortaya çıkarmak için kullanılır. Üç temel teknik vardır.

- Support Değeri ( x ve y nin birlikte görülme olasılığı)

Support (x, y) = Freq(x, y) / N 

x ve y nin birlikte görülme frekansı / tüm işlemler

- Confidence Değeri ( x satın alındığında y nin satın alınma olasılığı)

Confidence (x, y) = Freq (x, y) / Freq (x)

x ve y nin birlikte görülme frekansı / x in frekansı

- Lift Değeri (x satın alındığında y nin satın alınma olasılığı)

Lift = Support (x, y) / (Support (x) * Support (y))

x satın alındığında y nin satın alınma olasılığı lift kadar artar.

# 2. Content Based Filtering (İçerik Temelli Filtreleme)

Ürün içeriklerinin benzerlikleri üzerinden tavsiyeler geliştirilir.  Ürün(film, kitap, online alışveriş yapılan bir ürün …) metabilgileri bilgileri üzerinden benzerlikler hesaplanır ve ilgili ürün ile en alakalılar bulunarak tavsiye edilir. Film örneği üzerinden gidilirse; filmin konusu, kategorisi, oyuncu kadrosu, yönetmeni, beğenilme oranı gibi bilgilere metabilgiler denir. Ürünlerin birbirine yakınlığını ölçmek için:

- Metinleri matematiksel olarak temsil et (Metinleri vektörleştir)
- Benzerlik Hesapla

### Metinleri matematiksel olarak temsil etme (Metinleri vektörleştirme)

Matematiksel olarak ölçülebilir forma getirmektir. Yaygın kullanılan yollar:

- Count Vector (Word Count)
- TF-IDF

**Metinleri vektörleştirme** yaparken **ö**rneğin, bir film öneri tavsiyesi yapmak için ölkid uzaklıkları hesaplanır ve uzaklığı en küçük olan filmler birbirlerine en çok benzeyen filmlerdir deriz. Kullanıcının izlediği filme uzaklıkları en yakın olan filmleri tavsiye ederiz. Öklid uzaklığı için matematikte iki nokta arasındaki uzaklık formülüne bakabilirsiniz.

**Cosine Similarity (Kosinüs Benzerliği):**  A ve B vektörünün arasındaki açıyı bulmak için, A ve B çarpımı A ve B nin normlarının çarpımına bölünür. (Matematikte iki vektör arasındaki açının kosinüsünü bulma yöntemi)

### Count Vector (Word Count, Sayım Vektörü):

Bir ürün hakkındaki bilgilerde (yorum, açıklama gibi metinsel bilgi) bulunan tüm eşsiz terimleri sütunlara, tüm dökümanları da satırlara yerleştirilir.

Terimlerin dökümanlarda geçme frekanslarını hücrelere yerleştirilir. 

Sonrasında cosinüs benzerliği, öklid uzaklıı gibi yöntemler kullanılarak hesaplamalar yapılır.

### TF-IDF

Kelimelerin hem kendi metinlerinde hem de tüm corpusta  (odaklanılan veride) geçme frekansı üzerinden normalizasyon işlemi yapar. Count Vector yönteminden ortaya çıkabilecek yanlılıkları da giderir. Aşamaları:

- Count Vector hesaplanır (Kelimelerin her bir dökümandaki sayısı)
- TF-Term Frequency hesaplanır (t teriminin ilgili dökümandaki frekansı / dökümandaki toplam terim sayısı)
- IDF-Inverse Document Frequency hesaplanır.(1 + ln((toplam döküman sayısı + 1) / (içinde t terimi olan döküman sayısı + 1)))
- TF * IDF hesaplanır
- LF Normalizasyonu yapılır.(Satırların kareleri toplamının karekökünü bul, ilgili satırdaki tüm hücreleri bulduğun değere böl)

# 3. Item Based Collaborative Filtering (İşbirlikçi Filtreleme)

- **User Based Collaborative Filtering**
- **Item Based Collaborative Filtering**
- **Model Based (Matrix Factorization) Collaborative Filtering**

## **Item Based Collaborative Filtering (Öğe Tabanlı İşbirlikçi Filtreleme)**

Item benzerliği üzerinden öneriler yapılır. İzlenen bir filme ya da satın alınan bir ürüne beğenilme yapısına benzer beğenilme yapısı gösteren ürün ya da film aranır. Beğenilme - rating puanlarına göre korelasyon hesaplayarak benzer filmleri önerir.

## User **Based Collaborative Filtering (Kullanıcı Tabanlı İşbirlikçi Filtreleme)**

User (kullanıcı) bennzerlikleri üzerinden öneriler yapılır.

## Model **Based Collaborative Filtering (Matrix Factorization) (Model Tabanlı İşbirlikçi Filtreleme)**

Matrix Factorization yöntemi kullanılırken matrislerde boşlukları doldurmak için user lar ve movi ler için var old varsayılan latent feature ların ağırlıkları var olan veri üzerinden bulunur ve bu ağırlıklar ile var olmayan gözlemler için tahmin yapılır. İzlenecek yol:

- User-Item matrisini 2 tane daha az boyutlu matrislere ayrıştırır.
- 2 matristen User-Item matrisine gidişin latent factor ler ile gerçekleştiği varsayımında bulunur.
- Dolu olan gözlemler üzerinden latent factor lerin ağırlıklarını bulur.
- Bulunan ağırlıklar ile boş olan gözlemler doldurulur.
- Rating matrisinin iki factor matrisin çarpımı (dot product) ile oluştuğu varsayılır.
- Factor matrisler: user latent factors, movie latent factors
- Latent factors veya latent features: gizli faktörler ya da değişkenler
- Kullanıcıların ve filmlerin latent feature lar için skorlara sahip olduğu düşünülür.
- Bu ağırlıklar (skorlar) önce var olan veri üzerinden bulunur ve sonra Blank bölümler bu ağırlıklara göre doldurulur.

Latent Faktörlere örnek vermek istersek: film veri setini inceliyorsak, komedi, korku, macera, aksiyon, gerilim, gençlik, oyuncu kadrosu, yönetmen…

Matristeki user factors ve movie factors leri bulmak için:

- Var olan değerler üzerinden iteratif şekilde tüm p ve q lar bulunur ve sonra kullanılır.
- Başlangıçta rastgele p ve q değerleri ile rating matrisindeki değerler tahmin edilmeye çalışılır.
- Her iterasyonda hatalı tahminler düzenlenerek rating matristeki değerlere yaklaşılmaya çalışılır.
- Örneğin bir iterasyonda 5 e 3 deniyorsa sonrakinde 4 sonrakinde 5 denerek iyileştirme yapılır.
- Böylece belirli bir iterasyon sonucunda p ve q matrisleri doldurulmuş olur.
- Var olan p ve q lara göre boş gözlemler için tahmin yapılır.

**p ve q değerlerini değiştirmek için:**

**Gradient Descent Yöntemi:** fonkiyon minimizasyonu için kullanılan bir optimizasyon yöntemidir.

Gradyanın negatifi olarak tanımlanan en dik iniş yönünde iteratif olarak parametre değerlerini güncelleyerek ilgili fonksiyonun minimum değerini verecek parametreleri bulunur.
