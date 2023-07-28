###################################################################
# ASSOCIATION RULE LEARNING (BİRLİKTELİK KURALI ÖĞRENİMİ)
####################################################################

# 1. Veri Ön İşleme
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
# 3. Birliktelik Kurallarının Çıkarılması
# 4. Çalışmanın Scriptini Hazırlama
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak

####################################################################
# 1. Veri Ön İşleme
####################################################################

# !pip install mlxtend
import pandas as pd
pd.set_option("display.max_columns", None) # max column sayısı olmasın, tüm sütunları göster
pd.set_option("display.max_rows", None) # max satır sayısı olmasın, tüm satırları göster
pd.set_option("display.width", 500) # sütunları gösterirken alt alta gösterme genişlikte göster
pd.set_option("display.expand_frame_repr", False) # çıktının tek satırda olmasını sağlar
from mlxtend.frequent_patterns import apriori, association_rules

# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

df_ = pd.read_excel("Datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
# hata alınırsa yapılacaklar:

# pip install openpyxl
# df_ = pd.read_excel("datasets/online_retail_II.xlsx",
#                     sheet_name="Year 2010-2011", engine="openpyxl")

df.describe().T
df.isnull().sum()
df.shape

# eksik gözlemler gitsin - dropna
# min değerlerde eksiler olmasın - quantity ve price > 0
# iade olan ürünler görünmesin - invoice da c olanlar olmasın

#def retail_data_prep(dataframe):
    #dataframe.dropna(inplace=True)
    #dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    #dataframe =dataframe[dataframe["Quantity"] > 0]
    #dataframe = dataframe[dataframe["Price"] > 0]
    #return dataframe

# df = retail_data_prep(df)

# eşik değerleri getiren fonk:
# girilen değişkenin % 1 çeyrek değerini olustur q1 olsun
# girilen değişkenin % 99 çeyrek değerini oluştur q3 olsun
# iqr = q3 - q1
# up_limit = q3 + 1.5 * iqr
# low_limit = q1 - 1.5 * iqr
# uyarı: normalde q1 için % 25, q3 için %75 kullanılır
# şahsi tercihimiz çok fazla aykırı olanları çıkararak veriyi azıcık kırpmak
def outlier_thresholds(dataframe, veriable):
    quartile1 = dataframe[veriable].quantile(0.01)
    quartile3 = dataframe[veriable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# baskılama için:
# outlier dan bulunan alt ve üst limiti al
# dataframe de ilgili değişkende belirlenen alt limitten aşağıda olanları getir bunlara low limit ile değiştir
# dataframe de ilgili değişkende belirlenen üst limitten yukarıda olanları getir bunlara up limit ile değiştir

def replace_with_thresholds(dataframe, veriable):
    low_limit, up_limit = outlier_thresholds(dataframe, veriable)
    dataframe.loc[(dataframe[veriable] < low_limit), veriable] = low_limit
    dataframe.loc[(dataframe[veriable] > up_limit), veriable] = up_limit

# retail_data_prep i quantity ve price değişkenlerini replace_with_thresholds a uygulayarak tekrar yazdık

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)
df.isnull().sum()
df.describe().T


#############################################################################################################
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
#############################################################################################################
# satırda invoice, sütunda product olsun istiyoruz
# invoice sepet muamelesi görecek, sütunda ise bir faturada belirli bir ürünün olup olmaması 1-0 ile gösterilsin
# satırlarda kullanıcı id leri değil invoice id leri var
df.head()

# Description   NINE DRAWER OFFICE TIDY   SET 2 TEA TOWELS I LOVE LONDON    SPACEBOY BABY GIFT SET
# Invoice
# 536370                              0                                 1                       0
# 536852                              1                                 0                       1
# 536974                              0                                 0                       0
# 537065                              1                                 0                       0
# 537463                              0                                 0                       1


# işlemlerin hızlı gerçekleşmesi ve herkes tarafaından uygulanabilmesi için
# verisetini belirli bir ülkeye indirgeyerek yapıyoruz Fransa ya göre bakıyoruz
# Fransa müşterilerinin birliktelik kurallarını türetiyoruz
# Hiç müşterisi olmayan bir ülkeye pazara gircekse o ülkeye yakın bir ülke seçilir davranış yakınlığından faydalanır
df_fr = df[df["Country"] == "France"]

# invoice e göre grupby a al; önce faturaya göre kır sonra tanıma göre kır ve quantity nin sum ını al
# yani hangi üründen kaçar tane old bul
df_fr.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).head(20)

# unstack burada pivot eder yani isimleri değişken isimlerine çevir index based seçim yap
# satır ve sütunlardan 5 er tane getir
df_fr.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]

# applymap ile tüm gözlemleri gez ve 1 de eğer x > 0 ise
df_fr.groupby(["Invoice", "Description"]). \
    agg({"Quantity": "sum"}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]

# fonksiyon yazalım:
# yukarıdaki işlemi stockcode a göre yap
# eğer id true ise yani id verilirse stockcode göre yap
# id olmazsa yani öntanımlı değeri olan false göre ise description a göre yap
def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(["Invoice", "StockCode"])["Quantity"].sum().unstack().fillna(0) > 0
            #applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(["Invoice", "Description"])["Quantity"].sum().unstack().fillna(0) > 0
            #applymap(lambda x: 1 if x > 0 else 0)


fr_inv_pro_df = create_invoice_product_df(df_fr)

fr_inv_pro_df = create_invoice_product_df(df_fr, id=True)

# verilen dataframe den stockcode değişkeni seçilecek sorgulanmak istenen stockid girilince
# bunun decription ı gelecek
# çıktının içerisindeki değerlerden sadece string değere erişmek istediğimiz için valus[0] diyoruz
# ve bunu listeye çeviriyoruz
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


check_id(df_fr, 10120)

######################################################################################################
# 3. Birliktelik Kurallarının Çıkarılması
######################################################################################################
# apriori ile olası tüm ürün birlikteliklerinin support yani olasılıklarını bul
# dataframe i ver min support değeri varsa eşik değeri ver
# kullanmış old dataframe deki değişkenlerin ismini kullanmak istiyorsan true de
frequent_itemsets = apriori(fr_inv_pro_df,
                            min_support=0.01,
                            use_colnames=True)

# support değerine göre azalan şekilde sırala
frequent_itemsets.sort_values("support", ascending=False)

# metrik olarak support değeri giriliyor
# 0.01 min değer olarak giriliyor
rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

# support değeri, confidence ve lift değerleri belli değer üzerindekiler
rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]

check_id(df_fr, 21086)

rules[(rules["support"]>0.05) & ( rules["confidence"]>0.1) & (rules["lift"]>5)]. \
    sort_values("confidence", ascending=False)

##############################################################################################
# 4. Çalışmanın Scriptini Hazırlama
##############################################################################################

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0) > 0
            #applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0) > 0
            #applymap(lambda x: 1 if x > 0 else 0)


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


def create_rules(dataframe, id=True, country="France"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

df = df_.copy()

df = retail_data_prep(df)
rules = create_rules(df)

rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)].sort_values("confidence", ascending=False)

########################################################################################################################
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak
########################################################################################################################

# Örnek:
# Kullanıcı örnek ürün id: 22492
# 22492 ürününü almaya çalışan müşteriye tavsiye edilmesi gereken ürünleri bulmak için:
product_id = 22492
check_id(df, product_id)
sorted_rules = rules.sort_values("lift", ascending=False)

# tavsiye edilmesi gereken ürün listesi
recommendation_list = []

for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

recommendation_list[0] # ilk tavsiye edilmesi gereken ürün
recommendation_list[0:3] # tavsiye edilmesi gereken ilk 3 ürün

# önerilen ürünlerden birinin ne old görmek için:
check_id(df, 22326)

# tavsiye için fonks yazalım:
# kuralları , product_id öneri yapılması gereken, kaç tavsiyede bulunulması gerektiğini gir
# lift değerine göre azalan sırala
# bulduğun ürünleri listeye ekle
def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]
arl_recommender(rules, 22492, 1) # 22492 ürünü için 1 tane tavsiye ver
arl_recommender(rules, 22492, 2) # 22492 ürünü için 2 tane tavsiye ver
arl_recommender(rules, 22492, 3)# 22492 ürünü için 3 tane tavsiye ver





