
#########################
# İş Problemi
#########################

# Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
# Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca
# ulaşılmasını sağlamaktadır.
# Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak
# Association Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.


#########################
# Veri Seti
#########################
#Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır.
# Alınan her hizmetin tarih ve saat bilgisini içermektedir.

# UserId: Müşteri numarası
# ServiceId: Her kategoriye ait anonimleştirilmiş servislerdir.
# (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
# Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
# (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
# CategoryId: Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
# CreateDate: Hizmetin satın alındığı tarih




#######################################################################################################################
# GÖREV 1: Veriyi Hazırlama
#######################################################################################################################

# Adım 1: armut_data.csv dosyasınız okutunuz.

# !pip install mlxtend
import pandas as pd
pd.set_option("display.max_columns", None) # max column sayısı olmasın, tüm sütunları göster
pd.set_option("display.max_rows", None) # max satır sayısı olmasın, tüm satırları göster
pd.set_option("display.width", 500) # sütunları gösterirken alt alta gösterme genişlikte göster
pd.set_option("display.expand_frame_repr", False) # çıktının tek satırda olmasını sağlar
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_csv("Datasets/armut_data.csv")
df = df_.copy()
df.head()


# Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.
# ServiceID ve CategoryID'yi "_" ile birleştirerek hizmetleri temsil edecek yeni bir değişken oluşturunuz.

# 1. yol:
df["hizmet"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)
df.head(15)

# 2. yol merge ile:


# Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır.
# Association Rule Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir.
# Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir. Örneğin; 7256 id'li müşteri 2017'in 8.ayında aldığı 9_4, 46_4 hizmetleri bir sepeti;
# 2017’in 10.ayında aldığı  9_4, 38_4  hizmetleri başka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile tanımlanması gerekmektedir.
# Bunun için öncelikle sadece yıl ve ay içeren yeni bir date değişkeni oluşturunuz. UserID ve yeni oluşturduğunuz date değişkenini "_"
# ile birleştirirek ID adında yeni bir değişkene atayınız.


df["New_Date"] = pd.to_datetime(df['CreateDate'], format='%Y-%m').dt.to_period('M')
df.head(15)

df["SepetID"] = df["UserId"].astype(str) + "_" + df["New_Date"].astype(str)

df.head(15)


#############################################################################################################
# GÖREV 2: Birliktelik Kuralları Üretiniz
#############################################################################################################

# Adım 1: Aşağıdaki gibi sepet hizmet pivot table’i oluşturunuz.

# Hizmet         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# SepetID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..


df.groupby(["SepetID", "hizmet"]).agg({"New_Date": "sum"}).unstack().fillna(0).head(10)


# fonks olarak yazalım:
def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(["SepetID", "hizmet"])["New_Date"].sum().unstack().fillna(0) > 0

    else:
        return dataframe.groupby(["SepetID", "hizmet"])["New_Date"].sum().unstack().fillna(0) > 0

fr_inv_pro_df = create_invoice_product_df(df)

fr_inv_pro_df = create_invoice_product_df(df, id=True)

fr_inv_pro_df.head(10)

# Adım 2: Birliktelik kurallarını oluşturunuz.

requent_itemsets = apriori(fr_inv_pro_df,
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



rules[(rules["support"]>0.05) & ( rules["confidence"]>0.1) & (rules["lift"]>5)]. \
    sort_values("confidence", ascending=False)



#Adım 3: arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.


def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, hizmet in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]
arl_recommender(rules, "2_0", 3) # 22492 ürünü için 1 tane tavsiye ver

# check_id(df_fr, 21086)

df.loc[df["hizmet"] == "2_0"].sort_values("CreateDate", ascending=False)

df["CreateDate"].max() #Timestamp('2018-08-06 16:04:00')

#       UserId  ServiceId  CategoryId          CreateDate Hizmet        SepetID
#162519   10591          2           0 2018-08-06 14:43:00    2_0  10591_2018-08

rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0]).astype("unicode")
rules.loc[rules["antecedents"] == "2_0"]

recommendation_list = [rules["consequents"].iloc[i] for i, antecedent in enumerate(rules["antecedents"]) if
                       antecedent == "2_0"]
recommendation_list

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

arl_recommender(rules, 2_0, 5)

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


