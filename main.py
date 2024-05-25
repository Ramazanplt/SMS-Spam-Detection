#Gerekli kütüphaneleri import edelim
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#Veri setini yükleyelim
df = pd.read_csv('spam.csv', encoding='latin-1')

#İlk birkaç satırı görüntüle
print(df.head())

#Kolon isimlerini kontrol et
print(df.columns)

#Veri türlerini kontrol et
print(df.info())

#Gereksiz kolonları kaldır
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

#Etiket ve mesaj sütunlarını kontrol et ve Etiketleri sayısal değerlere dönüştür
print(df.head())
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

#Etiket ve mesaj sütunlarını kontrol et
print(df.head())

#NLTK paketlerini indirelim
nltk.download('punkt')
nltk.download('stopwords')

#Metin temizleme fonksiyonu
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

df['message'] = df['message'].apply(preprocess_text)
print(df.head())

#TF-IDF vektörizeri oluştur ve Mesajları vektörize et
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['message']).toarray()
y = df['label'].values

#Özellik ve etiketlerin boyutlarını kontrol et ve Veri setini böl
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Bölünmüş veri setlerinin boyutlarını kontrol et
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


#Modeli oluştur,eğit ve test seti ile değerlendir
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Doğruluk skoru
accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk Skoru: {accuracy}")

#Sınıflandırma raporu
print(classification_report(y_test, y_pred))

#Karışıklık matrisi
print(confusion_matrix(y_test, y_pred))

