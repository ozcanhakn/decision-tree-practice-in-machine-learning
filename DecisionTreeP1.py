import pandas as pd
import numpy as numpy
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


# Öncelikle veri setini doğrudan bir url'den çekelim
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(url, names=names)


X = data.iloc[:, :-1] # son sütün hariç hepsi özellik (features)
y = data.iloc[:, -1] # son sütün hedef değişkenimiz (target)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=42, stratify=y)
# stratify=y parametresi, sınıf dağılımının eğitim ve test kümelerinde benzer olmasını sağlar.
# Bu, dengesiz veri setlerinde önemlidir.

print(f"Eğitim seti boyutu (X_train): {X_train.shape}")
print(f"Test seti boyutu (X_test): {X_test.shape}")
print("\nVeri setinin ilk 5 satırı:")
print(data.head())


dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_classifier.fit(X_test, y_test)

y_pred = dt_classifier.predict(X_test)

accuracy=accuracy_score(y_test, y_pred)
print(f"Doğruluk:",accuracy)

# Sınıflandırma raporunu göster
print("\nSınıflandırma Raporu:")
# target_names parametresini belirtmek için 0 ve 1'in ne anlama geldiğini tanımlayalım
target_names_diabetes = ['No Diabetes (0)', 'Diabetes (1)']
print(classification_report(y_test, y_pred, target_names=target_names_diabetes))

# Karmaşıklık Matrisini görselleştir
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names_diabetes, yticklabels=target_names_diabetes)
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.title('Karmaşıklık Matrisi')
plt.show()

print("\nKarmaşıklık Matrisi:")
print(cm)
print(f"Doğru Pozitif (TP): {cm[1, 1]}") # Gerçekte 1 olan ve 1 tahmin edilenler
print(f"Doğru Negatif (TN): {cm[0, 0]}") # Gerçekte 0 olan ve 0 tahmin edilenler
print(f"Yanlış Pozitif (FP): {cm[0, 1]}") # Gerçekte 0 olan ve 1 tahmin edilenler (Tip I Hata)
print(f"Yanlış Negatif (FN): {cm[1, 0]}") # Gerçekte 1 olan ve 0 tahmin edil



# Karar Ağacını görselleştir
plt.figure(figsize=(20, 15))
plot_tree(dt_classifier,
          feature_names=X.columns, # Pandas DataFrame'den sütun isimlerini alıyoruz
          class_names=target_names_diabetes,
          filled=True,
          rounded=True,
          fontsize=10,
          impurity=False # Gini veya Entropy değerlerini gösterme
         )
plt.title("Pima Indians Diyabet Veri Seti Üzerine Eğitilmiş Karar Ağacı", fontsize=18)
plt.show()






