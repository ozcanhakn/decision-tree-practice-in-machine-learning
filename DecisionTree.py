import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# iris veri setini yükle
iris = load_iris()
X = iris.data # features
y = iris.target # target

# Özellik isimleri ve hedef sınıf isimleri
feature_names = iris.feature_names
target_name = iris.target_names

# Veri setini eğitim ve test setlerine ayırma
# %80 eğitim ve %20 test verisi

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)

print(f"Eğitim seti boyutu (X train): {X_train.shape}")
print(f"Test seti boyutu (X_test): {X_test.shape}")

# Karar ağacı sınıflandırıcısı oluştur
# random_state sabit tutularak sonuçların tekrarlanabilirliği için
dt_classifier = DecisionTreeClassifier(random_state=42)

# Modeli eğitim veri setiyle eğit
dt_classifier.fit(X_train, y_train)
print("Karar ağacı başarıyla eğitildi")

# Modelin performansını değerlendirme
y_pred = dt_classifier.predict(X_test)

# Doğruluk skorunu hesapla
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModelin Doğruluk Skoru: {accuracy:.2f}")

# Sınıflandırma raporunu göster
print("\nSınıflandırma raporu:")
print(classification_report(y_test, y_pred, target_names=target_name))

# Karar Ağacını görselleştirme
plt.figure(figsize=(15,10))
plot_tree(dt_classifier,
          feature_names=feature_names,
          class_names=target_name,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("İris veri seti üzerinde eğitilmiş karar ağacı", fontsize=16)
plt.show()
















