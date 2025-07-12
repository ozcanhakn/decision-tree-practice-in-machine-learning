from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Adım veri setini yükle X ve y 

cancer = load_breast_cancer()

X = cancer.data # features
y = cancer.target # target

feature_names = cancer.feature_names
target_names = cancer.target_names

# 2. adım veri setini ayır

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)

print(f"Eğitim seti boyutu (X_train): {X_train.shape}")
print(f"Test seti boyutu (X_test): {X_test.shape}")
print("\nÖzellik isimleri: ")
print(feature_names)
print("\nHedef sınıf isimleri")
print(target_names)


# 3. adım modeli eğit

dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_classifier.fit(X_train, y_train)
print("\nKarar ağacı başarıyla eğitildi")




# 4. doğruluk

y_pred = dt_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModelin Doğruluk Skoru: {accuracy:.2f}")

print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=target_names))


# 5. adım görselleştir

# Karmaşıklık Matrisini görselleştir
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.title('Karmaşıklık Matrisi')
plt.show()

print("\nKarmaşıklık Matrisi:")
print(cm)
print(f"Doğru Pozitif (TP - Benign doğru tahmin): {cm[1, 1]}") # Gerçekte 1 (benign) olan ve 1 tahmin edilenler
print(f"Doğru Negatif (TN - Malign doğru tahmin): {cm[0, 0]}") # Gerçekte 0 (malign) olan ve 0 tahmin edilenler
print(f"Yanlış Pozitif (FP - Malign yanlışlıkla Benign): {cm[0, 1]}") # Gerçekte 0 (malign) olan ve 1 tahmin edilenler (Tip I Hata)
print(f"Yanlış Negatif (FN - Benign yanlışlıkla Malign): {cm[1, 0]}") # Gerçekte 1 (benign) olan ve 0 tahmin edilenler (Tip II Hata)

# Karar Ağacını görselleştir
plt.figure(figsize=(25, 18))
plot_tree(dt_classifier,
          feature_names=feature_names, # Veri setinden gelen özellik isimleri
          class_names=target_names,    # Veri setinden gelen sınıf isimleri
          filled=True,
          rounded=True,
          fontsize=10,
          impurity=False # Gini veya Entropy değerlerini gösterme
         )
plt.title("Meme Kanseri Teşhis Veri Seti Üzerine Eğitilmiş Karar Ağacı", fontsize=20)
plt.show()



