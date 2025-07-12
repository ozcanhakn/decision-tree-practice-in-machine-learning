import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

column_names = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_commitment", "personal_status",
    "other_parties", "residence_since", "property_magnitude", "age",
    "other_payment_plans", "housing", "existing_credits", "job", "num_dependents",
    "own_telephone", "foreign_worker", "class" # 1: Good Risk, 2: Bad Risk
]
data = pd.read_csv(url, sep=' ', names=column_names)

# Hedef değişkeni 0 ve 1 olarak yeniden düzenle (1: İyi Kredi, 0: Kötü Kredi)
# Orijinalde 1=Good, 2=Bad idi. Biz 1=Good, 0=Bad yapalım.
data['class'] = data['class'].map({1: 1, 2: 0})

# Özellikleri (X) ve hedef değişkeni (y) ayır
X = data.drop('class', axis=1)
y = data['class']


# Kategorik ve Sayısal Sütunları Belirle
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42, stratify=y)

print(f"Eğitim seti boyutu (X_train): {X_train.shape}")
print(f"Test seti boyutu (X_test): {X_test.shape}")
print("\nVeri setinin ilk 5 satırı:")
print(data.head())
print("\nKategorik Özellikler:")
print(categorical_features.tolist())
print("\nSayısal Özellikler:")
print(numerical_features.tolist())


# Ön işleme adımlarını tanımla
# OneHotEncoder ile kategorik özellikleri dönüştür
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Diğer (sayısal) sütunları olduğu gibi bırak
)

# Pipeline oluştur: Önişleme -> Rastgele Orman Modeli
# Bu pipeline, model eğitiminde ve tahmin sırasında otomatik olarak veri ön işlemeyi uygulayacaktır.
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])

print("\nVeri ön işleme pipeline'ı başarıyla oluşturuldu.")

# Ayarlamak istediğimiz hiperparametreler ve denenecek değer aralıkları
# Pipeline'daki adımların parametrelerini 'adım_adı__parametre_adı' şeklinde belirtiriz.
param_grid = {
    'classifier__n_estimators': [100, 200, 300], # Ormandaki ağaç sayısı
    'classifier__max_depth': [None, 10, 20], # Her bir ağacın maksimum derinliği
    'classifier__min_samples_split': [2, 5, 10], # Bir düğümü bölmek için gereken minimum örnek sayısı
    'classifier__min_samples_leaf': [1, 2, 4] # Bir yaprak düğümde olması gereken minimum örnek sayısı
}

# GridSearchCV nesnesini oluştur
# Scoring: Finansal risk modellerinde sadece doğruluk yeterli olmayabilir.
# AUC-ROC (Area Under the Receiver Operating Characteristic Curve) gibi metrikler,
# sınıf dengesizliği olan durumlarda veya sınıflandırıcının iyi/kötü ayrım gücünü ölçmek için daha iyidir.
grid_search = GridSearchCV(estimator=model_pipeline, # Pipeline'ı tahminci olarak veriyoruz
                           param_grid=param_grid,
                           cv=5, # 5 katlı çapraz doğrulama
                           scoring='roc_auc', # ROC AUC skoruna göre en iyi modeli seç
                           n_jobs=-1, # Tüm işlemci çekirdeklerini kullan
                           verbose=2) # Detaylı ilerleme bilgisini göster

# GridSearch'ü eğitim verileri üzerinde çalıştır
print("\nHiperparametre ayarı başlatılıyor... Bu biraz zaman alabilir.")
grid_search.fit(X_train, y_train)

print(f"\nEn iyi parametreler: {grid_search.best_params_}")
print(f"En iyi çapraz doğrulama ROC AUC skoru: {grid_search.best_score_:.4f}")

# En iyi modeli al
best_model = grid_search.best_estimator_



# Ayarlanmış model ile test seti üzerinde tahmin yap
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1] # ROC eğrisi için olasılıklar

# Doğruluk skoru
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAyarlanmış Modelin Test Doğruluk Skoru: {accuracy:.4f}")

# Sınıflandırma Raporu
print("\nAyarlanmış Modelin Sınıflandırma Raporu:")
# target_names: 1=Good Credit, 0=Bad Credit
target_names_credit = ['Bad Credit (0)', 'Good Credit (1)']
print(classification_report(y_test, y_pred, target_names=target_names_credit))

# Karmaşıklık Matrisi
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names_credit, yticklabels=target_names_credit)
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.title('Ayarlanmış Rastgele Orman Modelinin Karmaşıklık Matrisi', fontsize=16)
plt.show()

print("\nKarmaşıklık Matrisi:")
print(cm)
print(f"Doğru Pozitif (TP - Gerçekte İyi, Tahmin Edilen İyi): {cm[1, 1]}")
print(f"Doğru Negatif (TN - Gerçekte Kötü, Tahmin Edilen Kötü): {cm[0, 0]}")
print(f"Yanlış Pozitif (FP - Gerçekte Kötü, Tahmin Edilen İyi - Tip I Hata): {cm[0, 1]}")
print(f"Yanlış Negatif (FN - Gerçekte İyi, Tahmin Edilen Kötü - Tip II Hata): {cm[1, 0]}")

# ROC Eğrisi ve AUC Skoru
# ROC AUC, bir sınıflandırıcının ayrım gücünün iyi bir ölçüsüdür.
# Özellikle finansal risk modellerinde hassasiyet ve duyarlılık tradeoff'unu anlamak için önemlidir.
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Yanlış Pozitif Oranı (False Positive Rate)')
plt.ylabel('Doğru Pozitif Oranı (True Positive Rate)')
plt.title('Alıcı İşletim Karakteristiği (ROC) Eğrisi', fontsize=16)
plt.legend(loc="lower right")
plt.show()

print(f"\nROC AUC Skoru: {roc_auc:.4f}")













