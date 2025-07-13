# Makine Öğrenmesinde Karar Ağaçları ile Gözetimli Öğrenme Pratikleri
Bu depo, makine öğrenimi alanında karar ağacı sınıflandırıcıları ve gözetimli öğrenme prensipleri üzerine yapılan dört uygulamalı pratik çalışmayı içermektedir. Amacımız, farklı veri setleri üzerinde karar ağaçlarının nasıl eğitildiğini, performanslarının nasıl değerlendirildiğini ve sonuçların nasıl yorumlandığını somut örneklerle göstermektir.

## 1. Pratik: Iris Çiçeği Veri Seti Üzerine Karar Ağacı Sınıflandırması
Bu bölümde, makine öğrenimi dünyasının "Merhaba Dünya" örneği olarak kabul edilen Iris Çiçeği Veri Seti kullanılarak bir karar ağacı sınıflandırıcısı eğitilmiştir.

**Amacı**
Iris çiçeğinin dört farklı özelliğine (çanak yaprağı uzunluğu/genişliği, taç yaprağı uzunluğu/genişliği) bakarak çiçeğin hangi türe (setosa, versicolor, virginica) ait olduğunu doğru bir şekilde tahmin eden bir model geliştirmektir.

**Kullanılan Kütüphaneler**
pandas
numpy
sklearn (özellikle datasets, model_selection, tree, metrics modülleri)
matplotlib.pyplot

**Uygulanan Adımlar**
Veri Seti Yükleme ve Ayırma: Iris veri seti sklearn.datasets modülünden yüklenmiş ve eğitim/test kümelerine ayrılmıştır. (Not: Eğitim seti boyutu bu pratikte %20, test seti %80 olarak ayarlanmıştır.)

Model Eğitimi: DecisionTreeClassifier kullanılarak model, eğitim verileri üzerinde eğitilmiştir.

Performans Değerlendirmesi: Modelin doğruluğu (accuracy_score) ve her bir sınıf için detaylı performans metrikleri (classification_report) hesaplanmıştır.

Karar Ağacı Görselleştirmesi: Oluşturulan karar ağacının yapısı matplotlib ile görselleştirilerek, modelin karar verme mekanizması anlaşılır hale getirilmiştir.


## 2. Pratik: Pima Indians Diyabet Veri Seti Üzerine Karar Ağacı Sınıflandırması
Bu bölümde, tıbbi tanısal ölçümlere dayanarak bir kişinin diyabet olup olmadığını tahmin etme problemi, Pima Indians Diyabet Veri Seti üzerinde incelenmiştir. Bu pratik, daha karmaşık ve dengesiz olabilecek gerçek dünya veri setleriyle çalışmaya bir örnek teşkil eder.

**Amacı**
Belirli sağlık verilerine (hamilelik sayısı, glikoz seviyesi, kan basıncı vb.) dayanarak bir bireyin diyabet hastası olup olmadığını tahmin eden bir sınıflandırma modeli geliştirmektir. Özellikle diyabet teşhisi gibi durumlarda, doğru pozitif ve doğru negatif tahminlerin önemi vurgulanmaktadır.

**Kullanılan Kütüphaneler**
pandas
numpy
sklearn (özellikle tree, metrics, model_selection modülleri)
matplotlib.pyplot
seaborn (görselleştirmeler için)

**Uygulanan Adımlar**
Veri Seti Yükleme ve Ayırma: Pima Indians Diyabet veri seti doğrudan bir URL'den pandas ile okunmuş ve eğitim/test kümelerine ayrılmıştır. stratify=y parametresi kullanılarak sınıf dağılımının dengeli olması sağlanmıştır.

Model Eğitimi: max_depth parametresi ayarlanmış bir DecisionTreeClassifier ile model eğitilmiştir. (Not: Bu pratikte modelin test verisi üzerinde eğitildiği görülmüştür. Genellikle model eğitim verisi üzerinde eğitilir, test verisi üzerinde değerlendirilir. Bu kısım gözden geçirilebilir.)

Performans Değerlendirmesi: Doğruluk skoru, classification_report ve özellikle karmaşıklık matrisi (confusion_matrix) ile model performansı detaylıca incelenmiştir. Tıbbi teşhis senaryolarında Yanlış Negatif (FN) ve Yanlış Pozitif (FP) hataların anlamları üzerine odaklanılmıştır.

Karar Ağacı Görselleştirmesi: Karar ağacı yapısı, özellik isimleri ve sınıf isimleri ile birlikte görselleştirilerek modelin şeffaflığı artırılmıştır.

## 3. Pratik: Meme Kanseri Teşhisinde Karar Ağacı Sınıflandırması

Meme Kanseri Teşhisinde Karar Ağacı Sınıflandırması
Bu proje, Scikit-learn (sklearn) kütüphanesinden alınan Meme Kanseri (Breast Cancer Wisconsin) veri seti üzerinde bir Karar Ağacı Sınıflandırıcısı uygulamasını içermektedir. Amaç, hücre özelliklerine dayanarak tümörleri iyi huylu (benign) veya kötü huylu (malignant) olarak sınıflandırmaktır.

### Proje Detayları & Veri Seti
load_breast_cancer() fonksiyonu ile yüklendi.

Özellikler (X) ve hedef (y) değişkenleri ayrıldı.

Özellik isimleri (feature_names) ve hedef sınıf isimleri (target_names) kullanıldı.

**Veri Bölme**
Veri seti, eğitim ve test kümelerine ayrıldı.

Eğitim seti boyutu %20, test seti boyutu %80 olarak belirlendi ve random_state=42 ile tekrar üretilebilirlik sağlandı.

**Model Eğitimi**
DecisionTreeClassifier kullanılarak bir karar ağacı modeli oluşturuldu.

max_depth=5 olarak ayarlanarak ağacın karmaşıklığı sınırlandırıldı.

Model, eğitim verisi (X_train, y_train) üzerinde eğitildi.

**Model Değerlendirmesi**
Modelin test seti üzerindeki tahminleri (y_pred) yapıldı.

Doğruluk skoru (accuracy_score) hesaplandı.

Sınıflandırma raporu (classification_report) ile kesinlik, duyarlılık ve F1-skoru gibi detaylı metrikler sunuldu.

Karmaşıklık Matrisi (confusion_matrix) oluşturularak görselleştirildi. Bu matris; Doğru Pozitif (TP), Doğru Negatif (TN), Yanlış Pozitif (FP) ve Yanlış Negatif (FN) değerlerini gösterir, özellikle tıbbi teşhislerde hata türlerinin analizi için önemlidir.

**Görselleştirme**
Karmaşıklık Matrisi seaborn.heatmap ile görselleştirilerek modelin tahmin performansı netleştirildi.

Eğitilen Karar Ağacı plot_tree fonksiyonu ile detaylıca görselleştirilerek, modelin hangi özelliklere göre karar aldığı anlaşıldı.


## 4. Pratik: Finans Sektöründe Kredi Riski Sınıflandırması: Rastgele Ormanlar
Bu depo, makine öğreniminin finans sektöründeki uygulamalarından kredi riski sınıflandırmasını ele alır. Rastgele Ormanlar modelini kullanarak, kredi başvuru sahiplerinin risk seviyelerini tahmin etmeyi hedefleriz. Bu pratik; veri ön işleme, hiperparametre optimizasyonu ve detaylı model değerlendirme adımlarını içerir.

### Proje Amacı ve Veri Seti
Amacımız, German Credit Data setini kullanarak kredi başvuru sahiplerinin riskini (iyi/kötü) sınıflandırmaktır. Bu veri seti, çeşitli finansal ve demografik özellikler barındırır.

**Uygulanan Adımlar**
**1. Veri Hazırlığı**
German Credit Data yüklendi.

Hedef değişken 1 (İyi Kredi Riski) ve 0 (Kötü Kredi Riski) olarak dönüştürüldü.

Veri, eğitim ve test kümelerine ayrıldı (%80 eğitim / %20 test, sınıf dağılımı korunarak).

**2. Veri Ön İşleme (Pipeline ve One-Hot Encoding)**
Kategorik özellikler OneHotEncoder ile sayısal formata çevrildi.

Ön işleme adımları ve Rastgele Orman sınıflandırıcısı bir Pipeline içinde birleştirildi.

**3. Model Eğitimi ve Hiperparametre Ayarı (GridSearchCV)**
RandomForestClassifier model olarak seçildi.

Model performansı için GridSearchCV ile hiperparametre ayarı yapıldı.

En iyi model, ROC AUC skoruna göre belirlendi.

**4. Model Performansı Değerlendirmesi**
Ayarlanmış modelin performansı test seti üzerinde değerlendirildi:

- Doğruluk Skoru
- Sınıflandırma Raporu
- Karmaşıklık Matrisi: Özellikle Yanlış Pozitif (FP) ve Yanlış Negatif (FN) hataları incelendi.
- ROC Eğrisi ve AUC Skoru

#### Finans Sektörü İçin Çıkarımlar
- Risk Yönetimi: Kredi riski tahminleri, finansal kayıpları azaltmada ve risk yönetimini güçlendirmede etkilidir.

- Optimizasyon: Hiperparametre ayarı, modelin gerçek dünya verilerindeki performansını optimize eder.

- Hata Maliyeti: Finansal modellerde FP ve FN hatalarının farklı maliyetleri olduğunu anlamak önemlidir.

- Yorumlanabilirlik: Karar ağaçları, finansal kararların şeffaflığını artırır.
