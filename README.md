# Makine Öğrenmesinde Karar Ağaçları ile Gözetimli Öğrenme Pratikleri
Bu depo, makine öğrenimi alanında karar ağacı sınıflandırıcıları ve gözetimli öğrenme prensipleri üzerine yapılan iki uygulamalı pratik çalışmayı içermektedir. Amacımız, farklı veri setleri üzerinde karar ağaçlarının nasıl eğitildiğini, performanslarının nasıl değerlendirildiğini ve sonuçların nasıl yorumlandığını somut örneklerle göstermektir.

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
