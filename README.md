# Credit Risk Model Validation & Stability Analysis 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Completed-green)
![Domain](https://img.shields.io/badge/Domain-Finance%20%26%20Risk-orange)

##  Proje Hakkında (Executive Summary)

Bu proje, bankacılık sektöründe kritik öneme sahip olan **Kredi Riski (Credit Risk)** modellerinin validasyon süreçlerini simüle etmek amacıyla geliştirilmiştir. 

Projenin temel amacı, sadece yüksek performanslı bir model geliştirmek değil; geliştirilen modelin zaman içindeki kararlılığını (**Stability**) ve veri yapısındaki değişimlere (**Data Drift**) karşı direncini matematiksel yöntemlerle kanıtlamaktır.

##  Kullanılan Teknolojiler ve Yöntemler

* **Algoritma:** XGBoost (Gradient Boosting)
* **Validasyon Metrikleri:**  **PSI (Population Stability Index):** Değişkenlerin dağılımındaki bozulmayı (Drift) ölçmek için.
* **Gini & ROC-AUC Consistency:** Eğitim ve validasyon setleri arasındaki performans farkını (Overfitting kontrolü) ölçmek için.
* **Veri Seti:** Give Me Some Credit (Kaggle)

##  Validasyon Sonuçları

### 1. Performans Tutarlılığı (Performance Consistency)
Modelin eğitim (Development) ve test (Validation) verileri üzerindeki performansı karşılaştırılmıştır.

| Metrik | Development (Eğitim) | Validation (OOT Simülasyonu) | Fark (Spread) | Durum |
| :--- | :--- | :--- | :--- | :--- |
| **Gini** | 0.7468 | 0.7307 | **0.0161** |  **PASS** |
| **AUC** | 0.8734 | 0.8653 | **0.0081** |  **PASS** |

> **Yorum:** Gini farkının %5'in altında olması, modelin "Overfitting" yapmadığını ve yeni müşteri verilerinde de istikrarlı çalıştığını göstermektedir.

### 2. Stabilite Analizi (PSI - Population Stability Index)
Modelde kullanılan değişkenlerin zaman içindeki dağılım kararlılığı PSI yöntemiyle test edilmiştir. (Eşik Değer: PSI < 0.10 -> Stabil)

| Değişken Adı | PSI Değeri | Durum |
| :--- | :--- | :--- |
| **DebtRatio** | 0.00045 | 🟢 Stabil |
| **MonthlyIncome** | 0.00036 | 🟢 Stabil |
| **RevolvingUtilization** | 0.00029 | 🟢 Stabil |

> **Yorum:** Tüm kritik değişkenlerin PSI değerleri "Yeşil" bölgededir. Veri dağılımında yapısal bir bozulma (Data Drift) gözlemlenmemiştir.

##  Kurulum ve Çalıştırma

Projeyi yerel bilgisayarınızda çalıştırmak için:

```bash
# 1. Repoyu klonlayın
git clone https://github.com/OmerFarukKos/CreditRisk-Stability-Analysis.git

# 2. Gerekli kütüphaneleri yükleyin
pip install pandas numpy xgboost scikit-learn

# 3. Analizi başlatın
python main.py
