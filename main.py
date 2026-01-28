import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings

# Uyarıları kapat ve pandas ayarları
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("--- CREDIT RISK MODEL VALIDATION PROJESİ BAŞLATILIYOR ---\n")

# --- 1. VERİ YÜKLEME VE HAZIRLIK ---
print("[1/4] Veri Yükleniyor ve Temizleniyor...")
# Dosya yolunu kontrol et, aynı klasörde olmalı
try:
    df = pd.read_csv('cs-training.csv')
except FileNotFoundError:
    print("HATA: 'cs-training.csv' dosyası bulunamadı! Lütfen dosyanın main.py ile aynı klasörde olduğundan emin ol.")
    exit()

# Gereksiz kolon temizliği
if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)

# Eksik Değer Doldurma
df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0)

target = 'SeriousDlqin2yrs'
features = [col for col in df.columns if col != target]

X = df[features]
y = df[target]

# Zaman Simülasyonu: %70 Geçmiş (Dev), %30 Gelecek (Validation)
X_dev, X_val, y_dev, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"      Development Seti: {X_dev.shape}")
print(f"      Validation Seti:  {X_val.shape}")

# --- 2. MODEL GELİŞTİRME (BASELINE) ---
print("\n[2/4] XGBoost Modeli Eğitiliyor...")
model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
    eval_metric='auc',
    use_label_encoder=False
)
model.fit(X_dev, y_dev)
print("      Model Eğitimi Tamamlandı.")

# --- 3. PERFORMANS VALİDASYONU (GINI & AUC) ---
print("\n[3/4] Performans Metrikleri Hesaplanıyor (Gini Consistency)...")
prob_dev = model.predict_proba(X_dev)[:, 1]
prob_val = model.predict_proba(X_val)[:, 1]

auc_dev = roc_auc_score(y_dev, prob_dev)
auc_val = roc_auc_score(y_val, prob_val)
gini_dev = 2 * auc_dev - 1
gini_val = 2 * auc_val - 1

print(f"      Development Gini: {gini_dev:.4f}")
print(f"      Validation Gini:  {gini_val:.4f}")
print(f"      Performans Farkı: {abs(gini_dev - gini_val):.4f}")

if abs(gini_dev - gini_val) < 0.05:
    print("      SONUÇ: Model Performansı Stabil (Pass).")
else:
    print("      SONUÇ: Modelde Overfitting Riski Var (Fail).")

# --- 4. STABİLİTE VALİDASYONU (PSI ANALİZİ) ---
print("\n[4/4] Stabilite Analizi (PSI) Çalıştırılıyor...")

def calculate_psi(expected, actual, buckettype='quantiles', buckets=10, axis=0):
    def psi(expected_array, actual_array, buckets):
        def scale_range (input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])
            breakpoints = np.unique(breakpoints)

        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            if a_perc == 0: a_perc = 0.0001
            if e_perc == 0: e_perc = 0.0001
            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return(value)

        psi_value = np.sum([sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents))])
        return(psi_value)

    if len(expected.shape) == 1:
        psi_values = psi(expected, actual, buckets)
    else:
        psi_values = np.empty(expected.shape[axis])
        for i in range(0, len(psi_values)):
            psi_values[i] = psi(expected[:,i], actual[:,i], buckets)

    return psi_values

psi_list = []
for feature in features:
    psi_val = calculate_psi(X_dev[feature], X_val[feature], buckettype='quantiles', buckets=10)
    
    if psi_val < 0.1:
        status = "Yeşil (Stabil)"
    elif psi_val < 0.25:
        status = "Sarı (Dikkat)"
    else:
        status = "Kırmızı (Alarm)"
        
    psi_list.append({'Değişken': feature, 'PSI': round(psi_val, 5), 'Durum': status})

psi_table = pd.DataFrame(psi_list).sort_values(by='PSI', ascending=False)
print("\n--- PSI RAPORU ---")
print(psi_table.to_string(index=False))
print("\n--- PROJE BAŞARIYLA TAMAMLANDI ---")