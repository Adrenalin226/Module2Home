import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)


print("="*60)
print("КРОК 1: Генерація синтетичного датасету вина")
print("="*60)

np.random.seed(42)
n_red = 500
n_white = 500

# Функція для генерації в діапазоні
def rnd(mean, std, n):
    return np.random.normal(mean, std, n)

data = {}

# Генерація кожної ознаки
data['Кислотність'] = np.concatenate([
    rnd(3.4, 0.1, n_red),
    rnd(3.15, 0.08, n_white)
])

data['Летка_кислотність'] = np.concatenate([
    rnd(0.5, 0.05, n_red),
    rnd(0.3, 0.04, n_white)
])

data['Лимонна_кислота'] = np.concatenate([
    rnd(0.30, 0.04, n_red),
    rnd(0.35, 0.05, n_white)
])

data['Залишковий_цукор'] = np.concatenate([
    rnd(2.5, 0.4, n_red),
    rnd(6.5, 1.0, n_white)
])

data['Хлориди'] = np.concatenate([
    rnd(0.085, 0.01, n_red),
    rnd(0.045, 0.008, n_white)
])

data['Вільний_SO2'] = np.concatenate([
    rnd(15, 4, n_red),
    rnd(40, 8, n_white)
])

data['Загальний_SO2'] = np.concatenate([
    rnd(60, 10, n_red),
    rnd(150, 20, n_white)
])

data['Щільність'] = np.concatenate([
    rnd(0.997, 0.001, n_red),
    rnd(0.9935, 0.001, n_white)
])

data['pH'] = np.concatenate([
    rnd(3.4, 0.05, n_red),
    rnd(3.2, 0.05, n_white)
])

data['Сульфати'] = np.concatenate([
    rnd(0.65, 0.05, n_red),
    rnd(0.45, 0.05, n_white)
])

data['Алкоголь'] = np.concatenate([
    rnd(10.5, 0.4, n_red),
    rnd(10.5, 0.4, n_white)
])

# Цільова змінна
data['Тип_вина'] = np.concatenate([
    np.zeros(n_red),
    np.ones(n_white)
])

df = pd.DataFrame(data)


print("\n" + "="*60)
print("КРОК 2: Дослідницький аналіз даних")
print("="*60)

print("Кількість зразків:", len(df))
print("Червоне:", (df['Тип_вина']==0).sum())
print("Біле:", (df['Тип_вина']==1).sum())

print("\nПерші 5 записів:")
print(df.head())

print("\nСтатистика:")
print(df.describe())

print("\nПропущені значення:")
print(df.isnull().sum())


print("\n" + "="*60)
print("КРОК 3: Підготовка даних")
print("="*60)

X = df.drop('Тип_вина', axis=1)
y = df['Тип_вина']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*60)
print("КРОК 4: Навчання моделей")
print("="*60)

model_lr = LogisticRegression(max_iter=1000, random_state=42)
model_lr.fit(X_train_scaled, y_train)

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train_scaled, y_train)

model_svm = SVC(kernel='rbf', probability=True, random_state=42)
model_svm.fit(X_train_scaled, y_train)


print("\n" + "="*60)
print("КРОК 5: Прогнозування")
print("="*60)

y_pred_lr = model_lr.predict(X_test_scaled)
y_pred_rf = model_rf.predict(X_test_scaled)
y_pred_svm = model_svm.predict(X_test_scaled)

# Ймовірності для ROC
y_proba_lr = model_lr.predict_proba(X_test_scaled)[:, 1]
y_proba_rf = model_rf.predict_proba(X_test_scaled)[:, 1]
y_proba_svm = model_svm.predict_proba(X_test_scaled)[:, 1]


print("\n" + "="*60)
print("КРОК 6: Оцінка моделей")
print("="*60)

def evaluate(model_name, y_true, y_pred):
    print(f"\n=== {model_name} ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print("Матриця помилок:")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["Червоне", "Біле"]))

evaluate("Логістична регресія", y_test, y_pred_lr)
evaluate("Random Forest", y_test, y_pred_rf)
evaluate("SVM", y_test, y_pred_svm)

feature_importance = pd.DataFrame({
    'Ознака': X.columns,
    'Важливість': model_rf.feature_importances_
}).sort_values('Важливість', ascending=False)

print("\n=== Важливість ознак (Random Forest) ===")
print(feature_importance)



print("\n" + "="*60)
print("КРОК 7: Візуалізація")
print("="*60)

fig = plt.figure(figsize=(18, 12))

# 1 — розподіл класів
plt.subplot(3, 3, 1)
df['Тип_вина'].value_counts().plot(kind='bar', color=['darkred','gold'])
plt.title("Розподіл типів вина")

# 2 — матриця LR
plt.subplot(3, 3, 2)
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt="d", cmap="Blues")
plt.title("Матриця помилок — Логістична регресія")

# 3 — RF
plt.subplot(3, 3, 3)
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Greens")
plt.title("Матриця помилок — Random Forest")

# 4 — SVM
plt.subplot(3, 3, 4)
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt="d", cmap="Oranges")
plt.title("Матриця помилок — SVM")

# 5 — ROC
plt.subplot(3, 3, 5)
fpr1, tpr1, _ = roc_curve(y_test, y_proba_lr)
fpr2, tpr2, _ = roc_curve(y_test, y_proba_rf)
fpr3, tpr3, _ = roc_curve(y_test, y_proba_svm)
plt.plot(fpr1, tpr1, label=f"LR AUC={auc(fpr1,tpr1):.2f}")
plt.plot(fpr2, tpr2, label=f"RF AUC={auc(fpr2,tpr2):.2f}")
plt.plot(fpr3, tpr3, label=f"SVM AUC={auc(fpr3,tpr3):.2f}")
plt.plot([0,1],[0,1],'--', color='gray')
plt.legend()
plt.title("ROC криві моделей")

# 6 — важливість ознак
plt.subplot(3, 3, 6)
plt.barh(feature_importance['Ознака'], feature_importance['Важливість'])
plt.title("Важливість ознак")

# 7 — розподіл кислотності
plt.subplot(3, 3, 7)
df[df['Тип_вина']==0]['Кислотність'].hist(alpha=.6, color='darkred')
df[df['Тип_вина']==1]['Кислотність'].hist(alpha=.6, color='gold')
plt.title("Розподіл кислотності")

# 8 — розподіл цукру
plt.subplot(3, 3, 8)
df[df['Тип_вина']==0]['Залишковий_цукор'].hist(alpha=.6, color='darkred')
df[df['Тип_вина']==1]['Залишковий_цукор'].hist(alpha=.6, color='gold')
plt.title("Розподіл залишкового цукру")

# 9 — scatter
plt.subplot(3, 3, 9)
plt.scatter(df[df['Тип_вина']==0]['Кислотність'],
            df[df['Тип_вина']==0]['Залишковий_цукор'],
            c='darkred', label="Червоне")
plt.scatter(df[df['Тип_вина']==1]['Кислотність'],
            df[df['Тип_вина']==1]['Залишковий_цукор'],
            c='gold', label="Біле")
plt.legend()
plt.title("Кислотність vs Цукор")

plt.tight_layout()
plt.show()



print("\n" + "="*60)
print("КРОК 8: Прогнозування нового зразка")
print("="*60)

new_wine = pd.DataFrame({
    'Кислотність': [3.2],
    'Летка_кислотність': [0.3],
    'Лимонна_кислота': [0.35],
    'Залишковий_цукор': [6.5],
    'Хлориди': [0.045],
    'Вільний_SO2': [40],
    'Загальний_SO2': [150],
    'Щільність': [0.994],
    'pH': [3.2],
    'Сульфати': [0.45],
    'Алкоголь': [10.5]
})

new_scaled = scaler.transform(new_wine)

pred = model_rf.predict(new_scaled)[0]
proba = model_rf.predict_proba(new_scaled)[0]

print("Характеристики нового зразка:")
print(new_wine)

print("\nРЕЗУЛЬТАТ:")
print("Тип вина:", "Біле" if pred == 1 else "Червоне")
print(f"Впевненість: {max(proba)*100:.1f}%")
print(f"Ймовірність червоного: {proba[0]*100:.1f}%")
print(f"Ймовірність білого: {proba[1]*100:.1f}%")



print("\n" + "="*60)
print("ВИСНОВКИ")
print("="*60)

