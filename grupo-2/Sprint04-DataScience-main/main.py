import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 1. GERAÇÃO DE DADOS SINTÉTICOS
# ============================================================================

np.random.seed(42)
n_samples = 500

# Features relevantes para gestão de estoque
dias_desde_pedido = np.random.randint(1, 60, n_samples)
quantidade_solicitada = np.random.randint(10, 500, n_samples)
velocidade_consumo = np.random.uniform(0.5, 5, n_samples)  # unidades/dia
dias_para_entrega_fornecedor = np.random.randint(1, 30, n_samples)
variabilidade_demanda = np.random.uniform(0, 1, n_samples)
estoque_atual = np.random.randint(0, 1000, n_samples)
temperatura_ambiente = np.random.uniform(15, 35, n_samples)  # crítica para medicamentos
umidade_relativa = np.random.uniform(30, 80, n_samples)
numero_pedidos_pendentes = np.random.randint(0, 20, n_samples)
tempo_medio_processamento = np.random.uniform(2, 24, n_samples)  # horas

# Criar DataFrame
X = pd.DataFrame({
    'dias_desde_pedido': dias_desde_pedido,
    'quantidade_solicitada': quantidade_solicitada,
    'velocidade_consumo': velocidade_consumo,
    'dias_para_entrega': dias_para_entrega_fornecedor,
    'variabilidade_demanda': variabilidade_demanda,
    'estoque_atual': estoque_atual,
    'temperatura': temperatura_ambiente,
    'umidade': umidade_relativa,
    'pedidos_pendentes': numero_pedidos_pendentes,
    'tempo_processamento': tempo_medio_processamento
})

# ============================================================================
# 2. DEFINIÇÃO DAS VARIÁVEIS ALVO
# ============================================================================

# Target Binário: Falta de Estoque Crítico (1) ou Não (0)
ponto_reordenacao = X['velocidade_consumo'] * X['dias_para_entrega'] * 2
y_binary = ((X['estoque_atual'] < ponto_reordenacao) |
            ((X['estoque_atual'] / (X['velocidade_consumo'] + 0.1)) < 5)).astype(int)

# Target Contínuo: Nível de Risco de Desabastecimento (0-100)
y_continuous = (
    (100 * X['velocidade_consumo'] / (X['estoque_atual'] + 1)) +
    (50 * X['variabilidade_demanda']) +
    (30 * (X['dias_para_entrega'] / 30)) +
    np.random.normal(0, 5, n_samples)
)
y_continuous = np.clip(y_continuous, 0, 100)

print("=" * 80)
print("PROJETO: GESTÃO DE ESTOQUE COM IA - LABORATÓRIO DASA")
print("=" * 80)
print(f"\nDados Gerados: {n_samples} amostras com {X.shape[1]} features")
print(f"Classes (Binário): {np.bincount(y_binary)}")
print(f"Risco Contínuo: Min={y_continuous.min():.2f}, Max={y_continuous.max():.2f}")

# ============================================================================
# 3. SPLIT DOS DADOS
# ============================================================================

X_train, X_test, y_train_binary, y_test_binary = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

_, _, y_train_cont, y_test_cont = train_test_split(
    X, y_continuous, test_size=0.2, random_state=42
)

# Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nDados de Treino: {X_train.shape[0]} | Dados de Teste: {X_test.shape[0]}")

# ============================================================================
# 4. MODELO KNN
# ============================================================================

print("\n" + "=" * 80)
print("4. K-NEAREST NEIGHBORS (KNN)")
print("=" * 80)

# Teste de diferentes valores de K
k_values = [3, 5, 7, 9, 11, 15]
knn_results = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train_binary)
    y_pred_knn = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test_binary, y_pred_knn)
    knn_results.append({'k': k, 'accuracy': acc})

best_k = max(knn_results, key=lambda x: x['accuracy'])['k']
print(f"\nTeste de K: {[(r['k'], f"{r['accuracy']:.4f}") for r in knn_results]}")
print(f"K Ótimo: {best_k}")

knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train_scaled, y_train_binary)
y_pred_knn = knn_final.predict(X_test_scaled)

knn_acc = accuracy_score(y_test_binary, y_pred_knn)
knn_prec = precision_score(y_test_binary, y_pred_knn, zero_division=0)
knn_rec = recall_score(y_test_binary, y_pred_knn, zero_division=0)
knn_f1 = f1_score(y_test_binary, y_pred_knn, zero_division=0)

print(f"\nMétricas KNN (k={best_k}):")
print(f"  Acurácia:  {knn_acc:.4f}")
print(f"  Precisão:  {knn_prec:.4f}")
print(f"  Recall:    {knn_rec:.4f}")
print(f"  F1-Score:  {knn_f1:.4f}")

# ============================================================================
# 5. REGRESSÃO LOGÍSTICA
# ============================================================================

print("\n" + "=" * 80)
print("5. REGRESSÃO LOGÍSTICA")
print("=" * 80)

lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train_binary)
y_pred_lr = lr.predict(X_test_scaled)

lr_acc = accuracy_score(y_test_binary, y_pred_lr)
lr_prec = precision_score(y_test_binary, y_pred_lr, zero_division=0)
lr_rec = recall_score(y_test_binary, y_pred_lr, zero_division=0)
lr_f1 = f1_score(y_test_binary, y_pred_lr, zero_division=0)

print(f"\nMétricas Regressão Logística:")
print(f"  Acurácia:  {lr_acc:.4f}")
print(f"  Precisão:  {lr_prec:.4f}")
print(f"  Recall:    {lr_rec:.4f}")
print(f"  F1-Score:  {lr_f1:.4f}")

print(f"\nCoeficientes (Importância das Features):")
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coeficiente': lr.coef_[0]
}).sort_values('Coeficiente', ascending=False)
print(coef_df.to_string(index=False))
print(f"Intercepto: {lr.intercept_[0]:.4f}")

# ============================================================================
# 6. RIDGE REGRESSION
# ============================================================================

print("\n" + "=" * 80)
print("6. RIDGE REGRESSION (Regressão Contínua)")
print("=" * 80)

alphas_ridge = [0.001, 0.01, 0.1, 1, 10, 100]
ridge_results = []

for alpha in alphas_ridge:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train_cont)
    y_pred_ridge = ridge.predict(X_test_scaled)
    mse = mean_squared_error(y_test_cont, y_pred_ridge)
    r2 = r2_score(y_test_cont, y_pred_ridge)
    ridge_results.append({'alpha': alpha, 'mse': mse, 'r2': r2})

print(f"\nPerformance Ridge com diferentes alphas:")
for r in ridge_results:
    print(f"  Alpha {r['alpha']:6.3f}: MSE={r['mse']:8.2f}, R²={r['r2']:.4f}")

best_alpha_ridge = min(ridge_results, key=lambda x: x['mse'])['alpha']
ridge_final = Ridge(alpha=best_alpha_ridge)
ridge_final.fit(X_train_scaled, y_train_cont)
y_pred_ridge_final = ridge_final.predict(X_test_scaled)

ridge_mse = mean_squared_error(y_test_cont, y_pred_ridge_final)
ridge_rmse = np.sqrt(ridge_mse)
ridge_mae = mean_absolute_error(y_test_cont, y_pred_ridge_final)
ridge_r2 = r2_score(y_test_cont, y_pred_ridge_final)

print(f"\nRidge Otimizado (alpha={best_alpha_ridge}):")
print(f"  MSE:  {ridge_mse:.4f}")
print(f"  RMSE: {ridge_rmse:.4f}")
print(f"  MAE:  {ridge_mae:.4f}")
print(f"  R²:   {ridge_r2:.4f}")

# ============================================================================
# 7. LASSO REGRESSION
# ============================================================================

print("\n" + "=" * 80)
print("7. LASSO REGRESSION")
print("=" * 80)

alphas_lasso = [0.001, 0.01, 0.1, 1, 10]
lasso_results = []

for alpha in alphas_lasso:
    lasso = Lasso(alpha=alpha, max_iter=5000)
    lasso.fit(X_train_scaled, y_train_cont)
    y_pred_lasso = lasso.predict(X_test_scaled)
    mse = mean_squared_error(y_test_cont, y_pred_lasso)
    r2 = r2_score(y_test_cont, y_pred_lasso)
    n_zeros = np.sum(lasso.coef_ == 0)
    lasso_results.append({'alpha': alpha, 'mse': mse, 'r2': r2, 'n_zeros': n_zeros})

print(f"\nPerformance Lasso com diferentes alphas:")
for r in lasso_results:
    print(f"  Alpha {r['alpha']:6.3f}: MSE={r['mse']:8.2f}, R²={r['r2']:.4f}, Features Removidas={r['n_zeros']}")

best_alpha_lasso = min(lasso_results, key=lambda x: x['mse'])['alpha']
lasso_final = Lasso(alpha=best_alpha_lasso, max_iter=5000)
lasso_final.fit(X_train_scaled, y_train_cont)
y_pred_lasso_final = lasso_final.predict(X_test_scaled)

lasso_mse = mean_squared_error(y_test_cont, y_pred_lasso_final)
lasso_rmse = np.sqrt(lasso_mse)
lasso_mae = mean_absolute_error(y_test_cont, y_pred_lasso_final)
lasso_r2 = r2_score(y_test_cont, y_pred_lasso_final)
lasso_removed = np.sum(lasso_final.coef_ == 0)

print(f"\nLasso Otimizado (alpha={best_alpha_lasso}):")
print(f"  MSE:  {lasso_mse:.4f}")
print(f"  RMSE: {lasso_rmse:.4f}")
print(f"  MAE:  {lasso_mae:.4f}")
print(f"  R²:   {lasso_r2:.4f}")
print(f"  Features com coeficiente zero: {lasso_removed}")

print(f"\nCoeficientes Lasso:")
coef_lasso_df = pd.DataFrame({
    'Feature': X.columns,
    'Coeficiente': lasso_final.coef_
}).sort_values('Coeficiente', ascending=False, key=abs)
print(coef_lasso_df.to_string(index=False))

# ============================================================================
# 8. REGRESSÃO POLINOMIAL
# ============================================================================

print("\n" + "=" * 80)
print("8. REGRESSÃO POLINOMIAL")
print("=" * 80)

# Regressão simples (baseline)
lr_simple = LinearRegression()
lr_simple.fit(X_train_scaled, y_train_cont)
y_pred_simple = lr_simple.predict(X_test_scaled)
simple_r2 = r2_score(y_test_cont, y_pred_simple)
simple_rmse = np.sqrt(mean_squared_error(y_test_cont, y_pred_simple))

print(f"\nRegressão Linear Simples:")
print(f"  R²:   {simple_r2:.4f}")
print(f"  RMSE: {simple_rmse:.4f}")

# Regressão Polinomial - Grau 2
poly_features_2 = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly2 = poly_features_2.fit_transform(X_train_scaled)
X_test_poly2 = poly_features_2.transform(X_test_scaled)

lr_poly2 = LinearRegression()
lr_poly2.fit(X_train_poly2, y_train_cont)
y_pred_poly2 = lr_poly2.predict(X_test_poly2)
poly2_r2 = r2_score(y_test_cont, y_pred_poly2)
poly2_rmse = np.sqrt(mean_squared_error(y_test_cont, y_pred_poly2))

print(f"\nRegressão Polinomial (Grau 2):")
print(f"  R²:   {poly2_r2:.4f}")
print(f"  RMSE: {poly2_rmse:.4f}")
print(f"  Melhora R²: {poly2_r2 - simple_r2:+.4f}")

# Regressão Polinomial - Grau 3
poly_features_3 = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly3 = poly_features_3.fit_transform(X_train_scaled)
X_test_poly3 = poly_features_3.transform(X_test_scaled)

lr_poly3 = LinearRegression()
lr_poly3.fit(X_train_poly3, y_train_cont)
y_pred_poly3 = lr_poly3.predict(X_test_poly3)
poly3_r2 = r2_score(y_test_cont, y_pred_poly3)
poly3_rmse = np.sqrt(mean_squared_error(y_test_cont, y_pred_poly3))

print(f"\nRegressão Polinomial (Grau 3):")
print(f"  R²:   {poly3_r2:.4f}")
print(f"  RMSE: {poly3_rmse:.4f}")
print(f"  Melhora R²: {poly3_r2 - simple_r2:+.4f}")

# ============================================================================
# 9. ÁRVORE DE DECISÃO E RANDOM FOREST
# ============================================================================

print("\n" + "=" * 80)
print("9. ÁRVORE DE DECISÃO E RANDOM FOREST")
print("=" * 80)

# Árvore de Decisão
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train_scaled, y_train_binary)
y_pred_dt = dt.predict(X_test_scaled)

dt_acc = accuracy_score(y_test_binary, y_pred_dt)
dt_prec = precision_score(y_test_binary, y_pred_dt, zero_division=0)
dt_rec = recall_score(y_test_binary, y_pred_dt, zero_division=0)
dt_f1 = f1_score(y_test_binary, y_pred_dt, zero_division=0)

print(f"\nÁrvore de Decisão (max_depth=5):")
print(f"  Acurácia:  {dt_acc:.4f}")
print(f"  Precisão:  {dt_prec:.4f}")
print(f"  Recall:    {dt_rec:.4f}")
print(f"  F1-Score:  {dt_f1:.4f}")

feature_importance_dt = pd.DataFrame({
    'Feature': X.columns,
    'Importância': dt.feature_importances_
}).sort_values('Importância', ascending=False)
print(f"\nImportância das Features (Árvore):")
print(feature_importance_dt.to_string(index=False))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train_binary)
y_pred_rf = rf.predict(X_test_scaled)

rf_acc = accuracy_score(y_test_binary, y_pred_rf)
rf_prec = precision_score(y_test_binary, y_pred_rf, zero_division=0)
rf_rec = recall_score(y_test_binary, y_pred_rf, zero_division=0)
rf_f1 = f1_score(y_test_binary, y_pred_rf, zero_division=0)

print(f"\nRandom Forest (n_estimators=100, max_depth=7):")
print(f"  Acurácia:  {rf_acc:.4f}")
print(f"  Precisão:  {rf_prec:.4f}")
print(f"  Recall:    {rf_rec:.4f}")
print(f"  F1-Score:  {rf_f1:.4f}")

feature_importance_rf = pd.DataFrame({
    'Feature': X.columns,
    'Importância': rf.feature_importances_
}).sort_values('Importância', ascending=False)
print(f"\nImportância das Features (Random Forest):")
print(feature_importance_rf.to_string(index=False))

# ============================================================================
# 10. COMPARAÇÃO DE TODOS OS MODELOS
# ============================================================================

print("\n" + "=" * 80)
print("10. COMPARAÇÃO FINAL DE TODOS OS MODELOS")
print("=" * 80)

# Classificadores
comparison_classifiers = pd.DataFrame({
    'Modelo': ['KNN', 'Regressão Logística', 'Árvore de Decisão', 'Random Forest'],
    'Acurácia': [knn_acc, lr_acc, dt_acc, rf_acc],
    'Precisão': [knn_prec, lr_prec, dt_prec, rf_prec],
    'Recall': [knn_rec, lr_rec, dt_rec, rf_rec],
    'F1-Score': [knn_f1, lr_f1, dt_f1, rf_f1]
})

print("\nClassificadores (Problema Binário - Falta de Estoque):")
print(comparison_classifiers.to_string(index=False))

# Modelos de Regressão
comparison_regression = pd.DataFrame({
    'Modelo': ['Linear Simples', 'Ridge', 'Lasso', 'Polinomial Grau 2', 'Polinomial Grau 3'],
    'R² Score': [simple_r2, ridge_r2, lasso_r2, poly2_r2, poly3_r2],
    'RMSE': [simple_rmse, ridge_rmse, lasso_rmse, poly2_rmse, poly3_rmse],
    'MAE': [
        mean_absolute_error(y_test_cont, y_pred_simple),
        ridge_mae,
        lasso_mae,
        mean_absolute_error(y_test_cont, y_pred_poly2),
        mean_absolute_error(y_test_cont, y_pred_poly3)
    ]
})

print("\n\nModelos de Regressão (Nível de Risco Contínuo):")
print(comparison_regression.to_string(index=False))

# ============================================================================
# 11. SALVANDO RESULTADOS
# ============================================================================

resultados = {
    'X_train': X_train,
    'X_test': X_test,
    'y_test_binary': y_test_binary,
    'y_test_cont': y_test_cont,
    'comparison_classifiers': comparison_classifiers,
    'comparison_regression': comparison_regression,
    'knn_final': knn_final,
    'lr': lr,
    'dt': dt,
    'rf': rf,
    'ridge_final': ridge_final,
    'lasso_final': lasso_final,
    'scaler': scaler
}

print("\n" + "=" * 80)
print("Análise completa finalizada com sucesso!")
print("=" * 80)