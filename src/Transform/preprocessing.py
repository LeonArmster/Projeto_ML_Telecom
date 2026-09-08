# ========================================
# Arquivo de preprocessamento de ML
# ========================================
# Bibliotecas
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
from src.Database.database_config import connect_database, read_query
from src.config import processed_dir

# Conectando ao banco
conexao = connect_database()
query = read_query('zeus.sql')

# Carregando no dataframe
df = pd.read_sql_query(query, con=conexao)

# Corrigindo os nomes
df['Status'] = df['Status'].replace({'Conclu+â-¡da':'Concluida',
                                     'N+â-úo Conclu+â-¡da':'Nao Concluida'})

df['Grupo_Info'] = df['Grupo_Info'].replace({'SIST+â-èMICO':'NAO_CONTAR'})


# Filtrando somente os dados que serão usados
df = df[df['Grupo_Info'].isin(['EFETIVO', 'CLIENTE', 'COMERCIAL', 'CAMPO', 'TECNICO'])]


# Criando a variável alvo
df['Target'] = df['Status'].str.strip().eq('Concluida').astype(int)


# Copiando df para fazer modelagem
df_model = df.copy()


# Removendo colunas que não vamos usar agora
cols_remove = [
    "Status",
    "Status_Final",
    "Duracao",
    "Termino_Atividade",
    "CHEGADA_SLOT",
    'Grupo_Info',
    'Atividade',
    'PON'
]

df_model = df_model.drop(columns=cols_remove, errors="ignore")


# Criando features de data e hora
df_model["Inicio_Previsto"] = pd.to_datetime(
    df_model["Inicio_Previsto"]
)


df_model["DT_Ag_Execucao"] = pd.to_datetime(
    df_model["DT_Ag_Execucao"],
    errors="coerce"
)


df_model["HORA_INICIO"] = df_model["Inicio_Previsto"].dt.hour

df_model["DIA_SEMANA"] = df_model["DT_Ag_Execucao"].dt.dayofweek

df_model["DIA_MES"] = df_model["DT_Ag_Execucao"].dt.day

df_model["MES"] = df_model["DT_Ag_Execucao"].dt.month

df_model["SEMANA"] = df_model["DT_Ag_Execucao"].dt.isocalendar().week.astype("int")

df_model["FDS"] = (
    df_model["DIA_SEMANA"] >= 5
).astype(int)


# Classificando periodo
df_model['SLOT'] = np.where(df_model['HORA_INICIO'] < 12, 'MANHA', 'TARDE')


# Ordenando os valores
df_model = df_model.sort_values(["DT_Ag_Execucao", "Inicio_Previsto"])


# Separando entre treino e teste pelo %
data_corte = df_model["DT_Ag_Execucao"].quantile(0.8)

train = df_model[df_model["DT_Ag_Execucao"] <= data_corte].copy()

test = df_model[df_model["DT_Ag_Execucao"] > data_corte].copy()


# Removendo a coluna dt_ag_execucao para evitar problema com o modelo
cols_remove = [
    "DT_Ag_Execucao",
    "Inicio_Previsto",
]

train = train.drop(columns=cols_remove, errors="ignore")
test = test.drop(columns=cols_remove, errors="ignore")


# Definindo x e y
y_train = train["Target"]
y_test = test["Target"]

X_train = train.drop(columns=["Target"])
X_test = test.drop(columns=["Target"])


# Identificando colunas categóricas
cat_cols = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()


# Tratando valores nulos para as colunas categórias
for col in cat_cols:
    X_train[col] = X_train[col].fillna("UNKNOWN").astype(str)
    X_test[col] = X_test[col].fillna("UNKNOWN").astype(str)


# Tratando valores nulos para as colunas numéricas
num_cols = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

for col in num_cols:
    X_train[col] = X_train[col].fillna(-999)
    X_test[col] = X_test[col].fillna(-999)




# Treinando o primeiro modelo
model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=100,
    random_seed=42
)


model.fit(
    X_train,
    y_train,
    cat_features=cat_cols,
    eval_set=(X_test, y_test),
    early_stopping_rounds=50
)



# Fazendo previsões
probabilidades = model.predict_proba(X_test)[:, 1]



pred = (probabilidades >= 0.5).astype(int)



print("Accuracy:", accuracy_score(y_test, pred))

print("Precision:", precision_score(y_test, pred))

print("Recall:", recall_score(y_test, pred))

print("F1:", f1_score(y_test, pred))

print("ROC-AUC:", roc_auc_score(y_test, probabilidades))

print("PR-AUC:", average_precision_score(y_test, probabilidades))

print(
    classification_report(
        y_test,
        pred,
        target_names=[
            "Não efetiva",
            "Efetiva"
        ]
    )
)


cm = confusion_matrix(y_test, pred)

cm




feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)


print(feature_importance.head(20))







from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42
)



model.fit(
    X_train,
    y_train,
    cat_features=cat_cols,
    eval_set=(X_val, y_val),
    early_stopping_rounds=50
)



# Fazendo previsões
probabilidades = model.predict_proba(X_test)[:, 1]



pred = (probabilidades >= 0.5).astype(int)



print("Accuracy:", accuracy_score(y_test, pred))

print("Precision:", precision_score(y_test, pred))

print("Recall:", recall_score(y_test, pred))

print("F1:", f1_score(y_test, pred))

print("ROC-AUC:", roc_auc_score(y_test, probabilidades))

print("PR-AUC:", average_precision_score(y_test, probabilidades))

print(
    classification_report(
        y_test,
        pred,
        target_names=[
            "Não efetiva",
            "Efetiva"
        ]
    )
)


cm = confusion_matrix(y_test, pred)

cm




feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)


print(feature_importance.head(20))


# Salvando o modelo
dados_para_salvar = {
    "modelo": model,
    "features": list(X_train.columns)
}
joblib.dump(dados_para_salvar, processed_dir/'modelo_rf.joblib')




# from sklearn.metrics import precision_recall_curve

# precision, recall, thresholds = precision_recall_curve(
#     y_test,
#     probabilidades
# )

# f1_scores = (
#     2 * precision * recall /
#     (precision + recall + 1e-10)
# )

# idx = np.argmax(f1_scores)

# melhor_threshold = thresholds[idx]

# print("Melhor threshold:", melhor_threshold)
# print("Precision:", precision[idx])
# print("Recall:", recall[idx])
# print("F1:", f1_scores[idx])



# pred_otimizado = (
#     probabilidades >= melhor_threshold
# ).astype(int)



# print(
#     classification_report(
#         y_test,
#         pred_otimizado,
#         target_names=[
#             "Não efetiva",
#             "Efetiva"
#         ]
#     )
# )






# analise_tecnico = (
#     df.groupby(["Matricula_Oper", "Nome_Oper"])["Target"]
#     .agg(
#         qtd_ordens="count",
#         efetividade="mean"
#     )
#     .sort_values("qtd_ordens", ascending=False)
# )

# analise_tecnico["efetividade"] *= 100

# print(analise_tecnico.head(20))





# df.groupby("Atividade")["Target"].agg(
#     qtd_ordens="count",
#     efetividade="mean"
# )