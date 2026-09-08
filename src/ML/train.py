# ===================================
# Treinamento para o ML
# ===================================
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
import os
import joblib
from src.Database.database_config import connect_database, read_query

# --------------------------------------
# Carregando DF
# --------------------------------------
# Conexão com o banco
conexao = connect_database()

# Lendo a query
query = read_query('query_train.sql')

# Carregando o df
df = pd.read_sql_query(query, conexao)


# --------------------------------------
# Preparando os dados
# --------------------------------------
# Ajustando o nome dos técnicos
df["nome_operador"] = df["nome_operador"].fillna("SEM_TECNICO").str.strip().str.upper()

# Ajustando as colunas temporais
df["data_agendamento"] = pd.to_datetime(df["data_agendamento"])

# Criando mes, ano e dia da semana
df["ano"] = df["data_agendamento"].dt.year
df["mes"] = df["data_agendamento"].dt.month
df["dia_semana"] = df["data_agendamento"].dt.dayofweek

# # Excluindo a coluna data_agendamento
# df.drop(columns=["data_agendamento"], inplace=True)


def criar_historico_temporal(df, group_cols, prefix, prior_rate=0.64, smoothing=20):
    """
    Calcula histórico somente até o dia anterior.

    group_cols:
        Colunas que definem o grupo.

    prefix:
        Prefixo das novas colunas.

    Exemplo:
        group_cols=["nome_operador"]
        prefix="tecnico"
    """

    df = df.copy()

    # Garantir ordenação temporal
    df = df.sort_values("data_agendamento")

    # Histórico diário por grupo
    daily = (
        df.groupby(
            group_cols + ["data_agendamento"],
            dropna=False,
            observed=True
        )
        .agg(
            _qtd=("_target", "size"),
            _sucessos=("_target", "sum")
        )
        .reset_index()
    )

    daily = daily.sort_values(
        group_cols + ["data_agendamento"]
    )

    # Histórico ANTES do dia atual
    grouped = daily.groupby(
        group_cols,
        dropna=False,
        observed=True
    )

    daily[f"{prefix}_qtd"] = (
        grouped["_qtd"].cumsum()
        - daily["_qtd"]
    )

    daily[f"{prefix}_sucessos"] = (
        grouped["_sucessos"].cumsum()
        - daily["_sucessos"]
    )

    # Bayesian smoothing
    daily[f"{prefix}_efetividade"] = (
        daily[f"{prefix}_sucessos"]
        + smoothing * prior_rate
    ) / (
        daily[f"{prefix}_qtd"]
        + smoothing
    )

    # Colunas que serão incorporadas ao dataframe original
    features = [
        *group_cols,
        "data_agendamento",
        f"{prefix}_qtd",
        f"{prefix}_efetividade"
    ]

    df = df.merge(
        daily[features],
        on=group_cols + ["data_agendamento"],
        how="left"
    )

    return df

# --------------------------------------------------------
# Histórico anterior à ordem
# --------------------------------------------------------
df["_target"] = df["efetividade"].astype("int8")

df = criar_historico_temporal(df, group_cols=["nome_operador"], prefix="tecnico")

df = criar_historico_temporal(df, group_cols=["cidade"],prefix="cidade")

df = criar_historico_temporal(df, group_cols=["tipo_servico"], prefix="servico")

df = criar_historico_temporal(df, group_cols=["dia_semana"], prefix="dia_semana")


# ----------------------------------------
# Separando as features
# ----------------------------------------
# Features categóricas
categorical_features = [
    "tipo_atividade",
    "slot",
    "detalhe_atividade",
    "tipo_servico",
    "cidade",
    "segmento",
    "categoria_cliente",
    "nome_operador",
    "cluster_zeus",
    "gerencia",
]

# Preenchendo os valores vazios
for column in categorical_features:
    df[column] = (
        df[column]
        .fillna("SEM_INFORMACAO")
        .astype(str)
        .str.strip()
    )


# Features numéricas
features_numericas = [
    "mes",
    "dia_semana",
    "tecnico_qtd",
    "tecnico_efetividade",
    "cidade_qtd",
    "cidade_efetividade",
    "servico_qtd",
    "servico_efetividade",
    "dia_semana_qtd",
    "dia_semana_efetividade"
]



# Features gerais
features = categorical_features + features_numericas

# Variável alvo
target = "efetividade"


# --------------------------------------
# Separando entre treinamento, validação e teste e preparando os modelos
# --------------------------------------
# df de treinamento
train_df = df[(df['ano'] == 2022) & (df["mes"] <= 9)]

# df de validação
validation_df = df[(df['ano'] == 2022) & (df["mes"] >= 10)]

# df de teste
test_df = df[df['ano'] == 2023]


# treino
X_train = train_df[features]
y_train = train_df[target]

# validação
X_validation = validation_df[features]
y_validation = validation_df[target]

# teste
X_test = test_df[features]
y_test = test_df[target]


# total de valores para as variáveis de treino, validação e teste
print()
print("Dataset:")
print(f"Treino:      {len(train_df):,}")
print(f"Validação:   {len(validation_df):,}")
print(f"Teste:       {len(test_df):,}")


# --------------------------------------
# Modelo de treinamnto
# --------------------------------------
model = CatBoostClassifier(
    iterations=800,
    learning_rate=0.05,
    depth=8,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=100,
    thread_count=-1,
)

model.fit(
    X_train,
    y_train,
    cat_features=categorical_features,
    eval_set=(X_validation, y_validation),
    use_best_model=True,
)


# ---------------------------------------
# Teste
# ---------------------------------------
probabilities = model.predict_proba(X_test)[:, 1]

predictions = (probabilities >= 0.5).astype(int)

# # ============================================================
# # CALIBRAÇÃO DAS PROBABILIDADES
# # ============================================================
# validation_probabilities = model.predict_proba(
#     X_validation
# )[:, 1]

# calibrator = LogisticRegression()

# calibrator.fit(
#     validation_probabilities.reshape(-1, 1),
#     y_validation,
# )

# calibrated_probabilities = calibrator.predict_proba(
#     probabilities.reshape(-1, 1)
# )[:, 1]


roc_auc = roc_auc_score(y_test, probabilities)
pr_auc = average_precision_score(y_test, probabilities)

print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC:  {pr_auc:.4f}")

print()
print("Matriz de confusão:")
print(confusion_matrix(y_test, predictions))

print()
print("Classification report:")
print(classification_report(y_test, predictions))


probabilities = model.predict_proba(X_test)[:, 1]

print(
    pd.Series(probabilities).describe(
        percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
)



importance = pd.DataFrame({
    "feature": features,
    "importance": model.get_feature_importance(),
})

importance = importance.sort_values(
    "importance",
    ascending=False,
)

print()
print("Importância das features:")
print(importance.to_string(index=False))







# Fazendo a avaliação de brier para diagnosticar problemas nas variáveis
y_pred_proba = model.predict_proba(X_test)[:, 1]


brier = brier_score_loss(
    y_test,
    y_pred_proba
)

print(f"Brier Score: {brier:.4f}")




df_calibracao = pd.DataFrame({
    "real": y_test.values,
    "probabilidade": y_pred_proba
})

df_calibracao["faixa"] = pd.cut(
    df_calibracao["probabilidade"],
    bins=[
        0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0
    ],
    include_lowest=True
)

resultado = (
    df_calibracao
    .groupby("faixa", observed=False)
    .agg(
        qtd_ordens=("real", "size"),
        taxa_real=("real", "mean"),
        probabilidade_media=("probabilidade", "mean")
    )
    .reset_index()
)

print(resultado)



y_val_proba = model.predict_proba(X_validation)[:, 1]


from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


calibrador_sigmoid = LogisticRegression()

calibrador_sigmoid.fit(
    y_val_proba.reshape(-1, 1),
    y_validation
)

y_test_proba_sigmoid = calibrador_sigmoid.predict_proba(
    y_pred_proba.reshape(-1, 1)
)[:, 1]


from sklearn.metrics import brier_score_loss

brier_original = brier_score_loss(
    y_test,
    y_pred_proba
)

brier_sigmoid = brier_score_loss(
    y_test,
    y_test_proba_sigmoid
)

print(f"Brier original: {brier_original:.4f}")
print(f"Brier sigmoid:  {brier_sigmoid:.4f}")



calibrador_isotonic = IsotonicRegression(
    out_of_bounds="clip"
)

calibrador_isotonic.fit(
    y_val_proba,
    y_validation
)

y_test_proba_isotonic = calibrador_isotonic.predict(
    y_pred_proba
)


brier_isotonic = brier_score_loss(
    y_test,
    y_test_proba_isotonic
)

print(f"Brier original:  {brier_original:.4f}")
print(f"Brier sigmoid:   {brier_sigmoid:.4f}")
print(f"Brier isotonic:  {brier_isotonic:.4f}")





df_calibracao = pd.DataFrame({
    "y": y_test,
    "proba_original": y_pred_proba,
    "proba_isotonic": y_test_proba_isotonic
})

df_calibracao["faixa"] = pd.cut(
    df_calibracao["proba_original"],
    bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    include_lowest=True
)

resultado_calibracao = (
    df_calibracao
    .groupby("faixa", observed=True)
    .agg(
        qtd_ordens=("y", "size"),
        taxa_real=("y", "mean"),
        prob_original=("proba_original", "mean"),
        prob_isotonic=("proba_isotonic", "mean")
    )
    .reset_index()
)

print(resultado_calibracao)











from xgboost import XGBClassifier

model_xgb = XGBClassifier(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)




encoder = ColumnTransformer(

    transformers=[

        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            ),
            features_categoricas
        ),

        (
            "numeric",
            "passthrough",
            features_numericas
        )

    ]

)


print("\nGerando encoding...")

X_train_encoded = encoder.fit_transform(
    X_train
)

X_val_encoded = encoder.transform(
    X_validation
)

X_test_encoded = encoder.transform(
    X_test
)


print("\nDimensões:")

print(
    "Train:",
    X_train_encoded.shape
)

print(
    "Validation:",
    X_val_encoded.shape
)

print(
    "Test:",
    X_test_encoded.shape
)