import pandas as pd
import numpy as np
from src.Database.database_config import connect_database, read_query

from xgboost import XGBClassifier

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

RANDOM_STATE = 42

SMOOTHING = 20
TAXA_GLOBAL = 0.64


# ============================================================
# 2. PREPARAÇÃO
# ============================================================

# --------------------------------------
# Carregando DF
# --------------------------------------
# Conexão com o banco
conexao = connect_database()

# Lendo a query
query = read_query('query_train.sql')

# Carregando o df
df = pd.read_sql_query(query, conexao)


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier


# ============================================================
# CONFIGURAÇÕES
# ============================================================

TARGET = "efetividade"

# ------------------------------------------------------------
# Rodadas da simulação
# ------------------------------------------------------------

RODADAS = [
    {
        "treino_inicio": "2023-06-01",
        "treino_fim": "2023-08-31",
        "teste_inicio": "2023-09-01",
        "teste_fim": "2023-09-30",
        "nome": "SETEMBRO/2023",
    },
    {
        "treino_inicio": "2023-07-01",
        "treino_fim": "2023-09-30",
        "teste_inicio": "2023-10-01",
        "teste_fim": "2023-10-31",
        "nome": "OUTUBRO/2023",
    },
    {
        "treino_inicio": "2023-08-01",
        "treino_fim": "2023-10-31",
        "teste_inicio": "2023-11-01",
        "teste_fim": "2023-11-30",
        "nome": "NOVEMBRO/2023",
    },
    {
        "treino_inicio": "2023-09-01",
        "treino_fim": "2023-11-30",
        "teste_inicio": "2023-12-01",
        "teste_fim": "2023-12-31",
        "nome": "DEZEMBRO/2023",
    },
]

JANELAS_GLOBAL = [7, 14, 30, 60, 90]
JANELAS_GRUPO = [30, 60, 90]

COLUNAS_GRUPO = {
    "tecnico": "nome_operador",
    "servico": "tipo_servico",
    "cidade": "cidade",
}


# ============================================================
# 1. PREPARAÇÃO DOS DADOS
# ============================================================

print("=" * 80)
print("PREPARAÇÃO DOS DADOS")
print("=" * 80)

# --------------------------------------
# Carregando DF
# --------------------------------------
# Conexão com o banco
conexao = connect_database()

# Lendo a query
query = read_query('query_train.sql')

# Carregando o df
df = pd.read_sql_query(query, conexao)


df["data_agendamento"] = pd.to_datetime(
    df["data_agendamento"],
    errors="coerce"
)

df = df.dropna(
    subset=["data_agendamento"]
)

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

df = df.dropna(
    subset=[TARGET]
)

df[TARGET] = df[TARGET].astype(int)

df = df.sort_values(
    "data_agendamento"
).reset_index(drop=True)

df["data_dia"] = (
    df["data_agendamento"]
    .dt.normalize()
)


print(
    f"Linhas totais: "
    f"{len(df):,}"
)

print(
    f"Data inicial: "
    f"{df['data_agendamento'].min()}"
)

print(
    f"Data final:   "
    f"{df['data_agendamento'].max()}"
)


# ============================================================
# 2. FEATURES DE DATA
# ============================================================

df["ano"] = (
    df["data_agendamento"].dt.year
)

df["mes"] = (
    df["data_agendamento"].dt.month
)

df["dia_mes"] = (
    df["data_agendamento"].dt.day
)

df["dia_semana"] = (
    df["data_agendamento"].dt.dayofweek
)

df["semana_ano"] = (
    df["data_agendamento"]
    .dt.isocalendar()
    .week
    .astype(int)
)

df["fim_de_semana"] = (
    df["dia_semana"] >= 5
).astype(int)


# ============================================================
# 3. EFETIVIDADE GLOBAL MÓVEL
# ============================================================

print()
print("=" * 80)
print("CRIANDO EFETIVIDADE GLOBAL MÓVEL")
print("=" * 80)


global_diario = (
    df.groupby(
        "data_dia",
        as_index=False
    )
    .agg(
        qtd_ordens=(TARGET, "size"),
        efetividade=(TARGET, "mean")
    )
    .sort_values("data_dia")
)


global_diario = (
    global_diario
    .set_index("data_dia")
)


for janela in JANELAS_GLOBAL:

    feature = (
        f"efetividade_global_{janela}d"
    )

    global_diario[feature] = (
        global_diario["efetividade"]
        .shift(1)
        .rolling(
            f"{janela}D",
            min_periods=1
        )
        .mean()
    )


features_global = [
    f"efetividade_global_{janela}d"
    for janela in JANELAS_GLOBAL
]


global_diario = (
    global_diario
    .reset_index()
)


df = df.merge(
    global_diario[
        ["data_dia"] + features_global
    ],
    on="data_dia",
    how="left"
)


# ============================================================
# 4. FUNÇÃO PARA EFETIVIDADE MÓVEL POR GRUPO
# ============================================================

def criar_efetividade_movel(
    dataframe,
    coluna_grupo,
    nome_grupo,
    janelas
):

    diario = (
        dataframe
        .groupby(
            [
                coluna_grupo,
                "data_dia"
            ],
            as_index=False
        )
        .agg(
            qtd_ordens=(TARGET, "size"),
            efetividade=(TARGET, "mean")
        )
        .sort_values(
            [
                coluna_grupo,
                "data_dia"
            ]
        )
    )

    # Evita problemas com NaN nos nomes dos grupos
    diario[coluna_grupo] = (
        diario[coluna_grupo]
        .fillna("__MISSING__")
        .astype(str)
    )

    for janela in janelas:

        feature = (
            f"{nome_grupo}_efetividade_{janela}d"
        )

        diario[feature] = (
            diario
            .groupby(coluna_grupo)["efetividade"]
            .transform(
                lambda x:
                x.shift(1)
                 .rolling(
                     janela,
                     min_periods=1
                 )
                 .mean()
            )
        )

    features = [
        f"{nome_grupo}_efetividade_{janela}d"
        for janela in janelas
    ]

    return (
        diario[
            [
                coluna_grupo,
                "data_dia"
            ] + features
        ],
        features
    )


# ============================================================
# 5. CRIAR FEATURES DE TÉCNICO / SERVIÇO / CIDADE
# ============================================================

print()
print("=" * 80)
print("CRIANDO EFETIVIDADE MÓVEL POR GRUPO")
print("=" * 80)


todas_features_moveis = []


for nome_grupo, coluna in COLUNAS_GRUPO.items():

    print(
        f"Criando: {nome_grupo}"
    )

    resultado, features = (
        criar_efetividade_movel(
            df,
            coluna,
            nome_grupo,
            JANELAS_GRUPO
        )
    )

    # Garantir mesmo tipo da chave
    df[coluna] = (
        df[coluna]
        .fillna("__MISSING__")
        .astype(str)
    )

    df = df.merge(
        resultado,
        on=[
            coluna,
            "data_dia"
        ],
        how="left"
    )

    todas_features_moveis.extend(
        features
    )


# ============================================================
# 6. HISTÓRICO ACUMULADO
# ============================================================

print()
print("=" * 80)
print("CRIANDO HISTÓRICO ACUMULADO")
print("=" * 80)


def criar_historico(
    dataframe,
    coluna_grupo,
    nome_grupo
):

    diario = (
        dataframe
        .groupby(
            [
                coluna_grupo,
                "data_dia"
            ],
            as_index=False
        )
        .agg(
            qtd_ordens=(TARGET, "size"),
            efetividade=(TARGET, "mean")
        )
        .sort_values(
            [
                coluna_grupo,
                "data_dia"
            ]
        )
    )

    diario[coluna_grupo] = (
        diario[coluna_grupo]
        .fillna("__MISSING__")
        .astype(str)
    )

    diario[
        f"{nome_grupo}_qtd_historica"
    ] = (
        diario
        .groupby(coluna_grupo)["qtd_ordens"]
        .transform(
            lambda x:
            x.shift(1)
             .expanding()
             .sum()
        )
    )

    diario[
        f"{nome_grupo}_efetividade_historica"
    ] = (
        diario
        .groupby(coluna_grupo)["efetividade"]
        .transform(
            lambda x:
            x.shift(1)
             .expanding()
             .mean()
        )
    )

    return diario[
        [
            coluna_grupo,
            "data_dia",
            f"{nome_grupo}_qtd_historica",
            f"{nome_grupo}_efetividade_historica",
        ]
    ]


for nome_grupo, coluna in COLUNAS_GRUPO.items():

    historico = criar_historico(
        df,
        coluna,
        nome_grupo
    )

    df = df.merge(
        historico,
        on=[
            coluna,
            "data_dia"
        ],
        how="left"
    )


# ============================================================
# 7. FEATURES RELATIVAS AO GLOBAL
# ============================================================

print()
print("=" * 80)
print("CRIANDO FEATURES RELATIVAS AO GLOBAL")
print("=" * 80)


features_relativas = []


for nome_grupo in [
    "tecnico",
    "servico",
    "cidade"
]:

    for janela in JANELAS_GRUPO:

        feature_grupo = (
            f"{nome_grupo}_efetividade_{janela}d"
        )

        feature_global = (
            f"efetividade_global_{janela}d"
        )

        feature_relativa = (
            f"{nome_grupo}_vs_global_{janela}d"
        )

        df[feature_relativa] = (
            df[feature_grupo]
            -
            df[feature_global]
        )

        features_relativas.append(
            feature_relativa
        )


# ============================================================
# 8. FEATURES DO MODELO
# ============================================================

categorical_features = [
    "tipo_atividade",
    "slot",
    "detalhe_atividade",
    "tipo_servico",
    "cidade",
    "segmento",
    "categoria_cliente",
    "cluster_zeus",
    "gerencia",
    "nome_operador",
]


numeric_features = [
    "ano",
    "mes",
    "dia_mes",
    "dia_semana",
    "semana_ano",
    "fim_de_semana",

    "tecnico_efetividade_historica",
    "tecnico_qtd_historica",

    "servico_efetividade_historica",
    "servico_qtd_historica",

    "cidade_efetividade_historica",
    "cidade_qtd_historica",

    # Global
    "efetividade_global_7d",
    "efetividade_global_14d",
    "efetividade_global_30d",
    "efetividade_global_60d",
    "efetividade_global_90d",

    # Técnico
    "tecnico_efetividade_30d",
    "tecnico_efetividade_60d",
    "tecnico_efetividade_90d",

    # Serviço
    "servico_efetividade_30d",
    "servico_efetividade_60d",
    "servico_efetividade_90d",

    # Cidade
    "cidade_efetividade_30d",
    "cidade_efetividade_60d",
    "cidade_efetividade_90d",

    # Relativas
    "tecnico_vs_global_30d",
    "tecnico_vs_global_60d",
    "tecnico_vs_global_90d",

    "servico_vs_global_30d",
    "servico_vs_global_60d",
    "servico_vs_global_90d",

    "cidade_vs_global_30d",
    "cidade_vs_global_60d",
    "cidade_vs_global_90d",
]


# ============================================================
# 9. DURACAO
# ============================================================

if "duracao" in df.columns:

    df["duracao"] = pd.to_numeric(
        df["duracao"],
        errors="coerce"
    )

    percentual_nulo = (
        df["duracao"].isna().mean()
    )

    print()
    print(
        f"Nulos em duracao: "
        f"{percentual_nulo:.2%}"
    )

    if percentual_nulo < 1:

        numeric_features.append(
            "duracao"
        )

    else:

        print(
            "duracao será ignorada."
        )


# ============================================================
# 10. CORREÇÃO DE TIPOS
# ============================================================

for coluna in categorical_features:

    df[coluna] = (
        df[coluna]
        .fillna("__MISSING__")
        .astype(str)
    )


for coluna in numeric_features:

    df[coluna] = pd.to_numeric(
        df[coluna],
        errors="coerce"
    )


# ============================================================
# 11. FUNÇÃO PARA TREINAR UMA RODADA
# ============================================================

def executar_rodada(
    dataframe,
    rodada
):

    print()
    print()
    print("#" * 80)
    print(
        f"SIMULAÇÃO: {rodada['nome']}"
    )
    print("#" * 80)

    treino_inicio = pd.Timestamp(
        rodada["treino_inicio"]
    )

    treino_fim = pd.Timestamp(
        rodada["treino_fim"]
    )

    teste_inicio = pd.Timestamp(
        rodada["teste_inicio"]
    )

    teste_fim = pd.Timestamp(
        rodada["teste_fim"]
    )

    # --------------------------------------------------------
    # Separação
    # --------------------------------------------------------

    mask_treino = (
        (dataframe["data_agendamento"] >= treino_inicio)
        &
        (dataframe["data_agendamento"] <= treino_fim)
    )

    mask_teste = (
        (dataframe["data_agendamento"] >= teste_inicio)
        &
        (dataframe["data_agendamento"] <= teste_fim)
    )

    treino = dataframe.loc[
        mask_treino
    ].copy()

    teste = dataframe.loc[
        mask_teste
    ].copy()

    print()
    print("PERÍODOS")

    print(
        f"Treino: "
        f"{treino_inicio.date()} "
        f"até "
        f"{treino_fim.date()}"
    )

    print(
        f"Teste:  "
        f"{teste_inicio.date()} "
        f"até "
        f"{teste_fim.date()}"
    )

    print()
    print(
        f"Ordens treino: "
        f"{len(treino):,}"
    )

    print(
        f"Ordens teste:  "
        f"{len(teste):,}"
    )

    print()
    print(
        f"Efetividade treino: "
        f"{treino[TARGET].mean():.2%}"
    )

    print(
        f"Efetividade teste:  "
        f"{teste[TARGET].mean():.2%}"
    )

    # --------------------------------------------------------
    # X / y
    # --------------------------------------------------------

    X_train = treino[
        categorical_features +
        numeric_features
    ]

    y_train = treino[
        TARGET
    ]

    X_test = teste[
        categorical_features +
        numeric_features
    ]

    y_test = teste[
        TARGET
    ]

    # --------------------------------------------------------
    # Preprocessor
    # --------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numeric_features
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    print()
    print("Transformando dados...")

    X_train_transformed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_test_transformed = (
        preprocessor.transform(
            X_test
        )
    )

    print(
        f"Shape treino: "
        f"{X_train_transformed.shape}"
    )

    print(
        f"Shape teste:  "
        f"{X_test_transformed.shape}"
    )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    print()
    print("Treinando XGBoost...")

    modelo = XGBClassifier(

        n_estimators=250,

        learning_rate=0.05,

        max_depth=7,

        min_child_weight=5,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="binary:logistic",

        eval_metric="logloss",

        tree_method="hist",

        max_bin=256,

        n_jobs=-1,

        random_state=42,
    )

    modelo.fit(
        X_train_transformed,
        y_train,
        verbose=False
    )

    # --------------------------------------------------------
    # Previsão
    # --------------------------------------------------------

    probabilidades = (
        modelo
        .predict_proba(
            X_test_transformed
        )[:, 1]
    )

    previsoes = (
        probabilidades >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # Métricas
    # --------------------------------------------------------

    auc = roc_auc_score(
        y_test,
        probabilidades
    )

    accuracy = accuracy_score(
        y_test,
        previsoes
    )

    precision = precision_score(
        y_test,
        previsoes,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        previsoes,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        previsoes,
        zero_division=0
    )

    brier = brier_score_loss(
        y_test,
        probabilidades
    )

    logloss = log_loss(
        y_test,
        probabilidades
    )

    prob_media = (
        probabilidades.mean()
    )

    efetividade_real = (
        y_test.mean()
    )

    erro_pp = (
        prob_media -
        efetividade_real
    ) * 100

    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print(
        f"RESULTADO: "
        f"{rodada['nome']}"
    )
    print("-" * 80)

    print(
        f"AUC:                 "
        f"{auc:.4f}"
    )

    print(
        f"Accuracy:            "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision:           "
        f"{precision:.4f}"
    )

    print(
        f"Recall:              "
        f"{recall:.4f}"
    )

    print(
        f"F1:                  "
        f"{f1:.4f}"
    )

    print(
        f"Brier:               "
        f"{brier:.4f}"
    )

    print(
        f"Log Loss:            "
        f"{logloss:.4f}"
    )

    print()

    print(
        f"Probabilidade média: "
        f"{prob_media:.2%}"
    )

    print(
        f"Efetividade real:    "
        f"{efetividade_real:.2%}"
    )

    print(
        f"Erro calibração:     "
        f"{erro_pp:+.2f} pp"
    )

    # --------------------------------------------------------
    # Resultado mensal
    # --------------------------------------------------------

    teste["probabilidade"] = (
        probabilidades
    )

    return {
        "periodo_treino_inicio":
            treino_inicio,

        "periodo_treino_fim":
            treino_fim,

        "periodo_teste":
            rodada["nome"],

        "qtd_treino":
            len(treino),

        "qtd_teste":
            len(teste),

        "efetividade_treino":
            treino[TARGET].mean(),

        "efetividade_real":
            efetividade_real,

        "probabilidade_media":
            prob_media,

        "erro_pp":
            erro_pp,

        "auc":
            auc,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "brier":
            brier,

        "logloss":
            logloss,
    }


# ============================================================
# 12. EXECUTAR TODAS AS RODADAS
# ============================================================

resultados = []


for rodada in RODADAS:

    resultado = executar_rodada(
        df,
        rodada
    )

    resultados.append(
        resultado
    )


# ============================================================
# 13. RESULTADO FINAL
# ============================================================

resultados_df = pd.DataFrame(
    resultados
)


print()
print()
print("=" * 100)
print("RESULTADO FINAL DA SIMULAÇÃO")
print("=" * 100)


colunas_exibicao = [
    "periodo_teste",
    "qtd_treino",
    "qtd_teste",
    "efetividade_treino",
    "efetividade_real",
    "probabilidade_media",
    "erro_pp",
    "auc",
    "brier",
    "logloss",
]


print(
    resultados_df[
        colunas_exibicao
    ].to_string(
        index=False
    )
)


# ============================================================
# 14. MÉDIA DOS RESULTADOS
# ============================================================

print()
print("=" * 100)
print("MÉDIA DAS RODADAS")
print("=" * 100)


print(
    f"AUC médio:       "
    f"{resultados_df['auc'].mean():.4f}"
)

print(
    f"Brier médio:     "
    f"{resultados_df['brier'].mean():.4f}"
)

print(
    f"Log Loss médio:  "
    f"{resultados_df['logloss'].mean():.4f}"
)

print(
    f"Erro médio:      "
    f"{resultados_df['erro_pp'].mean():+.2f} pp"
)

print(
    f"Erro absoluto:   "
    f"{resultados_df['erro_pp'].abs().mean():.2f} pp"
)


# ============================================================
# 15. SALVAR RESULTADOS
# ============================================================

# resultados_df.to_csv(
#     "simulacao_producao_2023.csv",
#     index=False
# )


# print()
# print(
#     "Resultado salvo em:"
# )

# print(
#     "simulacao_producao_2023.csv"
# )


# ============================================================
# 16. CONCLUSÃO AUTOMÁTICA
# ============================================================

print()
print("=" * 100)
print("CONCLUSÃO")
print("=" * 100)


erro_absoluto_medio = (
    resultados_df[
        "erro_pp"
    ]
    .abs()
    .mean()
)


if erro_absoluto_medio <= 3:

    print(
        "Excelente calibração: "
        "erro médio absoluto <= 3 pp."
    )

elif erro_absoluto_medio <= 5:

    print(
        "Boa calibração: "
        "erro médio absoluto <= 5 pp."
    )

elif erro_absoluto_medio <= 10:

    print(
        "Calibração aceitável, "
        "mas ainda pode ser melhorada."
    )

else:

    print(
        "Calibração ainda problemática."
    )


print()
print(
    "A simulação terminou."
)