import pandas as pd
import numpy as np
from src.Database.database_config import connect_database, read_query

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    log_loss
)

from xgboost import XGBClassifier


# ============================================================
# CONFIGURAÇÕES
# ============================================================

TARGET = "efetividade"
DATE_COL = "data_agendamento"

JANELA_TREINO_MESES = 6

DATAS_TESTE = [
    "2023-09",
    "2023-10",
    "2023-11",
    "2023-12",
]

RANDOM_STATE = 42


# ============================================================
# CÓPIA E PREPARAÇÃO
# ============================================================
# Conexão com o banco
conexao = connect_database()

# Lendo a query
query = read_query('query_train.sql')

# Carregando o df
df = pd.read_sql_query(query, conexao)

df[DATE_COL] = pd.to_datetime(df[DATE_COL])

df = df.sort_values(DATE_COL).reset_index(drop=True)

df[TARGET] = df[TARGET].astype(int)

print("=" * 70)
print("PREPARAÇÃO DOS DADOS")
print("=" * 70)

print(f"Linhas: {len(df):,}")
print(f"Data inicial: {df[DATE_COL].min()}")
print(f"Data final:   {df[DATE_COL].max()}")


# ============================================================
# FEATURES DE DATA
# ============================================================

df["ano"] = df[DATE_COL].dt.year
df["mes"] = df[DATE_COL].dt.month
df["dia"] = df[DATE_COL].dt.day
df["dia_semana"] = df[DATE_COL].dt.dayofweek
df["semana_ano"] = df[DATE_COL].dt.isocalendar().week.astype(int)
df["fim_semana"] = (df["dia_semana"] >= 5).astype(int)


# ============================================================
# FUNÇÃO PARA CALCULAR EFETIVIDADE MÓVEL GLOBAL
# ============================================================

def criar_rolling_global(df, janelas=(7, 14, 30, 60, 90)):

    diario = (
        df.groupby(DATE_COL)[TARGET]
        .agg(["sum", "count"])
        .sort_index()
    )

    diario["taxa"] = (
        diario["sum"] / diario["count"]
    )

    diario = diario.sort_index()

    for janela in janelas:

        diario[f"efetividade_global_{janela}d"] = (
            diario["taxa"]
            .shift(1)
            .rolling(f"{janela}D", min_periods=1)
            .mean()
        )

    colunas = [
        f"efetividade_global_{janela}d"
        for janela in janelas
    ]

    resultado = diario[colunas].reset_index()

    return resultado


print("\nCriando efetividade móvel global...")

rolling_global = criar_rolling_global(df)

df = df.merge(
    rolling_global,
    on=DATE_COL,
    how="left"
)


# ============================================================
# FUNÇÃO PARA FEATURES MÓVEIS POR GRUPO
# ============================================================

def criar_rolling_grupo(
    df,
    coluna_grupo,
    prefixo,
    janelas=(30, 60, 90)
):

    diario = (
        df.groupby([DATE_COL, coluna_grupo])[TARGET]
        .agg(["sum", "count"])
        .reset_index()
    )

    diario["taxa"] = (
        diario["sum"] / diario["count"]
    )

    diario = diario.sort_values(
        [coluna_grupo, DATE_COL]
    )

    for janela in janelas:

        diario[
            f"{prefixo}_efetividade_{janela}d"
        ] = (
            diario
            .groupby(coluna_grupo)["taxa"]
            .transform(
                lambda x:
                    x.shift(1)
                    .rolling(janela, min_periods=1)
                    .mean()
            )
        )

    colunas = [
        f"{prefixo}_efetividade_{janela}d"
        for janela in janelas
    ]

    return diario[
        [DATE_COL, coluna_grupo] + colunas
    ]


# ============================================================
# TÉCNICO
# ============================================================

print("Criando efetividade móvel do técnico...")

rolling_tecnico = criar_rolling_grupo(
    df,
    "nome_operador",
    "tecnico",
    (30, 60, 90)
)

df = df.merge(
    rolling_tecnico,
    on=[DATE_COL, "nome_operador"],
    how="left"
)


# ============================================================
# SERVIÇO
# ============================================================

print("Criando efetividade móvel do serviço...")

rolling_servico = criar_rolling_grupo(
    df,
    "tipo_servico",
    "servico",
    (30, 60, 90)
)

df = df.merge(
    rolling_servico,
    on=[DATE_COL, "tipo_servico"],
    how="left"
)


# ============================================================
# CIDADE
# ============================================================

print("Criando efetividade móvel da cidade...")

rolling_cidade = criar_rolling_grupo(
    df,
    "cidade",
    "cidade",
    (30, 60, 90)
)

df = df.merge(
    rolling_cidade,
    on=[DATE_COL, "cidade"],
    how="left"
)


# ============================================================
# EFETIVIDADE HISTÓRICA DO TÉCNICO
# ============================================================

print("Criando efetividade histórica do técnico...")

tecnico_historico = (
    df.groupby(["nome_operador", DATE_COL])[TARGET]
    .agg(["sum", "count"])
    .reset_index()
)

tecnico_historico["taxa"] = (
    tecnico_historico["sum"]
    / tecnico_historico["count"]
)

tecnico_historico = tecnico_historico.sort_values(
    ["nome_operador", DATE_COL]
)

tecnico_historico["tecnico_efetividade_historica"] = (
    tecnico_historico
    .groupby("nome_operador")["taxa"]
    .transform(
        lambda x:
            x.shift(1)
            .expanding(min_periods=1)
            .mean()
    )
)

df = df.merge(
    tecnico_historico[
        [
            DATE_COL,
            "nome_operador",
            "tecnico_efetividade_historica"
        ]
    ],
    on=[DATE_COL, "nome_operador"],
    how="left"
)


# ============================================================
# RELAÇÃO TÉCNICO x GLOBAL
# ============================================================

for janela in (30, 60, 90):

    df[f"tecnico_vs_global_{janela}d"] = (
        df[f"tecnico_efetividade_{janela}d"]
        -
        df[f"efetividade_global_{janela}d"]
    )


# ============================================================
# RELAÇÃO SERVIÇO x GLOBAL
# ============================================================

for janela in (30, 60, 90):

    df[f"servico_vs_global_{janela}d"] = (
        df[f"servico_efetividade_{janela}d"]
        -
        df[f"efetividade_global_{janela}d"]
    )


# ============================================================
# RELAÇÃO CIDADE x GLOBAL
# ============================================================

for janela in (30, 60, 90):

    df[f"cidade_vs_global_{janela}d"] = (
        df[f"cidade_efetividade_{janela}d"]
        -
        df[f"efetividade_global_{janela}d"]
    )


# ============================================================
# LIMPEZA
# ============================================================

# Duracao está 100% nula no dataset atual
if "duracao" in df.columns:
    if df["duracao"].isna().all():
        df = df.drop(columns=["duracao"])


# ============================================================
# FEATURES
# ============================================================

features_numericas = [
    "ano",
    "mes",
    "dia",
    "dia_semana",
    "semana_ano",
    "fim_semana",

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
    "tecnico_efetividade_historica",

    # Serviço
    "servico_efetividade_30d",
    "servico_efetividade_60d",
    "servico_efetividade_90d",

    # Cidade
    "cidade_efetividade_30d",
    "cidade_efetividade_60d",
    "cidade_efetividade_90d",

    # Relativos
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


features_categoricas = [
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


# ============================================================
# GARANTIR QUE SOMENTE COLUNAS EXISTENTES SEJAM UTILIZADAS
# ============================================================

features_numericas = [
    col
    for col in features_numericas
    if col in df.columns
]

features_categoricas = [
    col
    for col in features_categoricas
    if col in df.columns
]

FEATURES = features_numericas + features_categoricas


print("\nFeatures utilizadas:")
print(f"Numéricas: {len(features_numericas)}")
print(f"Categóricas: {len(features_categoricas)}")
print(f"Total: {len(FEATURES)}")


# ============================================================
# PIPELINE
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            features_numericas
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            features_categoricas
        )
    ]
)


model = XGBClassifier(
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
    random_state=RANDOM_STATE
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ============================================================
# SIMULAÇÃO DE PRODUÇÃO — 6 MESES
# ============================================================

resultados = []

predicoes = []

print("\n")
print("=" * 70)
print("SIMULAÇÃO DE PRODUÇÃO — JANELA DE 6 MESES")
print("=" * 70)


for mes_teste in DATAS_TESTE:

    data_teste = pd.Timestamp(
        f"{mes_teste}-01"
    )

    data_fim_treino = (
        data_teste
        - pd.Timedelta(days=1)
    )

    data_inicio_treino = (
        data_teste
        - pd.DateOffset(
            months=JANELA_TREINO_MESES
        )
    )

    data_fim_teste = (
        data_teste
        + pd.offsets.MonthEnd(0)
    )

    treino = df[
        (df[DATE_COL] >= data_inicio_treino)
        &
        (df[DATE_COL] <= data_fim_treino)
    ].copy()

    teste = df[
        (df[DATE_COL] >= data_teste)
        &
        (df[DATE_COL] <= data_fim_teste)
    ].copy()

    print("\n" + "-" * 70)

    print(
        f"TESTE: {mes_teste}"
    )

    print(
        f"Treino: "
        f"{data_inicio_treino.date()} "
        f"até "
        f"{data_fim_treino.date()}"
    )

    print(
        f"Teste:  "
        f"{data_teste.date()} "
        f"até "
        f"{data_fim_teste.date()}"
    )

    print(
        f"Linhas treino: {len(treino):,}"
    )

    print(
        f"Linhas teste:  {len(teste):,}"
    )

    if len(treino) == 0 or len(teste) == 0:

        print("Sem dados suficientes. Pulando.")

        continue

    X_train = treino[FEATURES]
    y_train = treino[TARGET]

    X_test = teste[FEATURES]
    y_test = teste[TARGET]

    print(
        f"Efetividade treino: "
        f"{y_train.mean():.2%}"
    )

    print(
        f"Efetividade teste:  "
        f"{y_test.mean():.2%}"
    )

    # --------------------------------------------------------
    # TREINAMENTO
    # --------------------------------------------------------

    pipeline.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # PREDIÇÃO
    # --------------------------------------------------------

    probabilidade = pipeline.predict_proba(
        X_test
    )[:, 1]

    predicao = (
        probabilidade >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # MÉTRICAS
    # --------------------------------------------------------

    auc = roc_auc_score(
        y_test,
        probabilidade
    )

    accuracy = accuracy_score(
        y_test,
        predicao
    )

    precision = precision_score(
        y_test,
        predicao,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predicao,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predicao,
        zero_division=0
    )

    brier = brier_score_loss(
        y_test,
        probabilidade
    )

    logloss = log_loss(
        y_test,
        probabilidade
    )

    prob_media = probabilidade.mean()

    efetividade_real = y_test.mean()

    erro_calibracao = (
        prob_media
        -
        efetividade_real
    )

    # --------------------------------------------------------
    # RESULTADOS
    # --------------------------------------------------------

    print("\nMÉTRICAS")

    print(
        f"AUC:                {auc:.4f}"
    )

    print(
        f"Accuracy:            {accuracy:.4f}"
    )

    print(
        f"Precision:           {precision:.4f}"
    )

    print(
        f"Recall:              {recall:.4f}"
    )

    print(
        f"F1:                  {f1:.4f}"
    )

    print(
        f"Brier:               {brier:.4f}"
    )

    print(
        f"Log Loss:            {logloss:.4f}"
    )

    print(
        f"Probabilidade média: {prob_media:.2%}"
    )

    print(
        f"Efetividade real:    {efetividade_real:.2%}"
    )

    print(
        f"Erro calibração:     "
        f"{erro_calibracao:+.2%}"
    )

    resultados.append({
        "mes": mes_teste,
        "treino_inicio": data_inicio_treino,
        "treino_fim": data_fim_treino,
        "teste_inicio": data_teste,
        "teste_fim": data_fim_teste,
        "qtd_treino": len(treino),
        "qtd_teste": len(teste),
        "efetividade_treino": y_train.mean(),
        "efetividade_real": efetividade_real,
        "probabilidade_media": prob_media,
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier": brier,
        "log_loss": logloss,
        "erro_calibracao": erro_calibracao,
        "erro_calibracao_abs": abs(
            erro_calibracao
        )
    })

    # --------------------------------------------------------
    # SALVAR PREDIÇÕES
    # --------------------------------------------------------

    teste_resultado = teste[
        [
            DATE_COL,
            TARGET
        ]
    ].copy()

    teste_resultado["probabilidade"] = (
        probabilidade
    )

    teste_resultado["predicao"] = (
        predicao
    )

    teste_resultado["mes_teste"] = (
        mes_teste
    )

    predicoes.append(
        teste_resultado
    )


# ============================================================
# RESULTADOS FINAIS
# ============================================================

df_resultados = pd.DataFrame(
    resultados
)

df_predicoes = pd.concat(
    predicoes,
    ignore_index=True
)


print("\n")
print("=" * 70)
print("RESULTADO FINAL — 6 MESES")
print("=" * 70)

print(
    df_resultados[
        [
            "mes",
            "qtd_treino",
            "qtd_teste",
            "efetividade_treino",
            "efetividade_real",
            "probabilidade_media",
            "auc",
            "brier",
            "log_loss",
            "erro_calibracao"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# MÉDIAS
# ============================================================

print("\n")
print("=" * 70)
print("MÉDIAS")
print("=" * 70)

print(
    f"AUC médio: "
    f"{df_resultados['auc'].mean():.4f}"
)

print(
    f"Brier médio: "
    f"{df_resultados['brier'].mean():.4f}"
)

print(
    f"Log Loss médio: "
    f"{df_resultados['log_loss'].mean():.4f}"
)

print(
    f"Erro calibração médio: "
    f"{df_resultados['erro_calibracao'].mean():+.2%}"
)

print(
    f"Erro calibração absoluto médio: "
    f"{df_resultados['erro_calibracao_abs'].mean():.2%}"
)


# ============================================================
# SALVAR RESULTADOS
# ============================================================

# df_resultados.to_csv(
#     "simulacao_producao_6_meses.csv",
#     index=False
# )

# df_predicoes.to_csv(
#     "predicoes_producao_6_meses.csv",
#     index=False
# )


# print("\nArquivos gerados:")

# print(
#     "simulacao_producao_6_meses.csv"
# )

# print(
#     "predicoes_producao_6_meses.csv"
# )




































































































































































import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    log_loss
)

from xgboost import XGBClassifier


# ============================================================
# CONFIGURAÇÕES
# ============================================================

TARGET = "efetividade"
DATE_COL = "data_agendamento"

JANELA_TREINO_MESES = 3
DIAS_TESTE = 7

# Vamos simular o período em que já temos o cenário de drift
DATA_INICIO_SIMULACAO = "2023-09-01"
DATA_FIM_SIMULACAO = "2023-12-31"

RANDOM_STATE = 42


# ============================================================
# PREPARAÇÃO
# ============================================================
# ============================================================
# CÓPIA E PREPARAÇÃO
# ============================================================
# Conexão com o banco
conexao = connect_database()

# Lendo a query
query = read_query('query_train.sql')

# Carregando o df
df = pd.read_sql_query(query, conexao)


df[DATE_COL] = pd.to_datetime(df[DATE_COL])

df = df.sort_values(DATE_COL).reset_index(drop=True)

df[TARGET] = df[TARGET].astype(int)


print("=" * 70)
print("PREPARAÇÃO DOS DADOS")
print("=" * 70)

print(f"Linhas: {len(df):,}")
print(f"Data inicial: {df[DATE_COL].min()}")
print(f"Data final:   {df[DATE_COL].max()}")


# ============================================================
# FEATURES DE DATA
# ============================================================

df["ano"] = df[DATE_COL].dt.year
df["mes"] = df[DATE_COL].dt.month
df["dia"] = df[DATE_COL].dt.day
df["dia_semana"] = df[DATE_COL].dt.dayofweek
df["semana_ano"] = df[DATE_COL].dt.isocalendar().week.astype(int)
df["fim_semana"] = (df["dia_semana"] >= 5).astype(int)


# ============================================================
# EFETIVIDADE MÓVEL GLOBAL
# ============================================================

def criar_rolling_global(
    df,
    janelas=(7, 14, 30, 60, 90)
):

    diario = (
        df.groupby(DATE_COL)[TARGET]
        .agg(["sum", "count"])
        .sort_index()
    )

    diario["taxa"] = (
        diario["sum"]
        / diario["count"]
    )

    for janela in janelas:

        diario[
            f"efetividade_global_{janela}d"
        ] = (
            diario["taxa"]
            .shift(1)
            .rolling(
                f"{janela}D",
                min_periods=1
            )
            .mean()
        )

    colunas = [
        f"efetividade_global_{janela}d"
        for janela in janelas
    ]

    return diario[colunas].reset_index()


print("\nCriando efetividade móvel global...")

rolling_global = criar_rolling_global(df)

df = df.merge(
    rolling_global,
    on=DATE_COL,
    how="left"
)


# ============================================================
# EFETIVIDADE MÓVEL POR GRUPO
# ============================================================

def criar_rolling_grupo(
    df,
    coluna_grupo,
    prefixo,
    janelas=(30, 60, 90)
):

    diario = (
        df.groupby(
            [DATE_COL, coluna_grupo]
        )[TARGET]
        .agg(["sum", "count"])
        .reset_index()
    )

    diario["taxa"] = (
        diario["sum"]
        / diario["count"]
    )

    diario = diario.sort_values(
        [coluna_grupo, DATE_COL]
    )

    for janela in janelas:

        diario[
            f"{prefixo}_efetividade_{janela}d"
        ] = (
            diario
            .groupby(coluna_grupo)["taxa"]
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

    colunas = [
        f"{prefixo}_efetividade_{janela}d"
        for janela in janelas
    ]

    return diario[
        [DATE_COL, coluna_grupo] + colunas
    ]


# ============================================================
# TÉCNICO
# ============================================================

print("Criando efetividade móvel do técnico...")

rolling_tecnico = criar_rolling_grupo(
    df,
    "nome_operador",
    "tecnico",
    (30, 60, 90)
)

df = df.merge(
    rolling_tecnico,
    on=[
        DATE_COL,
        "nome_operador"
    ],
    how="left"
)


# ============================================================
# SERVIÇO
# ============================================================

print("Criando efetividade móvel do serviço...")

rolling_servico = criar_rolling_grupo(
    df,
    "tipo_servico",
    "servico",
    (30, 60, 90)
)

df = df.merge(
    rolling_servico,
    on=[
        DATE_COL,
        "tipo_servico"
    ],
    how="left"
)


# ============================================================
# CIDADE
# ============================================================

print("Criando efetividade móvel da cidade...")

rolling_cidade = criar_rolling_grupo(
    df,
    "cidade",
    "cidade",
    (30, 60, 90)
)

df = df.merge(
    rolling_cidade,
    on=[
        DATE_COL,
        "cidade"
    ],
    how="left"
)


# ============================================================
# EFETIVIDADE HISTÓRICA DO TÉCNICO
# ============================================================

print("Criando efetividade histórica do técnico...")

tecnico_historico = (
    df.groupby(
        ["nome_operador", DATE_COL]
    )[TARGET]
    .agg(["sum", "count"])
    .reset_index()
)

tecnico_historico["taxa"] = (
    tecnico_historico["sum"]
    / tecnico_historico["count"]
)

tecnico_historico = tecnico_historico.sort_values(
    ["nome_operador", DATE_COL]
)

tecnico_historico[
    "tecnico_efetividade_historica"
] = (
    tecnico_historico
    .groupby("nome_operador")["taxa"]
    .transform(
        lambda x:
            x.shift(1)
            .expanding(min_periods=1)
            .mean()
    )
)

df = df.merge(
    tecnico_historico[
        [
            DATE_COL,
            "nome_operador",
            "tecnico_efetividade_historica"
        ]
    ],
    on=[
        DATE_COL,
        "nome_operador"
    ],
    how="left"
)


# ============================================================
# RELAÇÕES COM O GLOBAL
# ============================================================

for janela in (30, 60, 90):

    df[
        f"tecnico_vs_global_{janela}d"
    ] = (
        df[
            f"tecnico_efetividade_{janela}d"
        ]
        -
        df[
            f"efetividade_global_{janela}d"
        ]
    )

    df[
        f"servico_vs_global_{janela}d"
    ] = (
        df[
            f"servico_efetividade_{janela}d"
        ]
        -
        df[
            f"efetividade_global_{janela}d"
        ]
    )

    df[
        f"cidade_vs_global_{janela}d"
    ] = (
        df[
            f"cidade_efetividade_{janela}d"
        ]
        -
        df[
            f"efetividade_global_{janela}d"
        ]
    )


# ============================================================
# DURACAO
# ============================================================

if "duracao" in df.columns:

    if df["duracao"].isna().all():

        df = df.drop(
            columns=["duracao"]
        )


# ============================================================
# FEATURES NUMÉRICAS
# ============================================================

features_numericas = [

    "ano",
    "mes",
    "dia",
    "dia_semana",
    "semana_ano",
    "fim_semana",

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
    "tecnico_efetividade_historica",

    # Serviço
    "servico_efetividade_30d",
    "servico_efetividade_60d",
    "servico_efetividade_90d",

    # Cidade
    "cidade_efetividade_30d",
    "cidade_efetividade_60d",
    "cidade_efetividade_90d",

    # Relativos
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
# FEATURES CATEGÓRICAS
# ============================================================

features_categoricas = [

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


# ============================================================
# GARANTIR COLUNAS EXISTENTES
# ============================================================

features_numericas = [
    col
    for col in features_numericas
    if col in df.columns
]

features_categoricas = [
    col
    for col in features_categoricas
    if col in df.columns
]

FEATURES = (
    features_numericas
    +
    features_categoricas
)


print("\nFeatures utilizadas:")
print(
    f"Numéricas: {len(features_numericas)}"
)

print(
    f"Categóricas: {len(features_categoricas)}"
)

print(
    f"Total: {len(FEATURES)}"
)


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[

        (
            "num",
            "passthrough",
            features_numericas
        ),

        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            features_categoricas
        )
    ]
)


# ============================================================
# MODELO
# ============================================================

model = XGBClassifier(

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

    random_state=RANDOM_STATE
)


# ============================================================
# PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ============================================================
# DATAS DA SIMULAÇÃO
# ============================================================

data_inicio = pd.Timestamp(
    DATA_INICIO_SIMULACAO
)

data_fim = pd.Timestamp(
    DATA_FIM_SIMULACAO
)


# ============================================================
# SIMULAÇÃO WALK-FORWARD SEMANAL
# ============================================================

resultados = []

predicoes = []

data_teste_inicio = data_inicio

contador = 0


print("\n")
print("=" * 70)
print("SIMULAÇÃO DE PRODUÇÃO — 3 MESES / RETREINO SEMANAL")
print("=" * 70)


while data_teste_inicio <= data_fim:

    contador += 1

    # --------------------------------------------------------
    # PERÍODO DO TESTE
    # --------------------------------------------------------

    data_teste_fim = (
        data_teste_inicio
        + pd.Timedelta(
            days=DIAS_TESTE - 1
        )
    )

    # Não ultrapassar o fim da simulação
    if data_teste_fim > data_fim:

        data_teste_fim = data_fim


    # --------------------------------------------------------
    # PERÍODO DO TREINAMENTO
    # --------------------------------------------------------

    data_inicio_treino = (
        data_teste_inicio
        - pd.DateOffset(
            months=JANELA_TREINO_MESES
        )
    )

    data_fim_treino = (
        data_teste_inicio
        - pd.Timedelta(
            days=1
        )
    )


    # --------------------------------------------------------
    # SELECIONAR TREINO
    # --------------------------------------------------------

    treino = df[
        (df[DATE_COL] >= data_inicio_treino)
        &
        (df[DATE_COL] <= data_fim_treino)
    ].copy()


    # --------------------------------------------------------
    # SELECIONAR TESTE
    # --------------------------------------------------------

    teste = df[
        (df[DATE_COL] >= data_teste_inicio)
        &
        (df[DATE_COL] <= data_teste_fim)
    ].copy()


    if (
        len(treino) == 0
        or
        len(teste) == 0
    ):

        print(
            f"\nSemana {contador}: "
            f"sem dados. Pulando."
        )

        data_teste_inicio = (
            data_teste_inicio
            + pd.Timedelta(
                days=DIAS_TESTE
            )
        )

        continue


    # --------------------------------------------------------
    # DADOS
    # --------------------------------------------------------

    X_train = treino[FEATURES]

    y_train = treino[TARGET]

    X_test = teste[FEATURES]

    y_test = teste[TARGET]


    # --------------------------------------------------------
    # TREINAMENTO
    # --------------------------------------------------------

    pipeline.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # PREDIÇÃO
    # --------------------------------------------------------

    probabilidade = (
        pipeline
        .predict_proba(X_test)[:, 1]
    )

    predicao = (
        probabilidade >= 0.5
    ).astype(int)


    # --------------------------------------------------------
    # MÉTRICAS
    # --------------------------------------------------------

    auc = roc_auc_score(
        y_test,
        probabilidade
    )

    accuracy = accuracy_score(
        y_test,
        predicao
    )

    precision = precision_score(
        y_test,
        predicao,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predicao,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predicao,
        zero_division=0
    )

    brier = brier_score_loss(
        y_test,
        probabilidade
    )

    logloss = log_loss(
        y_test,
        probabilidade
    )

    prob_media = (
        probabilidade.mean()
    )

    efetividade_real = (
        y_test.mean()
    )

    erro_calibracao = (
        prob_media
        -
        efetividade_real
    )


    # --------------------------------------------------------
    # RESULTADO DA SEMANA
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    print(
        f"SEMANA {contador}"
    )

    print(
        f"Teste: "
        f"{data_teste_inicio.date()} "
        f"até "
        f"{data_teste_fim.date()}"
    )

    print(
        f"Treino: "
        f"{data_inicio_treino.date()} "
        f"até "
        f"{data_fim_treino.date()}"
    )

    print(
        f"Treino: {len(treino):,} linhas"
    )

    print(
        f"Teste:  {len(teste):,} linhas"
    )

    print(
        f"Efetividade treino: "
        f"{y_train.mean():.2%}"
    )

    print(
        f"Efetividade real:    "
        f"{efetividade_real:.2%}"
    )

    print(
        f"Probabilidade média: "
        f"{prob_media:.2%}"
    )

    print(
        f"AUC:       {auc:.4f}"
    )

    print(
        f"Brier:     {brier:.4f}"
    )

    print(
        f"Log Loss:  {logloss:.4f}"
    )

    print(
        f"Calibração: "
        f"{erro_calibracao:+.2%}"
    )


    # --------------------------------------------------------
    # ARMAZENAR RESULTADOS
    # --------------------------------------------------------

    resultados.append({

        "semana": contador,

        "teste_inicio":
            data_teste_inicio,

        "teste_fim":
            data_teste_fim,

        "treino_inicio":
            data_inicio_treino,

        "treino_fim":
            data_fim_treino,

        "qtd_treino":
            len(treino),

        "qtd_teste":
            len(teste),

        "efetividade_treino":
            y_train.mean(),

        "efetividade_real":
            efetividade_real,

        "probabilidade_media":
            prob_media,

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

        "log_loss":
            logloss,

        "erro_calibracao":
            erro_calibracao,

        "erro_calibracao_abs":
            abs(
                erro_calibracao
            )
    })


    # --------------------------------------------------------
    # PREDIÇÕES
    # --------------------------------------------------------

    resultado_teste = teste[
        [
            DATE_COL,
            TARGET
        ]
    ].copy()

    resultado_teste[
        "probabilidade"
    ] = probabilidade

    resultado_teste[
        "predicao"
    ] = predicao

    resultado_teste[
        "semana_teste"
    ] = contador

    resultado_teste[
        "data_treino_inicio"
    ] = data_inicio_treino

    resultado_teste[
        "data_treino_fim"
    ] = data_fim_treino

    predicoes.append(
        resultado_teste
    )


    # --------------------------------------------------------
    # AVANÇAR UMA SEMANA
    # --------------------------------------------------------

    data_teste_inicio = (
        data_teste_inicio
        + pd.Timedelta(
            days=DIAS_TESTE
        )
    )


# ============================================================
# DATAFRAMES FINAIS
# ============================================================

df_resultados = pd.DataFrame(
    resultados
)

df_predicoes = pd.concat(
    predicoes,
    ignore_index=True
)


# ============================================================
# RESULTADO FINAL
# ============================================================

print("\n")
print("=" * 70)
print("RESULTADO FINAL — SIMULAÇÃO SEMANAL")
print("=" * 70)


print(
    df_resultados[
        [
            "semana",
            "teste_inicio",
            "teste_fim",
            "qtd_treino",
            "qtd_teste",
            "efetividade_treino",
            "efetividade_real",
            "probabilidade_media",
            "auc",
            "brier",
            "log_loss",
            "erro_calibracao"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# MÉDIAS
# ============================================================

print("\n")
print("=" * 70)
print("MÉDIAS")
print("=" * 70)


print(
    f"AUC médio: "
    f"{df_resultados['auc'].mean():.4f}"
)

print(
    f"Brier médio: "
    f"{df_resultados['brier'].mean():.4f}"
)

print(
    f"Log Loss médio: "
    f"{df_resultados['log_loss'].mean():.4f}"
)

print(
    f"Erro calibração médio: "
    f"{df_resultados['erro_calibracao'].mean():+.2%}"
)

print(
    f"Erro calibração absoluto médio: "
    f"{df_resultados['erro_calibracao_abs'].mean():.2%}"
)


# ============================================================
# CALIBRAÇÃO GLOBAL
# ============================================================

prob_media_global = (
    df_predicoes["probabilidade"]
    .mean()
)

efetividade_real_global = (
    df_predicoes[TARGET]
    .mean()
)

erro_global = (
    prob_media_global
    -
    efetividade_real_global
)


print("\n")
print("=" * 70)
print("CALIBRAÇÃO GLOBAL")
print("=" * 70)

print(
    f"Probabilidade média: "
    f"{prob_media_global:.2%}"
)

print(
    f"Efetividade real:    "
    f"{efetividade_real_global:.2%}"
)

print(
    f"Erro:                "
    f"{erro_global:+.2%}"
)


# ============================================================
# SALVAR
# # ============================================================

# df_resultados.to_csv(
#     "simulacao_semanal_3_meses.csv",
#     index=False
# )

# df_predicoes.to_csv(
#     "predicoes_semanal_3_meses.csv",
#     index=False
# )


# print("\nArquivos gerados:")

# print(
#     "simulacao_semanal_3_meses.csv"
# )

# print(
#     "predicoes_semanal_3_meses.csv"
# )




































































































# ============================================================
# MODELO DE EFETIVIDADE
# WALK-FORWARD SEMANAL
#
# Estratégia:
#   - Treinamento: últimos 3 meses
#   - Teste: próxima semana
#   - Modelo congelado durante a semana
#   - Re-treinamento semanal
#   - Features históricas point-in-time
#   - Sem utilização de informações do próprio dia
# ============================================================

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from xgboost import XGBClassifier


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

RANDOM_STATE = 42

# Janela de treinamento
TRAIN_MONTHS = 3

# Período da simulação
SIMULATION_START = "2023-09-01"
SIMULATION_END = "2023-12-31"

# Diretório dos artefatos
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = (
    ARTIFACT_DIR /
    "modelo_efetividade_3m.joblib"
)


# ============================================================
# 2. CÓPIA DO DATAFRAME
# ============================================================
# Conexão com o banco
conexao = connect_database()

# Lendo a query
query = read_query('query_train.sql')

# Carregando o df
df = pd.read_sql_query(query, conexao)



# ============================================================
# 3. PREPARAÇÃO DOS DADOS
# ============================================================

print("=" * 70)
print("PREPARAÇÃO DOS DADOS")
print("=" * 70)


# ------------------------------------------------------------
# Padronizar nomes das colunas
# ------------------------------------------------------------

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)


# ------------------------------------------------------------
# Verificar colunas obrigatórias
# ------------------------------------------------------------

colunas_obrigatorias = [
    "data_agendamento",
    "efetividade",
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


colunas_faltantes = [
    coluna
    for coluna in colunas_obrigatorias
    if coluna not in df.columns
]


if colunas_faltantes:

    raise ValueError(
        "As seguintes colunas obrigatórias "
        "não existem no dataframe:\n\n"
        + "\n".join(colunas_faltantes)
    )


# ------------------------------------------------------------
# Converter data
# ------------------------------------------------------------

df["data_agendamento"] = pd.to_datetime(
    df["data_agendamento"],
    errors="coerce"
)


# ------------------------------------------------------------
# Remover datas inválidas
# ------------------------------------------------------------

df = df.dropna(
    subset=["data_agendamento"]
).copy()


# ------------------------------------------------------------
# Criar data sem horário
# ------------------------------------------------------------

df["data"] = (
    df["data_agendamento"]
    .dt.normalize()
)


# ------------------------------------------------------------
# Converter target
# ------------------------------------------------------------

df["efetividade"] = pd.to_numeric(
    df["efetividade"],
    errors="coerce"
)


df = df.dropna(
    subset=["efetividade"]
).copy()


df["efetividade"] = (
    df["efetividade"]
    .astype(int)
)


# ------------------------------------------------------------
# Ordenar
# ------------------------------------------------------------

df = df.sort_values(
    ["data"],
    kind="mergesort"
).reset_index(
    drop=True
)


print(
    f"Linhas: {len(df):,}"
)

print(
    f"Data inicial: "
    f"{df['data'].min():%Y-%m-%d}"
)

print(
    f"Data final:   "
    f"{df['data'].max():%Y-%m-%d}"
)

print(
    f"Efetividade geral: "
    f"{df['efetividade'].mean():.2%}"
)


# ============================================================
# 4. FEATURES TEMPORAIS
# ============================================================

print("\n" + "=" * 70)
print("FEATURES TEMPORAIS")
print("=" * 70)


df["ano"] = (
    df["data"]
    .dt.year
)

df["mes"] = (
    df["data"]
    .dt.month
)

df["dia"] = (
    df["data"]
    .dt.day
)

df["dia_semana"] = (
    df["data"]
    .dt.dayofweek
)

df["semana_ano"] = (
    df["data"]
    .dt.isocalendar()
    .week
    .astype(int)
)

df["fim_de_semana"] = (
    df["dia_semana"] >= 5
).astype(int)


# ============================================================
# 5. ROLLING GLOBAL
# ============================================================

def criar_rolling_global(
    dados,
    janela_dias
):
    """
    Calcula a efetividade global móvel.

    O shift(1) garante que o dia atual
    não seja utilizado para calcular
    a própria previsão.
    """

    diario = (
        dados
        .groupby(
            "data",
            as_index=False
        )
        .agg(
            qtd=(
                "efetividade",
                "size"
            ),
            sucessos=(
                "efetividade",
                "sum"
            )
        )
    )


    diario["efetividade_dia"] = (
        diario["sucessos"] /
        diario["qtd"]
    )


    diario = diario.sort_values(
        "data"
    )


    diario = diario.set_index(
        "data"
    )


    nome_feature = (
        f"efetividade_global_{janela_dias}d"
    )


    diario[nome_feature] = (
        diario["efetividade_dia"]
        .shift(1)
        .rolling(
            f"{janela_dias}D",
            min_periods=1
        )
        .mean()
    )


    return (
        diario[
            [nome_feature]
        ]
        .reset_index()
    )


# ------------------------------------------------------------
# Criar rolling global
# ------------------------------------------------------------

print(
    "Calculando efetividade global móvel..."
)


for janela in [
    7,
    14,
    30,
    60,
    90
]:

    rolling_global = (
        criar_rolling_global(
            df,
            janela
        )
    )


    df = df.merge(
        rolling_global,
        on="data",
        how="left"
    )


# ============================================================
# 6. ROLLING POR GRUPO
# ============================================================

def criar_rolling_grupo(
    dados,
    coluna_grupo,
    nome_feature,
    janela_dias
):
    """
    Calcula efetividade móvel por grupo
    utilizando dias de calendário.

    Exemplo:

        Técnico A
        últimos 30 dias
        = 78%

    O shift(1) evita usar o resultado
    do próprio dia.
    """

    diario = (
        dados
        .groupby(
            [
                coluna_grupo,
                "data"
            ],
            as_index=False
        )
        .agg(
            qtd=(
                "efetividade",
                "size"
            ),
            sucessos=(
                "efetividade",
                "sum"
            )
        )
    )


    diario["efetividade_dia"] = (
        diario["sucessos"] /
        diario["qtd"]
    )


    diario = diario.sort_values(
        [
            coluna_grupo,
            "data"
        ]
    )


    resultados = []


    for valor_grupo, grupo in diario.groupby(
        coluna_grupo,
        sort=False
    ):

        grupo = grupo.sort_values(
            "data"
        ).copy()


        serie = (
            grupo
            .set_index("data")[
                "efetividade_dia"
            ]
            .shift(1)
            .rolling(
                f"{janela_dias}D",
                min_periods=1
            )
            .mean()
        )


        grupo[
            f"{nome_feature}_{janela_dias}d"
        ] = serie.to_numpy()


        resultados.append(
            grupo[
                [
                    coluna_grupo,
                    "data",
                    f"{nome_feature}_{janela_dias}d"
                ]
            ]
        )


    if not resultados:

        return pd.DataFrame(
            columns=[
                coluna_grupo,
                "data",
                f"{nome_feature}_{janela_dias}d"
            ]
        )


    return pd.concat(
        resultados,
        ignore_index=True
    )


# ============================================================
# 7. ROLLING DO TÉCNICO
# ============================================================

print(
    "Calculando efetividade móvel do técnico..."
)


for janela in [
    30,
    60,
    90
]:

    temp = criar_rolling_grupo(
        df,
        "nome_operador",
        "tecnico_efetividade",
        janela
    )


    df = df.merge(
        temp,
        on=[
            "nome_operador",
            "data"
        ],
        how="left"
    )


# ============================================================
# 8. ROLLING DO SERVIÇO
# ============================================================

print(
    "Calculando efetividade móvel do serviço..."
)


for janela in [
    30,
    60,
    90
]:

    temp = criar_rolling_grupo(
        df,
        "tipo_servico",
        "servico_efetividade",
        janela
    )


    df = df.merge(
        temp,
        on=[
            "tipo_servico",
            "data"
        ],
        how="left"
    )


# ============================================================
# 9. ROLLING DA CIDADE
# ============================================================

print(
    "Calculando efetividade móvel da cidade..."
)


for janela in [
    30,
    60,
    90
]:

    temp = criar_rolling_grupo(
        df,
        "cidade",
        "cidade_efetividade",
        janela
    )


    df = df.merge(
        temp,
        on=[
            "cidade",
            "data"
        ],
        how="left"
    )


# ============================================================
# 10. FEATURES RELATIVAS
# ============================================================

print(
    "Calculando features relativas..."
)


for janela in [
    30,
    60,
    90
]:

    global_col = (
        f"efetividade_global_{janela}d"
    )

    tecnico_col = (
        f"tecnico_efetividade_{janela}d"
    )

    servico_col = (
        f"servico_efetividade_{janela}d"
    )

    cidade_col = (
        f"cidade_efetividade_{janela}d"
    )


    df[
        f"tecnico_vs_global_{janela}d"
    ] = (
        df[tecnico_col] -
        df[global_col]
    )


    df[
        f"servico_vs_global_{janela}d"
    ] = (
        df[servico_col] -
        df[global_col]
    )


    df[
        f"cidade_vs_global_{janela}d"
    ] = (
        df[cidade_col] -
        df[global_col]
    )


# ============================================================
# 11. HISTÓRICO CUMULATIVO POR GRUPO
# ============================================================

def criar_historico_grupo(
    dados,
    coluna_grupo,
    nome_feature
):
    """
    Efetividade histórica acumulada.

    O shift(1) garante que somente
    dados anteriores ao dia atual
    sejam utilizados.
    """

    diario = (
        dados
        .groupby(
            [
                coluna_grupo,
                "data"
            ],
            as_index=False
        )
        .agg(
            qtd=(
                "efetividade",
                "size"
            ),
            sucessos=(
                "efetividade",
                "sum"
            )
        )
    )


    diario = diario.sort_values(
        [
            coluna_grupo,
            "data"
        ]
    )


    resultados = []


    for valor_grupo, grupo in diario.groupby(
        coluna_grupo,
        sort=False
    ):

        grupo = grupo.sort_values(
            "data"
        ).copy()


        qtd_acumulada = (
            grupo["qtd"]
            .cumsum()
            .shift(1)
        )


        sucessos_acumulados = (
            grupo["sucessos"]
            .cumsum()
            .shift(1)
        )


        grupo[nome_feature] = np.where(
            qtd_acumulada > 0,
            sucessos_acumulados /
            qtd_acumulada,
            np.nan
        )


        resultados.append(
            grupo[
                [
                    coluna_grupo,
                    "data",
                    nome_feature
                ]
            ]
        )


    if not resultados:

        return pd.DataFrame(
            columns=[
                coluna_grupo,
                "data",
                nome_feature
            ]
        )


    return pd.concat(
        resultados,
        ignore_index=True
    )


# ------------------------------------------------------------
# Histórico
# ------------------------------------------------------------

historicos = [
    (
        "nome_operador",
        "tecnico_efetividade_historica"
    ),
    (
        "tipo_servico",
        "servico_efetividade_historica"
    ),
    (
        "cidade",
        "cidade_efetividade_historica"
    ),
]


for coluna, feature in historicos:

    temp = criar_historico_grupo(
        df,
        coluna,
        feature
    )


    df = df.merge(
        temp,
        on=[
            coluna,
            "data"
        ],
        how="left"
    )


# ============================================================
# 12. VOLUME DOS ÚLTIMOS 30 DIAS
# ============================================================

def criar_volume_grupo(
    dados,
    coluna_grupo,
    nome_feature,
    janela_dias
):
    """
    Volume de ordens do grupo nos últimos
    N dias de calendário.

    O shift(1) evita considerar o próprio dia.
    """

    diario = (
        dados
        .groupby(
            [
                coluna_grupo,
                "data"
            ],
            as_index=False
        )
        .size()
        .rename(
            columns={
                "size": "qtd"
            }
        )
    )


    diario = diario.sort_values(
        [
            coluna_grupo,
            "data"
        ]
    )


    resultados = []


    for valor_grupo, grupo in diario.groupby(
        coluna_grupo,
        sort=False
    ):

        grupo = grupo.sort_values(
            "data"
        ).copy()


        volume = (
            grupo
            .set_index("data")[
                "qtd"
            ]
            .shift(1)
            .rolling(
                f"{janela_dias}D",
                min_periods=1
            )
            .sum()
        )


        grupo[nome_feature] = (
            volume.to_numpy()
        )


        resultados.append(
            grupo[
                [
                    coluna_grupo,
                    "data",
                    nome_feature
                ]
            ]
        )


    if not resultados:

        return pd.DataFrame(
            columns=[
                coluna_grupo,
                "data",
                nome_feature
            ]
        )


    return pd.concat(
        resultados,
        ignore_index=True
    )


# ------------------------------------------------------------
# Volume técnico
# ------------------------------------------------------------

temp = criar_volume_grupo(
    df,
    "nome_operador",
    "tecnico_qtd_30d",
    30
)


df = df.merge(
    temp,
    on=[
        "nome_operador",
        "data"
    ],
    how="left"
)


# ------------------------------------------------------------
# Volume serviço
# ------------------------------------------------------------

temp = criar_volume_grupo(
    df,
    "tipo_servico",
    "servico_qtd_30d",
    30
)


df = df.merge(
    temp,
    on=[
        "tipo_servico",
        "data"
    ],
    how="left"
)


# ------------------------------------------------------------
# Volume cidade
# ------------------------------------------------------------

temp = criar_volume_grupo(
    df,
    "cidade",
    "cidade_qtd_30d",
    30
)


df = df.merge(
    temp,
    on=[
        "cidade",
        "data"
    ],
    how="left"
)


# ============================================================
# 13. FEATURES CATEGÓRICAS
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


# ============================================================
# 14. FEATURES NUMÉRICAS
# ============================================================

numeric_features = [

    # --------------------------------------------------------
    # Calendário
    # --------------------------------------------------------

    "ano",

    "mes",

    "dia",

    "dia_semana",

    "semana_ano",

    "fim_de_semana",


    # --------------------------------------------------------
    # Efetividade global móvel
    # --------------------------------------------------------

    "efetividade_global_7d",

    "efetividade_global_14d",

    "efetividade_global_30d",

    "efetividade_global_60d",

    "efetividade_global_90d",


    # --------------------------------------------------------
    # Técnico
    # --------------------------------------------------------

    "tecnico_efetividade_30d",

    "tecnico_efetividade_60d",

    "tecnico_efetividade_90d",

    "tecnico_efetividade_historica",


    # --------------------------------------------------------
    # Serviço
    # --------------------------------------------------------

    "servico_efetividade_30d",

    "servico_efetividade_60d",

    "servico_efetividade_90d",

    "servico_efetividade_historica",


    # --------------------------------------------------------
    # Cidade
    # --------------------------------------------------------

    "cidade_efetividade_30d",

    "cidade_efetividade_60d",

    "cidade_efetividade_90d",

    "cidade_efetividade_historica",


    # --------------------------------------------------------
    # Técnico vs cenário global
    # --------------------------------------------------------

    "tecnico_vs_global_30d",

    "tecnico_vs_global_60d",

    "tecnico_vs_global_90d",


    # --------------------------------------------------------
    # Serviço vs cenário global
    # --------------------------------------------------------

    "servico_vs_global_30d",

    "servico_vs_global_60d",

    "servico_vs_global_90d",


    # --------------------------------------------------------
    # Cidade vs cenário global
    # --------------------------------------------------------

    "cidade_vs_global_30d",

    "cidade_vs_global_60d",

    "cidade_vs_global_90d",


    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    "tecnico_qtd_30d",

    "servico_qtd_30d",

    "cidade_qtd_30d",
]


# ============================================================
# 15. VALIDAR FEATURES
# ============================================================

features = (
    categorical_features +
    numeric_features
)


features_inexistentes = [
    coluna
    for coluna in features
    if coluna not in df.columns
]


if features_inexistentes:

    raise ValueError(
        "As seguintes features não existem:\n\n"
        + "\n".join(features_inexistentes)
    )


# ============================================================
# 16. DATASET DO MODELO
# ============================================================

model_df = df[
    [
        "data",
        "efetividade"
    ] +
    features
].copy()


# ------------------------------------------------------------
# Categorias
# ------------------------------------------------------------

for coluna in categorical_features:

    model_df[coluna] = (
        model_df[coluna]
        .fillna("DESCONHECIDO")
        .astype(str)
    )


# ------------------------------------------------------------
# Numéricas
# ------------------------------------------------------------

for coluna in numeric_features:

    model_df[coluna] = pd.to_numeric(
        model_df[coluna],
        errors="coerce"
    )


model_df[numeric_features] = (
    model_df[numeric_features]
    .fillna(0)
)


print("\n" + "=" * 70)
print("DATASET FINAL")
print("=" * 70)

print(
    f"Linhas: "
    f"{len(model_df):,}"
)

print(
    f"Features categóricas: "
    f"{len(categorical_features)}"
)

print(
    f"Features numéricas: "
    f"{len(numeric_features)}"
)

print(
    f"Total de features: "
    f"{len(features)}"
)


# ============================================================
# 17. PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[

        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),

        (
            "num",
            "passthrough",
            numeric_features
        ),
    ]
)


# ============================================================
# 18. MODELO XGBOOST
# ============================================================

def criar_modelo():

    return XGBClassifier(

        n_estimators=250,

        max_depth=6,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        min_child_weight=5,

        gamma=0.0,

        reg_alpha=0.0,

        reg_lambda=1.0,

        objective="binary:logistic",

        eval_metric="logloss",

        random_state=RANDOM_STATE,

        n_jobs=-1,

        tree_method="hist"
    )


# ============================================================
# 19. PIPELINE
# ============================================================

def criar_pipeline():

    modelo = criar_modelo()


    pipeline = Pipeline(
        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                modelo
            ),
        ]
    )


    return pipeline


# ============================================================
# 20. MÉTRICAS
# ============================================================

def calcular_metricas(
    y_real,
    prob
):

    pred = (
        prob >= 0.5
    ).astype(int)


    auc = roc_auc_score(
        y_real,
        prob
    )


    brier = brier_score_loss(
        y_real,
        prob
    )


    logloss = log_loss(
        y_real,
        prob,
        labels=[
            0,
            1
        ]
    )


    accuracy = accuracy_score(
        y_real,
        pred
    )


    precision = precision_score(
        y_real,
        pred,
        zero_division=0
    )


    recall = recall_score(
        y_real,
        pred,
        zero_division=0
    )


    f1 = f1_score(
        y_real,
        pred,
        zero_division=0
    )


    return {

        "auc": auc,

        "brier": brier,

        "log_loss": logloss,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "prob_media": prob.mean(),

        "efetividade_real": y_real.mean(),

        "calibracao": (
            prob.mean() -
            y_real.mean()
        ),
    }


# ============================================================
# 21. TOP-K
# ============================================================

def calcular_top_k(
    y_real,
    prob
):

    resultado = []


    base = y_real.mean()


    ordem = np.argsort(
        -prob
    )


    y_ordenado = (
        y_real
        .iloc[ordem]
        .reset_index(drop=True)
    )


    prob_ordenado = (
        prob[ordem]
    )


    total = len(y_real)


    for percentual in [
        0.01,
        0.05,
        0.10,
        0.20,
        0.30,
    ]:

        n = max(
            1,
            int(
                total *
                percentual
            )
        )


        y_top = (
            y_ordenado
            .iloc[:n]
        )


        resultado.append(
            {

                "top": (
                    f"{percentual:.0%}"
                ),

                "qtd": n,

                "efetividade": (
                    y_top.mean()
                ),

                "ganho_vs_base": (
                    y_top.mean() -
                    base
                ),

                "prob_media": (
                    prob_ordenado[:n]
                    .mean()
                ),
            }
        )


    return pd.DataFrame(
        resultado
    )


# ============================================================
# 22. CALIBRAÇÃO
# ============================================================

def calcular_calibracao(
    y_real,
    prob
):

    temp = pd.DataFrame(
        {
            "real": y_real.to_numpy(),
            "prob": prob,
        }
    )


    bins = [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]


    labels = [
        "0-10%",
        "10-20%",
        "20-30%",
        "30-40%",
        "40-50%",
        "50-60%",
        "60-70%",
        "70-80%",
        "80-90%",
        "90-100%",
    ]


    temp["faixa"] = pd.cut(
        temp["prob"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )


    calibracao = (
        temp
        .groupby(
            "faixa",
            observed=False
        )
        .agg(
            qtd=(
                "real",
                "size"
            ),
            prob_media=(
                "prob",
                "mean"
            ),
            efetividade_real=(
                "real",
                "mean"
            )
        )
        .reset_index()
    )


    calibracao["erro"] = (
        calibracao["prob_media"] -
        calibracao["efetividade_real"]
    )


    return calibracao


# ============================================================
# 23. WALK-FORWARD SEMANAL
# ============================================================

print("\n" + "=" * 70)
print("WALK-FORWARD SEMANAL - 3 MESES")
print("=" * 70)


simulation_start = pd.Timestamp(
    SIMULATION_START
)


simulation_end = pd.Timestamp(
    SIMULATION_END
)


current_test_start = (
    simulation_start
)


resultados = []

topk_resultados = []

calibracao_resultados = []


semana = 1


while current_test_start <= simulation_end:

    # ========================================================
    # TESTE
    # ========================================================

    test_start = (
        current_test_start
    )


    test_end = min(
        test_start +
        timedelta(days=6),
        simulation_end
    )


    # ========================================================
    # TREINO
    # ========================================================

    train_end = (
        test_start -
        timedelta(days=1)
    )


    train_start = (
        test_start -
        pd.DateOffset(
            months=TRAIN_MONTHS
        )
    )


    # ========================================================
    # MÁSCARAS
    # ========================================================

    train_mask = (
        (model_df["data"] >= train_start) &
        (model_df["data"] <= train_end)
    )


    test_mask = (
        (model_df["data"] >= test_start) &
        (model_df["data"] <= test_end)
    )


    train = (
        model_df
        .loc[train_mask]
        .copy()
    )


    test = (
        model_df
        .loc[test_mask]
        .copy()
    )


    # ========================================================
    # VALIDAÇÃO
    # ========================================================

    if len(train) == 0:

        print(
            f"\nSemana {semana}: "
            f"sem dados de treinamento."
        )

        current_test_start = (
            test_end +
            timedelta(days=1)
        )

        semana += 1

        continue


    if len(test) == 0:

        current_test_start = (
            test_end +
            timedelta(days=1)
        )

        semana += 1

        continue


    # ========================================================
    # X / Y
    # ========================================================

    X_train = (
        train[features]
    )


    y_train = (
        train["efetividade"]
    )


    X_test = (
        test[features]
    )


    y_test = (
        test["efetividade"]
    )


    # ========================================================
    # TREINAR
    # ========================================================

    pipeline = (
        criar_pipeline()
    )


    pipeline.fit(
        X_train,
        y_train
    )


    # ========================================================
    # PREDIÇÃO
    # ========================================================

    prob = (
        pipeline
        .predict_proba(X_test)[:, 1]
    )


    # ========================================================
    # MÉTRICAS
    # ========================================================

    metricas = calcular_metricas(
        y_test,
        prob
    )


    resultados.append(
        {

            "semana": semana,

            "train_inicio": train_start,

            "train_fim": train_end,

            "test_inicio": test_start,

            "test_fim": test_end,

            "qtd_train": len(train),

            "qtd_test": len(test),

            "efetividade_train": (
                y_train.mean()
            ),

            **metricas,
        }
    )


    # ========================================================
    # TOP-K
    # ========================================================

    topk = calcular_top_k(
        y_test,
        prob
    )


    topk["semana"] = (
        semana
    )


    topk_resultados.append(
        topk
    )


    # ========================================================
    # CALIBRAÇÃO
    # ========================================================

    calib = calcular_calibracao(
        y_test,
        prob
    )


    calib["semana"] = (
        semana
    )


    calibracao_resultados.append(
        calib
    )


    # ========================================================
    # LOG
    # ========================================================

    print(
        f"\nSemana {semana:02d}"
    )


    print(
        f"Treino: "
        f"{train_start:%Y-%m-%d} "
        f"→ "
        f"{train_end:%Y-%m-%d}"
    )


    print(
        f"Teste:  "
        f"{test_start:%Y-%m-%d} "
        f"→ "
        f"{test_end:%Y-%m-%d}"
    )


    print(
        f"Train: "
        f"{len(train):,} | "
        f"Test: "
        f"{len(test):,}"
    )


    print(
        f"Efetividade treino: "
        f"{y_train.mean():.2%}"
    )


    print(
        f"Efetividade real: "
        f"{y_test.mean():.2%}"
    )


    print(
        f"Probabilidade média: "
        f"{prob.mean():.2%}"
    )


    print(
        f"AUC: "
        f"{metricas['auc']:.4f}"
    )


    print(
        f"Brier: "
        f"{metricas['brier']:.4f}"
    )


    print(
        f"Log Loss: "
        f"{metricas['log_loss']:.4f}"
    )


    print(
        f"Calibração: "
        f"{metricas['calibracao']:+.2%}"
    )


    # ========================================================
    # PRÓXIMA SEMANA
    # ========================================================

    current_test_start = (
        test_end +
        timedelta(days=1)
    )


    semana += 1


# ============================================================
# 24. CONSOLIDAR RESULTADOS
# ============================================================

if not resultados:

    raise RuntimeError(
        "Nenhum resultado foi gerado "
        "no walk-forward."
    )


resultados_df = pd.DataFrame(
    resultados
)


topk_df = pd.concat(
    topk_resultados,
    ignore_index=True
)


calibracao_df = pd.concat(
    calibracao_resultados,
    ignore_index=True
)


# ============================================================
# 25. RESUMO WALK-FORWARD
# ============================================================

print("\n" + "=" * 70)
print("RESUMO WALK-FORWARD - 3 MESES")
print("=" * 70)


auc_medio = (
    resultados_df["auc"]
    .mean()
)


brier_medio = (
    resultados_df["brier"]
    .mean()
)


logloss_medio = (
    resultados_df["log_loss"]
    .mean()
)


calibracao_media = (
    resultados_df["calibracao"]
    .mean()
)


erro_absoluto_calibracao = (
    resultados_df["calibracao"]
    .abs()
    .mean()
)


# ============================================================
# 26. PROBABILIDADE GLOBAL
# ============================================================

probabilidade_global = np.average(
    resultados_df["prob_media"],
    weights=resultados_df["qtd_test"]
)


efetividade_global_real = np.average(
    resultados_df["efetividade_real"],
    weights=resultados_df["qtd_test"]
)


# ============================================================
# 27. IMPRIMIR RESUMO
# ============================================================

print(
    f"AUC médio: "
    f"{auc_medio:.4f}"
)


print(
    f"Brier médio: "
    f"{brier_medio:.4f}"
)


print(
    f"Log Loss médio: "
    f"{logloss_medio:.4f}"
)


print(
    f"Calibração média: "
    f"{calibracao_media:+.2%}"
)


print(
    f"Erro absoluto de calibração: "
    f"{erro_absoluto_calibracao:.2%}"
)


print(
    f"\nProbabilidade global: "
    f"{probabilidade_global:.2%}"
)


print(
    f"Efetividade global real: "
    f"{efetividade_global_real:.2%}"
)


# ============================================================
# 28. TOP-K GLOBAL
# ============================================================

print("\n" + "=" * 70)
print("TOP-K GLOBAL")
print("=" * 70)


topk_resumo = (
    topk_df
    .groupby("top")
    .agg(
        qtd_media=(
            "qtd",
            "mean"
        ),
        efetividade=(
            "efetividade",
            "mean"
        ),
        ganho_vs_base=(
            "ganho_vs_base",
            "mean"
        ),
        prob_media=(
            "prob_media",
            "mean"
        )
    )
    .reset_index()
)


print(
    topk_resumo.to_string(
        index=False,
        formatters={

            "efetividade":
                "{:.2%}".format,

            "ganho_vs_base":
                "{:+.2%}".format,

            "prob_media":
                "{:.2%}".format,
        }
    )
)


# ============================================================
# 29. CALIBRAÇÃO GLOBAL
# ============================================================

print("\n" + "=" * 70)
print("CALIBRAÇÃO POR FAIXA DE PROBABILIDADE")
print("=" * 70)


calibracao_resumo = (
    calibracao_df
    .groupby(
        "faixa",
        observed=False
    )
    .agg(
        qtd=(
            "qtd",
            "sum"
        ),
        prob_media=(
            "prob_media",
            "mean"
        ),
        efetividade_real=(
            "efetividade_real",
            "mean"
        )
    )
    .reset_index()
)


calibracao_resumo["erro"] = (
    calibracao_resumo["prob_media"] -
    calibracao_resumo["efetividade_real"]
)


print(
    calibracao_resumo.to_string(
        index=False,
        formatters={

            "prob_media":
                "{:.2%}".format,

            "efetividade_real":
                "{:.2%}".format,

            "erro":
                "{:+.2%}".format,
        }
    )
)


# ============================================================
# 30. SALVAR RESULTADOS
# ============================================================

resultados_path = (
    ARTIFACT_DIR /
    "walk_forward_3m_semanal.csv"
)


topk_path = (
    ARTIFACT_DIR /
    "topk_3m_semanal.csv"
)


calibracao_path = (
    ARTIFACT_DIR /
    "calibracao_3m_semanal.csv"
)


# resultados_df.to_csv(
#     resultados_path,
#     index=False
# )


# topk_resumo.to_csv(
#     topk_path,
#     index=False
# )


# calibracao_resumo.to_csv(
#     calibracao_path,
#     index=False
# )


# ============================================================
# 31. TREINAMENTO FINAL
# ============================================================

print("\n" + "=" * 70)
print("TREINAMENTO FINAL")
print("=" * 70)


ultima_data = (
    model_df["data"]
    .max()
)


train_final_end = (
    ultima_data
)


train_final_start = (
    train_final_end -
    pd.DateOffset(
        months=TRAIN_MONTHS
    ) +
    timedelta(days=1)
)


train_final_mask = (
    (model_df["data"] >= train_final_start) &
    (model_df["data"] <= train_final_end)
)


train_final = (
    model_df
    .loc[train_final_mask]
    .copy()
)


X_final = (
    train_final[features]
)


y_final = (
    train_final["efetividade"]
)


print(
    f"Período: "
    f"{train_final_start:%Y-%m-%d} "
    f"→ "
    f"{train_final_end:%Y-%m-%d}"
)


print(
    f"Linhas: "
    f"{len(train_final):,}"
)


print(
    f"Efetividade: "
    f"{y_final.mean():.2%}"
)


# ============================================================
# 32. FIT FINAL
# ============================================================

modelo_final = (
    criar_pipeline()
)


modelo_final.fit(
    X_final,
    y_final
)


# ============================================================
# 33. ARTEFATO
# ============================================================

artefato = {

    # --------------------------------------------------------
    # Modelo
    # --------------------------------------------------------

    "model":
        modelo_final,


    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    "features":
        features,


    "categorical_features":
        categorical_features,


    "numeric_features":
        numeric_features,


    # --------------------------------------------------------
    # Configuração
    # --------------------------------------------------------

    "train_months":
        TRAIN_MONTHS,


    # --------------------------------------------------------
    # Período
    # --------------------------------------------------------

    "train_start":
        train_final_start,


    "train_end":
        train_final_end,


    # --------------------------------------------------------
    # Informações do treinamento
    # --------------------------------------------------------

    "training_rows":
        len(train_final),


    "training_effectiveness":
        y_final.mean(),


    # --------------------------------------------------------
    # Versão
    # --------------------------------------------------------

    "model_version":
        (
            f"efetividade_3m_"
            f"{train_final_end:%Y%m%d}"
        ),
}


# ============================================================
# 34. SALVAR MODELO
# ============================================================

# joblib.dump(
#     artefato,
#     MODEL_PATH
# )


# ============================================================
# 35. FINAL
# ============================================================

print("\n" + "=" * 70)
print("MODELO FINALIZADO")
print("=" * 70)


# print(
#     f"Arquivo: "
#     f"{MODEL_PATH}"
# )


print(
    f"Versão: "
    f"{artefato['model_version']}"
)


print(
    f"Treino: "
    f"{train_final_start:%Y-%m-%d} "
    f"→ "
    f"{train_final_end:%Y-%m-%d}"
)


print(
    f"Linhas: "
    f"{len(train_final):,}"
)


print(
    f"Efetividade treino: "
    f"{y_final.mean():.2%}"
)


print("\nArquivos gerados:")


print(
    f"- {MODEL_PATH}"
)


print(
    f"- {resultados_path}"
)


print(
    f"- {topk_path}"
)


print(
    f"- {calibracao_path}"
)