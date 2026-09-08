# ============================================================
# MODELO DE EFETIVIDADE - CATBOOST
# WALK-FORWARD SEMANAL - JANELA MÓVEL DE 3 MESES
# ============================================================

from pathlib import Path
import warnings
from src.Database.database_config import connect_database, read_query

import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings("ignore")










import pandas as pd
import numpy as np

# 90 dias de dados
datas = pd.date_range("2024-01-01", periods=90, freq="D")

# 7000 ordens por dia
df = pd.DataFrame({
    "data_agendamento": np.repeat(datas, 7000),
    "nome_operador": np.random.choice(["João", "Maria", "Carlos"], size=90*7000),
    "rede_acesso": np.random.choice(["Fibra", "Cabo", "Radio"], size=90*7000),
    "executadas": np.random.randint(1, 10, size=90*7000),
    "concluidas": np.random.randint(0, 10, size=90*7000)
})

df = df.sort_values("data_agendamento")
df["data_agendamento"] = pd.to_datetime(df["data_agendamento"])
df = df.set_index("data_agendamento")


g = df.groupby(["nome_operador", "rede_acesso"])

# Usar o índice diretamente dentro do rolling
concluidas_30d = (
    g["concluidas"]
    .shift(1)
    .rolling("30D", on=df.index)   # <-- ESSA LINHA É A CHAVE
    .sum()
)

executadas_30d = (
    g["executadas"]
    .shift(1)
    .rolling("30D", on=df.index)   # <-- ESSA LINHA É A CHAVE
    .sum()
)

df["efetividade_30d"] = concluidas_30d / executadas_30d


# ============================================================
# CONFIGURAÇÕES
# ============================================================

TRAIN_MONTHS = 3

SIMULATION_START = pd.Timestamp("2023-09-01")
SIMULATION_END = pd.Timestamp("2023-12-31")

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = (
    ARTIFACT_DIR
    / "modelo_efetividade_catboost_3m.joblib"
)


# ============================================================
# TARGET
# ============================================================

TARGET = "efetividade"


# ============================================================
# FEATURES CATEGÓRICAS
# ============================================================

CATEGORICAL_FEATURES = [
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
# FEATURES NUMÉRICAS
# ============================================================

NUMERIC_FEATURES = [

    # --------------------------------------------------------
    # GLOBAL
    # --------------------------------------------------------

    "efetividade_global_7d",
    "efetividade_global_14d",
    "efetividade_global_30d",
    "efetividade_global_60d",
    "efetividade_global_90d",

    # --------------------------------------------------------
    # TÉCNICO
    # --------------------------------------------------------

    "tecnico_efetividade_30d",
    "tecnico_efetividade_60d",
    "tecnico_efetividade_90d",

    # --------------------------------------------------------
    # SERVIÇO
    # --------------------------------------------------------

    "servico_efetividade_30d",
    "servico_efetividade_60d",
    "servico_efetividade_90d",

    # --------------------------------------------------------
    # CIDADE
    # --------------------------------------------------------

    "cidade_efetividade_30d",
    "cidade_efetividade_60d",
    "cidade_efetividade_90d",

    # --------------------------------------------------------
    # RELATIVOS
    # --------------------------------------------------------

    "tecnico_vs_global_30d",
    "tecnico_vs_global_60d",
    "tecnico_vs_global_90d",

    "servico_vs_global_30d",
    "servico_vs_global_60d",
    "servico_vs_global_90d",

    "cidade_vs_global_30d",
    "cidade_vs_global_60d",
    "cidade_vs_global_90d",

    # --------------------------------------------------------
    # HISTÓRICO
    # --------------------------------------------------------

    "tecnico_efetividade_historica",
    "servico_efetividade_historica",
    "cidade_efetividade_historica",

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    "tecnico_qtd_30d",
    "servico_qtd_30d",
    "cidade_qtd_30d",

    # --------------------------------------------------------
    # CALENDÁRIO
    # --------------------------------------------------------

    "ano",
    "mes",
    "dia",
    "dia_semana",
    "semana_ano",
    "fim_de_semana",
]


FEATURES = (
    CATEGORICAL_FEATURES
    + NUMERIC_FEATURES
)


# ============================================================
# CRIAR MODELO CATBOOST
# ============================================================

def criar_modelo():

    return CatBoostClassifier(

        # Aproximação ao XGBoost anterior
        iterations=250,
        depth=6,
        learning_rate=0.05,

        # Equivalente aproximado ao subsample
        bootstrap_type="Bernoulli",
        subsample=0.8,

        # Regularização
        l2_leaf_reg=1.0,

        # Objetivo
        loss_function="Logloss",
        eval_metric="Logloss",

        # Reprodutibilidade
        random_seed=42,

        # CPU
        thread_count=-1,

        # Sem output
        verbose=False,

        # Não criar arquivos temporários
        allow_writing_files=False,
    )


# ============================================================
# PREPARAR DADOS
# ============================================================

def preparar_dados(df):

    df = df.copy()

    print("\nPreparando dados...")

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    if "data_agendamento" not in df.columns:

        raise ValueError(
            "Coluna 'data_agendamento' não encontrada."
        )

    df["data_agendamento"] = pd.to_datetime(
        df["data_agendamento"],
        errors="coerce"
    )

    if df["data_agendamento"].isna().any():

        raise ValueError(
            "Existem datas inválidas em "
            "'data_agendamento'."
        )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    if TARGET not in df.columns:

        raise ValueError(
            f"Coluna '{TARGET}' não encontrada."
        )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    if df[TARGET].isna().any():

        raise ValueError(
            f"Existem valores inválidos em '{TARGET}'."
        )

    df[TARGET] = df[TARGET].astype(int)

    # --------------------------------------------------------
    # CATEGÓRICAS
    # --------------------------------------------------------

    for col in CATEGORICAL_FEATURES:

        if col not in df.columns:

            raise ValueError(
                f"Coluna categórica ausente: {col}"
            )

        df[col] = (
            df[col]
            .fillna("__MISSING__")
            .astype(str)
        )

    # --------------------------------------------------------
    # ORDENAÇÃO
    # --------------------------------------------------------

    df = (
        df
        .sort_values("data_agendamento")
        .reset_index(drop=True)
    )

    print(
        f"Linhas preparadas: {len(df):,}"
    )

    return df


# ============================================================
# ROLLING GLOBAL
# ============================================================

def criar_rolling_global(
    df,
    dias,
    coluna_saida
):

    """
    Calcula efetividade global móvel.

    Importante:
    - closed='left'
    - portanto o registro atual não participa
      do cálculo.
    """

    resultado = (
        df
        .set_index("data_agendamento")[TARGET]
        .rolling(
            f"{dias}D",
            closed="left"
        )
        .mean()
    )

    return (
        resultado
        .to_numpy()
    )


# ============================================================
# ROLLING POR GRUPO
# ============================================================

def calcular_rolling_grupo(
    df,
    grupo,
    dias,
    tipo="mean"
):

    """
    Calcula rolling temporal por grupo sem merge.

    O resultado mantém exatamente uma posição para
    cada linha original do dataframe.

    O dataframe precisa estar ordenado por:
        grupo + data_agendamento
    """

    valores = np.full(
        len(df),
        np.nan,
        dtype=np.float64
    )

    # Índices originais
    indices_originais = np.arange(
        len(df)
    )

    temp = df[
        [
            grupo,
            "data_agendamento",
            TARGET,
        ]
    ].copy()

    temp["_indice_original"] = (
        indices_originais
    )

    temp = temp.sort_values(
        [
            grupo,
            "data_agendamento",
        ]
    )

    # --------------------------------------------------------
    # Processar cada grupo
    # --------------------------------------------------------

    for _, grupo_df in temp.groupby(
        grupo,
        sort=False
    ):

        serie = (
            grupo_df
            .set_index("data_agendamento")[TARGET]
        )

        if tipo == "mean":

            rolling = (
                serie
                .rolling(
                    f"{dias}D",
                    closed="left"
                )
                .mean()
            )

        elif tipo == "count":

            rolling = (
                serie
                .rolling(
                    f"{dias}D",
                    closed="left"
                )
                .count()
            )

        else:

            raise ValueError(
                f"Tipo de rolling inválido: {tipo}"
            )

        idx = (
            grupo_df["_indice_original"]
            .to_numpy()
        )

        valores[idx] = (
            rolling
            .to_numpy()
        )

    return valores


# ============================================================
# HISTÓRICO ACUMULADO POR GRUPO
# ============================================================

def calcular_historico_grupo(
    df,
    grupo
):

    """
    Calcula efetividade histórica acumulada
    utilizando SOMENTE registros anteriores.
    """

    resultado = np.full(
        len(df),
        np.nan,
        dtype=np.float64
    )

    indices_originais = np.arange(
        len(df)
    )

    temp = df[
        [
            grupo,
            "data_agendamento",
            TARGET,
        ]
    ].copy()

    temp["_indice_original"] = (
        indices_originais
    )

    temp = temp.sort_values(
        [
            grupo,
            "data_agendamento",
        ]
    )

    for _, grupo_df in temp.groupby(
        grupo,
        sort=False
    ):

        valores = (
            grupo_df[TARGET]
            .shift(1)
        )

        historico = (
            valores
            .expanding()
            .mean()
        )

        idx = (
            grupo_df["_indice_original"]
            .to_numpy()
        )

        resultado[idx] = (
            historico
            .to_numpy()
        )

    return resultado


# ============================================================
# CRIAR FEATURES
# ============================================================

def criar_features(df):

    print("\n" + "=" * 70)
    print("CRIANDO FEATURES")
    print("=" * 70)

    df = df.copy()

    quantidade_original = len(df)

    # --------------------------------------------------------
    # Garantir ordenação temporal
    # --------------------------------------------------------

    df = (
        df
        .sort_values("data_agendamento")
        .reset_index(drop=True)
    )

    # Índice de segurança
    df["_indice_original"] = np.arange(
        len(df)
    )

    # ========================================================
    # GLOBAL
    # ========================================================

    print("\nCalculando efetividade global...")

    for dias in [7, 14, 30, 60, 90]:

        coluna = (
            f"efetividade_global_{dias}d"
        )

        df[coluna] = criar_rolling_global(
            df,
            dias,
            coluna
        )

    # ========================================================
    # TÉCNICO
    # ========================================================

    print(
        "\nCalculando features de técnico..."
    )

    grupo = "nome_operador"

    for dias in [30, 60, 90]:

        coluna = (
            f"tecnico_efetividade_{dias}d"
        )

        df[coluna] = calcular_rolling_grupo(
            df,
            grupo,
            dias,
            tipo="mean"
        )

    df[
        "tecnico_efetividade_historica"
    ] = calcular_historico_grupo(
        df,
        grupo
    )

    df[
        "tecnico_qtd_30d"
    ] = calcular_rolling_grupo(
        df,
        grupo,
        30,
        tipo="count"
    )

    # ========================================================
    # SERVIÇO
    # ========================================================

    print(
        "\nCalculando features de serviço..."
    )

    grupo = "tipo_servico"

    for dias in [30, 60, 90]:

        coluna = (
            f"servico_efetividade_{dias}d"
        )

        df[coluna] = calcular_rolling_grupo(
            df,
            grupo,
            dias,
            tipo="mean"
        )

    df[
        "servico_efetividade_historica"
    ] = calcular_historico_grupo(
        df,
        grupo
    )

    df[
        "servico_qtd_30d"
    ] = calcular_rolling_grupo(
        df,
        grupo,
        30,
        tipo="count"
    )

    # ========================================================
    # CIDADE
    # ========================================================

    print(
        "\nCalculando features de cidade..."
    )

    grupo = "cidade"

    for dias in [30, 60, 90]:

        coluna = (
            f"cidade_efetividade_{dias}d"
        )

        df[coluna] = calcular_rolling_grupo(
            df,
            grupo,
            dias,
            tipo="mean"
        )

    df[
        "cidade_efetividade_historica"
    ] = calcular_historico_grupo(
        df,
        grupo
    )

    df[
        "cidade_qtd_30d"
    ] = calcular_rolling_grupo(
        df,
        grupo,
        30,
        tipo="count"
    )

    # ========================================================
    # FEATURES RELATIVAS
    # ========================================================

    print(
        "\nCalculando features relativas..."
    )

    for grupo in [
        "tecnico",
        "servico",
        "cidade",
    ]:

        for dias in [30, 60, 90]:

            coluna_grupo = (
                f"{grupo}_efetividade_{dias}d"
            )

            coluna_global = (
                f"efetividade_global_{dias}d"
            )

            coluna_saida = (
                f"{grupo}_vs_global_{dias}d"
            )

            df[coluna_saida] = (
                df[coluna_grupo]
                - df[coluna_global]
            )

    # ========================================================
    # CALENDÁRIO
    # ========================================================

    print(
        "\nCalculando features de calendário..."
    )

    df["ano"] = (
        df["data_agendamento"]
        .dt.year
    )

    df["mes"] = (
        df["data_agendamento"]
        .dt.month
    )

    df["dia"] = (
        df["data_agendamento"]
        .dt.day
    )

    df["dia_semana"] = (
        df["data_agendamento"]
        .dt.dayofweek
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

    # ========================================================
    # LIMPEZA
    # ========================================================

    for col in NUMERIC_FEATURES:

        df[col] = (
            df[col]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
        )

    # Remover índice auxiliar
    df = df.drop(
        columns=["_indice_original"]
    )

    # ========================================================
    # VALIDAÇÃO DE GRANULARIDADE
    # ========================================================

    quantidade_final = len(df)

    print("\n" + "-" * 70)

    print(
        f"Linhas antes das features: "
        f"{quantidade_original:,}"
    )

    print(
        f"Linhas depois das features: "
        f"{quantidade_final:,}"
    )

    if quantidade_original != quantidade_final:

        raise RuntimeError(
            "ERRO CRÍTICO: a quantidade de linhas "
            "mudou durante a criação das features."
        )

    print(
        "OK: granularidade preservada."
    )

    print(
        f"Features totais: {len(FEATURES)}"
    )

    print(
        f"  Categóricas: "
        f"{len(CATEGORICAL_FEATURES)}"
    )

    print(
        f"  Numéricas: "
        f"{len(NUMERIC_FEATURES)}"
    )

    return df


# ============================================================
# PREPARAR TRAIN / TEST
# ============================================================

def preparar_train_test(
    train,
    test
):

    train = train.copy()
    test = test.copy()

    medianas = {}

    for col in NUMERIC_FEATURES:

        mediana = train[col].median()

        if pd.isna(mediana):

            mediana = 0.0

        medianas[col] = float(
            mediana
        )

        train[col] = (
            train[col]
            .fillna(mediana)
        )

        test[col] = (
            test[col]
            .fillna(mediana)
        )

    return (
        train,
        test,
        medianas
    )


# ============================================================
# TREINAR MODELO
# ============================================================

def treinar_modelo(
    train,
    test
):

    X_train = train[FEATURES]

    y_train = train[TARGET]

    X_test = test[FEATURES]

    model = criar_modelo()

    model.fit(
        X_train,
        y_train,
        cat_features=CATEGORICAL_FEATURES,
    )

    probabilidade = (
        model
        .predict_proba(X_test)[:, 1]
    )

    return (
        model,
        probabilidade
    )


# ============================================================
# MÉTRICAS
# ============================================================

def calcular_metricas(
    y_true,
    probabilidade
):

    predicao = (
        probabilidade >= 0.5
    ).astype(int)

    auc = roc_auc_score(
        y_true,
        probabilidade
    )

    brier = brier_score_loss(
        y_true,
        probabilidade
    )

    logloss = log_loss(
        y_true,
        probabilidade,
        labels=[0, 1]
    )

    accuracy = accuracy_score(
        y_true,
        predicao
    )

    precision = precision_score(
        y_true,
        predicao,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predicao,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predicao,
        zero_division=0
    )

    prob_media = (
        probabilidade.mean()
    )

    efetividade_real = (
        y_true.mean()
    )

    calibracao = (
        prob_media
        - efetividade_real
    )

    return {

        "auc": auc,

        "brier": brier,

        "logloss": logloss,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "prob_media": prob_media,

        "efetividade_real": (
            efetividade_real
        ),

        "calibracao": calibracao,
    }


# ============================================================
# TOP-K
# ============================================================

def calcular_top_k(
    y_true,
    probabilidade,
    data_inicio
):

    resultado = pd.DataFrame({

        "real": y_true.to_numpy(),

        "probabilidade": (
            probabilidade
        ),
    })

    resultado = (
        resultado
        .sort_values(
            "probabilidade",
            ascending=False
        )
        .reset_index(drop=True)
    )

    base = (
        resultado["real"].mean()
    )

    linhas = []

    for percentual in [
        0.01,
        0.05,
        0.10,
        0.20,
        0.30,
    ]:

        quantidade = max(
            1,
            int(
                len(resultado)
                * percentual
            )
        )

        top = resultado.iloc[
            :quantidade
        ]

        efetividade = (
            top["real"].mean()
        )

        prob_media = (
            top["probabilidade"].mean()
        )

        linhas.append({

            "data_inicio": data_inicio,

            "top_percentual": (
                percentual
            ),

            "qtd": quantidade,

            "efetividade": (
                efetividade
            ),

            "ganho_pp": (
                efetividade - base
            ),

            "prob_media": (
                prob_media
            ),
        })

    return pd.DataFrame(linhas)


# ============================================================
# CALIBRAÇÃO
# ============================================================

def calcular_calibracao(
    y_true,
    probabilidade,
    data_inicio
):

    resultado = pd.DataFrame({

        "real": y_true.to_numpy(),

        "probabilidade": (
            probabilidade
        ),
    })

    bins = np.arange(
        0,
        1.01,
        0.10
    )

    resultado["faixa"] = pd.cut(
        resultado["probabilidade"],
        bins=bins,
        include_lowest=True,
        right=True
    )

    calibracao = (
        resultado
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
                "probabilidade",
                "mean"
            ),

            efetividade_real=(
                "real",
                "mean"
            ),
        )
        .reset_index()
    )

    calibracao["erro_pp"] = (
        calibracao["prob_media"]
        - calibracao["efetividade_real"]
    )

    calibracao["data_inicio"] = (
        data_inicio
    )

    return calibracao


# ============================================================
# WALK-FORWARD
# ============================================================

def executar_walk_forward(df):

    resultados = []

    resultados_topk = []

    resultados_calibracao = []

    data_teste = (
        SIMULATION_START
    )

    semana = 1

    while data_teste <= SIMULATION_END:

        # ----------------------------------------------------
        # Janela de treino
        # ----------------------------------------------------

        inicio_treino = (
            data_teste
            - pd.DateOffset(
                months=TRAIN_MONTHS
            )
        )

        fim_treino = (
            data_teste
            - pd.Timedelta(
                days=1
            )
        )

        # ----------------------------------------------------
        # Janela de teste
        # ----------------------------------------------------

        fim_teste = min(

            data_teste
            + pd.Timedelta(days=6),

            SIMULATION_END
        )

        print("\n")

        print("=" * 70)

        print(
            f"SEMANA {semana}"
        )

        print("=" * 70)

        print(
            f"Treino: "
            f"{inicio_treino.date()} "
            f"→ "
            f"{fim_treino.date()}"
        )

        print(
            f"Teste:  "
            f"{data_teste.date()} "
            f"→ "
            f"{fim_teste.date()}"
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train = df[
            (
                df["data_agendamento"]
                >= inicio_treino
            )
            &
            (
                df["data_agendamento"]
                <= fim_treino
            )
        ].copy()

        # ----------------------------------------------------
        # TEST
        # ----------------------------------------------------

        test = df[
            (
                df["data_agendamento"]
                >= data_teste
            )
            &
            (
                df["data_agendamento"]
                <= fim_teste
            )
        ].copy()

        print(
            f"\nLinhas treino: "
            f"{len(train):,}"
        )

        print(
            f"Linhas teste:  "
            f"{len(test):,}"
        )

        if len(train) == 0:

            print(
                "Treino vazio. Semana ignorada."
            )

            data_teste = (
                fim_teste
                + pd.Timedelta(days=1)
            )

            semana += 1

            continue

        if len(test) == 0:

            print(
                "Teste vazio. Semana ignorada."
            )

            data_teste = (
                fim_teste
                + pd.Timedelta(days=1)
            )

            semana += 1

            continue

        # ----------------------------------------------------
        # PREPARAR FEATURES
        # ----------------------------------------------------

        (
            train,
            test,
            medianas
        ) = preparar_train_test(
            train,
            test
        )

        # ----------------------------------------------------
        # TREINAR
        # ----------------------------------------------------

        model, probabilidade = (
            treinar_modelo(
                train,
                test
            )
        )

        y_true = test[TARGET]

        # ----------------------------------------------------
        # MÉTRICAS
        # ----------------------------------------------------

        metricas = calcular_metricas(
            y_true,
            probabilidade
        )

        metricas["semana"] = semana

        metricas["data_inicio"] = (
            data_teste
        )

        metricas["data_fim"] = (
            fim_teste
        )

        metricas["qtd_train"] = (
            len(train)
        )

        metricas["qtd_test"] = (
            len(test)
        )

        metricas["efetividade_train"] = (
            train[TARGET].mean()
        )

        resultados.append(
            metricas
        )

        # ----------------------------------------------------
        # TOP-K
        # ----------------------------------------------------

        topk = calcular_top_k(
            y_true,
            probabilidade,
            data_teste
        )

        topk["semana"] = semana

        resultados_topk.append(
            topk
        )

        # ----------------------------------------------------
        # CALIBRAÇÃO
        # ----------------------------------------------------

        calibracao = calcular_calibracao(
            y_true,
            probabilidade,
            data_teste
        )

        calibracao["semana"] = (
            semana
        )

        resultados_calibracao.append(
            calibracao
        )

        # ----------------------------------------------------
        # LOG
        # ----------------------------------------------------

        print(
            f"\nEfetividade treino: "
            f"{metricas['efetividade_train']:.2%}"
        )

        print(
            f"Efetividade real: "
            f"{metricas['efetividade_real']:.2%}"
        )

        print(
            f"Probabilidade média: "
            f"{metricas['prob_media']:.2%}"
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
            f"LogLoss: "
            f"{metricas['logloss']:.4f}"
        )

        print(
            f"Calibração: "
            f"{metricas['calibracao']:+.2%}"
        )

        # ----------------------------------------------------
        # PRÓXIMA SEMANA
        # ----------------------------------------------------

        data_teste = (
            fim_teste
            + pd.Timedelta(days=1)
        )

        semana += 1

    resultados_df = pd.DataFrame(
        resultados
    )

    topk_df = pd.concat(
        resultados_topk,
        ignore_index=True
    )

    calibracao_df = pd.concat(
        resultados_calibracao,
        ignore_index=True
    )

    return (
        resultados_df,
        topk_df,
        calibracao_df
    )


# ============================================================
# RELATÓRIO
# ============================================================

def gerar_relatorio(
    resultados_df,
    topk_df,
    calibracao_df
):

    print("\n")

    print("=" * 70)
    print("RESULTADO GLOBAL - CATBOOST")
    print("=" * 70)

    # --------------------------------------------------------
    # MÉTRICAS
    # --------------------------------------------------------

    print(
        f"\nAUC médio: "
        f"{resultados_df['auc'].mean():.4f}"
    )

    print(
        f"Brier médio: "
        f"{resultados_df['brier'].mean():.4f}"
    )

    print(
        f"LogLoss médio: "
        f"{resultados_df['logloss'].mean():.4f}"
    )

    print(
        f"Calibração média: "
        f"{resultados_df['calibracao'].mean():+.2%}"
    )

    print(
        f"Erro absoluto médio de calibração: "
        f"{resultados_df['calibracao'].abs().mean():.2%}"
    )

    # --------------------------------------------------------
    # GLOBAL PONDERADO
    # --------------------------------------------------------

    probabilidade_global = np.average(

        resultados_df["prob_media"],

        weights=(
            resultados_df["qtd_test"]
        )
    )

    efetividade_global_real = np.average(

        resultados_df["efetividade_real"],

        weights=(
            resultados_df["qtd_test"]
        )
    )

    print(
        f"\nProbabilidade global: "
        f"{probabilidade_global:.2%}"
    )

    print(
        f"Efetividade global real: "
        f"{efetividade_global_real:.2%}"
    )

    # --------------------------------------------------------
    # TOP-K
    # --------------------------------------------------------

    print("\n")

    print("=" * 70)
    print("TOP-K GLOBAL")
    print("=" * 70)

    topk_global = (
        topk_df
        .groupby(
            "top_percentual"
        )
        .agg(

            qtd_media=(
                "qtd",
                "mean"
            ),

            efetividade=(
                "efetividade",
                "mean"
            ),

            ganho_pp=(
                "ganho_pp",
                "mean"
            ),

            prob_media=(
                "prob_media",
                "mean"
            ),
        )
        .reset_index()
    )

    for _, row in topk_global.iterrows():

        print(
            f"\nTop {row['top_percentual']:.0%}"
        )

        print(
            f"  Efetividade: "
            f"{row['efetividade']:.2%}"
        )

        print(
            f"  Ganho vs base: "
            f"{row['ganho_pp']:+.2%}"
        )

        print(
            f"  Probabilidade: "
            f"{row['prob_media']:.2%}"
        )

        print(
            f"  Qtd média: "
            f"{row['qtd_media']:,.0f}"
        )

    # --------------------------------------------------------
    # CALIBRAÇÃO
    # --------------------------------------------------------

    print("\n")

    print("=" * 70)
    print("CALIBRAÇÃO GLOBAL")
    print("=" * 70)

    calibracao_global = (
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
            ),

            erro_pp=(
                "erro_pp",
                "mean"
            ),
        )
        .reset_index()
    )

    print(
        calibracao_global.to_string(
            index=False
        )
    )

    return (
        topk_global,
        calibracao_global
    )


# ============================================================
# MODELO FINAL
# ============================================================

def treinar_modelo_final(df):

    print("\n")

    print("=" * 70)
    print("TREINAMENTO DO MODELO FINAL")
    print("=" * 70)

    data_final = (
        df["data_agendamento"].max()
    )

    inicio_final = (
        data_final
        - pd.DateOffset(
            months=TRAIN_MONTHS
        )
        + pd.Timedelta(days=1)
    )

    train_final = df[
        (
            df["data_agendamento"]
            >= inicio_final
        )
        &
        (
            df["data_agendamento"]
            <= data_final
        )
    ].copy()

    print(
        f"\nPeríodo: "
        f"{inicio_final.date()} "
        f"→ "
        f"{data_final.date()}"
    )

    print(
        f"Linhas: "
        f"{len(train_final):,}"
    )

    print(
        f"Efetividade: "
        f"{train_final[TARGET].mean():.2%}"
    )

    # --------------------------------------------------------
    # Medianas
    # --------------------------------------------------------

    medianas = {}

    for col in NUMERIC_FEATURES:

        mediana = (
            train_final[col]
            .median()
        )

        if pd.isna(mediana):

            mediana = 0.0

        medianas[col] = float(
            mediana
        )

        train_final[col] = (
            train_final[col]
            .fillna(mediana)
        )

    # --------------------------------------------------------
    # Modelo
    # --------------------------------------------------------

    model = criar_modelo()

    model.fit(

        train_final[FEATURES],

        train_final[TARGET],

        cat_features=(
            CATEGORICAL_FEATURES
        ),
    )

    # --------------------------------------------------------
    # Artifact
    # --------------------------------------------------------

    artifact = {

        "model": model,

        "categorical_features": (
            CATEGORICAL_FEATURES
        ),

        "numeric_features": (
            NUMERIC_FEATURES
        ),

        "features": FEATURES,

        "target": TARGET,

        "train_start": (
            inicio_final
        ),

        "train_end": (
            data_final
        ),

        "train_rows": (
            len(train_final)
        ),

        "train_effectiveness": (
            train_final[TARGET].mean()
        ),

        "numeric_medians": (
            medianas
        ),

        "train_months": (
            TRAIN_MONTHS
        ),

        "model_type": (
            "CatBoostClassifier"
        ),

        "model_version": (
            "efetividade_catboost_3m_"
            f"{data_final.strftime('%Y%m%d')}"
        ),
    }

    joblib.dump(
        artifact,
        MODEL_PATH
    )

    print(
        f"\nModelo salvo em:"
    )

    print(
        MODEL_PATH
    )

    return (
        model,
        artifact
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MODELO DE EFETIVIDADE - CATBOOST")
    print("=" * 70)

    # ========================================================
    # CARREGAMENTO
    # ========================================================
    #
    # Ajuste somente esta linha caso seu arquivo tenha
    # outro caminho/formato.
    #
    # ========================================================

    # Conexão com o banco
    conexao = connect_database()

    # Lendo a query
    query = read_query('query_train.sql')

    # Carregando o df
    df = pd.read_sql_query(query, conexao)

    print(
        f"\nLinhas originais: "
        f"{len(df):,}"
    )

    print(
        f"Data inicial: "
        f"{df['data_agendamento'].min()}"
    )

    print(
        f"Data final: "
        f"{df['data_agendamento'].max()}"
    )

    # ========================================================
    # PREPARAÇÃO
    # ========================================================

    df = preparar_dados(df)

    # ========================================================
    # FEATURES
    # ========================================================

    df = criar_features(df)

    # ========================================================
    # LIMITAR HISTÓRICO
    # ========================================================
    #
    # Precisamos de:
    #
    # 3 meses de treino
    # +
    # 90 dias de histórico para as rolling features
    #
    # ========================================================

    data_inicio_historico = (

        SIMULATION_START

        - pd.DateOffset(
            months=TRAIN_MONTHS
        )

        - pd.Timedelta(
            days=90
        )
    )

    df = df[
        df["data_agendamento"]
        >= data_inicio_historico
    ].copy()

    print(
        f"\nLinhas após filtro histórico: "
        f"{len(df):,}"
    )

    # ========================================================
    # WALK-FORWARD
    # ========================================================

    (
        resultados_df,
        topk_df,
        calibracao_df
    ) = executar_walk_forward(df)

    # ========================================================
    # RELATÓRIO
    # ========================================================

    (
        topk_global,
        calibracao_global
    ) = gerar_relatorio(

        resultados_df,

        topk_df,

        calibracao_df
    )

    # ========================================================
    # SALVAR RESULTADOS
    # ========================================================

    # resultados_path = (

    #     ARTIFACT_DIR
    #     / "walk_forward_catboost_3m_semanal.csv"
    # )

    # topk_path = (

    #     ARTIFACT_DIR
    #     / "topk_catboost_3m_semanal.csv"
    # )

    # calibracao_path = (

    #     ARTIFACT_DIR
    #     / "calibracao_catboost_3m_semanal.csv"
    # )

    # resultados_df.to_csv(
    #     resultados_path,
    #     index=False
    # )

    # topk_df.to_csv(
    #     topk_path,
    #     index=False
    # )

    # calibracao_df.to_csv(
    #     calibracao_path,
    #     index=False
    # )

    # print("\nArquivos salvos:")

    # print(
    #     resultados_path
    # )

    # print(
    #     topk_path
    # )

    # print(
    #     calibracao_path
    # )

    # ========================================================
    # MODELO FINAL
    # ========================================================

    treinar_modelo_final(df)

    print("\n")

    print("=" * 70)
    print("PROCESSO FINALIZADO")
    print("=" * 70)




# ============================================================
# TOP-K SEMANAL
# ============================================================

def calcular_top_k_semanal(
    y_true,
    proba,
    data_inicio,
    data_fim,
    ks=(0.01, 0.05, 0.10, 0.20, 0.30)
):
    """
    Calcula a efetividade dos Top-K% da semana.

    Top-K é definido pelas maiores probabilidades previstas
    pelo modelo.

    Retorna uma lista de dicionários.
    """

    resultado = []

    y_true = np.asarray(y_true)
    proba = np.asarray(proba)

    # Segurança
    if len(y_true) != len(proba):
        raise ValueError(
            f"y_true e proba possuem tamanhos diferentes: "
            f"{len(y_true)} != {len(proba)}"
        )

    qtd_total = len(y_true)

    if qtd_total == 0:
        return resultado

    # Efetividade real de toda a semana
    efetividade_semana = y_true.mean()

    # Ordena da maior probabilidade para a menor
    ordem = np.argsort(-proba)

    for k in ks:

        qtd_top = max(1, int(np.ceil(qtd_total * k)))

        indices_top = ordem[:qtd_top]

        y_top = y_true[indices_top]
        proba_top = proba[indices_top]

        efetividade_top = y_top.mean()
        probabilidade_top = proba_top.mean()

        ganho_pp = (
            efetividade_top - efetividade_semana
        ) * 100

        resultado.append({
            "data_inicio": data_inicio,
            "data_fim": data_fim,
            "top_k": f"{int(k * 100)}%",
            "percentual": k,
            "qtd_total_semana": qtd_total,
            "qtd_top_k": qtd_top,
            "efetividade_semana": efetividade_semana,
            "efetividade_top_k": efetividade_top,
            "probabilidade_media_top_k": probabilidade_top,
            "ganho_pp": ganho_pp
        })

    return resultado
