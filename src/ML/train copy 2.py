
import gc
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    log_loss,
)

from src.Database.database_config import (
    connect_database,
    read_query,
)


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_ARTIFACTS = Path("artifacts")

PASTA_ARTIFACTS.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# ARQUIVOS
# ============================================================

# ARQUIVO_MODELO = (
#     PASTA_ARTIFACTS
#     / "modelo_efetividade_catboost_3m_interacoes.joblib"
# )

# ARQUIVO_WALK_FORWARD = (
#     PASTA_ARTIFACTS
#     / "walk_forward_catboost_3m_interacoes.csv"
# )

# ARQUIVO_TOP_K = (
#     PASTA_ARTIFACTS
#     / "topk_catboost_3m_interacoes_semanal.csv"
# )

# ARQUIVO_TOP_K_RESUMO = (
#     PASTA_ARTIFACTS
#     / "topk_catboost_3m_interacoes_resumo.csv"
# )

# ARQUIVO_CALIBRACAO = (
#     PASTA_ARTIFACTS
#     / "calibracao_catboost_3m_interacoes.csv"
# )


# ============================================================
# SIMULAÇÃO
# ============================================================

DATA_INICIO_SIMULACAO = pd.Timestamp(
    "2023-09-01"
)

DATA_FIM_SIMULACAO = pd.Timestamp(
    "2023-12-31"
)


# ============================================================
# WALK-FORWARD
# ============================================================

MESES_TREINAMENTO = 3

DIAS_TESTE = 7


# ============================================================
# JANELAS
# ============================================================

JANELAS = [30, 60, 90]


# ============================================================
# TOP-K
# ============================================================

TOP_KS = (
    0.01,
    0.05,
    0.10,
    0.20,
    0.30,
)


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
    "tecnico_qtd_30d",
    "tecnico_efetividade_historica",

    # --------------------------------------------------------
    # SERVIÇO
    # --------------------------------------------------------

    "servico_efetividade_30d",
    "servico_efetividade_60d",
    "servico_efetividade_90d",
    "servico_qtd_30d",
    "servico_efetividade_historica",

    # --------------------------------------------------------
    # CIDADE
    # --------------------------------------------------------

    "cidade_efetividade_30d",
    "cidade_efetividade_60d",
    "cidade_efetividade_90d",
    "cidade_qtd_30d",
    "cidade_efetividade_historica",

    # --------------------------------------------------------
    # TÉCNICO VS GLOBAL
    # --------------------------------------------------------

    "tecnico_vs_global_30d",
    "tecnico_vs_global_60d",
    "tecnico_vs_global_90d",

    # --------------------------------------------------------
    # SERVIÇO VS GLOBAL
    # --------------------------------------------------------

    "servico_vs_global_30d",
    "servico_vs_global_60d",
    "servico_vs_global_90d",

    # --------------------------------------------------------
    # CIDADE VS GLOBAL
    # --------------------------------------------------------

    "cidade_vs_global_30d",
    "cidade_vs_global_60d",
    "cidade_vs_global_90d",

    # --------------------------------------------------------
    # TÉCNICO × SERVIÇO
    # --------------------------------------------------------

    "tecnico_servico_efetividade_30d",
    "tecnico_servico_efetividade_60d",
    "tecnico_servico_efetividade_90d",
    "tecnico_servico_qtd_30d",

    # --------------------------------------------------------
    # CIDADE × SERVIÇO
    # --------------------------------------------------------

    "cidade_servico_efetividade_30d",
    "cidade_servico_efetividade_60d",
    "cidade_servico_efetividade_90d",
    "cidade_servico_qtd_30d",

    # --------------------------------------------------------
    # SERVIÇO × SLOT
    # --------------------------------------------------------

    "servico_slot_efetividade_30d",
    "servico_slot_efetividade_60d",
    "servico_slot_efetividade_90d",
    "servico_slot_qtd_30d",

    # --------------------------------------------------------
    # TÉCNICO × SLOT
    # --------------------------------------------------------

    "tecnico_slot_efetividade_30d",
    "tecnico_slot_efetividade_60d",
    "tecnico_slot_efetividade_90d",
    "tecnico_slot_qtd_30d",

    # --------------------------------------------------------
    # TÉCNICO × SERVIÇO VS SERVIÇO
    # --------------------------------------------------------

    "tecnico_servico_vs_servico_30d",
    "tecnico_servico_vs_servico_60d",
    "tecnico_servico_vs_servico_90d",

    # --------------------------------------------------------
    # CIDADE × SERVIÇO VS SERVIÇO
    # --------------------------------------------------------

    "cidade_servico_vs_servico_30d",
    "cidade_servico_vs_servico_60d",
    "cidade_servico_vs_servico_90d",

    # --------------------------------------------------------
    # SERVIÇO × SLOT VS SERVIÇO
    # --------------------------------------------------------

    "servico_slot_vs_servico_30d",
    "servico_slot_vs_servico_60d",
    "servico_slot_vs_servico_90d",

    # --------------------------------------------------------
    # TÉCNICO × SLOT VS TÉCNICO
    # --------------------------------------------------------

    "tecnico_slot_vs_tecnico_30d",
    "tecnico_slot_vs_tecnico_60d",
    "tecnico_slot_vs_tecnico_90d",

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


# ============================================================
# COLUNAS OBRIGATÓRIAS
# ============================================================

COLUNAS_OBRIGATORIAS = [

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


# ============================================================
# VALIDAÇÃO
# ============================================================

def validar_colunas(df):

    faltantes = [
        coluna
        for coluna in COLUNAS_OBRIGATORIAS
        if coluna not in df.columns
    ]

    if faltantes:

        raise ValueError(
            "\nColunas obrigatórias ausentes:\n"
            + "\n".join(
                f" - {coluna}"
                for coluna in faltantes
            )
        )


# ============================================================
# PREPARAÇÃO
# ============================================================

def preparar_dados(df):

    print("\n" + "=" * 80)
    print("PREPARAÇÃO DOS DADOS")
    print("=" * 80)

    validar_colunas(df)

    df = df.copy()

    df["data_agendamento"] = pd.to_datetime(
        df["data_agendamento"],
        errors="coerce",
    )

    df["efetividade"] = pd.to_numeric(
        df["efetividade"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "data_agendamento",
            "efetividade",
        ]
    )

    valores_target = set(
        df["efetividade"].unique()
    )

    valores_invalidos = (
        valores_target
        - {0, 1}
    )

    if valores_invalidos:

        raise ValueError(
            "A coluna efetividade possui "
            f"valores diferentes de 0/1: "
            f"{valores_invalidos}"
        )

    df["efetividade"] = (
        df["efetividade"]
        .astype(np.int8)
    )

    for coluna in CATEGORICAL_FEATURES:

        df[coluna] = (
            df[coluna]
            .fillna("DESCONHECIDO")
            .astype(str)
        )

    df = (
        df
        .sort_values(
            "data_agendamento"
        )
        .reset_index(drop=True)
    )

    print(
        f"Linhas preparadas: {len(df):,}"
    )

    print(
        "Data inicial:",
        df["data_agendamento"].min()
    )

    print(
        "Data final:",
        df["data_agendamento"].max()
    )

    return df


# ============================================================
# ROLLING GLOBAL
# ============================================================

def criar_rolling_global(
    datas,
    valores,
    dias,
):

    datas = pd.to_datetime(
        datas
    )

    valores = np.asarray(
        valores,
        dtype=np.float32,
    )

    serie = pd.Series(
        valores,
        index=datas,
    )

    resultado = (
        serie
        .rolling(
            f"{dias}D",
            closed="left",
            min_periods=1,
        )
        .mean()
        .to_numpy(
            dtype=np.float32
        )
    )

    return resultado


# ============================================================
# CRIA CÓDIGOS DE GRUPO
# ============================================================

def criar_codigos_grupo(
    df,
    grupos,
):

    """
    Cria um código inteiro para cada combinação
    de grupos.

    Exemplo:

        nome_operador + tipo_servico

    sem criar strings concatenadas.
    """

    codigo = np.zeros(
        len(df),
        dtype=np.int64,
    )

    multiplicador = 1

    for coluna in reversed(grupos):

        valores = pd.factorize(
            df[coluna],
            sort=False,
        )[0]

        quantidade = (
            int(valores.max()) + 1
            if len(valores) > 0
            else 1
        )

        codigo += (
            valores.astype(
                np.int64
            )
            *
            multiplicador
        )

        multiplicador *= (
            quantidade + 1
        )

    return codigo


# ============================================================
# ROLLING POR GRUPO — MEMÓRIA OTIMIZADA
# ============================================================

def criar_rolling_grupo_otimizado(
    df,
    grupos,
    prefixo,
):

    """
    Implementação otimizada para grandes volumes.

    Não utiliza:

        groupby(...).groups
        merge
        MultiIndex gigantesco

    O cálculo utiliza códigos inteiros dos grupos
    e arrays NumPy.
    """

    n = len(df)

    resultado = {}

    for dias in JANELAS:

        resultado[
            f"{prefixo}_efetividade_{dias}d"
        ] = np.full(
            n,
            np.nan,
            dtype=np.float32,
        )

    resultado[
        f"{prefixo}_qtd_30d"
    ] = np.zeros(
        n,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Código dos grupos
    # --------------------------------------------------------

    codigo_grupo = criar_codigos_grupo(
        df,
        grupos,
    )

    # --------------------------------------------------------
    # Ordenação
    # --------------------------------------------------------

    datas = (
        df["data_agendamento"]
        .values
        .astype("datetime64[D]")
    )

    valores = (
        df["efetividade"]
        .values
        .astype(np.float32)
    )

    ordem = np.lexsort(
        (
            datas,
            codigo_grupo,
        )
    )

    codigo_ordenado = (
        codigo_grupo[ordem]
    )

    datas_ordenadas = (
        datas[ordem]
    )

    valores_ordenados = (
        valores[ordem]
    )

    # --------------------------------------------------------
    # Identifica início dos grupos
    # --------------------------------------------------------

    inicio_grupo = np.empty(
        n,
        dtype=bool,
    )

    inicio_grupo[0] = True

    inicio_grupo[1:] = (
        codigo_ordenado[1:]
        !=
        codigo_ordenado[:-1]
    )

    indices_inicio = np.flatnonzero(
        inicio_grupo
    )

    indices_fim = np.empty_like(
        indices_inicio
    )

    if len(indices_inicio) > 1:

        indices_fim[:-1] = (
            indices_inicio[1:]
        )

    indices_fim[-1] = n

    # --------------------------------------------------------
    # Cálculo por grupo
    # --------------------------------------------------------

    for inicio, fim in zip(
        indices_inicio,
        indices_fim,
    ):

        tamanho = fim - inicio

        if tamanho <= 1:

            continue

        datas_grupo = (
            datas_ordenadas[
                inicio:fim
            ]
        )

        valores_grupo = (
            valores_ordenados[
                inicio:fim
            ]
        )

        # ----------------------------------------------------
        # Soma acumulada
        # ----------------------------------------------------

        soma = np.cumsum(
            valores_grupo,
            dtype=np.float64,
        )

        soma = np.concatenate(
            (
                np.array(
                    [0.0],
                    dtype=np.float64,
                ),
                soma,
            )
        )

        # ----------------------------------------------------
        # Datas únicas do grupo
        # ----------------------------------------------------

        inicio_dia = np.empty(
            tamanho,
            dtype=bool,
        )

        inicio_dia[0] = True

        inicio_dia[1:] = (
            datas_grupo[1:]
            !=
            datas_grupo[:-1]
        )

        indices_dias = np.flatnonzero(
            inicio_dia
        )

        datas_unicas = (
            datas_grupo[
                indices_dias
            ]
        )

        fim_dias = np.empty_like(
            indices_dias
        )

        if len(indices_dias) > 1:

            fim_dias[:-1] = (
                indices_dias[1:]
            )

        fim_dias[-1] = tamanho

        # ----------------------------------------------------
        # Calcula cada janela
        # ----------------------------------------------------

        for dias in JANELAS:

            saida = np.full(
                tamanho,
                np.nan,
                dtype=np.float32,
            )

            for pos_dia, inicio_dia_pos in enumerate(
                indices_dias
            ):

                fim_dia_pos = (
                    fim_dias[pos_dia]
                )

                data_atual = (
                    datas_unicas[
                        pos_dia
                    ]
                )

                data_limite = (
                    data_atual
                    -
                    np.timedelta64(
                        dias,
                        "D",
                    )
                )

                esquerda = np.searchsorted(
                    datas_grupo,
                    data_limite,
                    side="left",
                )

                # ------------------------------------------------
                # IMPORTANTE:
                #
                # fim_dia_pos representa o início
                # da observação atual.
                #
                # Portanto, todas as ordens do
                # próprio dia ficam fora.
                # ------------------------------------------------

                soma_janela = (
                    soma[fim_dia_pos]
                    -
                    soma[esquerda]
                )

                quantidade_janela = (
                    fim_dia_pos
                    -
                    esquerda
                )

                if quantidade_janela > 0:

                    media = (
                        soma_janela
                        /
                        quantidade_janela
                    )

                    saida[
                        inicio_dia_pos:
                        fim_dia_pos
                    ] = np.float32(
                        media
                    )

            resultado[
                f"{prefixo}_efetividade_{dias}d"
            ][
                ordem[
                    inicio:fim
                ]
            ] = saida

        # ----------------------------------------------------
        # Volume 30d
        # ----------------------------------------------------

        saida_qtd = np.zeros(
            tamanho,
            dtype=np.float32,
        )

        for pos_dia, inicio_dia_pos in enumerate(
            indices_dias
        ):

            fim_dia_pos = (
                fim_dias[pos_dia]
            )

            data_atual = (
                datas_unicas[
                    pos_dia
                ]
            )

            data_limite = (
                data_atual
                -
                np.timedelta64(
                    30,
                    "D",
                )
            )

            esquerda = np.searchsorted(
                datas_grupo,
                data_limite,
                side="left",
            )

            quantidade = (
                fim_dia_pos
                -
                esquerda
            )

            saida_qtd[
                inicio_dia_pos:
                fim_dia_pos
            ] = quantidade

        resultado[
            f"{prefixo}_qtd_30d"
        ][
            ordem[
                inicio:fim
            ]
        ] = saida_qtd

    # --------------------------------------------------------
    # Libera memória
    # --------------------------------------------------------

    del codigo_grupo
    del ordem
    del codigo_ordenado
    del datas_ordenadas
    del valores_ordenados

    gc.collect()

    return resultado


# ============================================================
# HISTÓRICO CUMULATIVO
# ============================================================

def criar_historico(
    df,
    grupo,
    coluna_saida,
):

    # --------------------------------------------------------
    # Soma acumulada
    # --------------------------------------------------------

    soma_acumulada = (
        df
        .groupby(
            grupo,
            sort=False,
            dropna=False,
        )["efetividade"]
        .cumsum()
    )

    # --------------------------------------------------------
    # Quantidade anterior
    # --------------------------------------------------------

    quantidade_anterior = (
        df
        .groupby(
            grupo,
            sort=False,
            dropna=False,
        )
        .cumcount()
    )

    # --------------------------------------------------------
    # Remove a própria observação
    # --------------------------------------------------------

    soma_anterior = (
        soma_acumulada
        -
        df["efetividade"]
    )

    # --------------------------------------------------------
    # Média
    # --------------------------------------------------------

    resultado = (
        soma_anterior
        /
        quantidade_anterior.replace(
            0,
            np.nan,
        )
    )

    return resultado.astype(
        np.float32
    )


# ============================================================
# CRIA FEATURES
# ============================================================

def criar_features(df):

    print("\n" + "=" * 80)
    print("CRIAÇÃO DAS FEATURES")
    print("=" * 80)

    quantidade_original = len(df)

    df = df.copy()

    # ========================================================
    # HISTÓRICO CUMULATIVO
    # ========================================================

    print(
        "\n[1/8] Históricos acumulados..."
    )

    df[
        "tecnico_efetividade_historica"
    ] = criar_historico(
        df,
        "nome_operador",
        "tecnico_efetividade_historica",
    )

    df[
        "servico_efetividade_historica"
    ] = criar_historico(
        df,
        "tipo_servico",
        "servico_efetividade_historica",
    )

    df[
        "cidade_efetividade_historica"
    ] = criar_historico(
        df,
        "cidade",
        "cidade_efetividade_historica",
    )

    gc.collect()

    # ========================================================
    # CALENDÁRIO
    # ========================================================

    print(
        "\n[2/8] Calendário..."
    )

    df["ano"] = (
        df["data_agendamento"]
        .dt.year
        .astype(np.int16)
    )

    df["mes"] = (
        df["data_agendamento"]
        .dt.month
        .astype(np.int8)
    )

    df["dia"] = (
        df["data_agendamento"]
        .dt.day
        .astype(np.int8)
    )

    df["dia_semana"] = (
        df["data_agendamento"]
        .dt.dayofweek
        .astype(np.int8)
    )

    df["semana_ano"] = (
        df["data_agendamento"]
        .dt.isocalendar()
        .week
        .astype(np.int8)
    )

    df["fim_de_semana"] = (
        df["dia_semana"] >= 5
    ).astype(np.int8)

    # ========================================================
    # GLOBAL
    # ========================================================

    print(
        "\n[3/8] Global..."
    )

    for dias in [7, 14, 30, 60, 90]:

        print(
            f"  - Global {dias}d"
        )

        df[
            f"efetividade_global_{dias}d"
        ] = criar_rolling_global(
            df["data_agendamento"],
            df["efetividade"],
            dias,
        )

    gc.collect()

    # ========================================================
    # IMPORTANTE:
    #
    # A partir daqui só precisamos dos últimos 90 dias
    # anteriores ao início da simulação.
    #
    # Isso reduz brutalmente o volume usado pelas
    # interações.
    # ========================================================

    data_minima_features = (
        DATA_INICIO_SIMULACAO
        -
        pd.Timedelta(
            days=90
        )
    )

    print(
        "\nMantendo dados necessários "
        "para as features móveis..."
    )

    mascara_features = (
        df["data_agendamento"]
        >= data_minima_features
    )

    df = df.loc[
        mascara_features
    ].copy()

    print(
        f"Data mínima das features: "
        f"{data_minima_features.date()}"
    )

    print(
        f"Linhas após redução temporal: "
        f"{len(df):,}"
    )

    gc.collect()

    # ========================================================
    # TÉCNICO
    # ========================================================

    print(
        "\n[4/8] Técnico..."
    )

    resultado = criar_rolling_grupo_otimizado(
        df,
        ["nome_operador"],
        "tecnico",
    )

    for coluna, valores in resultado.items():

        df[coluna] = valores

    del resultado

    gc.collect()

    # ========================================================
    # SERVIÇO
    # ========================================================

    print(
        "\n[5/8] Serviço..."
    )

    resultado = criar_rolling_grupo_otimizado(
        df,
        ["tipo_servico"],
        "servico",
    )

    for coluna, valores in resultado.items():

        df[coluna] = valores

    del resultado

    gc.collect()

    # ========================================================
    # CIDADE
    # ========================================================

    print(
        "\n[6/8] Cidade..."
    )

    resultado = criar_rolling_grupo_otimizado(
        df,
        ["cidade"],
        "cidade",
    )

    for coluna, valores in resultado.items():

        df[coluna] = valores

    del resultado

    gc.collect()

    # ========================================================
    # INTERAÇÕES
    # ========================================================

    print(
        "\n[7/8] Interações..."
    )

    # --------------------------------------------------------
    # Técnico × Serviço
    # --------------------------------------------------------

    print(
        "\n  Técnico × Serviço..."
    )

    resultado = criar_rolling_grupo_otimizado(
        df,
        [
            "nome_operador",
            "tipo_servico",
        ],
        "tecnico_servico",
    )

    for coluna, valores in resultado.items():

        df[coluna] = valores

    del resultado

    gc.collect()

    # --------------------------------------------------------
    # Cidade × Serviço
    # --------------------------------------------------------

    print(
        "\n  Cidade × Serviço..."
    )

    resultado = criar_rolling_grupo_otimizado(
        df,
        [
            "cidade",
            "tipo_servico",
        ],
        "cidade_servico",
    )

    for coluna, valores in resultado.items():

        df[coluna] = valores

    del resultado

    gc.collect()

    # --------------------------------------------------------
    # Serviço × Slot
    # --------------------------------------------------------

    print(
        "\n  Serviço × Slot..."
    )

    resultado = criar_rolling_grupo_otimizado(
        df,
        [
            "tipo_servico",
            "slot",
        ],
        "servico_slot",
    )

    for coluna, valores in resultado.items():

        df[coluna] = valores

    del resultado

    gc.collect()

    # --------------------------------------------------------
    # Técnico × Slot
    # --------------------------------------------------------

    print(
        "\n  Técnico × Slot..."
    )

    resultado = criar_rolling_grupo_otimizado(
        df,
        [
            "nome_operador",
            "slot",
        ],
        "tecnico_slot",
    )

    for coluna, valores in resultado.items():

        df[coluna] = valores

    del resultado

    gc.collect()

    # ========================================================
    # RELATIVAS
    # ========================================================

    print(
        "\n[8/8] Features relativas..."
    )

    # --------------------------------------------------------
    # Técnico vs Global
    # --------------------------------------------------------

    for dias in JANELAS:

        df[
            f"tecnico_vs_global_{dias}d"
        ] = (
            df[
                f"tecnico_efetividade_{dias}d"
            ]
            -
            df[
                f"efetividade_global_{dias}d"
            ]
        ).astype(np.float32)

    # --------------------------------------------------------
    # Serviço vs Global
    # --------------------------------------------------------

    for dias in JANELAS:

        df[
            f"servico_vs_global_{dias}d"
        ] = (
            df[
                f"servico_efetividade_{dias}d"
            ]
            -
            df[
                f"efetividade_global_{dias}d"
            ]
        ).astype(np.float32)

    # --------------------------------------------------------
    # Cidade vs Global
    # --------------------------------------------------------

    for dias in JANELAS:

        df[
            f"cidade_vs_global_{dias}d"
        ] = (
            df[
                f"cidade_efetividade_{dias}d"
            ]
            -
            df[
                f"efetividade_global_{dias}d"
            ]
        ).astype(np.float32)

    # --------------------------------------------------------
    # Técnico × Serviço vs Serviço
    # --------------------------------------------------------

    for dias in JANELAS:

        df[
            f"tecnico_servico_vs_servico_{dias}d"
        ] = (
            df[
                f"tecnico_servico_efetividade_{dias}d"
            ]
            -
            df[
                f"servico_efetividade_{dias}d"
            ]
        ).astype(np.float32)

    # --------------------------------------------------------
    # Cidade × Serviço vs Serviço
    # --------------------------------------------------------

    for dias in JANELAS:

        df[
            f"cidade_servico_vs_servico_{dias}d"
        ] = (
            df[
                f"cidade_servico_efetividade_{dias}d"
            ]
            -
            df[
                f"servico_efetividade_{dias}d"
            ]
        ).astype(np.float32)

    # --------------------------------------------------------
    # Serviço × Slot vs Serviço
    # --------------------------------------------------------

    for dias in JANELAS:

        df[
            f"servico_slot_vs_servico_{dias}d"
        ] = (
            df[
                f"servico_slot_efetividade_{dias}d"
            ]
            -
            df[
                f"servico_efetividade_{dias}d"
            ]
        ).astype(np.float32)

    # --------------------------------------------------------
    # Técnico × Slot vs Técnico
    # --------------------------------------------------------

    for dias in JANELAS:

        df[
            f"tecnico_slot_vs_tecnico_{dias}d"
        ] = (
            df[
                f"tecnico_slot_efetividade_{dias}d"
            ]
            -
            df[
                f"tecnico_efetividade_{dias}d"
            ]
        ).astype(np.float32)

    # ========================================================
    # VALIDAÇÃO
    # ========================================================

    quantidade_final = len(df)

    print("\n" + "=" * 80)
    print("VALIDAÇÃO DAS FEATURES")
    print("=" * 80)

    print(
        f"Linhas após corte temporal: "
        f"{quantidade_final:,}"
    )

    if quantidade_final == 0:

        raise ValueError(
            "Nenhuma linha disponível "
            "após o corte temporal."
        )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    features_faltantes = [

        coluna

        for coluna in (
            CATEGORICAL_FEATURES
            +
            NUMERIC_FEATURES
        )

        if coluna not in df.columns
    ]

    if features_faltantes:

        raise ValueError(
            "\nFeatures não criadas:\n"
            +
            "\n".join(
                f" - {coluna}"
                for coluna in features_faltantes
            )
        )

    print(
        "OK: todas as features foram criadas."
    )

    print(
        f"\nFeatures categóricas: "
        f"{len(CATEGORICAL_FEATURES)}"
    )

    print(
        f"Features numéricas: "
        f"{len(NUMERIC_FEATURES)}"
    )

    print(
        f"Total: "
        f"{len(CATEGORICAL_FEATURES) + len(NUMERIC_FEATURES)}"
    )

    gc.collect()

    return df


# ============================================================
# PREPARA X E Y
# ============================================================

def preparar_X_y(df):

    features = (
        CATEGORICAL_FEATURES
        +
        NUMERIC_FEATURES
    )

    X = df[
        features
    ].copy()

    y = df[
        "efetividade"
    ].copy()

    for coluna in CATEGORICAL_FEATURES:

        X[coluna] = (
            X[coluna]
            .fillna("DESCONHECIDO")
            .astype(str)
        )

    for coluna in NUMERIC_FEATURES:

        X[coluna] = pd.to_numeric(
            X[coluna],
            errors="coerce",
        )

    return X, y


# ============================================================
# MODELO
# ============================================================

def criar_modelo():

    return CatBoostClassifier(

        iterations=250,

        depth=6,

        learning_rate=0.05,

        bootstrap_type="Bernoulli",

        subsample=0.8,

        l2_leaf_reg=1.0,

        loss_function="Logloss",

        eval_metric="Logloss",

        random_seed=42,

        thread_count=-1,

        verbose=False,

        allow_writing_files=False,
    )


# ============================================================
# MÉTRICAS
# ============================================================

def calcular_metricas(
    y_true,
    proba,
):

    auc = roc_auc_score(
        y_true,
        proba,
    )

    brier = brier_score_loss(
        y_true,
        proba,
    )

    logloss = log_loss(
        y_true,
        proba,
        labels=[0, 1],
    )

    calibracao = (
        proba.mean()
        -
        y_true.mean()
    ) * 100

    return {

        "auc": auc,

        "brier": brier,

        "logloss": logloss,

        "calibracao": calibracao,

    }


# ============================================================
# TOP-K
# ============================================================

def calcular_top_k_semanal(
    y_true,
    proba,
    data_inicio,
    data_fim,
):

    y_true = np.asarray(
        y_true
    )

    proba = np.asarray(
        proba
    )

    quantidade = len(
        y_true
    )

    if quantidade == 0:

        return []

    efetividade_semana = (
        y_true.mean()
    )

    ordem = np.argsort(
        -proba
    )

    resultados = []

    for k in TOP_KS:

        qtd_top = max(
            1,
            int(
                np.ceil(
                    quantidade * k
                )
            ),
        )

        indices = ordem[
            :qtd_top
        ]

        y_top = y_true[
            indices
        ]

        proba_top = proba[
            indices
        ]

        efetividade_top = (
            y_top.mean()
        )

        probabilidade_top = (
            proba_top.mean()
        )

        ganho_pp = (
            efetividade_top
            -
            efetividade_semana
        ) * 100

        resultados.append({

            "data_inicio":
                data_inicio,

            "data_fim":
                data_fim,

            "top_k":
                f"{int(k * 100)}%",

            "percentual":
                k,

            "qtd_total_semana":
                quantidade,

            "qtd_top_k":
                qtd_top,

            "efetividade_semana":
                efetividade_semana,

            "efetividade_top_k":
                efetividade_top,

            "probabilidade_media_top_k":
                probabilidade_top,

            "ganho_pp":
                ganho_pp,
        })

    return resultados


# ============================================================
# CALIBRAÇÃO
# ============================================================

def calcular_calibracao(
    y_true,
    proba,
    data_inicio,
    data_fim,
):

    y_true = np.asarray(
        y_true
    )

    proba = np.asarray(
        proba
    )

    resultados = []

    limites = np.arange(
        0,
        1.01,
        0.10,
    )

    for i in range(
        len(limites) - 1
    ):

        inferior = limites[i]

        superior = limites[i + 1]

        if i == len(limites) - 2:

            mascara = (
                (proba >= inferior)
                &
                (proba <= superior)
            )

        else:

            mascara = (
                (proba >= inferior)
                &
                (proba < superior)
            )

        quantidade = (
            mascara.sum()
        )

        if quantidade == 0:

            probabilidade_media = np.nan

            efetividade_real = np.nan

            erro_pp = np.nan

        else:

            probabilidade_media = (
                proba[
                    mascara
                ].mean()
            )

            efetividade_real = (
                y_true[
                    mascara
                ].mean()
            )

            erro_pp = (
                probabilidade_media
                -
                efetividade_real
            ) * 100

        resultados.append({

            "data_inicio":
                data_inicio,

            "data_fim":
                data_fim,

            "bucket":
                f"{int(inferior * 100)}-"
                f"{int(superior * 100)}%",

            "qtd":
                quantidade,

            "probabilidade_media":
                probabilidade_media,

            "efetividade_real":
                efetividade_real,

            "erro_pp":
                erro_pp,
        })

    return resultados


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print(
        "CATBOOST — EFETIVIDADE"
    )
    print(
        "WALK-FORWARD 3 MESES + INTERAÇÕES"
    )
    print("=" * 80)

    # ========================================================
    # 1. BANCO
    # ========================================================

    print("\n" + "=" * 80)
    print("CARREGAMENTO DO BANCO")
    print("=" * 80)

    print(
        "\nConectando ao banco..."
    )

    conexao = connect_database()



    print(
        "Lendo query_train.sql..."
    )

    query = read_query(
        "query_train.sql"
    )

    print(
        "Executando query..."
    )

    df = pd.read_sql_query(
        query,
        conexao,
    )


    print(
        f"\nLinhas carregadas: "
        f"{len(df):,}"
    )

    # ========================================================
    # 2. PREPARAÇÃO
    # ========================================================

    df = preparar_dados(
        df
    )

    # ========================================================
    # 3. FEATURES
    # ========================================================

    df = criar_features(
        df
    )

    # ========================================================
    # 4. X/Y
    # ========================================================

    X, y = preparar_X_y(
        df
    )

    print("\n" + "=" * 80)
    print("PREPARAÇÃO DO MODELO")
    print("=" * 80)

    print(
        f"X: {X.shape}"
    )

    print(
        f"y: {y.shape}"
    )

    # ========================================================
    # 5. WALK-FORWARD
    # ========================================================

    resultados_walk_forward = []

    resultados_top_k = []

    resultados_calibracao = []

    data_inicio_teste = (
        DATA_INICIO_SIMULACAO
    )

    numero_semana = 1

    indices_categoricos = [
        X.columns.get_loc(
            coluna
        )
        for coluna in CATEGORICAL_FEATURES
    ]

    while (
        data_inicio_teste
        <= DATA_FIM_SIMULACAO
    ):

        # ----------------------------------------------------
        # TESTE
        # ----------------------------------------------------

        data_fim_teste = min(

            data_inicio_teste
            +
            pd.Timedelta(
                days=DIAS_TESTE - 1
            ),

            DATA_FIM_SIMULACAO,
        )

        # ----------------------------------------------------
        # TREINO
        # ----------------------------------------------------

        data_inicio_treino = (
            data_inicio_teste
            -
            pd.DateOffset(
                months=MESES_TREINAMENTO
            )
        )

        data_fim_treino = (
            data_inicio_teste
            -
            pd.Timedelta(
                days=1
            )
        )

        # ----------------------------------------------------
        # MÁSCARAS
        # ----------------------------------------------------

        mascara_treino = (
            (
                df["data_agendamento"]
                >= data_inicio_treino
            )
            &
            (
                df["data_agendamento"]
                <= data_fim_treino
            )
        )

        mascara_teste = (
            (
                df["data_agendamento"]
                >= data_inicio_teste
            )
            &
            (
                df["data_agendamento"]
                <= data_fim_teste
            )
        )

        X_treino = X.loc[
            mascara_treino
        ]

        y_treino = y.loc[
            mascara_treino
        ]

        X_teste = X.loc[
            mascara_teste
        ]

        y_teste = y.loc[
            mascara_teste
        ]

        if len(X_treino) == 0:

            raise ValueError(
                f"Treino vazio na semana "
                f"{numero_semana}."
            )

        if len(X_teste) == 0:

            break

        # ----------------------------------------------------
        # MODELO
        # ----------------------------------------------------

        print("\n" + "-" * 80)

        print(
            f"SEMANA {numero_semana}"
        )

        print("-" * 80)

        print(
            f"Treino: "
            f"{data_inicio_treino.date()} "
            f"→ "
            f"{data_fim_treino.date()}"
        )

        print(
            f"Teste:  "
            f"{data_inicio_teste.date()} "
            f"→ "
            f"{data_fim_teste.date()}"
        )

        print(
            f"Linhas treino: "
            f"{len(X_treino):,}"
        )

        print(
            f"Linhas teste: "
            f"{len(X_teste):,}"
        )

        print(
            "Treinando CatBoost..."
        )

        modelo = criar_modelo()

        modelo.fit(
            X_treino,
            y_treino,
            cat_features=indices_categoricos,
        )

        # ----------------------------------------------------
        # PREDIÇÃO
        # ----------------------------------------------------

        proba = (
            modelo
            .predict_proba(
                X_teste
            )[:, 1]
        )

        # ----------------------------------------------------
        # MÉTRICAS
        # ----------------------------------------------------

        metricas = calcular_metricas(
            y_teste.to_numpy(),
            proba,
        )

        # ----------------------------------------------------
        # TOP-K
        # ----------------------------------------------------

        resultados_top_k.extend(

            calcular_top_k_semanal(

                y_teste.to_numpy(),

                proba,

                data_inicio_teste,

                data_fim_teste,
            )
        )

        # ----------------------------------------------------
        # CALIBRAÇÃO
        # ----------------------------------------------------

        resultados_calibracao.extend(

            calcular_calibracao(

                y_teste.to_numpy(),

                proba,

                data_inicio_teste,

                data_fim_teste,
            )
        )

        # ----------------------------------------------------
        # RESULTADO
        # ----------------------------------------------------

        resultados_walk_forward.append({

            "semana":
                numero_semana,

            "data_inicio_treino":
                data_inicio_treino,

            "data_fim_treino":
                data_fim_treino,

            "data_inicio_teste":
                data_inicio_teste,

            "data_fim_teste":
                data_fim_teste,

            "qtd_treino":
                len(X_treino),

            "qtd_teste":
                len(X_teste),

            "efetividade_treino":
                y_treino.mean(),

            "efetividade_teste":
                y_teste.mean(),

            "probabilidade_media":
                proba.mean(),

            "auc":
                metricas["auc"],

            "brier":
                metricas["brier"],

            "logloss":
                metricas["logloss"],

            "calibracao":
                metricas["calibracao"],
        })

        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print(
            f"\nEfetividade treino: "
            f"{y_treino.mean():.2%}"
        )

        print(
            f"Efetividade real:    "
            f"{y_teste.mean():.2%}"
        )

        print(
            f"Probabilidade média: "
            f"{proba.mean():.2%}"
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
            f"{metricas['calibracao']:+.2f}%"
        )

        # ----------------------------------------------------
        # Próxima semana
        # ----------------------------------------------------

        data_inicio_teste = (
            data_fim_teste
            +
            pd.Timedelta(
                days=1
            )
        )

        numero_semana += 1

        # ----------------------------------------------------
        # Libera memória
        # ----------------------------------------------------

        del (
            X_treino,
            y_treino,
            X_teste,
            y_teste,
            modelo,
            proba,
        )

        gc.collect()

    # ========================================================
    # WALK-FORWARD
    # ========================================================

    df_walk_forward = pd.DataFrame(
        resultados_walk_forward
    )

    df_walk_forward.to_csv(
        ARQUIVO_WALK_FORWARD,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # RESUMO
    # ========================================================

    print("\n\n" + "=" * 80)
    print("RESUMO WALK-FORWARD")
    print("=" * 80)

    print(
        f"\nAUC médio: "
        f"{df_walk_forward['auc'].mean():.4f}"
    )

    print(
        f"Brier médio: "
        f"{df_walk_forward['brier'].mean():.4f}"
    )

    print(
        f"LogLoss médio: "
        f"{df_walk_forward['logloss'].mean():.4f}"
    )

    print(
        f"Calibração média: "
        f"{df_walk_forward['calibracao'].mean():+.2f}%"
    )

    print(
        f"Erro absoluto médio: "
        f"{df_walk_forward['calibracao'].abs().mean():.2f}%"
    )

    # ========================================================
    # TOP-K
    # ========================================================

    df_top_k = pd.DataFrame(
        resultados_top_k
    )

    df_top_k.to_csv(
        ARQUIVO_TOP_K,
        index=False,
        encoding="utf-8-sig",
    )

    resumo_top_k = (
        df_top_k
        .groupby("top_k")
        .agg(

            semanas=(
                "data_inicio",
                "count"
            ),

            ganho_pp_medio=(
                "ganho_pp",
                "mean"
            ),

            ganho_pp_mediano=(
                "ganho_pp",
                "median"
            ),

            ganho_pp_min=(
                "ganho_pp",
                "min"
            ),

            ganho_pp_max=(
                "ganho_pp",
                "max"
            ),

            efetividade_top_k_media=(
                "efetividade_top_k",
                "mean"
            ),

            probabilidade_media=(
                "probabilidade_media_top_k",
                "mean"
            ),

            qtd_media=(
                "qtd_top_k",
                "mean"
            ),
        )
        .reset_index()
    )


    print("\n\n" + "=" * 80)
    print("RESUMO TOP-K")
    print("=" * 80)

    print(
        resumo_top_k.to_string(
            index=False
        )
    )

    # ========================================================
    # CALIBRAÇÃO
    # ========================================================

    df_calibracao = pd.DataFrame(
        resultados_calibracao
    )

    # ========================================================
    # MODELO FINAL
    # ========================================================

    print("\n\n" + "=" * 80)
    print("TREINAMENTO FINAL")
    print("=" * 80)

    data_inicio_final = (
        DATA_FIM_SIMULACAO
        -
        pd.DateOffset(
            months=MESES_TREINAMENTO
        )
        +
        pd.Timedelta(
            days=1
        )
    )

    data_fim_final = (
        DATA_FIM_SIMULACAO
    )

    mascara_final = (
        (
            df["data_agendamento"]
            >= data_inicio_final
        )
        &
        (
            df["data_agendamento"]
            <= data_fim_final
        )
    )

    X_final = X.loc[
        mascara_final
    ]

    y_final = y.loc[
        mascara_final
    ]

    print(
        f"Período: "
        f"{data_inicio_final.date()} "
        f"→ "
        f"{data_fim_final.date()}"
    )

    print(
        f"Linhas: "
        f"{len(X_final):,}"
    )

    print(
        f"Efetividade: "
        f"{y_final.mean():.2%}"
    )

    modelo_final = criar_modelo()

    print(
        "\nTreinando modelo final..."
    )

    modelo_final.fit(
        X_final,
        y_final,
        cat_features=indices_categoricos,
    )

    # ========================================================
    # ARTIFACT
    # ========================================================

    artifact = {

        "modelo":
            modelo_final,

        "categorical_features":
            CATEGORICAL_FEATURES,

        "numeric_features":
            NUMERIC_FEATURES,

        "features":
            (
                CATEGORICAL_FEATURES
                +
                NUMERIC_FEATURES
            ),

        "data_inicio_treino":
            data_inicio_final,

        "data_fim_treino":
            data_fim_final,

        "qtd_treino":
            len(X_final),

        "efetividade_treino":
            y_final.mean(),

        "modelo_tipo":
            "CatBoostClassifier",

        "janela_treinamento_meses":
            MESES_TREINAMENTO,

        "periodicidade_retreino":
            "semanal",

        "seed":
            42,

        "features_interacao":
            [
                "tecnico_servico",
                "cidade_servico",
                "servico_slot",
                "tecnico_slot",
            ],
    }

    # ========================================================
    # SALVA
    # ========================================================


    # ========================================================
    # FINAL
    # ========================================================


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    main()
