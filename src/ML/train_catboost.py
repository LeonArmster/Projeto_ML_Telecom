# ===================================
# Arquivo de treinamento do modelo catboost
# ====================================
# Bibliotecas
import gc
import warnings
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from src.Database.database_config import connect_database, read_query

# Configurando o warning
warnings.filterwarnings("ignore")


# ------------------------------------
# Conectando ao banco e carregando df
# ------------------------------------
# Conexão com o anco
conexao = connect_database()

# lendo a query
query = read_query('query_train.sql')

# Carregando df
df = pd.read_sql_query(query, conexao)


# ---------------------------------------
# Preparando os dados para o treinamento
# ---------------------------------------
# Convertendo data agendamento para data
df['data_agendamento'] = pd.to_datetime(df['data_agendamento'])

# Ordenando por data
df = df.sort_values('data_agendamento')


def calcular_efetividade_media(df, coluna_data, colunas_agrup, dias_movel, nome_padrao):

    # Histórico diário
    df_calculo = df.groupby([coluna_data] + colunas_agrup).agg(qtd_ordens=('efetividade','size'), qtd_efetivas=('efetividade', sum)).reset_index()

    # Colocando em ordem crescente
    df_calculo = df_calculo.sort_values([coluna_data] + colunas_agrup)

    # Histórico móvel da quantidade de ordens
    historico = df_calculo.set_index(coluna_data).groupby(colunas_agrup)['qtd_ordens'].rolling(dias_movel, closed='left', min_periods=1).sum().reset_index().rename(columns={'qtd_ordens':f'qtd_ordens_{dias_movel}_{nome_padrao}'})

    # Trazendo a coluna com a qtd valoers
    df_calculo = pd.merge(df_calculo,historico, on=[coluna_data] + colunas_agrup, how='left')

    # Histórico móvel de quantidade de efetivas
    historico = df_calculo.set_index(coluna_data).groupby(colunas_agrup)['qtd_efetivas'].rolling(dias_movel, closed='left', min_periods=1).sum().reset_index().rename(columns={'qtd_efetivas':f'qtd_efetivas_{dias_movel}_{nome_padrao}'})

    # Trazendo a coluna com a qtd valoers
    df_calculo = pd.merge(df_calculo,historico, on=[coluna_data] + colunas_agrup, how='left')

    # Efetividade histórica
    df_calculo[f'efetividade_{nome_padrao}_{dias_movel}'] = df_calculo[f'qtd_efetivas_{dias_movel}_{nome_padrao}'] / df_calculo[f'qtd_ordens_{dias_movel}_{nome_padrao}'].replace(0, np.nan)


    return df_calculo


coluna_data = 'data_agendamento'
colunas_agrup = ['nome_operador', 'segmento']
dias_movel = '7D'
nome_padrao = 'tecnico_segmento'



df_tec_segmento = calcular_efetividade_media(df, coluna_data=coluna_data, colunas_agrup=colunas_agrup, dias_movel=dias_movel, nome_padrao=nome_padrao)