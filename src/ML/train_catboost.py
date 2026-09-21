# ===================================
# Arquivo de treinamento do modelo catboost
# ====================================
# Bibliotecas
import gc
import warnings
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from src.Database.database_config import connect_database, read_query
from src.config import processed_dir, raw_dir
from src.Load.bulk_insert import executar_bulk_insert
from src.Load.truncate_table import executar_truncate_table

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

# Criando as colunas de data
df["ano"] = df["data_agendamento"].dt.year.astype(np.int16)

df["mes"] = df["data_agendamento"].dt.month.astype(np.int8)

df["dia"] = df["data_agendamento"].dt.day.astype(np.int8)

df["dia_semana"] = df["data_agendamento"].dt.dayofweek.astype(np.int8)

df["semana_ano"] = df["data_agendamento"].dt.isocalendar().week.astype(np.int8)

df["fim_de_semana"] = (df["dia_semana"] >= 5).astype(np.int8)



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

    # Trazendo a coluna com a qtd valores
    df_calculo = pd.merge(df_calculo,historico, on=[coluna_data] + colunas_agrup, how='left')

    # Efetividade histórica
    df_calculo[f'efetividade_{nome_padrao}_{dias_movel}'] = df_calculo[f'qtd_efetivas_{dias_movel}_{nome_padrao}'] / df_calculo[f'qtd_ordens_{dias_movel}_{nome_padrao}'].replace(0, np.nan)

    # Excluindo as colunas de qtd básica
    df_calculo = df_calculo.drop(columns=['qtd_ordens', 'qtd_efetivas'])

    return df_calculo



# efetividade media tecnico
# 7 dias
coluna_data = 'data_agendamento'
colunas_agrup = ['nome_operador']
dias_movel = '7D'
nome_padrao = 'tecnico'

df_tec_7d = calcular_efetividade_media(df, coluna_data=coluna_data, colunas_agrup=colunas_agrup, dias_movel=dias_movel, nome_padrao=nome_padrao)


# 15 dias
coluna_data = 'data_agendamento'
colunas_agrup = ['nome_operador']
dias_movel = '15D'
nome_padrao = 'tecnico'

df_tec_15d = calcular_efetividade_media(df, coluna_data=coluna_data, colunas_agrup=colunas_agrup, dias_movel=dias_movel, nome_padrao=nome_padrao)


# 30 dias
coluna_data = 'data_agendamento'
colunas_agrup = ['nome_operador']
dias_movel = '30D'
nome_padrao = 'tecnico'

df_tec_30d = calcular_efetividade_media(df, coluna_data=coluna_data, colunas_agrup=colunas_agrup, dias_movel=dias_movel, nome_padrao=nome_padrao)



# efetividade media tipo_servico
# 7 dias
coluna_data = 'data_agendamento'
colunas_agrup = ['tipo_servico']
dias_movel = '7D'
nome_padrao = 'tipo_servico'

df_servico_7d = calcular_efetividade_media(df, coluna_data=coluna_data, colunas_agrup=colunas_agrup, dias_movel=dias_movel, nome_padrao=nome_padrao)


# 15 dias
coluna_data = 'data_agendamento'
colunas_agrup = ['tipo_servico']
dias_movel = '15D'
nome_padrao = 'tipo_servico'

df_servico_15d = calcular_efetividade_media(df, coluna_data=coluna_data, colunas_agrup=colunas_agrup, dias_movel=dias_movel, nome_padrao=nome_padrao)


# 30 dias
coluna_data = 'data_agendamento'
colunas_agrup = ['tipo_servico']
dias_movel = '30D'
nome_padrao = 'tipo_servico'

df_servico_30d = calcular_efetividade_media(df, coluna_data=coluna_data, colunas_agrup=colunas_agrup, dias_movel=dias_movel, nome_padrao=nome_padrao)





# efetividade media tecnico x segmento
# 7 dias
coluna_data = 'data_agendamento'
colunas_agrup = ['nome_operador', 'segmento']
dias_movel = '7D'
nome_padrao = 'tecnico_segmento'

df_tec_segmento_7d = calcular_efetividade_media(df, coluna_data=coluna_data, colunas_agrup=colunas_agrup, dias_movel=dias_movel, nome_padrao=nome_padrao)


# 15 dias
coluna_data = 'data_agendamento'
colunas_agrup = ['nome_operador', 'segmento']
dias_movel = '15D'
nome_padrao = 'tecnico_segmento'

df_tec_segmento_15d = calcular_efetividade_media(df, coluna_data=coluna_data, colunas_agrup=colunas_agrup, dias_movel=dias_movel, nome_padrao=nome_padrao)


# 30 dias
coluna_data = 'data_agendamento'
colunas_agrup = ['nome_operador', 'segmento']
dias_movel = '30D'
nome_padrao = 'tecnico_segmento'

df_tec_segmento_30d = calcular_efetividade_media(df, coluna_data=coluna_data, colunas_agrup=colunas_agrup, dias_movel=dias_movel, nome_padrao=nome_padrao)


# efetividade media tecnico x tipo_atividade
# 7 dias
coluna_data = 'data_agendamento'
colunas_agrup = ['nome_operador', 'tipo_servico']
dias_movel = '7D'
nome_padrao = 'tecnico_atividade'

df_tec_atividade_7d = calcular_efetividade_media(df, coluna_data=coluna_data, colunas_agrup=colunas_agrup, dias_movel=dias_movel, nome_padrao=nome_padrao)

# 15 dias
coluna_data = 'data_agendamento'
colunas_agrup = ['nome_operador', 'tipo_servico']
dias_movel = '15D'
nome_padrao = 'tecnico_atividade'

df_tec_atividade_15d = calcular_efetividade_media(df, coluna_data=coluna_data, colunas_agrup=colunas_agrup, dias_movel=dias_movel, nome_padrao=nome_padrao)

# 30 dias
coluna_data = 'data_agendamento'
colunas_agrup = ['nome_operador', 'tipo_servico']
dias_movel = '30D'
nome_padrao = 'tecnico_atividade'

df_tec_atividade_30d = calcular_efetividade_media(df, coluna_data=coluna_data, colunas_agrup=colunas_agrup, dias_movel=dias_movel, nome_padrao=nome_padrao)


# efetividade media tecnico x cidade
# 07 dias
coluna_data = 'data_agendamento'
colunas_agrup = ['nome_operador', 'cidade']
dias_movel = '7D'
nome_padrao = 'tecnico_cidade'

df_tec_cidade_7d = calcular_efetividade_media(df, coluna_data=coluna_data, colunas_agrup=colunas_agrup, dias_movel=dias_movel, nome_padrao=nome_padrao)


# 15 dias
coluna_data = 'data_agendamento'
colunas_agrup = ['nome_operador', 'cidade']
dias_movel = '15D'
nome_padrao = 'tecnico_cidade'

df_tec_cidade_15d = calcular_efetividade_media(df, coluna_data=coluna_data, colunas_agrup=colunas_agrup, dias_movel=dias_movel, nome_padrao=nome_padrao)


# 30 dias
coluna_data = 'data_agendamento'
colunas_agrup = ['nome_operador', 'cidade']
dias_movel = '30D'
nome_padrao = 'tecnico_cidade'

df_tec_cidade_30d = calcular_efetividade_media(df, coluna_data=coluna_data, colunas_agrup=colunas_agrup, dias_movel=dias_movel, nome_padrao=nome_padrao)



# efetividade media atividade x slot
# 7 dias
coluna_data = 'data_agendamento'
colunas_agrup = ['tipo_servico', 'slot']
dias_movel = '7D'
nome_padrao = 'atividade_slot'

df_atividade_slot_7d = calcular_efetividade_media(df, coluna_data=coluna_data, colunas_agrup=colunas_agrup, dias_movel=dias_movel, nome_padrao=nome_padrao)


# 15 dias
coluna_data = 'data_agendamento'
colunas_agrup = ['tipo_servico', 'slot']
dias_movel = '15D'
nome_padrao = 'atividade_slot'

df_atividade_slot_15d = calcular_efetividade_media(df, coluna_data=coluna_data, colunas_agrup=colunas_agrup, dias_movel=dias_movel, nome_padrao=nome_padrao)


# 30 dias
coluna_data = 'data_agendamento'
colunas_agrup = ['tipo_servico', 'slot']
dias_movel = '30D'
nome_padrao = 'atividade_slot'

df_atividade_slot_30d = calcular_efetividade_media(df, coluna_data=coluna_data, colunas_agrup=colunas_agrup, dias_movel=dias_movel, nome_padrao=nome_padrao)


# efetividade media cidade x atividade
# 7 dias
coluna_data = 'data_agendamento'
colunas_agrup = ['cidade', 'tipo_servico']
dias_movel = '7D'
nome_padrao = 'cidade_atividade'

df_cidade_atividade_7d = calcular_efetividade_media(df, coluna_data=coluna_data, colunas_agrup=colunas_agrup, dias_movel=dias_movel, nome_padrao=nome_padrao)


# 15 dias
coluna_data = 'data_agendamento'
colunas_agrup = ['cidade', 'tipo_servico']
dias_movel = '15D'
nome_padrao = 'cidade_atividade'

df_cidade_atividade_15d = calcular_efetividade_media(df, coluna_data=coluna_data, colunas_agrup=colunas_agrup, dias_movel=dias_movel, nome_padrao=nome_padrao)


# 30 dias
coluna_data = 'data_agendamento'
colunas_agrup = ['cidade', 'tipo_servico']
dias_movel = '30D'
nome_padrao = 'cidade_atividade'

df_cidade_atividade_30d = calcular_efetividade_media(df, coluna_data=coluna_data, colunas_agrup=colunas_agrup, dias_movel=dias_movel, nome_padrao=nome_padrao)



# Juntando os dfs para fazer o treinamento
# Criando uma copia
df_treinamento = df.copy()

# Juntando os dfs
df_treinamento = pd.merge(df_treinamento, df_tec_7d, on=['data_agendamento', 'nome_operador'], how='left')

df_treinamento = pd.merge(df_treinamento, df_tec_15d, on=['data_agendamento', 'nome_operador'], how='left')

df_treinamento = pd.merge(df_treinamento, df_tec_30d, on=['data_agendamento', 'nome_operador'], how='left')

df_treinamento = pd.merge(df_treinamento, df_servico_7d, on=['data_agendamento','tipo_servico'], how='left')

df_treinamento = pd.merge(df_treinamento, df_servico_15d, on=['data_agendamento','tipo_servico'], how='left')

df_treinamento = pd.merge(df_treinamento, df_servico_30d, on=['data_agendamento', 'tipo_servico'], how='left')

df_treinamento = pd.merge(df_treinamento, df_atividade_slot_7d, on=['data_agendamento', 'tipo_servico', 'slot'], how='left')

df_treinamento = pd.merge(df_treinamento, df_atividade_slot_15d, on=['data_agendamento', 'tipo_servico', 'slot'], how='left')

df_treinamento = pd.merge(df_treinamento, df_atividade_slot_30d, on=['data_agendamento', 'tipo_servico', 'slot'], how='left')

df_treinamento = pd.merge(df_treinamento, df_tec_atividade_7d, on=['data_agendamento', 'nome_operador', 'tipo_servico'], how='left')

df_treinamento = pd.merge(df_treinamento, df_tec_atividade_15d, on=['data_agendamento', 'nome_operador', 'tipo_servico'], how='left')

df_treinamento = pd.merge(df_treinamento, df_tec_atividade_30d, on=['data_agendamento', 'nome_operador', 'tipo_servico'], how='left')

df_treinamento = pd.merge(df_treinamento, df_tec_segmento_7d, on=['data_agendamento', 'nome_operador', 'segmento'], how='left')

df_treinamento = pd.merge(df_treinamento, df_tec_segmento_15d, on=['data_agendamento', 'nome_operador', 'segmento'], how='left')

df_treinamento = pd.merge(df_treinamento, df_tec_segmento_30d, on=['data_agendamento', 'nome_operador', 'segmento'], how='left')

df_treinamento = pd.merge(df_treinamento, df_cidade_atividade_7d, on=['data_agendamento', 'cidade', 'tipo_servico'], how='left')

df_treinamento = pd.merge(df_treinamento, df_cidade_atividade_15d, on=['data_agendamento', 'cidade', 'tipo_servico'], how='left')

df_treinamento = pd.merge(df_treinamento, df_cidade_atividade_30d, on=['data_agendamento', 'cidade', 'tipo_servico'], how='left')



# --------------------------------------------
# Preparação para treinamento
# --------------------------------------------
# Separando entre treino e teste com a data limite
data_limite_treino = pd.Timestamp('2023-09-01')

#df_treinamento['ordem_id'] = df_treinamento['ordem_id'].astype('category')

df_treino = df_treinamento[df_treinamento['data_agendamento'] < data_limite_treino]
df_teste = df_treinamento[df_treinamento['data_agendamento'] >= data_limite_treino]
ordem_teste = df_teste[['ordem_id', 'atividade_id']]

# Variável alvo de teste e treinamento
y_train = df_treino['efetividade']
y_teste = df_teste['efetividade']

# Features de treinamento e teste
x_train = df_treino.drop(columns=['ordem_id', 'atividade_id', 'data_agendamento','efetividade', 'duracao', 'status_ordem', 'rede_acesso'])
x_teste = df_teste.drop(columns=['ordem_id', 'atividade_id', 'data_agendamento','efetividade', 'duracao', 'status_ordem', 'rede_acesso'])


# Features categoricas
colunas_categoricas = [
    'tipo_atividade',
    'slot',
    'detalhe_atividade',
    'tipo_servico',
    'cidade',
    'segmento',
    'categoria_cliente',
    'nome_operador',
    'cluster_zeus',
    'gerencia'
]

# Preenchendo os vazios dessas colunas
for coluna in colunas_categoricas:

    x_train[coluna] = x_train[coluna].fillna('SEM_INFORMACAO').astype(str)

    x_teste[coluna] = x_teste[coluna].fillna('SEM_INFORMACAO').astype(str)



# -----------------------------------------------------
# Treinamento
# -----------------------------------------------------
# Configuração de treinamento
modelo = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    loss_function='Logloss',
    eval_metric='AUC',
    cat_features=colunas_categoricas,
    random_seed=42,
    verbose=100
)

# Treinando o modelo
modelo.fit(
    x_train,
    y_train,
    cat_features=colunas_categoricas,
    eval_set=(x_teste, y_teste),
    use_best_model=True
)


# ------------------------------
# Testando o modelo
# ------------------------------
probabilidade = modelo.predict_proba(x_teste)[:, 1]




# ---------------------------------
# Avaliando o modelo
# ---------------------------------
# Métricas
roc_auc = roc_auc_score(y_teste, probabilidade)

brier = brier_score_loss(y_teste, probabilidade)

logloss = log_loss(y_teste, probabilidade)

print('=' * 60)
print('RESULTADOS DO CATBOOST')
print('=' * 60)

print(f'ROC-AUC : {roc_auc:.4f}')
print(f'Brier   : {brier:.4f}')
print(f'LogLoss : {logloss:.4f}')


print(f'Melhor iteração: {modelo.get_best_iteration()}')



# Importância das Features
importancia = pd.DataFrame({'feature': x_train.columns, 'importance': modelo.get_feature_importance()})

importancia = importancia.sort_values('importance', ascending=False).reset_index(drop=True)

print('=' * 60)
print('TOP 30 FEATURES')
print('=' * 60)

print(importancia.head(30).to_string(index=False))





# ----------------------------------------
# Análise por faixa de probabilidade
# ----------------------------------------

df_calibracao = pd.DataFrame({'real': y_teste.values, 'probabilidade': probabilidade})

# Criando faixas de 10%
faixas = np.arange(0, 1.1, 0.1)

df_calibracao['faixa'] = pd.cut(df_calibracao['probabilidade'], bins=faixas, include_lowest=True)

# Agrupando
calibracao = (df_calibracao.groupby('faixa', observed=True).agg(
        qtd_ordens=('real', 'size'),
        qtd_efetivas=('real', 'sum'),
        probabilidade_media=('probabilidade', 'mean'),
        efetividade_real=('real', 'mean')
    )
    .reset_index()
)

# Convertendo para %
calibracao['probabilidade_media'] = (calibracao['probabilidade_media'] * 100).round(2)

calibracao['efetividade_real'] = (calibracao['efetividade_real'] * 100).round(2)

print('=' * 80)
print('CALIBRAÇÃO POR FAIXA DE PROBABILIDADE')
print('=' * 80)

print(calibracao.to_string(index=False))





# ---------------------------------------------------------------
# Criando a tabela que subirá no banco com as predições
# ---------------------------------------------------------------
# preparando o df de precição
df_predicoes = df_teste

# Adicionando a coluna de probabilidade
df_predicoes['probabilidade'] = probabilidade

df_predicoes['efetividade'] = df_predicoes['efetividade'].fillna(0)

# Subindo o df com as probabilidades no banco
df_predicoes.to_csv(processed_dir/'tb_stg_ml_predicoes_efetividade.csv', index=False, sep='|')

#df_predicoes.head(10000).to_sql('tb_stg_ml_predicoes_efetividade', conexao, if_exists='replace', index=False)


arquivo = '/Data/Processed/tb_stg_ml_predicoes_efetividade.csv'

executar_truncate_table(conexao, query='truncate_table.sql', tabela='tb_stg_ml_predicoes_efetividade')

executar_bulk_insert(conexao, query='bulk_insert.sql', tabela='tb_stg_ml_predicoes_efetividade',arquivo=arquivo)













# Classificando a importância das colunas
pool_ordem = Pool(
    data=x_teste,
    cat_features=colunas_categoricas
)

shap_values  = modelo.get_feature_importance(type='ShapValues', data=pool_ordem)


print(shap_values.shape)
print(x_teste.shape)


shap_features = shap_values[:, :-1]
valor_base = shap_values[:, -1]


df_shap = pd.DataFrame(
    shap_features,
    columns=x_teste.columns
)



df_shap.insert(
    0,
    'ordem_id',
    ordem_teste.values
)


df_valores = df_shap.melt(
    id_vars='ordem_id',
    var_name='feature',
    value_name='valor'
)


df_shap = pd.DataFrame({'feature': ordem.columns, 'valor': ordem.iloc[0].values, 'shap': shap_features})

df_shap['shap_abs'] = df_shap['shap'].abs()

df_shap = df_shap.sort_values('shap_abs', ascending=False)

print('Valor base:', valor_base)
print()
print(df_shap.to_string(index=False))