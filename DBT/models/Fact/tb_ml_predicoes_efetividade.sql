{{ config(
    materialized='incremental',
    unique_key=['ordem_id', 'atividade_id'],
    incremental_strategy='merge'
) }}

SELECT
    ordem_id,
    atividade_id,
    data_agendamento,
    tipo_atividade,
    slot,
    tipo_servico,
    cidade,
    segmento,
    categoria_cliente,
    nome_operador,
    cluster_zeus,
    gerencia,

    probabilidade,

    CASE
        WHEN probabilidade > 0.70 THEN 2
        WHEN probabilidade > 0.40 THEN 1
        ELSE 0
    END AS classificacao

FROM {{ source(
    'telecom',
    'tb_stg_ml_predicoes_efetividade'
) }}