{{ config(
    materialized='table'
)}}

SELECT
    ordem_id,
    atividade_id,
    tipo_atividade,
    slot,
    status_ordem,
    detalhe_atividade,
    status_final,
    grupo_resultado,
    tipo_servico,
    cidade,
    data_agendamento,
    segmento,
    categoria_cliente,
    duracao,
    rede_acesso,
    matricula_operador,
    nome_operador,
    cluster_zeus,
    gerencia,
    efetividade,
    ano,
    mes,
    dia,
    dia_semana
FROM {{ ref('int_ordens_efetividade') }}