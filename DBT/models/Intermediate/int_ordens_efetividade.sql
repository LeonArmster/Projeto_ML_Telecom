{{ config(
    materialized='ephemeral'
)}}

SELECT
    ordem_id,
    atividade_id,
    tipo_atividade,
    slot,
    CASE
        WHEN status_ordem = 'N+â-úo Conclu+â-¡da' THEN 'Não Concluída'
        WHEN status_ordem = 'Conclu+â-¡da' THEN 'Concluída'
        ELSE status_ordem
    END AS status_ordem,
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
    CASE
        WHEN grupo_resultado = 'EFETIVO' THEN 1
        ELSE 0
    END AS efetividade,
    YEAR(data_agendamento) AS ano,
    MONTH(data_agendamento) AS mes,
    DAY(data_agendamento) AS dia,
    DATEPART(WEEKDAY, data_agendamento) AS dia_semana

FROM {{ ref('stg_ordens') }}