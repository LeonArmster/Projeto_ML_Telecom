SELECT
    [PON] AS ordem_id,
    [Atividade] AS atividade_id,

    TRIM([TP_Atividade]) AS tipo_atividade,
    TRIM([Intervalo]) AS slot,
    TRIM([Status]) AS status_ordem,
    TRIM([Detalhe_Atividade]) AS detalhe_atividade,

    TRIM([Status_Final]) AS status_final,
    TRIM([Grupo_Info]) AS grupo_resultado,
    TRIM([Tipo_Servico]) AS tipo_servico,

    TRIM([Cidade]) AS cidade,
    [DT_Ag_Execucao] AS data_agendamento,

    TRIM([Segmento]) AS segmento,
    TRIM([Categoria_Cliente]) AS categoria_cliente,

    [Duracao] AS duracao,
    [Rede_Acesso] AS rede_acesso,

    [Matricula_Oper] AS matricula_operador,
    TRIM([Nome_Oper]) AS nome_operador,

    TRIM([CLUSTER_ZEUS]) AS cluster_zeus,
    TRIM([GERENCIA]) AS gerencia

FROM {{ source('telecom', 'tb_efetividade') }}