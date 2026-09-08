SELECT
    tipo_atividade,
    slot,
    status_ordem,
    detalhe_atividade,
    tipo_servico,
    cidade,
    data_agendamento,
    segmento,
    categoria_cliente,
    duracao,
    rede_acesso,
    nome_operador,
    cluster_zeus,
    gerencia,
    efetividade
FROM TELECOM.DBO.fact_efetividade
WHERE 
    nome_operador NOT IN (
    'Application(osb)',
    'osb',
    'Message engine'
    )
    AND
    DATEDIFF(YEAR, data_agendamento, GETDATE()) = 3
