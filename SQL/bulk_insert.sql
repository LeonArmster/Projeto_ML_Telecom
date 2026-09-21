BULK INSERT {tabela} -- Nome da tabela que fará a inserção dos dados
FROM '{arquivo}'     -- Nome do arquivo que fará e leitura para a inserção dos dados
WITH (
    FIRSTROW = 2,
    FIELDTERMINATOR = '|',
    ROWTERMINATOR = '0x0d0a',
    TABLOCK
)