USE TELECOM
GO

/****** Object:  Table [dbo].[Tb_Efetividade_Brasil]    Script Date: 04/07/2026 22:21:27 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

IF OBJECT_ID('dbo.Tb_Efetividade','u') IS NULL
BEGIN
	CREATE TABLE [dbo].[Tb_Efetividade](
		[PON] [varchar](max) NULL,
		[Atividade] [bigint] NULL,
		[Inicio_Previsto] [varchar](max) NULL,
		[Termino_Atividade] [varchar](max) NULL,
		[TP_Atividade] [varchar](max) NULL,
		[Intervalo] [varchar](max) NULL,
		[Status] [varchar](max) NULL,
		[Cidade] [varchar](max) NULL,
		[Detalhe_Atividade] [varchar](max) NULL,
		[DT_Ag_Execucao] [varchar](max) NULL,
		[Segmento] [varchar](max) NULL,
		[Categoria_Cliente] [varchar](max) NULL,
		[Duracao] [varchar](max) NULL,
		[Rede_Acesso] [varchar](max) NULL,
		[Matricula_Oper] [varchar](max) NULL,
		[Nome_Oper] [varchar](max) NULL,
		[Status_Final] [varchar](max) NULL,
		[Grupo_Info] [varchar](max) NULL,
		[Tipo_Servico] [varchar](max) NULL,
		[CLUSTER_ZEUS] [varchar](max) NULL,
		[GERENCIA] [varchar](max) NULL,
		[ORIGEM_ORDENS] [varchar](max) NULL,
		[CARGA_CONF] [varchar](max) NULL,
		[MATRICULA_CONF] [varchar](max) NULL,
		[DIA_ATUAL] [varchar](max) NULL,
		[Dia] [bigint] NULL,
		[Semana] [bigint] NULL,
		[Mes] [bigint] NULL,
		[Ano] [bigint] NULL,
		[INICIO_SLOT] [varchar](max) NULL,
		[FIM_SLOT] [varchar](max) NULL,
		[CHEGADA_SLOT] [varchar](max) NULL,
		[NUMERO_ORDEM] [varchar](max) NULL,
		[CANAL] [varchar](max) NULL,
		[DIV_GER_CIDADE] [varchar](max) NULL,
		[DEALER] [varchar](max) NULL,
		[LOGIN] [varchar](max) NULL,
		[NOME_LOGIN] [varchar](max) NULL,
		[DT_ABERTURA] [varchar](max) NULL,
		[BASE] [varchar](max) NULL,
		[CANALNIVEL1] [varchar](max) NULL
	) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
END
GO