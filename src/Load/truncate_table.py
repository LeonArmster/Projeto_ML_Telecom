# ==============================
# Configurações para truncate table
# ==============================
# Bibliotecas
from src.Database.database_config import read_query, connect_database
from sqlmodel import text
import logging

# Função para realizar o truncate table

# # conectando ao banco
# engine = connect_database()

# # Tabela que será carregado
# tabela = 'Treinamento.DBO.Stg_Tb_Efetividade_Geral'


def executar_truncate_table(engine, query:str, tabela:str):
    """
    Objetivo:
    Função para executar o truncate table no banco de dados.

    Parâmetros:
    - engine: Conexão com o banco de dados.
    - query: Query de truncate table a ser executada.
    - tabela: Nome da tabela onde os dados serão truncados.

    Retorna:
    - None: A função não retorna nenhum valor, mas executa o truncate table no banco de dados.
    """

    # Nome da query
    query = read_query(query)

    # Configurando a query com os parâmetros
    query = query.format(tabela=tabela)

    # Conectando ao banco
    with engine.connect() as conexao:

        # Realizando o truncate table
        try:
            logging.info("Iniciando o Truncate Table.")
            conexao.execute(text(query))
            logging.info("Truncate table realizado com sucesso.")

        # Caso ocorra algum erro, ele será logado e a exceção será levantada
        except Exception as e:
            logging.error(f"Erro ao realizar truncate table: {e}")
            raise