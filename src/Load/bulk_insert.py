# ==============================
# Configurações para Bulk Insert
# ==============================
# Bibliotecas
from src.Database.database_config import read_query, connect_database
from sqlmodel import text
import logging

# Função para realizar o bulk insert
# # conectando ao banco
# engine = connect_database()

# # Tabela que será carregado
# tabela = 'Treinamento.DBO.Stg_Tb_Efetividade_Geral'



def executar_bulk_insert(engine, query:str, tabela:str, arquivo:str):
    """
    Objetivo:
    Função para executar o bulk insert no banco de dados.

    Parâmetros:
    - engine: Conexão com o banco de dados.
    - query: Query de bulk insert a ser executada.
    - tabela: Nome da tabela onde os dados serão inseridos.
    - arquivo: Caminho do arquivo CSV a ser carregado.
    #### Exemplo de arquivo
    ##### arquivo = (arquivo_docker/'Arquivo_Processado.csv').as_posix()

    Retorna:
    - None: A função não retorna nenhum valor, mas executa o bulk insert no banco de dados.
    """

    query = read_query(query)

    query = query.format(
        tabela=tabela,
        arquivo=arquivo
    )

    try:
        logging.info("Iniciando o Bulk Insert.")

        with engine.connect() as conexao:

            conexao.exec_driver_sql(query)

            conexao.commit()

            logging.info("Bulk insert commitado.")

            resultado = conexao.exec_driver_sql(f"SELECT COUNT(*) FROM {tabela}")

            quantidade = resultado.scalar()

            logging.info(
                f"Registros na tabela: {quantidade}"
            )

    except Exception as e:
        logging.error(f"Erro ao realizar bulk insert: {e}")
        raise