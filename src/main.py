# ===================================
# Projeto ETL - Extração, Transformação e Carga de Dados
# ===================================
# Bibliotecas
from src.Extract.extract import extract_csv
from src.Load.bulk_insert import executar_bulk_insert
from src.Load.truncate_table import executar_truncate_table
from src.Logger.config_logger import configure_logger
from src.Database.database_config import connect_database
from src.config import processed_dir
import logging


def main():
    # Configurando o Logger
    configure_logger()
    logging.info("Iniciando o processo ETL.")

    # Conectando ao banco de dados
    logging.info("Estabelecendo conexão com o banco de dados.")
    engine = connect_database()
    logging.info("Conexão com o banco de dados estabelecida com sucesso.")

    # Transformando os dados do csv para inserção correta no banco
    extract_csv('Efetividade_2022-2023.csv')

    # Tabela principal onde os dados serão inseridos
    tabela = 'TELECOM.DBO.Tb_Efetividade'

    # Query de truncate table
    query = 'truncate_table.sql'

    # Executando o Truncate Table
    executar_truncate_table(engine, query, tabela)

    # Query de bulk insert
    query = 'bulk_insert.sql'

    # Caminho do arquivo processado para o bulk insert
    arquivo = r'/Data/Processed/Arquivo_Processado.csv'

    # Executando o Bulk Insert
    executar_bulk_insert(engine, query, tabela, arquivo)



if __name__ == "__main__":
    main()

