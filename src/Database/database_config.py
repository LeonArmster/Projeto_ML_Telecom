# =============================
# Arquivo de configuração com o banco de dados
# =============================
from sqlmodel import create_engine
import urllib
from src.config import db_login, db_database, db_host, db_senha, sql_dir

def connect_database():
    """
    Objetivo: Função para conectar ao banco de dados

    return: Retorna conexão com o banco de dados
    """

    # Configurando o driver do banco
    db_driver = '{ODBC Driver 18 for SQL Server}'

    # Montando a URL com codificação segura para caracteres especiais
    params = urllib.parse.quote_plus(
        f'DRIVER={db_driver};SERVER={db_host};DATABASE={db_database};UID={db_login};PWD={db_senha};TrustServerCertificate=yes;'
    )
    engine = create_engine(f'mssql+pyodbc:///?odbc_connect={params}')

    return engine



def read_query(query_file:str):
    """
    Objetivo: Ler o arquivo na pasta SQL que contem a query que será utilizada

    query_file: nome do arquivo que será lido
    type: str

    return: retorna o arquivo lido e carregado
    type: str
    """

    with open(file=sql_dir/query_file, mode='r', encoding='utf-8') as file:
        query = file.read()

    return query