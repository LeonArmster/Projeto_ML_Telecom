# =====================================
# Arquivo de configuração do logger
# =====================================
import logging
from logging.handlers import RotatingFileHandler
from src.config import log_dir


# Criando a função para confugurar o logger
def configure_logger():
    """
    Objetivo:
    Configura o logger para o projeto ETL.
    """

    # Configuração do logger
    log_path = log_dir/'etl_log.log'

    # criando conector do logger
    logger = logging.getLogger()

    # Definindo o nível do log
    logger.setLevel(logging.INFO)


    # Configurando a gestão do logger para evitar multiplos handlers
    if not logger.handlers:
        # Criando um manipulador de arquivos rotativo
        handler = RotatingFileHandler(log_path, maxBytes=5*1024*1024, backupCount=5)

        # Definindo o formato do log
        formato_log = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Aplicando a formatação ao manipulador e adicionando ao logger
        handler.setFormatter(formato_log)

        # Adicionando o manipulador ao logger
        logger.addHandler(handler)


    return logger