# ===========================
# Bibliotecas
# ===========================
import pandas as pd
from src.config import raw_dir, processed_dir
import logging

# Extraindo dados do CSV
def extract_csv(file_name: str) -> pd.DataFrame:
    """
    Objetivo:
    Função para extrair dados do arquivo CSV e ajustar para o modelo de inserção de dados.
    
    Parâmetros:
    file_name: Nome do arquivo CSV a ser lido. O arquivo deve estar localizado no diretório raw_data_dir.
    """

    try:
        logging.info(F"Iniciando leitura do CSV. {file_name}")
        df = pd.read_csv(raw_dir / file_name, low_memory=False, encoding='latin-1')
        logging.info("Transformando dados do DataFrame para o formato de inserção.")
        df = df.astype(str)
        df.to_csv(processed_dir / 'Arquivo_Processado.csv', index=False, sep=';', encoding='utf-8', lineterminator='\n')
        logging.info("Arquivo processado e salvo com sucesso.")
    except Exception as e:
        logging.error(f"Erro ao extrair dados do CSV: {e}")
        raise