# ==========================
# Arquivo de configuração
# ==========================
from pathlib import Path
from dotenv import load_dotenv
import os

# Diretorio Principal
main_dir = Path(__file__).resolve().parent.parent

# Diretorio do SQL
sql_dir = main_dir/'SQL'

# Diretorio de arquivos
data_dir = main_dir/'Data'

raw_dir = data_dir/'Raw'

processed_dir = data_dir/'Processed'

# Diretório de Log
log_dir = main_dir/'Log'

# Arquivo Env
env_file = main_dir/'src/.env'

# Login e senha de configuração do banco
load_dotenv(env_file)

db_login = os.getenv('DB_USER')
db_senha = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_database = os.getenv('DB_NAME')
db_port = os.getenv('DB_PORT')

