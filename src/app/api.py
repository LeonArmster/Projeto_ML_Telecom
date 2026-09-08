# ========================================
# Arquivo de configuração da api
# ========================================
# Bibliotecas
import pandas as pd
import numpy as np
from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager
import joblib
from src.app.schemas import OrdersRequest
from src.app.prediction_service import PredictionService
from src.config import processed_dir
from src.Database.database_config import connect_database
from sqlalchemy.orm import Session
from sqlalchemy import text, bindparam

# Variável global para armazenar a instância do nosso serviço
prediction_service = None


# ===================================================
# Configuração da Sessão do Banco para o FastAPI
# ===================================================
engine = connect_database()

def get_db():
    """Gera uma sessão de banco de dados para cada requisição da API"""
    with Session(engine) as session:
        yield session




def aplicar_feature_engineering_lote(df: pd.DataFrame) -> pd.DataFrame:
    df["Inicio_Previsto"] = pd.to_datetime(df["Inicio_Previsto"])
    df["DT_Ag_Execucao"] = pd.to_datetime(df["DT_Ag_Execucao"], errors="coerce")
    
    df["HORA_INICIO"] = df["Inicio_Previsto"].dt.hour
    df["DIA_SEMANA"] = df["DT_Ag_Execucao"].dt.dayofweek
    df["DIA_MES"] = df["DT_Ag_Execucao"].dt.day
    df["MES"] = df["DT_Ag_Execucao"].dt.month
    df["SEMANA"] = df["DT_Ag_Execucao"].dt.isocalendar().week.astype("int")
    df["FDS"] = (df["DIA_SEMANA"] >= 5).astype(int)
    
    df['SLOT'] = np.where(df['HORA_INICIO'] < 12, 'MANHA', 'TARDE')
    
    # Tratamento de Nulos
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].fillna("UNKNOWN").astype(str)
        
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in num_cols:
        df[col] = df[col].fillna(-999)
        
    return df





@asynccontextmanager
async def lifespan(app: FastAPI):
    global prediction_service
    print("Iniciando o carregamento do modelo...")
    
    # IMPORTANTE: Ajuste o caminho se a pasta models estiver em outro lugar
    prediction_service = PredictionService(processed_dir/'modelo_rf.joblib')
    
    print("Modelo carregado com sucesso!")
    yield
    print("Limpando memória...")
    prediction_service = None

app = FastAPI(
    title='Api de Efetividade de Ordens',
    version='1.0.0',
    lifespan=lifespan
)


@app.get('/health')
def health():
    return {
        'status':'ok'
    }



@app.post('/api/v1/predictions/orders')
def predict_orders(request: OrdersRequest, db: Session = Depends(get_db)):
    # 1. Extrai todos os IDs enviados pelo Node
    incoming_orders = {order.order_id: order.technician_id for order in request.orders}
    order_ids = list(incoming_orders.keys())
    
    if not order_ids:
        return {"predictions": []}

    # 2. Faz uma ÚNICA consulta no banco de dados para todos os IDs
    # Usamos o tuple_ to SQLAlchemy lidar perfeitamente com a cláusula IN no SQL Server
    query = text("""
        SELECT * 
        FROM [TELECOM].[dbo].[Tb_Efetividade] 
        WHERE PON IN :order_ids
    """).bindparams(bindparam('order_ids', expanding=True))
    
    resultado = db.execute(query, {"order_ids": order_ids}).fetchall()
    
    if not resultado:
        return {"predictions": []}

    # 3. Converte o resultado inteiro para um DataFrame do Pandas
    df_ords = pd.DataFrame([row._asdict() for row in resultado])
    
    # 4. Atualiza a matrícula do técnico com a que veio do portal Node para cada ordem
    # Mapeia o dicionário de ordens/técnicos vindos da requisição
    df_ords['Matricula_Oper'] = df_ords['PON'].map(incoming_orders)
    
    # 5. Aplica a engenharia de features em LOTE (muito mais rápido)
    df_processed = aplicar_feature_engineering_lote(df_ords)
    
    # 6. Garante que as colunas estão na ordem exata que o modelo espera
    df_processed = df_processed[prediction_service.expected_features]
    
    # 7. Faz a PREVISÃO EM MASSA (Uma única chamada para o modelo CatBoost!)
    # predict_proba retorna um array com [[prob_0, prob_1], [prob_0, prob_1], ...]
    probabilities = prediction_service.model.predict_proba(df_processed)[:, 1]
    
    # 8. Monta a lista de respostas para devolver ao Node
    predictions = []
    for i, row in df_ords.iterrows():
        prob = probabilities[i]
        predictions.append({
            "order_id": row['PON'],
            "technician_id": incoming_orders[row['PON']],
            "success_probability": float(prob),
            "success_percentage": round(float(prob) * 100, 2)
        })
        
    return {"predictions": predictions}