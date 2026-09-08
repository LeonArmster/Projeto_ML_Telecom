# =============================================
# Criando o schema de recebimento dos dados
# =============================================
# Bibliotecas
from pydantic import BaseModel
from typing import List


class OrderRequest(BaseModel):
    order_id: str
    technician_id: str

class OrdersRequest(BaseModel):
    orders: List[OrderRequest]

class PredictionResponse(BaseModel):
    order_id: str
    technician_id: str
    success_probability: float
    success_percentage: float