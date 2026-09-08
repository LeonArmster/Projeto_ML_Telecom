# ===================================================
# Criando o serviço de previsão
# ===================================================
# Bibliotecas
import joblib
import pandas as pd

class PredictionService:
    def __init__(self, model_path: str):
        # 1. Carrega o dicionário inteiro do arquivo .joblib
        saved_data = joblib.load(model_path)
        
        # 2. Separa o modelo e a lista de features dinamicamente
        self.model = saved_data["modelo"]
        self.expected_features = saved_data["features"]

    def predict(self, raw_data_from_db: dict) -> float:
        # Transforma os dados brutos em um DataFrame de uma linha
        df = pd.DataFrame([raw_data_from_db])
        
        # Filtra e ordena as colunas exatamente como o modelo espera
        df = df[self.expected_features]
        
        # Calcula a probabilidade da ordem ser concluída (classe 1)
        probability = self.model.predict_proba(df)[0][1]
        
        return probability