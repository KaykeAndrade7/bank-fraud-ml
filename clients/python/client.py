import requests

class FraudClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')

    # Testar se a API esta online
    def healthcheck(self):
        try:
            r = requests.get(f'{self.base_url}')
            return r.json()
        except Exception as e:
            return {'error': str(e)}
        
    # Previsao individual
    def predict_single(self, transaction: dict):
        url = f'{self.base_url}/predict'
        response = requests.post(url, json=transaction)

        if response.status_code != 200:
            raise Exception(f'Erro na API: {response.text}')
        
        return response.json()
    
    # Previsao em pacote 
    def predict_batch(self, transaction: list[dict]):
        url = f'{self.base_url}/predict-bash'
        response = requests.post(url, json=transaction)

        if response.status_code != 200:
            raise Exception(f'Erro na API: {response.text}')
        
        return response.json()