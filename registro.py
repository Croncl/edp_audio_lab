"""
registro.py: camada fina sobre o MLflow. Com ativo=False (ou sem mlflow instalado) não faz nada,
então qualquer script pode chamar o registro sem se preocupar.

    r = Registro(True, "treino_unet", "unet_v1", {"lr": 1e-3})
    r.metricas({"val_loss": 0.3}, passo=1)
    r.arquivo("modelos/unet_v1/best.pt")
    r.fim()

Os dados ficam em ./mlflow.db (métricas) e ./mlartifacts (arquivos). Para ver:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import os


class Registro:
    def __init__(self, ativo, experimento, nome, params=None):
        self.ativo, self.mlflow = ativo, None
        if not ativo:
            return
        try:
            import mlflow
        except ImportError:
            print("mlflow não instalado (pip install mlflow); seguindo sem registro.")
            self.ativo = False
            return
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        mlflow.set_experiment(experimento)
        mlflow.start_run(run_name=nome)
        self.mlflow = mlflow
        if params:
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})

    def params(self, d):
        if self.ativo:
            self.mlflow.log_params({k: str(v)[:250] for k, v in d.items()})

    def metricas(self, d, passo=None):
        if self.ativo:
            self.mlflow.log_metrics({k: float(v) for k, v in d.items()}, step=passo)

    def arquivo(self, caminho, pasta=None):
        if self.ativo and os.path.exists(caminho):
            self.mlflow.log_artifact(str(caminho), pasta)

    def fim(self):
        if self.ativo:
            self.mlflow.end_run()
