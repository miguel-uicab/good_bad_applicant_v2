```
cd deployment/mlflow
docker compose -f docker-compose.yaml up -d
docker compose ps
```


```
python src/models/train_model.py   --config configs/model_config.yaml   --data data/processed/df_cleaned_featured.sav   --models-dir models   --mlflow-tracking-uri http://localhost:5555
```