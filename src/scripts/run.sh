#!/bin/bash

python -m src.scripts.main --task_type classification --model_name KNN
python -m src.scripts.main --task_type classification --model_name PassiveAggressive
python -m src.scripts.main --task_type classification --model_name LogisticRegression
python -m src.scripts.main --task_type classification --model_name DecisionTree
python -m src.scripts.main --task_type classification --model_name RandomForest
python -m src.scripts.main --task_type classification --model_name XGBoost
python -m src.scripts.main --task_type classification --model_name LightGBM
python -m src.scripts.main --task_type classification --model_name ExtraTrees


python -m src.scripts.main --task_type regression --model_name KNN
python -m src.scripts.main --task_type regression --model_name PassiveAggressive
python -m src.scripts.main --task_type regression --model_name DecisionTree
python -m src.scripts.main --task_type regression --model_name RandomForest
python -m src.scripts.main --task_type regression --model_name XGBoost
python -m src.scripts.main --task_type regression --model_name LightGBM
python -m src.scripts.main --task_type regression --model_name ExtraTrees

python -m src.scripts.main --task_type classification --model_name SVM
python -m src.scripts.main --task_type regression --model_name SVM
