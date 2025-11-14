# IE4483-Sentiment-Analysis-Project
Two sentiment analysis pipelines using a Gemma-based transformer model and a TF-IDF + Feed-Forward Neural Network (MLP). Includes full preprocessing, hyperparameter tuning (Optuna), evaluation scripts, and reproducible training code.

## Gemma-based Transformer Pipeline

* Custom dataset preprocessing (data_gemma.py)
* Model definiton (model_gemma.py)
* Evaluation utilities (eval_gemma.py)
* Custom trainer class (train_gemma.py)
* Training start point (main_gemma.py)
* Output directory for metrics and plots (gemma_eval folder)
* Model Prediction results on test.json (submission_gemma.csv)

## TF-IDF Pipeline

* Custom dataset preprocessing (data.py)
* Custom TF-IDF script (features_tfidf.py)
* Model definition (models_tfidf.py)
* Evalution utilities (eval_tfidf.py)
* Custom trainer class (train_tfidf.py)
* Training start point (main_tfidf.py)
* Package requirements for Venv (requirements.txt)
* Output directory for metrics and plots (tfidf_eval folder)
* Model Prediction results on test.json (submission.csv)

## Data 

The repository includes:

train.json: Labeled training reviews
test.json: Unlabeled test reviews
Both pipelines consume the same dataset for fair comparison.

