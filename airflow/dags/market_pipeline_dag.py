from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Default settings for the pipeline
default_args = {
    'owner': 'project_team',
    'depends_on_past': False,
    'start_date': datetime(2026, 5, 9),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'market_movement_pipeline',
    default_args=default_args,
    description='Automated pipeline for financial data ingestion, processing, and RNN training',
    schedule_interval=timedelta(days=1), # Set to run daily
    catchup=False,
) as dag:

    # 1. Scrape the live data
    task_ingest = BashOperator(
        task_id='data_ingestion',
        bash_command='python ../../data_ingestion.py', 
    )

    # 2. Convert text to sentiment labels
    task_sentiment = BashOperator(
        task_id='sentiment_labeling',
        bash_command='python ../../sentiment_analysis.py',
    )

    # 3. Construct the structured time-series datasets
    task_build_dataset = BashOperator(
        task_id='time_series_construction',
        bash_command='python ../../build_dataset.py',
    )

    # 4. Train models and track with MLflow
    task_train = BashOperator(
        task_id='model_training_and_evaluation',
        bash_command='python ../../train_models.py',
    )

    # Define the exact execution order using bitshift operators
    task_ingest >> task_sentiment >> task_build_dataset >> task_train