# import the libraries

from datetime import timedelta
# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG
# Operators; we need this to write tasks!
from airflow.operators.bash_operator import BashOperator
# This makes scheduling easy
from airflow.utils.dates import days_ago
#defining DAG arguments

# You can override them on a per-task basis during operator initialization
default_args = {
    'owner': 'Javier Jaime',
    'start_date': days_ago(0),
    'email': ['jjaime@thinkuity.com'],
    'email_on_failure': True,
    'email_on_retry': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# define the DAG
dag = DAG(
    dag_id='ETL_toll_data',
    default_args=default_args,
    description='Apache Airflow Final Assignment',
    schedule_interval=timedelta(days=1),
)

# define the tasks

# define the first task named unzip_data
unzip_data = BashOperator(
    task_id='unzip_data',
    bash_command='cd /home/project/airflow/dags/finalassignment/staging;\
        tar -xzf tolldata.tgz -C /home/project/airflow/dags/finalassignment/staging',
    dag=dag,
)

# define the second task
extract_data_from_csv = BashOperator(
    task_id='extract_data_from_csv',
    bash_command='cut -d"," -f1-4 < vehicle-data.txt > csv-data.csv',
    dag=dag,
)

# define the third task
extract_data_from_tsv = BashOperator(
    task_id='extract_data_from_tsv',
    bash_command='tr "\t" "," < tollplaza-data.tsv | cut -d"," -f5-7 > tsv_data.csv',
    dag = dag,
)

# define the fourth task
extract_data_from_fixed_width = BashOperator(
    task_id='extract_data_from_fixed_width',
    bash_command='tr -s "[:space:]" < payment-data.txt | cut -d" " -f11,12 > tempt.txt, ; tr" ""," < temp.txt > fixed_width_data.csv',
    dag=dag,
)

# define the fifth task
consolidate_data = BashOperator(
    task_id='consolidate_data',
    bash_command='paste -d, csv_data.csv tsv_data.csv fixed_width_data.csv > extracted_data.csv',
    dag=dag,
)

# define the data transformation
transform_data = BashOperator(
    task_id='transform_data',
    bash_command='paste -d, <(cut -d, f1,2,3 extracted_data.csv) <(cut -d, -f4 extracted_data.csv | tr "[a-z]" "[A-Z]"'
                 '<(cut -d, -f5,6,7,8,9 extracted_data.csv',
    dag=dag,
)

# define the task pipeline
unzip_data >> extract_data_from_csv >> extract_data_from_tsv >> extract_data_from_fixed_width >> consolidate_data >> transform_data