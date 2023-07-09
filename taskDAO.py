from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os

# Configuração do banco de dados

db_username = os.environ.get('DB_USERNAME', 'root')
db_password = os.environ.get('DB_PASSWORD', 'QweBHU*')
db_host = os.environ.get('DB_HOST', 'localhost')
db_port = os.environ.get('DB_PORT', '3306')
db_name = os.environ.get('DB_NAME', 'evasionwatch')

db_url = f'mysql+pymysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}'

engine = create_engine(db_url)
Session = sessionmaker(bind=engine)
session = Session()


def updateProgress(uuid, progress):
    query = text("UPDATE task SET progress = :progress WHERE uuid = :uuid")
    session.execute(query, {"progress": progress, "uuid": uuid})
    session.commit()
    session.close()

def updateSituationToFinished(uuid):
    query = text("UPDATE task SET situation = 'SUCCESS', progress = 100 WHERE uuid = :uuid")
    session.execute(query, {"uuid": uuid})
    session.commit()
    session.close()

def updateSituationToFailed(uuid, error):
    query = text("UPDATE task SET situation = 'ERROR', exception_msg = :error WHERE uuid = :uuid")
    session.execute(query, {"error": error, "uuid": uuid})
    session.commit()
    session.close()