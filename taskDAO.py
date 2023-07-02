from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Configuração do banco de dados
db_host = 'localhost'
db_port = '3306'
db_name = 'evasionwatch'
db_username = 'root'
db_password = 'QweBHU*'

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