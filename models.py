from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum, Float, Boolean, Double
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from flask_login import UserMixin
from enum import Enum as PythonEnum

Base = declarative_base()

class StudentData(Base):
    __tablename__ = 'student_data'

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, nullable=False)
    name = Column(String, nullable=False)
    studentid = Column(String, nullable=False, unique=True)
    #Features:
    descricao_modalidade_curso = Column(String, nullable=False)
    descricao_cota = Column(String, nullable=False)
    nivel_ensino = Column(String, nullable=False)
    descricao_situacao_matricula = Column(String, nullable=False)
    descricao_estado_civil = Column(String, nullable=False)
    descricao_turno = Column(String, nullable=False)
    descricao_naturalidade = Column(String, nullable=False)
    coeficiente_rendimento = Column(String, nullable=False)
    descricao_cor = Column(String, nullable=False)
    sexo = Column(String, nullable=False)
    descricao_curso = Column(String, nullable=False)
    percentual_frequencia = Column(String, nullable=False)
    descricao_tipo_escola_origem = Column(String, nullable=False)
    descricao_renda_per_capita = Column(String, nullable=False)

    csv_import_history_id = Column(Integer, ForeignKey('csv_import_history.id'))
    csv_import_history = relationship("CsvImportHistory", back_populates="student_data")

class StudentDataTemp(Base):
    __tablename__ = 'student_data_temp'

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, nullable=False)
    name = Column(String, nullable=False)
    studentid = Column(String, nullable=False, unique=True)
    #Features:
    descricao_modalidade_curso = Column(String, nullable=False)
    descricao_cota = Column(String, nullable=False)
    nivel_ensino = Column(String, nullable=False)
    descricao_situacao_matricula = Column(String, nullable=False)
    descricao_estado_civil = Column(String, nullable=False)
    descricao_turno = Column(String, nullable=False)
    descricao_naturalidade = Column(String, nullable=False)
    coeficiente_rendimento = Column(String, nullable=False)
    descricao_cor = Column(String, nullable=False)
    sexo = Column(String, nullable=False)
    descricao_curso = Column(String, nullable=False)
    percentual_frequencia = Column(String, nullable=False)
    descricao_tipo_escola_origem = Column(String, nullable=False)
    descricao_renda_per_capita = Column(String, nullable=False)

class AnalysisResultHistory(Base):
    __tablename__ = 'analysis_result_history'

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, nullable=False)
    studentid = Column(String, nullable=False)
    evaded = Column(Boolean, nullable=False)

class TrainingHistory(Base):
    __tablename__ = 'training_history'

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, nullable=False)
    accuracy = Column(Double, nullable=False)
    f1score = Column(Double, nullable=False)
    recall = Column(Double, nullable=False)
    kappa = Column(Double, nullable=False)
    model_score = Column(Double, nullable=False)
    feature_importances = Column(String, nullable=False)

class CsvImportHistory(Base):
    __tablename__ = 'csv_import_history'

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, nullable=False)
    situation = Column(Enum('RUNNING', 'SUCCESS', 'ERROR'), nullable=False)
    file_name = Column(String, nullable=False)
    file_size = Column(Float, nullable=False)
    row_count = Column(Integer, nullable=False, default=0)

    student_data = relationship("StudentData", cascade="all, delete", back_populates="csv_import_history")

    user_id = Column(Integer, ForeignKey('user.id'))
    user = relationship("User", backref="csv_import_histories")


class User(Base, UserMixin):
    __tablename__ = 'user'

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, nullable=False)
    user_role = Column(Enum('BASIC_USER', 'ADMIN_USER'), nullable=False)
    login = Column(String, nullable=False, unique=True)
    password = Column(String, nullable=False)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False, unique=True)


class UserRoleEnum(PythonEnum):
    BASIC_USER = 'BASIC_USER'
    ADMIN_USER = 'ADMIN_USER'


class SituationEnum(PythonEnum):
    RUNNING = 'RUNNING'
    SUCCESS = 'SUCCESS'
    ERROR = 'ERROR'

class TaskOperationEnum:
    IA_TRAIN = "IA_TRAIN"
    CUSTOMIZED_ANALYSIS = "CUSTOMIZED_ANALYSIS"
    FULL_ANALYSIS = "FULL_ANALYSIS"
    FAST_ANALYSIS = "FAST_ANALYSIS"
