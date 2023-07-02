from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from flask_login import UserMixin
from enum import Enum as PythonEnum

Base = declarative_base()

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

    csv_import_history_temp_id = Column(Integer, ForeignKey('csv_import_history_temp.id'))
    csv_import_history_temp = relationship("CsvImportHistoryTemp", back_populates="student_data_temp")


class CsvImportHistoryTemp(Base):
    __tablename__ = 'csv_import_history_temp'

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, nullable=False)
    situation = Column(Enum('RUNNING', 'SUCCESS', 'ERROR'), nullable=False)
    file_name = Column(String, nullable=False)
    file_size = Column(Float, nullable=False)
    row_count = Column(Integer, nullable=False, default=0)

    student_data_temp = relationship("StudentDataTemp", cascade="all, delete", back_populates="csv_import_history_temp")

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
