from sqlalchemy import Column, Integer, String, Text, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from werkzeug.security import generate_password_hash, check_password_hash
from globals import encrypt_data, decrypt_data

Base = declarative_base()

class User(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    username = Column(String(150), unique=True, nullable=False)
    password_hash = Column(String(150), nullable=False)
    encrypted_info = Column(String(500), nullable=True)
    data_directory = Column(String(500), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def store_encrypted_info(self, info):
        self.encrypted_info = encrypt_data(info)

    def retrieve_encrypted_info(self):
        if self.encrypted_info:
            return decrypt_data(self.encrypted_info)
        return None

class Thread(Base):
    __tablename__ = 'thread'
    id = Column(Integer, primary_key=True)
    thread_id = Column(String(150), unique=True, nullable=False)

class Transcript(Base):
    __tablename__ = 'transcript'
    id = Column(Integer, primary_key=True)
    thread_id = Column(String(150), nullable=False)
    timestamp = Column(DateTime, default=func.now(), unique=True)
    content = Column(Text, nullable=False)

class Report(Base):
    __tablename__ = 'report'
    id = Column(Integer, primary_key=True)
    thread_id = Column(String(150), nullable=False)
    timestamp = Column(DateTime, default=func.now(), unique=True)
    content = Column(Text, nullable=False)

class Speakers(Base):
    __tablename__ = 'speakers'
    id = Column(Integer, primary_key=True)
    thread_id = Column(String(150), nullable=False)
    timestamp = Column(DateTime, default=func.now(), unique=True)
    speakers_list = Column(Text, nullable=False)  # Store as JSON string

