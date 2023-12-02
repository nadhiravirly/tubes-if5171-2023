# models.py
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Date, Numeric

Base = declarative_base()

class Price(Base):
    __tablename__ = 'price'
    date = Column(Date, primary_key=True)
    #open = Column(Numeric)
    #high = Column(Numeric)
    #low = Column(Numeric)
    close = Column(Numeric)
    #adj_close = Column(Numeric)
    #volume = Column(Numeric)
