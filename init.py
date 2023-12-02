import csv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from models.models import Base, Price
from datetime import datetime

def convert_to_float(value):
    return None if value == '-' else float(value.replace(',', ''))

def init_db():
    # Konfigurasi database
    engine = create_engine('sqlite:///price.db')
    Base.metadata.create_all(engine)

    # Membuat sesi
    session = Session(engine)

    # Membaca data dari file CSV
    with open('data/database_fix.csv', 'r', encoding='utf-8', errors='replace') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # Convert the string date to a Python date object
            date_str = row['Date']
            date_object = datetime.strptime(date_str, '%m/%d/%Y').date()

            #new_price = Price(date=row['Date'], open=row['Open'], high=row['High'], low=row['Low'], close=row['Close*'], adj_close=row['Adj Close**'], volume=row['Volume'])
            #new_price = Price(date=date_object, open=row['Open'], high=row['High'], low=row['Low'], close=row['Close*'], adj_close=row['Adj Close**'], volume=row['Volume'])
            
            # Remove commas from the volume string and convert to float
            # Convert all attributes to float, replacing '-' with None
            #open_value = convert_to_float(row['Open'])
            #high_value = convert_to_float(row['High'])
            #low_value = convert_to_float(row['Low'])
            close_value = convert_to_float(row['Close*'])
            #adj_close_value = convert_to_float(row['Adj Close**'])
            #volume_value = convert_to_float(row['Volume'])

            new_price = Price(
                date=date_object,
                #open=open_value,
                #high=high_value,
                #low=low_value,
                close=close_value,
                #adj_close=adj_close_value,
                #volume=volume_value
            )
            
            session.add(new_price)

    session.commit()

if __name__ == "__main__":
    init_db()
