import sys
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,date,open,high,low,volume,name):
        self.date=date
        self.open=open
        self.high=high
        self.low=low
        self.volume=volume
        self.name=name
        
    def get_data_as_dataframe(self):
        try:
            import pandas as pd
            custom_data_input_dict={
                "date":[self.date],
                "open":[self.open],
                "high":[self.high],
                "low":[self.low],
                "volume":[self.volume],
                "name":[self.name]
                
            }
            df=pd.DataFrame(custom_data_input_dict)
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['day_of_week'] = df['date'].dt.dayofweek
            df['week_of_year'] = df['date'].dt.isocalendar().week
            df['quarter'] = df['date'].dt.quarter
            df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
            df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
            df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
            df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
            df.drop(columns=['date'], inplace=True)
            
            data=pd.read_csv('artifacts/processed_sp500.csv')
            data=pd.concat([data,df],ignore_index=True)
            
            cols = ['open', 'high', 'low', 'close']
            windows = [5, 10, 20]

            for col in cols:
                for w in windows:
                    data[f'{col}_EMA_{w}'] = data[col].ewm(span=w, adjust=False).mean()
                    
            
            self.open_ema_5 = data['open_EMA_5'].values[-1]
            self.open_ema_10 = data['open_EMA_10'].values[-1]
            self.open_ema_20 = data['open_EMA_20'].values[-1]
            self.high_ema_5 = data['high_EMA_5'].values[-1]
            self.high_ema_10 = data['high_EMA_10'].values[-1]
            self.high_ema_20 = data['high_EMA_20'].values[-1]
            self.low_ema_5 = data['low_EMA_5'].values[-1]
            self.low_ema_10 = data['low_EMA_10'].values[-1]
            self.low_ema_20 = data['low_EMA_20'].values[-1]
            self.close_ema_5 = data['close_EMA_5'].values[-1]
            self.close_ema_10 = data['close_EMA_10'].values[-1]
            self.close_ema_20 = data['close_EMA_20'].values[-1]
            
            df['open_EMA_5'] = self.open_ema_5
            df['open_EMA_10'] = self.open_ema_10
            df['open_EMA_20'] = self.open_ema_20
            df['high_EMA_5'] = self.high_ema_5
            df['high_EMA_10'] = self.high_ema_10
            df['high_EMA_20'] = self.high_ema_20
            df['low_EMA_5'] = self.low_ema_5
            df['low_EMA_10'] = self.low_ema_10
            df['low_EMA_20'] = self.low_ema_20
            df['close_EMA_5'] = self.close_ema_5
            df['close_EMA_10'] = self.close_ema_10
            df['close_EMA_20'] = self.close_ema_20

            return df
        except Exception as e:
            raise CustomException(e,sys)
       