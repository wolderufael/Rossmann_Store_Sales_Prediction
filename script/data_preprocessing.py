import pandas as pd
import logging

class Preprocessor:
    def create_feature(self,df):
        logging.info("feature engineering")
        # creating "isWeekend" column to the data 
        for raw in df['DayOfWeek']:
            if raw in [1,2,3,4,5]:
                df.at[raw, 'isWeekend']==0
            else:
                df.at[raw, 'isWeekend']==1
        
        #creating number of days to holidays column to the data 
        df['Date']=pd.to_datetime(df['Date'])
        de
        
        
        