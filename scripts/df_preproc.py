
import pandas as pd

class Df_preprocesing:
    def __init__(self) -> None:
        pass
    
    def drop_columns(self,df,columns:list)->pd.DataFrame:
        return df.drop(columns=columns)