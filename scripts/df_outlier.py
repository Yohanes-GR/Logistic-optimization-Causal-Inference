import pandas as pd
import numpy as np
from logger import get_logger

my_logger = get_logger("DfOutlier")
my_logger.debug("Loaded successfully!")


class DfOutlier:
  """
      Give an overview for a given data frame, 
      percentage of outliers in each column and 
      has methods for removeing or replacing outliers.
  """

  def __init__(self, df: pd.DataFrame) -> None:
    self.df = df

  def count_outliers(self, Q1, Q3, IQR):
    cut_off = IQR * 1.5
    temp_df = (self.df < (Q1 - cut_off)) | (self.df > (Q3 + cut_off))
    return [len(temp_df[temp_df[col] == True]) for col in temp_df]

  def calc_skew(self):
    return [self.df[col].skew() for col in self.df]

  def percentage(self, list):
    return [str(round(((value / self.df.shape[0]) * 100), 2)) + '%' for value in list]

  def getOverview(self) -> None:

    _labels = [column for column in self.df]
    Q1 = self.df.quantile(0.25)
    _median = self.df.quantile(0.5)
    Q3 = self.df.quantile(0.75)
    IQR = Q3 - Q1
    _skew = self.calc_skew()
    _outliers = self.count_outliers(Q1, Q3, IQR)

    columns = [
      'label',
      'number_of_outliers',
      'percentage_of_outliers',
      'skew',
      'Q1',
      'Median',
      'Q3'
    ]
    data = zip(
      _labels,
      _outliers,
      self.percentage(_outliers),
      _skew,
      Q1,
      _median,
      Q3,
    )
    new_df = pd.DataFrame(data=data, columns=columns)
    new_df.set_index('label', inplace=True)
    new_df.sort_values(by=["number_of_outliers"], inplace=True)
    return new_df


  def fix_outlier(self):
    column_name=list(self.df.columns[2:])
    for i in column_name:
        upper_quartile=self.df[i].quantile(0.75)
        lower_quartile=self.df[i].quantile(0.25)
        self.df[i]=np.where(self.df[i]>upper_quartile,self.df[i].median(),np.where(self.df[i]<lower_quartile,self.df[i].median(),self.df[i]))
    return self.df 


  def replace_outliers_with_mean(self, columns):
    for col in columns:
      Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
      IQR = Q3 - Q1
      cut_off = IQR * 1.5
      lower, upper = Q1 - cut_off, Q3 + cut_off

      self.df[col] = np.where(self.df[col] > upper, upper, self.df[col])
      self.df[col] = np.where(self.df[col] < lower, lower, self.df[col])