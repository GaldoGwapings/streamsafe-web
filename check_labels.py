
import pandas as pd

df = pd.read_csv('RiverQuality.csv')
print("Unique Labels found in CSV:")
print(df['RiskLevel'].unique())
