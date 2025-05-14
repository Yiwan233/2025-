import pandas as pd

# df1=pd.read_csv('Q2data.csv')
# # df2=pd.read_csv('centrality_analysis_with_coach_effect.csv')
# df3=pd.read_csv('new_data.csv')

# df3=df3.drop_duplicates(['Year', 'NOC'])
# print(df3.head(10))

# # df1.merge(df2, on=['Year', 'NOC'], how='inner')
# df1.merge(df3, on=['Year', 'NOC'], how='inner').to_csv('final_data.csv', index=False)

# df1=pd.read_csv('final_merged_data.csv')
# df1.drop_duplicates(subset=['NOC', 'Year'])
# df1.to_csv('final_merged_data_filled.csv', index=False)

# df1=pd.read_csv('NR.csv')
# df2=pd.read_csv('CA.csv')

# df1=df1.drop('Predicted_CA',axis=1)

# df1.merge(df2, on=['Country'], how='inner')
# df1.drop_duplicates(['Year', 'NOC'])
# df1.to_csv('testdata.csv', index=False)

df1=pd.read_csv('testdata.csv')
df2=pd.read_csv('centrality_analysis_with_coach_effect.csv')

df1.merge(df2, on=['Year', 'NOC'], how='inner').to_csv('final_data1.csv', index=False)