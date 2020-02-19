import pandas as pd

if1 = r'../input/gwlevel/CaracteristiquesPEM_clean.csv'  # data set 1 - raw
# data set 1 - clean
#if1 = r'../output/gwlevel/Ministry of Hydraulic/df_filtered_All_wells_wells.csv'
df1 = pd.read_csv(if1, encoding="ISO-8859-1")
#df1_rm_dup = df1.drop_duplicates(['Name PEM', 'Date'])

# data set 2'  # data set 2
if2 = r'../output/gwlevel/Alan Fryar/df_filtered Alan Fryar All wells.csv'
df2_org = pd.read_csv(if2, encoding="ISO-8859-1")

# data set 3'  # data set 3
if3 = r'../output/gwlevel/PEUEMOABERIAF/df_filtered PEUEMOABERIAF All wells.csv'
df3_org = pd.read_csv(if3, encoding="ISO-8859-1")

# data set 4'  # data set 4
# if4 = r'../input/gwlevel/Michel Wakil Donnees Puits Copie de PEM FALMEY DOSSO GAYA hp.csv'
# df4 = pd.read_csv(if4, encoding="ISO-8859-1")


# Diff between dataset 1 and dataset 2 ========================================
cols = ['IRH PEM', 'Name PEM']
df1 = df1[cols]
df2 = df2_org[cols]
df12 = df2.merge(df1, how='outer',
                 indicator=True).loc[lambda x: x['_merge'] == 'left_only']
df12_diff = pd.merge(df12, df2_org, on=['IRH PEM'], how='inner')
df12_diff.to_csv('../output/gwlevel/df12_dif.csv', encoding="ISO-8859-1")

# Diff between dataset 1 and dataset 3 ========================================
cols = ['Name PEM']
df1 = df1[cols]
df3 = df3_org[cols]
df13 = df3.merge(df1, how='outer',
                 indicator=True).loc[lambda x: x['_merge'] == 'left_only']
df13_diff = pd.merge(df13, df3_org, on=['Name PEM'], how='inner')
df13_diff.to_csv('../output/gwlevel/df13_dif.csv', encoding="ISO-8859-1")


def Diff(li1, li2):
    return (list(set(li1) - set(li2)))


name1 = list(df1['Name PEM'])
name2 = list(df2_org['Name PEM'])
name3 = list(df3_org['Name PEM'])
# name4 = list(df4['Name PEM'])

name_diff2 = Diff(name2, name1)
name_diff3 = Diff(name3, name1)
# name_diff4 = Diff(name4, name1)
# print(name_diff)
print(f'number of new observations from dataset 2 = {len(name_diff2)} \n')
print(f'number of new observations from dataset 3 = {len(name_diff3)} \n')
# print(f'number of new observations from dataset 4 = {len(name_diff4)} \n')

# Merge two datasets

df_new = pd.concat([df1, df2_org, df3_org])
print(df_new.shape)
df_new_rm_dup = df_new.drop_duplicates(
    subset=['Name PEM', 'XCoor', 'YCoor', 'Date', 'Equipped depth'])
# df_new_rm_dup = df_new.drop_duplicates(
#    subset = ['Name PEM', 'Date'])

print(df_new_rm_dup.shape)
ofile = '../output/gwlevel/gwlevel_clean_merged_all.csv'
df_new_rm_dup.to_csv(ofile)
