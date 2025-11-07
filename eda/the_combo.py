'''
The big dataset

From Lock:
Families below poverty, persons below 150% of poverty, less than 9th
grade ed, less than HS ed, at least bachelor's degree ed, median
family income, median household income, vast majority income, 
non-english language, population (all), unemployed

From SASUMMARY:
Gross domestic product

From SAGDP1:
Real GDP

From KFF:
Gun violence death rate

Everything is going to be an average from the years 2019-2023 per
state
'''

import pandas as pd

pd.set_option('display.max_columns', None)

# families below poverty
fams_below = pd.read_csv('fams_below_pov.csv')
# print(fams_below.head())
fams_below.drop(['FIPS', 'Rank within US (of 52 states)'], axis=1, inplace=True)
# print(fams_below.head())

fams_below.drop([0, 1], axis=0, inplace=True)
fams_below.rename(columns={'Value (Percent)': 'Fams Below Pov %'}, inplace=True)
# print(fams_below.head())

## done

# families below 150% of poverty
below150 = pd.read_csv('below150.csv')
below150.drop(['FIPS', 'Rank within US (of 52 states)'], axis=1, inplace=True)
below150.drop([0, 1], axis=0, inplace=True)
below150.rename(columns={'Value (Percent)': 'Ppl (<150% Of Pov) %'}, inplace=True)

# print(below150.head())

## done

# persons below poverty
ppl_below = pd.read_csv('ppl_below.csv')
ppl_below.drop(['FIPS', 'Rank within US (of 52 states)'], axis=1, inplace=True)
ppl_below.drop([0, 1], axis=0, inplace=True)
ppl_below.rename(columns={'Value (Percent)': 'Ppl (Below Poverty) %'}, inplace=True)

# print(ppl_below.head())

## done

poverty = pd.merge(fams_below, below150, on='State', how='outer')
poverty_df = pd.merge(poverty, ppl_below, on='State', how='outer')

##########################################################################################################################################

## education time

# below 9th ed
below_9 = pd.read_csv('9th_less_ed.csv')
below_9.drop(['FIPS', 'Rank within US (of 52 states)'], axis=1, inplace=True)
below_9.drop([0, 1], axis=0, inplace=True)
below_9.rename(columns={'Value (Percent)': 'Below 9th Ed %'}, inplace=True)
# print(below_9.head())

# below hs ed

below_hs = pd.read_csv('hs_less_ed.csv')
below_hs.drop(['FIPS', 'Rank within US (of 52 states)'], axis=1, inplace=True)
below_hs.drop([0, 1], axis=0, inplace=True)
below_hs.rename(columns={'Value (Percent)': 'Below HS Ed %'}, inplace=True)

# print(below_hs.head())

bach = pd.read_csv('bach_ed.csv')
bach.drop(['FIPS', 'Rank within US (of 52 states)'], axis=1, inplace=True)
bach.drop([0, 1], axis=0, inplace=True)
bach.rename(columns={'Value (Percent)': 'Below Bach Ed %'}, inplace=True)

# print(bach.head())

education = pd.merge(below_9, below_hs, on='State', how='outer')
education_df = pd.merge(education, bach, on='State', how='outer')

# print(education_df.head())

##########################################################################################################################################

med_fam = pd.read_csv('med_fam_income.csv')
med_fam.drop(['FIPS', 'Rank within US (of 52 states)'], axis=1, inplace=True)
med_fam.drop([0, 1], axis=0, inplace=True)
med_fam.rename(columns={'Value (Dollars)': 'Median Fam Income (Dollars)'}, inplace=True)
# print(med_fam.head())

med_house = pd.read_csv('med_house.csv')
med_house.drop(['FIPS', 'Rank within US (of 52 states)'], axis=1, inplace=True)
med_house.drop([0, 1], axis=0, inplace=True)
med_house.rename(columns={'Value (Dollars)': 'Median Household Income (Dollars)'}, inplace=True)

# print(med_house.head())

vast_maj = pd.read_csv('vas_maj.csv')
vast_maj.drop(['FIPS', 'Rank within US (of 52 states)'], axis=1, inplace=True)
vast_maj.drop([0, 1], axis=0, inplace=True)
vast_maj.rename(columns={'Value (Dollars)': 'Vast Majority Income (Dollars)'}, inplace=True)

# print(vast_maj.head())

income = pd.merge(med_fam, med_house, on='State', how='outer')
income_df = pd.merge(income, vast_maj, on='State', how='outer')

# print(income_df.head())

##########################################################################################################################################

non_eng = pd.read_csv('non_eng.csv')
non_eng.drop(['FIPS', 'Rank within US (of 51 states)'], axis=1, inplace=True)
non_eng.drop([0, 1], axis=0, inplace=True)
non_eng.rename(columns={'Value (Percent)': 'Language Isolation %'}, inplace=True)

# print(non_eng.head())

unemp = pd.read_csv('unemp.csv')
unemp.drop(['FIPS', 'Rank within US (of 52 states)'], axis=1, inplace=True)
unemp.drop([0, 1], axis=0, inplace=True)
unemp.rename(columns={'Value (Percent)': 'Unemployed %'}, inplace=True)

# print(unemp.head())

other = pd.merge(non_eng, unemp, on='State', how='outer')

##########################################################################################################################################

# merging all

all_1 = pd.merge(poverty_df, education_df, on='State', how='outer')

all_2 = pd.merge(income_df, other, on='State', how='outer')

all_df = pd.merge(all_1, all_2, on='State', how='outer')

all_df.to_csv('combo.csv', index=False)


##########################################################################################################################################
##########################################################################################################################################

# SASUMMARY

# GDP

GDP = pd.read_csv('gdp.csv')
GDP.drop(['GeoFips', '1998', '1999', '2000', '2001', '2002', 
          '2003', '2004', '2005', '2006', '2007', '2008',
          '2009', '2010', '2011', '2012', '2013', '2014',
          '2015', '2016', '2017', '2018', '2024'], axis=1, inplace=True)
GDP.drop([0], axis=0, inplace=True)
GDP.rename(columns={'GeoName': 'State'}, inplace=True)
GDP['avg GDP from 2019-2023'] = GDP[['2019', '2020', '2021', '2022', '2023']].mean(axis=1)
GDP.drop(['2019', '2020', '2021', '2022', '2023'], axis=1, inplace=True)

# print(GDP.head())

# Real GDP

realGDP = pd.read_csv('realGDP.csv')
realGDP.drop(['GeoFips', '1998', '1999', '2000', '2001', '2002', 
          '2003', '2004', '2005', '2006', '2007', '2008',
          '2009', '2010', '2011', '2012', '2013', '2014',
          '2015', '2016', '2017', '2018', '2024'], axis=1, inplace=True)
realGDP.drop([0], axis=0, inplace=True)
realGDP.rename(columns={'GeoName': 'State'}, inplace=True)
realGDP['avg realGDP from 2019-2023'] = realGDP[['2019', '2020', '2021', '2022', '2023']].mean(axis=1)
realGDP.drop(['2019', '2020', '2021', '2022', '2023'], axis=1, inplace=True)

# print(realGDP.head())

gdp_df = pd.merge(GDP, realGDP, on='State', how='outer')

all_df = pd.merge(all_df, gdp_df, on='State', how='outer')

# print(all_df.head())

all_df.to_csv('combo.csv', index=False)

##########################################################################################################################################
##########################################################################################################################################

# gun violence death rate


nine = pd.read_csv('2019.csv')
nine.rename(columns={'Firearms Death Rate per 100,000': '19'}, inplace=True)

zero = pd.read_csv('2020.csv')
zero.rename(columns={'Firearms Death Rate per 100,000': '20'}, inplace=True)

one = pd.read_csv('2021.csv')
one.rename(columns={'Firearms Death Rate per 100,000': '21'}, inplace=True)

two = pd.read_csv('2022.csv')
two.rename(columns={'Firearms Death Rate per 100,000': '22'}, inplace=True)

three = pd.read_csv('2023.csv')
three.rename(columns={'Firearms Death Rate per 100,000': '23'}, inplace=True)

nine_zero = pd.merge(nine, zero, on='Location', how='outer')
one_two = pd.merge(one, two, on='Location', how='outer')
nine_two = pd.merge(nine_zero, one_two, on='Location', how='outer')

target = pd.merge(nine_two, three, on='Location', how='outer')

target['g_va death rate'] = target[['19', '20', '21', '22', '23']].mean(axis=1)
target.drop(['19', '20', '21', '22', '23'], axis=1, inplace=True)

target.rename(columns={'Location': 'State'}, inplace=True)

# print(target.head())

all_df = pd.merge(all_df, target, on='State', how='outer')

# print(all_df.head())

all_df.to_csv('combo.csv', index=False)

print(all_df.head())

new_df = all_df.drop(['Families (Below Poverty)', 'People (<150% Of Poverty)', 
            'People (Below Poverty)', 'People (Education: Less Than 9th Grade)', 
            'People(Education: Less Than High School)', "People (Education: At Least Bachelor's Degree)", 
            'Households (language Isolation)', 'People (Unemployed)'], axis=1)

new_df.to_csv('percents.csv', index=False)