#!/usr/bin/python






"""
    EXPLORING ENRON DATASET
"""
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
get_ipython().run_line_magic('matplotlib', 'inline')

from time import time 
import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn import cross_validation

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )


from matplotlib import pyplot

import pandas as pd
import seaborn as sb
import matplotlib.pyplot
from io import StringIO
from tabulate import tabulate

pd.set_option('display.width', 500) ## in order to get nicer display of pandas dataframes on screen
#pd.options.display.float_format = '{:,.0f}'.format




print("            ")


"""
SIZE, AVAILABLE FEATURES
"""
Enron_df = pd.DataFrame.from_records(data_dict).T
print("Features_list :")
Enron_df.info()

"""
POIs in the DATASET
"""

print("    ")
print("NUMBER OF POI IN THE SAMPLE: ",Enron_df['poi'].value_counts())

POI=()
POI=(Enron_df[Enron_df['poi']== True])
print(tabulate(POI.iloc[0:-1,13:14], headers='keys', tablefmt='presto',floatfmt=".0f"))




"""
DATASET CLEANING - Numeric values as numbers
"""
clefs=Enron_df.columns.tolist()
for i in clefs :
    if i !='email_address':
        Enron_df[i]=Enron_df[i].astype(float)

"""
DATASET CLEANING - NAN values in features
"""
valeursnan = Enron_df.isnull().sum()

ax=Enron_df.isnull().sum().plot.barh(width=0.75,figsize=(15, 9))
for i, v in enumerate(valeursnan):
    ax.text(v + 3, i + .25, str(v), color='darkblue', fontweight='bold')
pyplot.title('Number of incomplete data per feature, out of 146 records', fontsize=16)
pyplot.xlabel('number of null values', fontsize=12)
pyplot.ylabel('features', fontsize=12)
pyplot.show()

"""
DATASET CLEANING Enquiry about high number of Nan Values features
"""
print('non NA loan advances' )

print('restricted_stock_deferred')
print(tabulate(Enron_df[Enron_df['restricted_stock_deferred']<0 ].iloc[0:-1,15:16], headers='keys', tablefmt='fancy_grid',floatfmt=".0f"))
print(tabulate(Enron_df[Enron_df['restricted_stock_deferred']>0].iloc[0:-1,15:16], headers='keys', tablefmt='fancy_grid'))
print('non null loan advances')
print(tabulate(Enron_df[Enron_df['loan_advances']>0].iloc[0:-1,10:11], headers='keys', tablefmt='fancy_grid',floatfmt=".0f"))
print(tabulate(Enron_df[Enron_df['loan_advances']<0].iloc[0:-1,10:11], headers='keys', tablefmt='fancy_grid',floatfmt=".0f"))
print('non null director fees')
print(tabulate(Enron_df[Enron_df['director_fees']>0].iloc[0:-1,3:4], headers='keys', tablefmt='fancy_grid',floatfmt=".0f"))


"""
Remove column 'loan_advances' from dataset
"""
Enron_df1=Enron_df.drop(['loan_advances'], axis=1)



"""
DATASET CLEANING - Replace all Nan values in numeric features by 0 
"""

Enron2=Enron_df1.fillna(0)
Enron3=Enron2.T
print(Enron2.info()) ## check that all data are now numeric, except email addresses

#print(tabulate(Enron3.iloc[0:-1,0:3], headers='keys', tablefmt='fancy_grid'))# check that loan_advances has disapperaed from features

"""
DATASET CLEANING - IDENTIFY INCOMPLETE LINES
"""
notrecorded = []
for person in Enron3:
    
    n = 0
    for key, value in Enron3[person].iteritems():
        if value == 'NaN' or value == 0:
            n += 1
       
        if n > 17:
            if person not in notrecorded :
                notrecorded.append(person)


for nom in notrecorded :
        print("incomplete lines: ", Enron2.loc[nom][['poi','total_payments']])               


"""
Remove incomplete lines from dataset
"""

Enron2.drop(labels='GRAMM WENDY L',axis=0, inplace=True)
Enron2.drop(labels='LOCKHART EUGENE E',axis=0,inplace=True)
Enron2.drop(labels='WHALEY DAVID A',axis=0,inplace=True)
Enron2.drop(labels='WROBEL BRUCE',axis=0,inplace=True)
Enron2.drop(labels='THE TRAVEL AGENCY IN THE PARK',axis=0, inplace=True)
Enron3=Enron2.T

print("Enron2",Enron2.shape) 
#
"""
DATASET CLEANING - Find outliers (1/2)

First identify outliers (small an enormous values)
"""
sb.lmplot(x='bonus', y= 'salary', data=Enron2, palette='Set1',size=5)
pyplot.title('Salary/Bonus', fontsize=18)
pyplot.xlabel('Bonus', fontsize=16)
pyplot.ylabel('Salary', fontsize=16)
pyplot.show()


print("bigest salaries, bonus, with POI indication")
outhigh=Enron2[Enron2['bonus']>1000000 ].iloc[0:-1,[0,15,12]]
print(tabulate(outhigh.sort_values(by=['bonus','salary'],ascending=False), headers='keys', tablefmt='fancy_grid',floatfmt=".0f"))
Enron2.drop(labels='TOTAL',axis=0, inplace=True)#remove TOTAL from dataset
#print("zero salaries")
#print(tabulate(Enron2[(Enron2['salary']==0)][['salary','bonus','poi','other','total_payments']], headers='keys', tablefmt='fancy_grid',floatfmt=".0f"))



"""
DATASET CLEANING - Find outliers (2/2)

See new graph after remove of non significant datapoints
"""
sb.lmplot(x='salary', y= 'bonus', hue='poi', data=Enron2, palette='bright',size=8,markers=['P','o'])
pyplot.title('salary/bonus for POI and non-POI', fontsize=18)
pyplot.xlabel('salary', fontsize=16)
pyplot.ylabel('bonus', fontsize=16)
pyplot.show()

print(Enron3.shape)


"""
-----------------------------------------------------------------------------------------------------------------
ADDITIONAL FEATURES CREATION
"""
# function for any ratio
def ratiocomp(x, y):
    if x == 0 or y == 0:
        ratio = 0
    else :
        ratio= float(x) / float(y)
        
    return ratio

# add new features in Enron2 dataset ratio bonus/salary

for i in Enron3:
    
    row = Enron3[i]
    
    x= row['bonus']
    y = row['salary']
    bonus_salary_ratio= ratiocomp(x, y)
    Enron3.loc['ratio_bonus_salary',i] = bonus_salary_ratio
    

    rec = row['from_messages']
    poif = row['from_poi_to_this_person']
    receivedfromPOI_ratio= ratiocomp(poif, rec)
    Enron3.loc['part_from_POI',i] = receivedfromPOI_ratio
    
    sent = row['to_messages']
    pois = row['from_this_person_to_poi']
    receivedfromPOI_ratio= ratiocomp(pois, sent)
    Enron3.loc['part_to_POI',i] = receivedfromPOI_ratio
    
Enron2=Enron3.T

print(tabulate(Enron2.iloc[0:5,20:23], headers='keys', tablefmt='fancy_grid'))# check that loan_advances has disapperaed from features


"""
-----------------------------------------------------------------------------------------------------------------
FEATURE_FORMAT.py
Extract features and labels from dataset for local testing
"""
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!

features_list =['poi', 'bonus', 'deferral_payments', 'deferred_income', 'director_fees', 'exercised_stock_options', 'expenses', 'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 'long_term_incentive', 'other', 'restricted_stock', 'restricted_stock_deferred', 'salary', 
                'shared_receipt_with_poi', 'to_messages', 'total_payments', 'total_stock_value']

#remove outliers identified above :
data_dict.pop('TOTAL',0)
data_dict.pop('GRAMM WENDY L',0)
data_dict.pop('LOCKHART EUGENE E',0)
data_dict.pop('WHALEY DAVID A',0)
data_dict.pop('WROBEL BRUCE',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)

#remove loan advances identified as incomplete above:
for name in data_dict:
    data_dict[name].pop('loan_advances',0)

#recreate features created before...
def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    fraction = 0.
    if poi_messages != 'NaN' and all_messages != 'NaN':
        fraction = float(poi_messages)/all_messages


    return fraction

for name in data_dict:

    row = data_dict[name]

    from_poi_to_this_person = row["from_poi_to_this_person"]
    to_messages = row["to_messages"]
    fraction_from_poi = computeFraction(from_poi_to_this_person,to_messages )
    
    data_dict[name]["fraction_from_poi"] = fraction_from_poi
  
    from_this_person_to_poi = row["from_this_person_to_poi"]
    from_messages = row["from_messages"]
    fraction_to_poi = computeFraction(from_this_person_to_poi, from_messages)

    data_dict[name]["fraction_to_poi"] = fraction_to_poi
    
    bonus=row['bonus']
    salaire=row['salary']
    bonus_salary= computeFraction(bonus,salaire)
    data_dict[name]["bonus_vs_salary"] = bonus_salary
    
my_dataset = data_dict

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)




