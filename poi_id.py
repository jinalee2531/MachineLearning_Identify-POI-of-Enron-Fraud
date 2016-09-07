
# coding: utf-8

#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
sys.path.append("../../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
### transfomr data_dict to pandas data frame    
df_org = pd.DataFrame(data_dict).T
print "Raw data has %d features and %d rows." %(df_org.shape[1],df_org.shape[0])

features_to_exclude = ['poi', 'email_address']

def update_features_list(df_org, features_to_exclude):
	"""
	Updates features_list excluding up-to-date features to exclude list.
	"""
    features_to_exclude= set(features_to_exclude)
    features_list_no_label = [col for col in df_org.columns if col not in features_to_exclude]

    # locate 'poi' to the first index of the features_list
    #features_list=set(['poi'])
    features_list = ['poi']
    features_list.extend(features_list_no_label)
    print "from ->", features_list_no_label
    print
    print "to ->", features_list
    print
    print len(df_org.columns), len(features_list_no_label), len(features_list)
    return features_list


# In[3]:

features_list=update_features_list(df_org, features_to_exclude)
print features_list


# ## Task 2: Remove outliers


df = df_org[features_list]

print "Data with features selected has %d features and %d rows." %(df.shape[1],df.shape[0])


import numpy as np

###### filling missing value with 0
def NaN_to_Zero(data):
    if data=="NaN": return 0
    else : return data

###### exploring missinsg values.

def processing_nan(df, col_name):
	"""
	Returns count of missing values and datapoints that substitutes missing value for 0 of designated feature.
	"""
    #col_filtered = df[df[col_name]!='NaN'][col_name]
    col_filtered = df[col_name].fillna(0)
    missing_cnt = float(sum([1 for val in df[col_name] if val=='NaN']))
    orginal_cnt = df.shape[0]
    
    return col_filtered, orginal_cnt, missing_cnt



from collections import defaultdict

###### do statstics of missing values of input dataset
def exploring_missing_values(df):
    print "**** Missing Values Exploration*****\n"
    missing_dict = defaultdict(dict)
    for col in df.columns:
        #if col not in features_to_exclude:
        df_poi, poi_cnt, poi_missing_cnt = processing_nan(df[df['poi']==True],col)
        df_non_poi, non_poi_cnt, non_poi_missing_cnt = processing_nan(df[df['poi']!=True],col)

        missing_dict[col]={"poi_missing":round(poi_missing_cnt,0), 
                           "poi_missing_prop": round(poi_missing_cnt/poi_cnt,2),
                           "non_poi_missing":round(non_poi_missing_cnt,0), 
                           "non_poi_missing_prop": round(non_poi_missing_cnt/non_poi_cnt,2)}
            
    print "Number of PoI : %d, Number of non-PoI: %d\n" %(poi_cnt, non_poi_cnt)
    print pd.DataFrame(missing_dict).T
    print
    

exploring_missing_values(df)


# In[9]:

"""
exploring_features(df[features_list])
"""

# In[10]:

### identify outlier values for each feature

col_names=["total_payments", "expenses", "from_messages",            "long_term_incentive", "restricted_stock","salary",           "to_messages","total_stock_value", "loan_advances"]

###### Methods to explore outliers
### print out index, 'poi' label, and value for max and min record of each feature.
def identifying_outliers(df, col_names):
    for col_name in col_names:
        idx_max = df[col_name].fillna(0).argmax()
        idx_min = df[col_name].fillna(0).argmin()
        print "*** %s\n max : %s, %s, %s" %(col_name, idx_max, df.loc[idx_max,'poi'], df.loc[idx_max, col_name])
        #print df.loc[idx_max]
        #print
        print " min : %s, %s, %s" %(idx_min, df.loc[idx_min,'poi'], df.loc[idx_min, col_name])
        #print
        #print df.loc[idx_min]

identifying_outliers(df, col_names)


# In[11]:

#remove outlier : 'TOTAL'
print df.shape
df=df.drop(["TOTAL"])
print df.shape

###### exploring outliers after removing "total"
identifying_outliers(df, col_names)


# In[12]:

## exclude features which its data is missing more than 70% in both 'poi' and 'non-poi'
features_to_exclude.extend(['deferral_payments', 'director_fees','loan_advances',                            'restricted_stock_deferred'])#, 'deferred_income'])

features_list=update_features_list(df, features_to_exclude)
print features_list

#df_selected_features=df[features_list]
df_selected=df[features_list]

#print df.shape,df_selected_features.shape,df_selected.shape


# In[13]:

### re-exploring after removing outlier
# exploring_features(df_selected)


# In[14]:

### processing missing rows in features regarding mail : fill with median values
if True:
    from collections import defaultdict

    to_exclude = features_to_exclude#+['deferred_income']

    features_mail = ['from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi',                     'shared_receipt_with_poi', 'to_messages']

    med_poi = defaultdict(float)
    med_non_poi = defaultdict(float)

    a= df[features_mail+['poi']]
    a[features_mail]= a[features_mail].astype(float)

    meds= a.groupby('poi').median()
    print meds.T

    for f in features_mail:
        med_poi[f] = meds.loc[True,f]
        med_non_poi[f] = meds.loc[False,f]

    ### spliting data frame by 'poi' values    
    pois = df_selected[df_selected['poi']==True]
    non_pois = df_selected[df_selected['poi']==False]
    
    ### replace NaN to median values of each 'poi' class
    for f in features_mail:
        #print
       #print pois[f]
        pois[f] = pois[f].apply(lambda x: med_poi[f] if x=="NaN" else x)
        #rint pois[f]
        non_pois[f] = non_pois[f].apply(lambda x: med_non_poi[f] if x=="NaN" else x)
        
    df_2 = pd.concat((pois,non_pois), axis=0)
    print df_2.shape
    df_selected = df_2
    
    
    #exploring_features(df_selected)
    exploring_missing_values(df_selected[features_list])
    #identifying_outliers(df_selected, df_selected.columns)


# # Task 3: Create new feature(s)

# In[15]:

### Store to my_dataset for easy export below.

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
my_dataset=None
def create_features_labels(df_selected, features_list):
    my_dataset = df_selected.T.to_dict()
    print features_list
    #print my_dataset[my_dataset.keys()[0]]


    ### Extract features and labels from dataset for local testing
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    return my_dataset, labels, features

my_dataset,labels, features=create_features_labels(df_selected,features_list)


# In[16]:

##### getting importance of features
def feature_importance(features,labels):
    features_low_importance=[]
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(class_weight='balanced')
    print clf
    clf.fit(features,labels)
    feature_importances = dict()
    print
    print "features: ", clf.max_features_
    print
    print "importance of features"
    for idx, val in sorted(enumerate(clf.feature_importances_), key=lambda variable: -variable[1]):
        print "%s : %.4f" %(features_list[idx+1], val)
        feature_importances[features_list[idx+1]] = val
        if val<.0001: 
            features_low_importance.append(features_list[idx+1])
    return feature_importances, features_low_importance

feature_importances, features_low_importance= feature_importance(features,labels)
print features_low_importance


# In[17]:

df_selected['to_poi_ratio'] = df_selected['from_this_person_to_poi']/df_selected['from_messages']
df_selected['from_poi_ratio'] = df_selected['from_poi_to_this_person']/df_selected['to_messages']

### update features_list
features_list=update_features_list(df_selected, features_to_exclude)

### create new dataset with updated features
my_dataset,labels, features=create_features_labels(df_selected,features_list)

### computing importance of features
feature_importances, features_low_importance= feature_importance(features,labels)

features_to_exclude.extend(features_low_importance)
features_list=update_features_list(df_selected, features_to_exclude)

### Extract features and labels from dataset for local testing
my_dataset,labels, features = create_features_labels(df_selected,features_list)


# In[18]:

### split train and test set

from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
## creating StratifiedShuffleSplit object to use in GridSearch
s= StratifiedShuffleSplit(labels, n_iter=3, test_size=.3, random_state=0)


# # Task 4: Try a varity of classifiers
# 
# > Please name your classifier clf for easy export below.
# > Note that if you want to do PCA or other multi-stage operations,
# >  you'll need to use Pipelines. For more info:
# >  http://scikit-learn.org/stable/modules/pipeline.html
# > Provided to give you a starting point. Try a variety of classifiers.

# In[31]:

from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import RandomizedPCA
from sklearn.metrics import precision_score, recall_score, make_scorer
from sklearn.metrics import classification_report as rp


precisions = []
recalls = []

### creating scoring function
def my_scorer_func (test_label, prediction):
    precision = precision_score(test_label, prediction, labels=None, pos_label=1,                                average='binary', sample_weight=None)
    recall = recall_score(test_label, prediction, labels=None, pos_label=1,                          average='binary', sample_weight=None)
    score = 1/(((1/precision)+(1/recall))/2)
    
    precisions.append(precision)
    recalls.append(recall)
    ### put weight if both precision and recall are over 0.3
    if precision >.3 and recall>.3:
        score *= 100
    #print "precision: %.4f, recall: %.4f" %(precision, recall)
   
    return score

my_scorer = make_scorer(my_scorer_func, greater_is_better= True)


# In[32]:

### build classification a model for each input parameter.
### and find the best model with the best score.

def run_training(algo, parameters, scorer=my_scorer):
    clf = GridSearchCV(algo, parameters, scoring =scorer, cv = s)                               
    clf.fit(features, labels)

    print
    print "Estimator: ", clf.best_estimator_
    #print
    #print clf.best_params_
    print
    #print "mean score: %.4f" %(clf.best_score_)
    
    for result in clf.grid_scores_ :
        m = result.mean_validation_score
        std = np.std(result.cv_validation_scores)
        
        if abs(clf.best_score_-m) < .00001:
            print ">> mean score : %.4f, std: %.4f" %(m,std)
   
    return clf.best_estimator_    


# In[33]:

##### Gaussain Naive Bayes

from sklearn.naive_bayes import GaussianNB
precisions = []
recalls = []

algo = GaussianNB()
parameters={}
run_training(algo, parameters, scorer=my_scorer)
print "precision: %.4f, recall: %.4f" %(np.mean(precisions), np.mean(recalls))


##### classifer built on PCA ####
precisions = []
recalls = []
estimators = [('reduce_dim', RandomizedPCA()), ('algo', algo)]
parameters = dict(reduce_dim__n_components=[2, 5, 10]
                 , reduce_dim__whiten=[True,False])

clf = Pipeline(estimators)
best_clf = run_training(clf, parameters, scorer=my_scorer)

print "precision: %.4f, recall: %.4f" %(np.mean(precisions), np.mean(recalls))


# In[34]:

##### Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier
algo = DecisionTreeClassifier()

parameters ={'class_weight': ('balanced',None), 
             'max_features' : range(2,len(features_list)),
             'min_samples_split' : range(2,11)
             }
precisions = []
recalls = []

best_clf = run_training(algo, parameters, scorer=my_scorer)

print "precision: %.4f, recall: %.4f" %(np.mean(precisions), np.mean(recalls))


# In[35]:
"""
###### Visualize decision tree

from IPython.display import Image
from sklearn.externals.six import StringIO  
from sklearn import tree
import pydot
import os


with open("enron.dot", 'w') as f:
    f = tree.export_graphviz(best_clf, out_file=f)
os.unlink('enron.dot')

dot_data = StringIO()

tree.export_graphviz(best_clf, out_file=dot_data,  
                         feature_names=features_list[1:],
                         class_names = ['non-poi','poi'],
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())  
"""

# In[36]:

##### classifer built on PCA ####
parameters = dict(reduce_dim__n_components=[2, 5, 10]
                  , reduce_dim__whiten=(True,False)
                  , algo__class_weight=('balanced',None)
                  , algo__max_features = (None, 'auto','log2')
                  , algo__min_samples_split=range(2,11)
                 )


precisions = []
recalls = []

estimators = [('reduce_dim', RandomizedPCA()), ('algo', algo)]
clf = Pipeline(estimators)
best_clf = run_training(clf, parameters, scorer=my_scorer)


print "precision: %.4f, recall: %.4f" %(np.mean(precisions), np.mean(recalls))


# In[37]:

##### Random Forest

from sklearn.ensemble import RandomForestClassifier

algo =  RandomForestClassifier()
parameters =dict( class_weight = ('balanced',None)
                 , max_features = (None, 'auto','log2')
                 , min_samples_split = range(2,11)
             )

precisions = []
recalls = []

best_clf = run_training(algo, parameters, scorer=my_scorer)

print "precision: %.4f, recall: %.4f" %(np.mean(precisions), np.mean(recalls))


# In[38]:

##### classifer built on PCA ####
parameters = dict(reduce_dim__n_components=[2, 5, 10]
                  , reduce_dim__whiten=(True,False)
                  , algo__class_weight=('balanced',None)
                  , algo__max_features = (None, 'auto','log2')
                  , algo__min_samples_split=range(2,11)
                  #, algo__n_estimators = (10,20)
                  #, algo__n_jobs = [5]
                 )
precisions = []
recalls = []
estimators = [('reduce_dim', RandomizedPCA()), ('algo', algo)]
clf = Pipeline(estimators)
run_training(clf, parameters, scorer=my_scorer)
print "precision: %.4f, recall: %.4f" %(np.mean(precisions), np.mean(recalls))


# In[39]:

##### ADABOOST

from sklearn.ensemble import AdaBoostClassifier

algo =  AdaBoostClassifier()
parameters =dict(n_estimators= range(50, 201, 50)             
                 , learning_rate= [.1, .4, .7, 1.0]
                 #,'algorithm' : ('SAMME', 'SAMME.R') 
                )
precisions = []
recalls = []
run_training(algo, parameters, scorer=my_scorer)
print "precision: %.4f, recall: %.4f" %(np.mean(precisions), np.mean(recalls))

##### classifer built on PCA ####
parameters = dict(reduce_dim__n_components=[2, 5, 10]
                  , reduce_dim__whiten=(True,False)
                  , algo__n_estimators= range(50, 201, 50)
                  , algo__learning_rate = [.1, .4, .7, 1.0]
                 )
precisions = []
recalls = []
estimators = [('reduce_dim', RandomizedPCA()), ('algo', algo)]
clf = Pipeline(estimators)
run_training(clf, parameters, scorer=my_scorer)

print "precision: %.4f, recall: %.4f" %(np.mean(precisions), np.mean(recalls))


# # Task 5: Tune your classifier to achieve better than .3 precision and recall 
# > using our testing script. Check the tester.py script in the final project
# > folder for details on the evaluation method, especially the test_classifier
# > function. Because of the small size of the dataset, the script uses
# > stratified shuffle split cross validation. For more info: 
# > http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# 

# # Task 6: Dump your classifier, dataset, and features_list
# 
# > so anyone can check your results. You do not need to change anything below, but make sure  that the version of poi_id.py that you submit can be run on its own and generates the necessary .pkl files for validating your results.
# 

# In[40]:

print best_clf

    
dump_classifier_and_data(best_clf, my_dataset, features_list)

