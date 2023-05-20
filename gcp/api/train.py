########### IMPORT DEPENDENCIES ###########
import pandas as pd
import pickle
import sklearn
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline, make_pipeline
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score



########### GET DATA #######################
df = pd.read_csv('../data/train.csv')

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_df[features])
y = train_df["Survived"]


########### SPLIT DATA ######################
X_train, X_val, y_train, y_val = train_test_split(X,y, train_size=0.7, random_state=0)


########### MODEL SET-UP ####################
clf1 = RandomForestClassifier()
clf2 = ExtraTreesClassifier()

voting_clf = VotingClassifier(estimators=[('rfc', clf1), ('etc', clf2)])

pipeline = Pipeline([('voting_clf', voting_clf)])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)


########### PREDICTIONS ######################
predictions = pipeline.predict(X_val)
accuracy = accuracy_score(y_val, predictions)


########### SAVE MODEL #######################

filename = 'model.pkl'
pickle.dump(pipeline, open(filename, 'wb'))