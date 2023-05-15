import pandas as pd 
import numpy as np
import os 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings 
from sklearn.preprocessing import LabelEncoder,StandardScaler  
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,GridSearchCV 
from sklearn.metrics import classification_report,confusion_matrix
from imblearn.over_sampling import SMOTE
warnings.filterwarnings("ignore")
path="E:/datasets"
dataset_path=os.path.join(path,"bank_marketing.csv") 
class Classifier:
	def __init__(self):
		pass 	
	def data_and_info(self):
		self.data=pd.read_csv(dataset_path)	
		nullvalues=self.data.isnull().sum()
		data_info=self.data.info()
		print(self.data.describe())
		self.data.drop("Unnamed: 0",axis=1,inplace=True)
		self.data.dropna(inplace=True)
		return nullvalues,data_info 
	def heatmap(self):
		self.corr=self.data.corr()
		sns.heatmap(self.corr,annot=True)
		plt.title("Co relation heatmap")
		plt.show()
	def data_and_preprocessing(self): 
		self.features=list(self.data.columns)
		self.features.remove("loan")
		self.x_data=self.data[self.features].values
		self.y_data=self.data['loan'].values
		x=self.data["loan"].value_counts()
		print(x)
		lab=LabelEncoder()
		self.y_data=lab.fit_transform(self.y_data)
		categorical_data=[]
		for feature in self.features:
			if type(self.data[feature].values[0])==str:
				categorical_data.append(feature)
		numerical_data=list(set(self.features)-set(categorical_data))
		x_cat_data=self.data[categorical_data].values
		for i in range(len(x_cat_data[0])):
			x_cat_data[:,i]=lab.fit_transform(x_cat_data[:,i])
		cat_df=pd.DataFrame(x_cat_data,columns=categorical_data)
		num_data=self.data[numerical_data]
		for feature in cat_df.columns:
		     num_data[feature]=cat_df[feature]
		x_data=num_data 
		stand=StandardScaler()
		self.x_data=stand.fit_transform(x_data.values)
		return self.x_data,self.y_data
	def up_sampling(self):
		sm=SMOTE(random_state=101)
		self.x_newdata,self.y_newdata=sm.fit_resample(self.x_data, self.y_data)
		print("Up smamping values of the data ")
		print(self.x_newdata.shape,self.y_newdata.shape)	
	def spliting_data(self):
		self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.x_newdata,self.y_newdata,random_state=101,train_size=0.78)
		return self.x_train,self.x_test,self.y_test,self.y_train	
	def logistic_regression(self):
		logreg=LogisticRegression(max_iter=len(self.x_train))
		logreg.fit(self.x_train, self.y_train)
		print("The classification report of the classifier \n")
		self.y_pred=logreg.predict(self.x_test)
		cls_report=classification_report(self.y_test,self.y_pred)
		print(cls_report)
		sns.heatmap(confusion_matrix(self.y_test,self.y_pred),annot=True)
		plt.title("LogisticRegression")
		plt.show()
	def tree_classifier(self):
		tree=DecisionTreeClassifier(max_depth=10)
		tree.fit(self.x_train, self.y_train)
		score=tree.score(self.x_test,self.y_test)
		y_pred=tree.predict(self.x_test)
		cls_report=classification_report(self.y_test, y_pred)
		print(cls_report)
		sns.heatmap(confusion_matrix(self.y_test,y_pred),annot=True)
		plt.show()
	def forest(self):
		forest=RandomForestClassifier(max_depth=10)
		forest.fit(self.x_train,self.y_train)
		y_pred=forest.predict(self.x_test)
		cls_report=classification_report(self.y_test,y_pred)
		sns.heatmap(confusion_matrix(self.y_test,y_pred),annot=True)
		print(cls_report)
		plt.show()

	def svm(self):
		svm=SVC()
		C=[0.001,0.01,0.1,1,10,100]
		gamma=[0.001,0.01,0.1,1,10,100]
		parm_grid={"C":C,"gamma":gamma}
		grid_search=GridSearchCV(svm,param_grid=parm_grid,cv=5)
		grid_search.fit(self.x_train,self.y_train)
		print("The best score:{:.2f}".format(
			grid_search.score(self.x_test, self.y_test)))
		print((grid_search.best_params_)) 
	def svm_grid(self):
		svm=SVC(C=10,gamma=1)
		svm.fit(self.x_train,self.y_train)
		y_pred=svm.predict(self.x_test)
		cls_report=classification_report(self.y_test,y_pred)
		print(cls_report)
		sns.heatmap(confusion_matrix(self.y_test,y_pred),annot=True)
		plt.show()
		
clas=Classifier()
clas.data_and_info() 
clas.data_and_preprocessing()
clas.up_sampling()	
clas.spliting_data()
clas.svm_grid()