import numpy as np
import pandas as pd
import random
import math
from collections import Counter

def modelling(train, validation):
	
	# No resampling
	train_input = train.drop(['Headline', 'Body ID', 'Stance', 'Headline words', 'articleBody', 'articleBody words'], axis=1)
	train_input_standardised = (train_input - train_input.min())/(train_input.max() - train_input.min())
	train_output_agree = (train['Stance']=='agree')*1
	train_output_disagree = (train['Stance']=='disagree')*1
	train_output_discuss = (train['Stance']=='discuss')*1
	train_output_unrelated = (train['Stance']=='unrelated')*1

	# Resample data for linear & logistic regression due to highly unbalanced distribution of stances
	train_unrelated_resampled = train[train['Stance']=='unrelated'].sample(frac=0.0229, random_state=12345)
	train_agree_resampled = train[train['Stance']=='agree'].sample(frac=0.2285, random_state=12345)
	train_disagree_resampled = train[train['Stance']=='disagree']
	train_discuss_resampled = train[train['Stance']=='discuss'].sample(frac=0.0917, replace=True, random_state=12345)

	train_stances_resampled = pd.concat([train_unrelated_resampled, train_agree_resampled, train_disagree_resampled, train_discuss_resampled]).reset_index(drop=True)

	train_input_resampled = train_stances_resampled.drop(['Headline', 'Body ID', 'Stance', 'Headline words', 'articleBody', 'articleBody words'], axis=1)
	train_input_standardised_resampled = (train_input_resampled - train_input_resampled.min())/(train_input_resampled.max() - train_input_resampled.min())
	train_output_agree_resampled = (train_stances_resampled['Stance']=='agree')*1
	train_output_disagree_resampled = (train_stances_resampled['Stance']=='disagree')*1
	train_output_discuss_resampled = (train_stances_resampled['Stance']=='discuss')*1
	train_output_unrelated_resampled = (train_stances_resampled['Stance']=='unrelated')*1

	validation_input = validation.drop(['Headline', 'Body ID', 'Stance', 'Headline words', 'articleBody', 'articleBody words'], axis=1)
	validation_input_standardised = (validation_input - validation_input.min())/(validation_input.max() - validation_input.min())
	validation_output_agree = (validation['Stance']=='agree')*1
	validation_output_disagree = (validation['Stance']=='disagree')*1
	validation_output_discuss = (validation['Stance']=='discuss')*1
	validation_output_unrelated = (validation['Stance']=='unrelated')*1


	##### Linear Regression #####
	print('##### Linear Regression #####')

	def GradientDescent(input_df, output_df, alpha=0.03):
		import time
		start = time.time()
		x = input_df.copy()
		y = output_df.copy()
		x.insert(0, 'intercept', 1)
		y = np.array(y)
		features = input_df.columns.tolist()
		theta = [np.ones(len(features)+1)*0.5] # Initiate theta_0 to theta_n (+1 for the intercept)
		m = len(x)
		
		while True:
			hypothesis = np.dot(x, theta[-1])
			loss = hypothesis - y
			gradient = np.dot(x.transpose().reset_index(drop=True), loss)/m
			theta_iter = np.array(theta[-1] - alpha*gradient)
			
			if max(np.absolute(np.array(theta_iter) - theta[-1])) < 0.005:
				break
			else:
				theta.append(theta_iter)
		return theta
		
	agree_theta_trained_linreg = GradientDescent(train_input_standardised_resampled, train_output_agree_resampled)
	disagree_theta_trained_linreg = GradientDescent(train_input_standardised_resampled, train_output_disagree_resampled)
	discuss_theta_trained_linreg = GradientDescent(train_input_standardised_resampled, train_output_discuss_resampled)
	unrelated_theta_trained_linreg = GradientDescent(train_input_standardised_resampled, train_output_unrelated_resampled)

	validation_agree_results_linreg = np.insert(validation_input_standardised.values, 0, 1, axis=1).dot(np.array(agree_theta_trained_linreg[-1]))
	validation_disagree_results_linreg = np.insert(validation_input_standardised.values, 0, 1, axis=1).dot(np.array(disagree_theta_trained_linreg[-1]))
	validation_discuss_results_linreg = np.insert(validation_input_standardised.values, 0, 1, axis=1).dot(np.array(discuss_theta_trained_linreg[-1]))
	validation_unrelated_results_linreg = np.insert(validation_input_standardised.values, 0, 1, axis=1).dot(np.array(unrelated_theta_trained_linreg[-1]))

	validation_model_results_linreg = pd.DataFrame({'agree': validation_agree_results_linreg, 'disagree': validation_disagree_results_linreg, 'discuss': validation_discuss_results_linreg, 'unrelated': validation_unrelated_results_linreg})

	from sklearn.metrics import confusion_matrix
	cnf_matrix_linreg = confusion_matrix(validation['Stance'], validation_model_results_linreg.idxmax(axis=1), labels=['agree', 'disagree', 'discuss', 'unrelated'])
	print('Confusion matrix: \n' + str(cnf_matrix_linreg))

	from sklearn.metrics import precision_recall_fscore_support
	print('Precision recall F1: \n' + str(precision_recall_fscore_support(validation['Stance'], validation_model_results_linreg.idxmax(axis=1))))


	##### Logistic Regression #####
	print('##### Logistic Regression #####')

	def LogisticRegression(input_df, output_df, alpha=0.03):
		import time
		start = time.time()
		x = input_df.copy()
		y = output_df.copy()
		x.insert(0, 'intercept', 1)
		y = np.array(y)
		features = input_df.columns.tolist()
		theta = [np.ones(len(features)+1)*0.5] # Initiate theta_0 to theta_n (+1 for the intercept)
		m = len(x)    
		
		while True:
			hypothesis = 1/(1+1/np.exp(np.dot(x, theta[-1])))
			loss = hypothesis - y
			gradient = np.dot(x.transpose().reset_index(drop=True), loss)/m
			theta_iter = np.array(theta[-1] - alpha*gradient)
			
			if max(np.absolute(np.array(theta_iter) - theta[-1])) < 0.005:
				break
			else:
				theta.append(theta_iter)
		return theta

	agree_theta_trained_logreg = LogisticRegression(train_input_standardised_resampled, train_output_agree_resampled)
	disagree_theta_trained_logreg = LogisticRegression(train_input_standardised_resampled, train_output_disagree_resampled)
	discuss_theta_trained_logreg = LogisticRegression(train_input_standardised_resampled, train_output_discuss_resampled)
	unrelated_theta_trained_logreg = LogisticRegression(train_input_standardised_resampled, train_output_unrelated_resampled)

	validation_agree_results_logreg = 1/(1+1/np.exp((np.insert(validation_input_standardised.values, 0, 1, axis=1).dot(np.array(agree_theta_trained_logreg[-1])))))
	validation_disagree_results_logreg = 1/(1+1/np.exp((np.insert(validation_input_standardised.values, 0, 1, axis=1).dot(np.array(disagree_theta_trained_logreg[-1])))))
	validation_discuss_results_logreg = 1/(1+1/np.exp((np.insert(validation_input_standardised.values, 0, 1, axis=1).dot(np.array(discuss_theta_trained_logreg[-1])))))
	validation_unrelated_results_logreg = 1/(1+1/np.exp((np.insert(validation_input_standardised.values, 0, 1, axis=1).dot(np.array(unrelated_theta_trained_logreg[-1])))))

	validation_model_results_logreg = pd.DataFrame({'agree': validation_agree_results_logreg, 'disagree': validation_disagree_results_logreg, 'discuss': validation_discuss_results_logreg, 'unrelated': validation_unrelated_results_logreg})

	cnf_matrix_logreg = confusion_matrix(validation['Stance'], validation_model_results_logreg.idxmax(axis=1), labels=['agree', 'disagree', 'discuss', 'unrelated'])
	print('Confusion matrix: \n' + str(cnf_matrix_logreg))

	print('Precision recall F1 score: \n' + str(precision_recall_fscore_support(validation['Stance'], validation_model_results_logreg.idxmax(axis=1), labels=['agree', 'disagree', 'discuss', 'unrelated'])))


	##### Random Forest #####
	print('##### Random Forest #####')

	from sklearn.ensemble import RandomForestClassifier
	rfc = RandomForestClassifier()
	rfc.fit(train_input_standardised, train['Stance'])

	rfc_results = rfc.predict(validation_input_standardised)

	cnf_matrix_rfc = confusion_matrix(validation['Stance'], rfc_results, labels=['agree', 'disagree', 'discuss', 'unrelated'])
	print('Confusion matrix: \n' + str(cnf_matrix_logreg))
	
	print('Precision recall F1 score: \n' + str(precision_recall_fscore_support(validation['Stance'], rfc_results, labels=['agree', 'disagree', 'discuss', 'unrelated'])))

	print('Feature Importance: \n' + str(pd.DataFrame({'Features': train_input_standardised.columns.tolist(), 'Importance': rfc.feature_importances_}).sort_values('Importance', ascending=False)))

