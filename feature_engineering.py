import numpy as np
import pandas as pd
import random
import math
from collections import Counter

def feature_engineering(train_bodies, train_stances, train_bodies_vector, train_stances_vector, validation_bodies, validation, validation_bodies_vector, validation_vector):
	
	print('###### Training dataset feature engineering #####')
	# Function to convert a document into lower case and remove all symbols
	def clean_doc(string):
		cleaned_string = string.lower().replace(',', ' ').replace('.', ' ').replace(';', ' ').replace(':', ' ').replace('(', '').replace(')', '').replace('[','').replace(']','').replace('\'','').replace('\"','').replace('‘','').replace('’','').replace('“','').replace('”','').replace('/','').replace('?','').replace('!','').replace('%','').replace('&','').replace('-','').replace('$','').replace('—','')
		return cleaned_string
	
	##### Cosine Similarity #####
	def cosine_similarity(x,y):
		dot_prod = sum(i[0] * i[1] for i in zip(x,y))
		len_x = np.sqrt(sum(i[0] * i[1] for i in zip(x,x)))
		len_y = np.sqrt(sum(i[0] * i[1] for i in zip(y,y)))
		sim = dot_prod/(len_x * len_y)
		return sim
	
	# Calculate cosine similarity for each stance vs. its corresponding article
	train_cosine_sim = []

	import time
	start = time.time()
	for i in range(len(train_stances)):
		body_id = train_stances.loc[i, 'Body ID']
		index = train_bodies[train_bodies['Body ID']==body_id].index[0]
		train_cosine_sim.append(cosine_similarity(train_bodies_vector[index], train_stances_vector[i]))
	end = time.time()
	print('Time elapsed for cosine similarity calculation: ' + str(end-start))
	
	train_stances['Cosine similarity'] = train_cosine_sim
	train_stances = train_stances.merge(train_bodies, on='Body ID', how='left')
	
	
	##### KL-Divergence without smoothing #####
	def KLDivergence(df):
		import time
		start = time.time()
		
		kl_div = []
		for i in range(len(df)):
			cross_entropy = 0
			cleaned_headline = clean_doc(df.loc[i, 'Headline']).split()
			cleaned_body = clean_doc(df.loc[i, 'articleBody']).split()
			for word in df.loc[i, 'Headline words']: # Loop through each word in the headline
				p_headline = cleaned_headline.count(word)/len(cleaned_headline) # Calculate probability of word occurence in the headline
				p_body = cleaned_body.count(word)/len(cleaned_body) # Calculate probability of word occurence in the article
				if p_body == 0: # If p_body is zero then move onto next word, since log(0) will return error
					val = 0
				else:
					val = p_headline * math.log(p_body)
				cross_entropy = cross_entropy - val
			kl_div.append(cross_entropy)
		
		end = time.time()
		print('Time elapsed for KL Divergence: ' + str(end - start))
		
		return kl_div
	
	train_stances['KL-divergence with no smoothing'] = KLDivergence(train_stances)
	
	
	##### Dirichlet Smoothing #####
	# Aggregate a list of all word occurences from article bodies
	article_body_words_all = []

	for i in range(len(train_stances)):
		cleaned_body = clean_doc(train_stances.loc[i, 'articleBody']).split()
		article_body_words_all.extend(cleaned_body)
	total_words = len(article_body_words_all)

	# Estimate mu: # of word occurences in collection divided by number of documents
	mu = int(len(article_body_words_all)/len(train_stances))
	
	# Calculate KL-divergence for Dirichlet smoothing
	c = Counter(article_body_words_all) # Initiate a counter to count number of word occurences for each word

	def KLDivergenceDirichlet(df):
		import time
		start = time.time()
		kl_div_dirichlet = []
		
		for i in range(len(df)):
			cross_entropy = 0
			for word in df.loc[i, 'Headline words']: # Loop through each word in the headline
				cleaned_headline = clean_doc(df.loc[i, 'Headline']).split()
				cleaned_body = clean_doc(df.loc[i, 'articleBody']).split()
				lambda_dirichlet = len(cleaned_body)/(len(cleaned_body) + mu) # Calculate the lambda for Dirichlet smoothing
				p_headline = cleaned_headline.count(word)/len(cleaned_headline) # Calculate probability of word occurence in the headline
				p_body = cleaned_body.count(word)/len(cleaned_body) * lambda_dirichlet # Calculate SCALED probability of word occurence in the article
				p_collection = c[word]/total_words * (1 - lambda_dirichlet) # Calculate SCALED probability of word occurence in the collection
				p_d = p_body + p_collection
				if p_d == 0: # If zero then move onto next word, since log(0) will return error
					val = 0
				else: val = p_headline * math.log(p_d)
				cross_entropy = cross_entropy - val
			kl_div_dirichlet.append(cross_entropy)
    
		end = time.time()
		print('Time elapsed for KL Divergence with Dirichlet Smoothing: ' + str(end - start))
		
		return kl_div_dirichlet
	
	train_stances['KL-divergence with Dirichlet smoothing'] = KLDivergenceDirichlet(train_stances)
	
	
	##### Jaccard Similarity #####
	def Jaccard_sim(headline_words, body_words):
		intersection = len(headline_words & body_words)
		union = len(headline_words | body_words)
		return (intersection/union)
	
	# Calculate Jaccard similarity (1 - Jaccard distance) between all new stances and their corresponding headlines
	jaccard_sim = []
	for i in range(len(train_stances)):
		jaccard_sim.append(Jaccard_sim(set(train_stances.loc[i, 'Headline words']), set(train_stances.loc[i, 'articleBody words'])))
		
	train_stances['Jaccard similarity'] = jaccard_sim
	
	
	##### Discussion words #####
	# Define list of words that would help to indicate the 'discussion' stance in documents, as a result of the ranking
	discussion_words = ['according', 'apparently', 'appears', 'believed', 'investigating', 'investigation', 'ongoing', 'reported', 'reportedly', 'claim', 'claims', 'claimed', 'implies', 'imply', 'told', 'allegedly', 'said', 'statement', 'stated', 'states', 'spokesman', 'unknown']
	
	# Calculate the 'discuss' stance word occurences (as a proportion of body length) for all documents
	for word in discussion_words:
		discussion_word_count = train_stances['articleBody'].str.count(word)
		body_length = train_stances['articleBody'].apply(lambda x: len(clean_doc(x).split()))
		train_stances['discuss: ' + str(word)] = discussion_word_count/body_length
		
	
	##### Disagree words #####
	
	# Define list of words that would help to indicate the 'disagree' stance in headlines, as a result of the ranking
	disagree_words = ['may', 'can', 'could', 'doubt', 'doubts', 'reported', 'no', 'experts', 'still', 'not']
	
	# Calculate the 'disagree' stance word occurences (as a proportion of headline length) for all headlines
	for word in disagree_words:
		disagree_word_count = train_stances['Headline'].str.count(word)
		headline_length = train_stances['Headline'].apply(lambda x: len(clean_doc(x).split()))
		train_stances['disagree: ' + str(word)] = disagree_word_count/headline_length
	
	
	
	############ Validation ############
	print('###### Validation dataset feature engineering #####')
	
	val_cosine_sim = []
	for i in range(len(validation)):
		body_id = validation.loc[i, 'Body ID']
		index = validation_bodies[validation_bodies['Body ID']==body_id].index[0]
		val_cosine_sim.append(cosine_similarity(validation_bodies_vector[index], validation_vector[i]))

	validation['Cosine similarity'] = val_cosine_sim
	validation = validation.merge(validation_bodies, on='Body ID', how='left')

	# Calculate KL-divergence for no smoothing
	validation['KL-divergence with no smoothing'] = KLDivergence(validation)

	# Calculate KL-divergence for Dirichlet smoothing
	validation['KL-divergence with Dirichlet smoothing'] = KLDivergenceDirichlet(validation)

	# Calculate Jaccard similarity (1 - Jaccard distance) between all new stances and their corresponding headlines
	jaccard_sim = []
	for i in range(len(validation)):
		jaccard_sim.append(Jaccard_sim(set(validation.loc[i, 'Headline words']), set(validation.loc[i, 'articleBody words'])))
	validation['Jaccard similarity'] = jaccard_sim

	# Calculate the 'discuss' stance word occurences (as a proportion of body length) for all documents
	for word in discussion_words:
		discussion_word_count = validation['articleBody'].str.count(word)
		body_length = validation['articleBody'].apply(lambda x: len(clean_doc(x).split()))
		validation['discuss: ' + str(word)] = discussion_word_count/body_length

	# Calculate the 'disagree' stance word occurences (as a proportion of headline length) for all headlines
	for word in disagree_words:
		disagree_word_count = validation['Headline'].str.count(word)
		headline_length = validation['Headline'].apply(lambda x: len(clean_doc(x).split()))
		validation['disagree: ' + str(word)] = disagree_word_count/headline_length
	
	
	return train_stances, validation