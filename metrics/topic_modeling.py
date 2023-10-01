import numpy as np
import scipy
import gensim
import gensim.corpora as corpora
import sklearn
from sklearn.model_selection import KFold 
from scipy.spatial import distance
from math import log2

##############################################Reproduction of the method in the paper: how many topics, but I think their method is useless
#Pre-process text for lda
def createInput(text):
  id2word = corpora.Dictionary(text) # Create the Dictionary and Corpus needed for Topic Modeling
  texts = text # Create Corpus
  corpus = [id2word.doc2bow(text) for text in texts]   # Term Document Frequency
  return id2word, corpus

#Run lda model
def lda(text, k_topics):  
  id2word, corpus = createInput(text)
  lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=k_topics, #va être remplacé par k=number of topics
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=False)
  return lda_model

#Extract top n terms from each topic
def extract_terms(lda_model,nb_words):
  s = [[token for token, score in lda_model.show_topic(i, topn=nb_words)] for i in range(0, lda_model.num_topics)]
  return s

#Create the ranking set
def ranking_set(text, k_topics, t):
  lda_model = lda(text,k_topics)
  s = extract_terms(lda_model,t)
  return s

#Compute agreement_score
def agreement_score(k_topics,t,S0,S):
  matrix = []
  for i in range(k_topics): #Pour chaque topic i dans S0
    s0_list_scores = []
    for j in range(k_topics): #Pour chaque topic j dans S
      sum_jaccard_score = 0  
      for d in range(1,t+1):#Pour chaque profondeur t
        jaccard_score = len(set.intersection(set(S0[i][:d]), set(S[j][:d]))) / len(set(S0[i][:d]+S[j][:d]))
        sum_jaccard_score += jaccard_score
      average_jaccard_score = sum_jaccard_score * (1/t)
      s0_list_scores.append(average_jaccard_score)
    matrix.append(s0_list_scores)

  similarity_matrix = np.array(matrix)
  row_ind, col_ind = scipy.optimize.linear_sum_assignment(similarity_matrix)
  agreement_score = similarity_matrix[row_ind, col_ind].sum() / k_topics
  return agreement_score


def stability_topics(lyrics_column, k_min, k_max, t):
    stab_by_topics = {}
    #k_min - Minimum number of topics to evaluate
    #k_max - Maximum number of topics to evaluate
    #t - Chose t top terms
    for k_topics in range(k_min,k_max):
        #1.Create the reference ranking set S0
        S0 = ranking_set(lyrics_column, k_topics, t)
        #2. Cross validation to create S1, S2
        kf = KFold(10)
        stability_list = []
        for sample1_index, sample2_index in kf.split(lyrics_column):
            sample1, sample2 = lyrics_column.loc(axis=0)[sample1_index], lyrics_column.loc(axis=0)[sample2_index]
            #For sample 1
            S1 = ranking_set(sample1, k_topics, t)
            S2 = ranking_set(sample2, k_topics, t)
            #3-Calculate stability
            agreeS1 = agreement_score(k_topics,t,S0,S1)
            agreeS2 = agreement_score(k_topics,t,S0,S2) 
            stability = (agreeS1+agreeS2)/2
            stability_list.append(stability)
        stability_kfold = sum(stability_list)/10
        stab_by_topics[k_topics] = stability_kfold
      return stab_by_topics


if __name__ == "__main__":
  #Load dataset
  top5_df = pd.read_csv("/content/df.csv") #put path to csv file here
  
  #Run LDA
  #Find an optimal number of topics
  stab_by_topics = stability_topics(top5_df["word_list_lemmatized"], 4, 22, 20)
  print(stab_by_topic) #chose 4 topics

  #Run LDA
  lda_model = (top5_df["word_list_lemmatized"], 4)

  #GET DOCUMENTS TOPICS
  #https://stackoverflow.com/questions/66403628/how-to-change-topic-list-from-gensim-lda-get-document-topics-to-a-dataframe
  top5_df['topics'] = lda_model.get_document_topics(corpus, minimum_probability=0.0)
  sf = pd.DataFrame(data=top5_df['topics'])
  af = pd.DataFrame()
  for i in range(5):
      af[str(i)]=[]
  frames = [sf,af]
  af = pd.concat(frames).fillna(0)
  for i in range(1480):
      for j in range(len(top5_df['topics'][i])):
          af[str(top5_df['topics'][i][j][0])].loc[i] = top5_df['topics'][i][j][1]

  
  #Add them to the dataframe
  top5_df_topics = pd.merge(top5_df.reset_index(),af.reset_index(),how='inner', right_on='index', left_on='index')

  #Compute cosine distance between topics
  topic0 = lda_model.get_topics()[0]
  topic1 = lda_model.get_topics()[1]
  topic2 = lda_model.get_topics()[2]
  topic3 = lda_model.get_topics()[3]
  topic4 = lda_model.get_topics()[4]
  
  cos04 = distance.cosine(topic0,topic4)
  cos14 = distance.cosine(topic1,topic4)
  cos24 = distance.cosine(topic2,topic4)
  cos34 = distance.cosine(topic3,topic4)

  top5_df_topics['distTop'] = cos04*top5_df_topics['0'] + cos14*top5_df_topics['1']+cos24*top5_df_topics['2']+cos34*top5_df_topics['3']


  #Compute KL divergence between topics

  # calculate the KL divergence
  def kl_divergence(p, q):
  	return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))
  
  df_topics = top5_df_topics[['0_y', '1_y', '2_y', '3_y']]
  df_topics.loc['mean'] = df_topics.mean()

  list_div = []
  for index, row in df_topics.iterrows():
    div = kl_divergence(row, df_topics.loc['mean'])
    list_div.append(div)

  top5_df_topics['kl_div'] = list_div



