#The Effect of the Typicality of Song Lyrics on Song Popularity



This repository contains the code needed to reproduce the experiments of the paper "The Effect of the Typicality of Song Lyrics on Song Popularity:
A Natural Language Processing Analysis of the British Top Singles Chart". In this paper, we look at the relationship between song lyrics' typicality and their popularity. To do so, we define different metrics to measure the typicality of the lyrics of a song and then perform a GLMM analysis to assess the relationship between these metrics and popularity. 

#Dataset

We collect all the songs that reached one of the top five spots in the UK Official Singles Chart Top 100 between January 1999 and December 2013. The file "songs_used.csv" contains the list of songs used in our analysis without lyrics (for copyright reasons). 

#Metrics
The folder metrics contain the code needed to reproduce our typicality metrics:
<ul>
    <li>lexical_repetition.py: contains the code needed to recreate the variables for lexical repetition (h-point, repetition of the chorus) and complexity (variety and complexity)</li>
    <li>sentiment_analysis.py: contains the code to train 2 DistilBert models for predicting valence and arousal in our datasets. The training data used is the MoodyLyrics dataset (Çano and Morisio, 2017)</li>
    <li>topic_modeling.py: contains the code needed for our topic modeling and for identifying the optimal number of topics</li>
</ul>

The file glmm_analysis.py contains the glmm models to fit to analyze the relationship between metrics' typicality and lyrics. 
And the archive.zip folder contains the 2 distilbert models used to annotate our own dataset for valance and arousal










