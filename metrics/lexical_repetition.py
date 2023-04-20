from collections import Counter


##############################################Compute variety = number of unique words in a song divided by the total number of words
#Count the number of words in a song and the number of unique words in a song
def count_unique_words(column):
  nb_unique_list = []
  nb_words_list = []
  for i, string in column.iteritems():
    words_list = string.str.lower().str.split()
    unique = []
    nb_words = 0
    for word in words_list:
      nb_words+=1
      if word not in unique:
        unique.append(word)
    nb_unique_list.append(len(unique))
    nb_words_list.append(nb_words)

  return nb_words_list, nb_unique_list

def variety(nb_words_col, nb_unique_col):
  variety_col = nb_unique_col / nb_words_col * 100
  return variety_col



##############################################Compute complexity = average number of characters per words

#Count the average number of characters per word in a song
def avgNbChar(column):
  avg_char_list = []
  for i, string in column.iteritems():
    words_list = string.str.split()
    nbChar = 0
    i =0
    for word in words_list:
      nbChar += len(word)
      i+=1
    avg_char_list.append(nbChar/i)
  return avg_char_list



##############################################Compute repetition of the chorus
#Count the number of occurences of a word in a song. Return a list of values
def countOccurence(column, word):
  nbWord_list = []
  for i, string in column.iteritems():
    nbWord = string.count(word)
    nbWord_list.append(nbWord)

  return nbWord_list



##############################################Compute the H-Point of a song
#Count the number of time each word of a song is occuring in the song and return an ordered list for each song. 
def freqWords(column):
  count_list = []
  for index, string in column.iteritems():
    counter = Counter(string.split())
    count = counter.most_common()
    count_list.append(count)

  return count_list

#Compute the H-Point of a song
def getHPoint(column):
  hpoint_list = []
  for index, count_list in column.iteritems():
    hpoint = 0
    i = 0
    #If r=f(r)
    for elem in count_list:
      i += 1
      diff = elem[1] - i
      if diff == 0:
        hpoint = i
        break

    #If r=f(r) does not exist
    if hpoint == 0:
      j = 0
      last_value = 0
      for value in count_list:
        j += 1
        if j > 1:
          last_value = count_list[j-2][1]

        if (value[1] < j) & (last_value > (j-1)):
          hpoint = ((last_value*j) - (value[1]*(j-1))) / (j - (j-1) + last_value - value[1])
          break
  
    hpoint_list.append(hpoint)
  return hpoint_list