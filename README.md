# jean_bartik_computing_symposium_rankin
Introduction to Natural Language Processing 
A workshop at the Jean Bartik Computing Symposium 2020. 
February 27-28, 2020
United States Naval Academy

# Installation Requirements / Setup Instructions



1.	Install mini-conda.  Link:  https://docs.conda.io/en/latest/miniconda.html#linux-installers
        
        Should be Python 3.6 or higher

2.	Pull Github repo. Link: https://github.com/1fmusic/jean_bartik_computing_symposium_rankin

        a)	If you have a github account, fork & clone the repo
        b)	If you don’t have a github account, download as zip:
        
![example of how to clone or download a repository][image]
        
[image]:https://github.com/1fmusic/jean_bartik_computing_symposium_rankin/blob/master/Picture1.png
       
3.	Start Anaconda Prompt (miniconda3) 
        On Windows: Search for ‘anaconda’ in the ‘Search Windows’ bar  
        On Mac/Linux: Open Terminal
        
4.	Create an environment with .yml file
        Navigate inside the repo folder  
        `$ conda env create -f environment.yml`  
        `$ conda activate jbcs2020`  
        `$ jupyter notebook`  
        A browser will open showing the directory tree. Click on the notebook you want to open.  
        At end of session, `$ conda deactivate`  

# Order of the files: (keep reading for descriptions) 

1. [environment.yml][3]
2. [ted_clean_jbcs.ipynb][1] 
3. [topic_modeling_ted_jbcs.nb][10] 
4. [topic_modeling_tSNE_tutorial3.ipynb][9]



[1]: https://github.com/1fmusic/jean_bartik_computing_symposium_rankin/blob/master/ted_clean_jbcs.ipynb
[10]: https://github.com/1fmusic/jean_bartik_computing_symposium_rankin/blob/master/ted_modeling_ted_jbcs.ipynb

[3]: https://github.com/1fmusic/jean_bartik_computing_symposium_rankin/blob/master/environment.yml
[9]: https://github.com/1fmusic/jean_bartik_computing_symposium_rankin/blob/master/topic_modeling_tSNE_tutorial3.ipynb



This repo contains Ipython/Jupyter notebooks for basic exploration of transcripts of Ted Talks using Natural Language Processing (NLP), topic modeling, and I also link to the code for a recommender that lets you enter key words from the title of a talk and finds 5 talks that are similar.  
    The data consists of transcripts from Ted and TedX talks. Thanks to the lovely Rounak Banik and his web scraping I was able to dowload transcripts from 2467 Ted and TedX talks from 355 different Ted events. I downloaded this corpus from Kaggle, along with metadata about every talk. I encourage you to go to [kaggle][kaggle_link] and download it so that he can get credit for his scraping rather than put it in this repo.  
    
[kaggle_link]: https://www.kaggle.com/rounakbanik/ted-talks
    

The initial cleaning and exploration are done in 
   
[ted_clean_jbcs.ipynb][1] 
    
   Start by importing the csv files and looking at the raw data. Create a variable that holds only the transcripts called 'talks'. Below is a sample of the transcript from the most popular (highest views) Ted Talk. 'Do Schools Kill Creativity? by Sir Ken Robinson. 

    Good morning. How are you?(Laughter)It\\'s been great, hasn\\'t it? 

The first thing I saw when looking at these transcripts was that there were a lot of parentheticals for various non-speech sounds. For example, (Laughter) or (applause) or (Music).  There were even some cute little notes when the lyrics of a performance were transcribed
    
    someone like him ♫♫ He was tall and strong,
 
   I decided that I wanted to look at only the words that the speaker said, and remove these words in parentheses. Although, it would be interesting to collect these non-speech events and keep a count in the main matrix, especially for things like 'laughter' or applause or multimedia (present/not present) in making recommendations or calculating the popularity of a talk.
   
    Lucky for me, all of the parentheses contained these non-speech sounds and any of the speaker's words that required parenthesis were in brackets, so I just removed them with a simple regular expression. Thank you, Ted transcribers, for making my life a little easier!!!
    
    clean_parens = re.sub(r'\\([^)]*\\)', ' ', text)
    
# Cleaning Text with NLTK
Four important steps for cleaning the text and getting it into a format that we can analyze:
1)tokenize
2)lemmatize
3)remove stop words/punctuation
4)vectorize
    
[NLTK (Natural Language ToolKit)][2] is a python library for NLP. I found it very easy to use and highly effective.
    
[2]: http://www.nltk.org/
    
 * **tokenize**- This is the process of splitting up the document (talk) into words. There are a few tokenizers in NLTK, and one called **wordpunct** was my favorite because it separated the punctuation as well.
    ```
    from nltk.tokenize import wordpunct_tokenize
    doc_words2 = [wordpunct_tokenize(docs[fileid]) for fileid in fileids]
    print('\\n-----\\n'.join(wordpunct_tokenize(docs[1])))
 
    OUTPUT:
    Good
    morning
    .
    How
    are
    you
    ?
    ```
   
The notes were easy to remove by adding them to my stop words. Stopwords are the words that don't give us much information, (i.e., the, and, it, she, as) along with the punctuation. We want to remove these from our text, too. 
    
* We can do this by importing NLTKs list of **stopwords** and then adding to it. I added a lot of words and little things that weren't getting picked up, but this is a sample of my list. I went through many iterations of cleaning in order to figure out which words to add to my stopwords.

```
      from nltk.corpus import stopwords,
      stop = stopwords.words('english')
      stop += ['.',\" \\'\", 'ok','okay','yeah','ya','stuff','?']
```
**Lemmatization** - In this step, we get each word down to its root form. I chose the lemmatizer over the stemmer because it was more conservative and was able to change the ending to the appropriate one (i.e. children-->child, capacities-->capacity). This was at the expense of missing a few obvious ones (starting, unpredictability).

```
        from nltk.stem import WordNetLemmatizer
        lemmizer = WordNetLemmatizer()
        clean_words = []
      
        for word in docwords2:
       
            #remove stop words
            if word.lower() not in stop:
                low_word = lemmizer.lemmatize(word)
     
                #another shot at removing stopwords
                if low_word.lower() not in stop:
                    clean_words.append(low_word.lower())
```
   
 Now we have squeaky clean text! Here's the same excerpt that I showed you at the top of the README.
 
 ```
    good morning great blown away whole thing fact leaving three theme running conference relevant want talk one extraordinary evidence human creativity
```
    
As you can see it no longer makes a ton of sense, but it will still be very informative once we process these words over the whole corpus of talks.

* **Vectorization** is the important step of turning our words into numbers. The method that gave me the best results was count vectorizer. This function takes each word in each document and counts the number of times the word appears. You end up with each word as your columns and each row is a document (talk), so the data is the frequency of each word in each document. As you can imagine, there will be a large number of zeros in this matrix; we call this a sparse matrix. 
    
```
    c_vectorizer = CountVectorizer(ngram_range=(1,3), 
                                 stop_words='english',
                                 max_df = 0.6, 
                                 max_features=2000)
    
    # call `fit` to build the vocabulary
    c_vectorizer.fit(cleaned_talks)
    
    # finally, call `transform` to convert text to a bag of words
    c_x = c_vectorizer.transform(cleaned_talks)
```
    
# Now we are ready for topic modeling!
Open:

[topic_modeling_ted_jbcs.ipynb][10] 
    
    
First get the cleaned_talks from the previous step. Then import the models

```
    from sklearn.decomposition import LatentDirichletAllocation
```
    
We will try each of these models and tune the hyperparameters to see which one gives us the best topics (ones that make sense to you). It's an art.
    
This is the main format of calling the model, but I put it into a function along with the vectorizers so that I could easily manipulate the paremeters like 'number of topics, number of iterations (max_iter),n-gram size (ngram_min,ngram_max), number of features (max_df): 

```
    lda = LatentDirichletAllocation(n_components=topics,
                                        max_iter=iters,
                                        random_state=42,
                                        learning_method='online',
                                        n_jobs=-1)
       
    lda_dat = lda.fit_transform(vect_data)
```
The functions will print the topics and the most frequent 20 words in each topic.
    
The best parameter to tweak is the number of topics, higher is more narrow, but I decided to stay with a moderate number (20) because I didn't want the recommender to be too specific in the recommendations.

Once we get the topics that look good, we can do some clustering to improve it further. However, as you can see, these topics are already pretty good, so we will just assign the topic with the highest score to each document. 
```
    topic_ind = np.argmax(lda_data, axis=1)
```

Then we can use some visualization tools to 'see' what our clusters look like. The pyLDAviz is really fun and easy to export or use interactively outside of the notebook. 
