# NPFL128-Project

# Exploring linguistics and deep learning techniques to enhance football predictions

## Before running the code:
Download this data and place it to the data/ folder https://www.uschovna.cz/zasilka/QE3V9A25E6VT3LPZ-I9S/.
(I already put the data/reddit.json to git)


## What I found:
Using text data helps, even though the quality of them wasn't the best and I didn't have much of it.

## What I did: 
0) Read papers (Literature overview.docx) and prepared my old project numerical data (dataset/*_comprehensive)
1) Scraped reddit data (reddit_scraper.py, data/reddit.json)
2) Focused more on hyperparameters of the MLP, where I found less is more. Data is too simple to use big topologies. 
3) Came up with two model approaches (approach_sketches, models.py) and tested them out (main.py)
4) Approach 1 is more modifiable (we can use/not use LSTM with various timestep size, use/not use TextBlob features of sentiment analysis)
5) Approach 2 has one fixed model. Use BERT to encode first 512 tokens of reddit data, concat this vector with numerical data encoded by MLP.
Use one more MLP layer to predict from this concatenated representation.
6) For both of them I chose a callback to restore best weights from validation set. 
On the Approach 2 model, because of the slow training, I set patience to 5 to end training when val loss does not improve.
7) Evaluated the results (Evaluation.xlsx)

## Possible problems:
Too little text data. Too little of preprocessing. Missed some better models/approaches to use. Better finetuning.
Too little context window.


