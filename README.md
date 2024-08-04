# NPFL128-Project

# Exploring linguistics and deep learning techniques to enhance football predictions
## What I found:
Using text data helps, even though the quality of them wasn't the best and I didn't have much of it.

## What I did: 
0) Read papers (Literature overview.docx) and prepared my old project numerical data (dataset/*_comprehensive)
1) Scraped reddit data (reddit_scraper.py, data/reddit.json)
2) Focused more on hyperparameters of the MLP, where I found less is more. Data is too simple to use big topologies. 
3) Came up with two model approaches (approach_sketches, models.py) and tested them out (main.py)
3.1) Approach 1 is more modifiable (we can use/not use LSTM with various timestep size, use/not use TextBlob features)
3.2) Approach 2 has one fixed model
3.3) For both of them I chose a callback to restore best weights from validation set. 
On the Approach 2 model, because of the slow training, I set patience to 5 to end training when val loss does not improve.
4) Evaluated the results (Evaluation.xlsx)