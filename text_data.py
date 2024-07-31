import nltk
from textblob import TextBlob
import numpy as np
import pandas as pd
from collections import Counter
nltk.download('stopwords')
from tqdm.auto import tqdm
tqdm.pandas()
from models import BertEncoder


def row_to_bert_vector(row, encoded_texts):
    player_club = row['club']
    match_date = row['date']
    modeled_club = player_club
    flip_sentiment = False
    if player_club not in encoded_texts.keys():
        flip_sentiment = True
        if player_club == row.homeTeam:
            modeled_club = row.awayTeam
        else:
            modeled_club = row.homeTeam

    # Get all dates for the club that are before the match date
    eligible_match_dates = sorted([date for date in encoded_texts[modeled_club] if date < match_date], reverse=True)
    date_for_consideration = eligible_match_dates[0]

    input_ids = encoded_texts[modeled_club][date_for_consideration][0]

    res = {'gameId': f"{row.slug}_{row.date}", "flipSentiment": int(flip_sentiment)}
    for i in range(input_ids.shape[1]):
        res[f"input_ids_{i}"] = float(input_ids[0][i])

    return pd.Series(res)


def extract_sentiment_features(corpus):
    analysis = TextBlob(corpus)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    return polarity, subjectivity


def xs_ys_from_text(row, sentiments, timesteps=None):
    """
    Aggregates text data from Reddit for dates before the match date.

    :param row: A row from the DataFrame representing a game.
    :param reddit_data: A dictionary with Reddit textual data structured as
                        {team_slug: {datetime_object: "text_corpus"}}.
    :param timesteps: Optional. The number of time steps (dates) to consider before the match.
    :return: Aggregated textual data.
    """
    player_club = row['club']
    match_date = row['date']
    flip_sentiment = False
    modeled_club = player_club
    if player_club not in sentiments.keys():
        flip_sentiment = True
        if player_club == row.homeTeam:
            modeled_club = row.awayTeam
        else:
            modeled_club = row.homeTeam

    # Get all dates for the club that are before the match date
    eligible_match_dates = sorted([date for date in sentiments[modeled_club] if date < match_date], reverse=True)

    # Limit to the specified number of timesteps if provided
    if timesteps is not None:
        eligible_match_dates = eligible_match_dates[:timesteps]

    # Aggregate text data from eligible dates
    #aggregated_text = "\n____\n".join([sentiments[modeled_club][date] for date in eligible_dates])
    res = {"gameId": f"{row.slug}_{row.date}"}
    timeframes = [1, 2, 4, 8]
    for timeframe in timeframes:
        dates_subset = eligible_match_dates[:timeframe]
        polarities = [sentiments[modeled_club][date][0] for date in dates_subset]
        subjectivities = [sentiments[modeled_club][date][1] for date in dates_subset]
        polarity = np.mean(polarities)
        if flip_sentiment:
            polarity *= -1
        res[f"polarity_{timeframe}"] = polarity
        res[f"subjectivity_{timeframe}"] = np.mean(subjectivities)

    return pd.Series(res)


def get_common_words_set(word_count_lists):
    total_word_count_dict = {}
    for vocab in word_count_lists:
        for word, count in vocab.items():
            if word in total_word_count_dict:
                total_word_count_dict[word] += count
            else:
                total_word_count_dict[word] = count

    total_vocab = list(total_word_count_dict)


def describe_reddit_data(reddit_json):
    stats = {key: {} for key in reddit_json.keys()}
    print("Reddit data is composed of 100 top comments of each relevant thread and then top 10 replies to each.")
    print("This makes up to 1000 comments per match.")
    print("Some comments are more relevant to the prediction task than others, as shown below")
    print("We won't however adress this problem and just assume that the most popular comments have the useful info.")
    all_teams_text = ""
    for team, team_dict in reddit_json.items():
        stats[team]["number_of_matches"] = len(team_dict)
        messages_flattened = [msg for corpus in team_dict.values() for msg in corpus.split("____")]
        team_text = "".join(messages_flattened)
        all_teams_text += team_text
        stats[team]["avg_thread_len"] = len(team_text) / len(team_dict)
        stats[team]["comment_count"] = len(messages_flattened)
        if team == "arsenal-london":
            print("There is text with clear useful team-wide information: ", messages_flattened[3])
            print("There is text which has predictive value for individual players: ", messages_flattened[13])
            print("There is text which is straight up jokes and not very useful for predictions: ", messages_flattened[4])
            print("There is text which is not related to team performance: ", messages_flattened[33])
            print("There is text about too specific situations of the game: ",  messages_flattened[16])

        stats[team]["avg_comment_len"] = sum([len(msg) for msg in messages_flattened]) / len(messages_flattened)

        def sorted_vocabulary(string):
            # Split the string into words and convert to lowercase
            estop_palabras = set(nltk.corpus.stopwords.words('english'))
            estop_palabras.add("____")
            words = string.lower().split()
            words = [w for w in words if w not in estop_palabras]

            # Create a Counter object to count the occurrences of each word
            word_counts = Counter(words)

            # Convert the Counter object to a list of tuples (word, count)
            vocabulary = list(word_counts.items())

            # Sort the vocabulary list by the words (first element of each tuple)
            vocabulary.sort(key=lambda x: x[1], reverse=True)

            return vocabulary

        stats[team]["vocab"] = sorted_vocabulary(team_text)
        for k, stat in stats[team].items():
            if k != "vocab":
                print(f"Stat: {k} for {team}: ", stat)
            else:
                continue

    domain_common_words = [w for w,count in sorted_vocabulary(all_teams_text)[:180]] #180 is the id where I found first NE
    for team in stats:
        team_specific_vocab = []
        for w, count in stats[team]["vocab"]:
            if w not in domain_common_words:
                team_specific_vocab.append((w,count))
        stats[team]["team_specific_vocab"] = team_specific_vocab
        print(f"Team ({team}) specific vocabulary (mainly named entities): {team_specific_vocab[:10]}")

    return stats


def create_and_save_textblob(game_stats, reddit_json, positions):
    text_feature_pool = {}

    sentiments = {team: {date: extract_sentiment_features(text) for date, text in team_d.items()}
                  for team, team_d in reddit_json.items()}

    text_args = {'sentiments': sentiments, 'axis': 1}
    for position in positions:
        tqdm.pandas()
        text_df = game_stats[game_stats.position == position].progress_apply(xs_ys_from_text, **text_args)
        text_feature_pool[position] = text_df


def create_and_save_bert(game_stats, reddit_json, positions):
    bert_encoder = BertEncoder()

    def encode_text(text):
        inputs = bert_encoder.tokenizer(text, return_tensors='pt', max_length=512, truncation=True,
                                        padding='max_length')
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        return input_ids, attention_mask

    encoded_texts = {team: {match_date: encode_text(match_text)
                            for match_date, match_text in reddit_json[team].items()}
                     for team in reddit_json
                     }

    bert_args = {'encoded_texts': encoded_texts, 'axis': 1}
    bert_feature_pool = {}
    for position in positions:
        tqdm.pandas()
        bert_df = game_stats[game_stats.position == position].progress_apply(row_to_bert_vector, **bert_args)
        bert_df = bert_df.set_index("gameId")
        bert_feature_pool[position] = bert_df
        bert_df.to_csv(f"dataset/bert/{position}.csv")