from textblob import TextBlob
import numpy as np
import pandas as pd


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
    eligible_dates = sorted([date for date in sentiments[modeled_club] if date < match_date], reverse=True)

    # Limit to the specified number of timesteps if provided
    if timesteps is not None:
        eligible_dates = eligible_dates[:timesteps]

    # Aggregate text data from eligible dates
    #aggregated_text = "\n____\n".join([sentiments[modeled_club][date] for date in eligible_dates])
    res = {"gameId": f"{row.slug}_{row.date}"}
    timeframes = [1, 2, 4, 8, 16]
    for timeframe in timeframes:
        dates_subset = eligible_dates[:timeframe]
        polarities = [sentiments[modeled_club][date][0] for date in dates_subset]
        subjectivities = [sentiments[modeled_club][date][1] for date in dates_subset]
        polarity = np.mean(polarities)
        if flip_sentiment:
            polarity *= -1
        res[f"polarity_{timeframe}"] = polarity
        res[f"subjectivity_{timeframe}"] = np.mean(subjectivities)

    return pd.Series(res)