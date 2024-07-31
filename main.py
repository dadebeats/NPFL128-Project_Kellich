from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Input, Dropout
import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from common import load_gamestats
import json
from text_data import extract_sentiment_features, xs_ys_from_text, describe_reddit_data, row_to_bert_vector
from tqdm.auto import tqdm
from reddit_scraper import team_subreddits
from models import BertEncoder, create_model2, BertLayer
import tensorflow as tf
import torch

pd.set_option('display.max_columns', 5)


def prepare_data(df, position, concat_text=False, use_bert=False):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    data = df.drop(columns=["totalScore", "devFromL40"]).select_dtypes(include=numerics).fillna(
        df.drop(columns=["totalScore", "devFromL40"]).select_dtypes(include=numerics).median()).fillna(0)
    data = data.iloc[:, :-12]
    target = df["totalScore"].replace([np.inf, -np.inf], df["totalScore"].median()).fillna(45)
    l40s = df["L40"].replace([np.inf, -np.inf], df["L40"].median()).fillna(45)

    col_trans = ColumnTransformer(
        [("scaler", RobustScaler(), [i for i in range(data.shape[1])])],
        remainder="passthrough"
    )

    data = pd.DataFrame(col_trans.fit_transform(data), columns=data.columns, index=data.index)

    if concat_text:
        text_df = pd.read_csv(f"dataset/textual/{position}.csv", index_col="gameId")
        data = text_df.merge(data, left_index=True, right_index=True, how='inner')
        target = df["totalScore"]
        l40s = l40s[:len(data)]

    if use_bert:
        bert_df = pd.read_csv(f"dataset/bert/{position}.csv", index_col="gameId")
        data = bert_df.merge(data, left_index=True, right_index=True, how='inner')
        target = df["totalScore"]
        l40s = l40s[:len(data)]

    x_train, x_temp, t_train, t_temp, _, l40s_temp = train_test_split(data, target, l40s, train_size=0.7, shuffle=False)
    x_val, x_test, t_val, t_test, _, l40s_test = train_test_split(x_temp, t_temp, l40s_temp, train_size=0.5,
                                                                  shuffle=False)

    return x_train, x_val, x_test, t_train, t_val, t_test, l40s_test


def create_sequences(df: pd.DataFrame, series_target: pd.Series, timestep: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences of a specified length for LSTM model training, ensuring each player has enough data points.

    :param df: DataFrame containing features for each player and game.
    :param series_target: Series containing the target variable for each player and game.
    :param timestep: Length of the sequence to be created for each sample.
    :return: A tuple of numpy arrays (sequences, targets) ready for LSTM training.
    """
    sequences = []
    targets = []
    target_ids = []
    df_data = df.copy()
    # Create new columns for Player and Date by splitting the index
    player_names = df_data.index.map(lambda x: x.split('_')[0])
    game_dates = df_data.index.map(lambda x: x.split('_')[1])

    # Assign these columns to the DataFrame
    df_data['Player'] = player_names
    df_data['Date'] = pd.to_datetime(game_dates)
    series_target.index = df_data.index  # Ensure target series has the same index for alignment

    # Group by player
    grouped = df_data.groupby('Player')

    for player, group in grouped:
        # Ensure the group is sorted by date
        group = group.sort_values(by='Date')

        # Check if the group has enough data points
        if len(group) > timestep:
            # Iterate over the group to create sequences
            for i in range(len(group) - timestep):
                seq = group.iloc[i:i + timestep].drop(columns=['Player', 'Date']).values  # Extract sequence of features
                target_index = group.index[
                    i + timestep]  # Get the index of the target corresponding to the end of the sequence
                target = series_target.loc[target_index]  # Get the target value

                target_ids.append(target_index)
                sequences.append(seq)
                targets.append(target)

    # Convert lists to numpy arrays
    sequences = np.array(sequences)
    targets = np.array(targets)

    return sequences, targets, target_ids


if __name__ == "__main__":
    # Originální data "game_stats", ze kterých je napočítán dataset v proměnné "feature_pool"
    # Řádek je jeden zápas z pohledu jednoho hráče a obsahuje statistiky hráče v zápase a cílovou veličinu
    # Game_stats budeme potřebovat minimálně jako index pro vytváření features z textu
    game_stats = load_gamestats(is_clustering=False)
    positions = ["Defender", "Midfielder", "Forward"]

    # Have to limit ourselves to the teams that we scraped text data for
    game_stats = game_stats[
        (game_stats.homeTeam.isin(list(team_subreddits.keys()))) |
        (game_stats.awayTeam.isin(list(team_subreddits.keys())))
        ]
    print("Original data/Game stats snippet:")
    print(game_stats.head())

    # Dataset, který jsem používal v bakalářské práci
    # Řádek je jeden zápas z pohledu jednoho hráče a obsahuje *agregované statistiky před začátkem zápasu a cíl. vel.
    # *Např. L40_mean je průměr skóre hráče za posledních 40 zápasů
    feature_pool = {}
    for position in positions:
        df = pd.read_csv(f"dataset/2024-03-12_comprehensive/{position}.csv", index_col="gameId")
        df.columns = df.columns.str.replace('<', '_under_')
        # needs to replace < because it interferes with some models.
        df = df[~df.index.to_series().astype(str).str.contains('2020|2019')]
        feature_pool[position] = df
    print("Feature pool snippet:")
    print(feature_pool["Forward"].head())
    # Data nascrapovaná z redditu:
    reddit_json = json.load(open('data/reddit.json'))
    print("Reddit data description:")
    # describe_reddit_data(reddit_json)

    # Pokud nastavíme nenulové, data se níže v training loopu zakódují jako timeseries a využije se LSTM, před standardním MLP
    lstm_timesteps = 0
    # Switch, jestli chceme využívat textová data, nebo ne
    use_text_data = False
    create_bert_features = False
    use_bert_model = True
    print(lstm_timesteps, use_text_data, create_bert_features)

    # 1) Využití lexicon based algoritmu z TextBlobu
    if use_text_data:
        text_feature_pool = {}

        sentiments = {team: {date: extract_sentiment_features(text) for date, text in team_d.items()}
                      for team, team_d in reddit_json.items()}

        text_args = {'sentiments': sentiments, 'axis': 1}
        for position in positions:
            tqdm.pandas()
            text_df = game_stats[game_stats.position == position].progress_apply(xs_ys_from_text, **text_args)
            text_feature_pool[position] = text_df
    # 2) Využití BERT encoderu
    if create_bert_features:
        # Prepare bert vectors for each team/match_date
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

    # Instanciace zatím prázdného modelu
    model = Sequential()
    data_dim = 658
    if use_bert_model:
        data_dim += 1
    if use_text_data:
        data_dim += 10

    if lstm_timesteps:
        model.add(Input(shape=(lstm_timesteps, data_dim)))
        model.add(LSTM(400, dropout=0.5, unroll=True))
    else:
        model.add(Input(shape=[data_dim]))

    # Přidání Dense vrstev Multi Layer Perceptronu
    for i in range(4):
        model.add(Dense(100, activation="relu"))
        model.add(Dropout(0.5))
    model.add(Dense(1, activation=None))

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mean_squared_error'])

    for position in positions:
        # Rozdělení dat do potřebných množin
        x_train, x_val, x_test, t_train, t_val, t_test, _ = prepare_data(feature_pool[position],
                                                                         position,
                                                                         concat_text=False,
                                                                         use_bert=False)

        if lstm_timesteps:
            # Zakódujeme data do timeseries formátu pro LSTM
            # Timeseries datapointů vznikne méně než bylo původních dat, čím větší je lstm_timesteps, tím méně datapointů
            x_train, t_train, _ = create_sequences(x_train, t_train, lstm_timesteps)
            x_test, t_test, _ = create_sequences(x_test, t_test, lstm_timesteps)
            x_val, t_val, _ = create_sequences(x_val, t_val, lstm_timesteps)

        cut_data_to_lstm_size = False
        if cut_data_to_lstm_size:
            _, _, lstm_data_index = create_sequences(x_test, t_test, 2)
            x_test, t_test = x_test.loc[lstm_data_index].sort_index().astype('float32'), t_test.loc[
                lstm_data_index].sort_index().astype('float32')

        if use_bert_model:
            hidden_dim = 128
            output_dim = 1  # Assuming binary classification or regression
            model = create_model2(data_dim, hidden_dim, output_dim)
            bert_df = pd.read_csv(f"dataset/bert/{position}.csv", index_col="gameId")
            flip_sentiment_col = bert_df["flipSentiment"]
            bert_df = bert_df.drop(columns=["flipSentiment"])
            bert_train, bert_temp = train_test_split(bert_df, train_size=0.7, shuffle=False)
            bert_val, bert_test = train_test_split(bert_temp, train_size=0.5, shuffle=False)

            att_mask_train = pd.DataFrame(np.ones(bert_train.shape), index=bert_train.index, columns=bert_train.columns)
            att_mask_val = pd.DataFrame(np.ones(bert_val.shape), index=bert_val.index, columns=bert_val.columns)
            att_mask_test = pd.DataFrame(np.ones(bert_test.shape), index=bert_test.index, columns=bert_test.columns)

            x_train["flipSentiment"] = flip_sentiment_col
            x_val["flipSentiment"] = flip_sentiment_col
            x_test["flipSentiment"] = flip_sentiment_col

            train_inputs = [bert_train,
                            att_mask_train,
                            x_train]
            val_inputs = [bert_val,
                          att_mask_val,
                          x_val]
            test_inputs = [bert_test,
                           att_mask_test,
                           x_test]
            # Nešlo mi zapnout trénování pomocí GPU, stáhnul jsem si CUDA, env. proměnné nastavené, ale nefunuguje
            # Tady už je trénování tak pomalé, že by se to vyplatilo umět
            # TODO: make sure GPU is used for training
            model.fit(train_inputs, t_train, epochs=20,
                      batch_size=50,
                      validation_data=(val_inputs, t_val))

            model.save(f"model2_{position}.keras")
            test_loss = model.evaluate(test_inputs, t_test)
            print(f'Test loss {position}(RMSE): {np.sqrt(test_loss)}')
        else:
            # Nafitujeme model, přičemž také sledujeme validační chybu
            model.fit(x_train, t_train, epochs=50, batch_size=8, validation_data=(x_val, t_val))
            test_loss = model.evaluate(x_test, t_test)
            print("Test data size:", len(x_test))
            print(f'Test loss {position}(RMSE): {np.sqrt(test_loss)}')

        average_train_target = np.mean(t_train)
        baseline_predictions = np.full(shape=t_test.shape, fill_value=average_train_target)
        baseline_rmse = np.sqrt(np.mean((t_test - baseline_predictions) ** 2))
        print(f'Baseline RMSE (predicting average of train targets): {baseline_rmse}')
        # predictions = model.predict(x_test)
