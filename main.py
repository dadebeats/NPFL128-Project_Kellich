import argparse
import json
import os

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm

from text_data import describe_reddit_data, create_and_save_bert, create_and_save_textblob
from reddit_scraper import team_subreddits
from models import create_model2, create_model1
from common import load_gamestats

pd.set_option('display.max_columns', 5)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--dropout", type=float, default=0.2)

parser.add_argument("--lstm_timesteps", type=int, default=0,
                    help="Set zero to not use LSTM, else set size of sequence u want to train/predict from.")
parser.add_argument("--use_bert", type=bool, default=True,
                    help="Whether to use bert or not, if BERT is used we can't use LSTM")
parser.add_argument("--use_textblob", type=bool, default=False,
                    help="Whether to use textblob features or not.")


def prepare_data(df, concat_text=False):
    """
    Functions which cleans up data (fillna, include numerics only), scales them and splits to train/val/test sets.
    :param df:
    :param position:
    :param concat_text: Whether we want to use textblob features too
    :return:
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    filtered_df = df.drop(columns=["totalScore", "devFromL40"]).select_dtypes(include=numerics)
    df_median = filtered_df.median()
    data = filtered_df.fillna(df_median).fillna(0)

    if not concat_text:
        text_blob_feature_count = 10
        # delete last ten features as those are the ones from TextBlob
        data = data.iloc[:, :-text_blob_feature_count]

    infinities = [np.inf, -np.inf]
    average_score = 45  # players score roughly 45
    # Try to fill with median
    target = df["totalScore"].replace(infinities, df["totalScore"].median()).fillna(average_score)
    l40s = df["L40"].replace(infinities, df["L40"].median()).fillna(average_score)

    # Scale features
    col_trans = ColumnTransformer(
        [("scaler", RobustScaler(), [i for i in range(data.shape[1])])],
        remainder="passthrough"
    )
    data = pd.DataFrame(col_trans.fit_transform(data), columns=data.columns, index=data.index)

    # Split data
    x_train, x_temp, t_train, t_temp, _, l40s_temp = train_test_split(data, target, l40s, train_size=0.7, shuffle=False)
    x_val, x_test, t_val, t_test, _, l40s_test = train_test_split(x_temp, t_temp, l40s_temp, train_size=0.5,
                                                                  shuffle=False)

    return x_train, x_val, x_test, t_train, t_val, t_test, l40s_test


def load_feature_pool() -> pd.DataFrame:
    fp = {}
    for position in positions:
        df = pd.read_csv(f"dataset/2024-03-12_comprehensive/{position}.csv", index_col="gameId")
        df.columns = df.columns.str.replace('<', '_under_')
        df = df[~df.index.to_series().astype(str).str.contains('2020|2019')]
        fp[position] = df
    return fp


def create_sequences(df: pd.DataFrame, series_target: pd.Series, timestep: int):
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
    args = parser.parse_args()
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
    feature_pool = load_feature_pool()

    print("Feature pool snippet:")
    print(feature_pool["Forward"].head())
    # Data nascrapovaná z redditu:
    reddit_json = json.load(open('data/reddit.json'))
    print("Reddit data description:")
    describe_reddit_data(reddit_json)

    # Konfigurace programu
    use_text = args.use_textblob
    use_bert_model = args.use_bert
    lstm_timesteps = args.lstm_timesteps
    lstm_data_index = None # will be set later if needed
    if use_bert_model and lstm_timesteps:
        raise NotImplementedError("Can't use LSTM (model1) and BERT (model2) at the same time")
    print("LSTM:", lstm_timesteps, "Use textblob:", use_text, "Use BERT:", use_bert_model)

    # 1) Využití lexicon based algoritmu z TextBlobu - pro zrychlení zakomentované - v gitu jsou už potř. soubory
    # create_and_save_textblob(game_stats, reddit_json, positions)

    # 2) Využití BERT encoderu - pro zrychlení zakomentované - v gitu jsou už potř. soubory
    # create_and_save_bert(game_stats, reddit_json, positions)

    data_dim = 660
    if use_bert_model:
        data_dim += 1
    if use_text:
        data_dim += 10

    for position in positions:
        # Rozdělení dat do potřebných množin
        x_train, x_val, x_test, t_train, t_val, t_test, _ = prepare_data(feature_pool[position], concat_text=use_text)

        if lstm_timesteps:
            # Zakódujeme data do timeseries formátu pro LSTM
            # Timeseries datapointů vznikne méně než bylo původních dat, čím větší je lstm_timesteps, tím méně dat
            x_train, t_train, _ = create_sequences(x_train, t_train, lstm_timesteps)
            x_test, t_test, _ = create_sequences(x_test, t_test, lstm_timesteps)
            x_val, t_val, _ = create_sequences(x_val, t_val, lstm_timesteps)

        else:
            # Pokud využíváme jiný model než LSTM osekneme test. data
            # pro vzájemné porovnání je potřeba testovat na stejných datech
            lstm_steps_to_compare_with = 2
            _, _, lstm_data_index = create_sequences(x_test, t_test, lstm_steps_to_compare_with)
            x_test = x_test.loc[lstm_data_index]
            t_test = t_test.loc[lstm_data_index]

        # Rozdvojka na model1 a model2 ze složky "approach_sketches"
        if not use_bert_model:
            model1 = create_model1(args.lstm_timesteps, data_dim, args.hidden_dim, args.num_layers, args.dropout)
            model1.fit(x_train, t_train,
                       epochs=200,
                       batch_size=50,
                       validation_data=(x_val, t_val),
                       callbacks=[EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)])
            test_loss = model1.evaluate(x_test, t_test)
            print(f'Test loss {position}(RMSE): {np.sqrt(test_loss)}')
        else:
            model = create_model2(data_dim, args.hidden_dim, args.num_layers, args.dropout)

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

            # Omezení testovacích dat:
            if lstm_data_index is not None:
                bert_test = bert_test.loc[lstm_data_index]
                att_mask_test = att_mask_test.loc[lstm_data_index]

            test_inputs = [bert_test,
                           att_mask_test,
                           x_test]

            # Nešlo mi zapnout trénování pomocí GPU, stáhnul jsem si CUDA, env. proměnné nastavené, ale nefunuguje
            # Tady už je trénování tak pomalé, že by se to vyplatilo umět
            # EDIT: udělal jsem alternativní verzi v pytorchi, kde se mi povedlo zapnout GPU
            # EDIT: výsledkem byl pomalejší trénink než přes CPU

            model.fit(train_inputs, t_train,
                      epochs=200,
                      batch_size=20,
                      validation_data=(val_inputs, t_val),
                      callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

            model.save(f"model2_{position}.keras")
            test_loss = model.evaluate(test_inputs, t_test)
            print(f'Test loss {position}(RMSE): {np.sqrt(test_loss)}')

        print("Test data size:", len(x_test))
        average_train_target = np.mean(t_train)
        baseline_predictions = np.full(shape=t_test.shape, fill_value=average_train_target)
        baseline_rmse = np.sqrt(np.mean((t_test - baseline_predictions) ** 2))
        print(f'Baseline RMSE (predicting average of train targets): {baseline_rmse}')
        # predictions = model.predict(x_test)
