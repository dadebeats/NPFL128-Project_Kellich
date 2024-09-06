import argparse
import json
import os
from typing import Dict, Tuple, List, Union

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping

from text_data import describe_reddit_data, create_and_save_bert, create_and_save_textblob
from reddit_scraper import team_subreddits
from models import create_model2, create_model1
from common import load_gamestats, positions


def load_data() -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Loads and preprocesses game stats and feature pool data.

    :return: Tuple containing game stats and feature pool data.
    """
    game_stats = load_gamestats(is_clustering=False)

    # Filter game stats to relevant teams
    game_stats = game_stats[
        (game_stats.homeTeam.isin(list(team_subreddits.keys()))) |
        (game_stats.awayTeam.isin(list(team_subreddits.keys())))
    ]
    print("Original data/Game stats snippet:")
    print(game_stats.head())

    feature_pool = load_feature_pool()
    print("Feature pool snippet:")
    print(feature_pool["Forward"].head())

    # Load and describe Reddit data
    reddit_json = json.load(open('data/reddit.json'))
    print("Reddit data description:")
    describe_reddit_data(reddit_json)

    return game_stats, feature_pool


def configure_model_settings(args: argparse.Namespace) -> Tuple[int, bool, bool, Union[None, int]]:
    """
    Configures model settings based on input arguments.

    :param args: Command-line arguments.
    :return: Tuple containing data dimension, use_text, use_bert_model, and lstm_timesteps.
    """
    use_text = args.use_textblob
    use_bert_model = args.use_bert
    lstm_timesteps = args.lstm_timesteps

    if use_bert_model and lstm_timesteps:
        raise NotImplementedError("Can't use LSTM (model1) and BERT (model2) at the same time")

    data_dim = 660
    if use_bert_model:
        data_dim += 1
    if use_text:
        data_dim += 10

    print("LSTM:", lstm_timesteps, "Use textblob:", use_text, "Use BERT:", use_bert_model)
    return data_dim, use_text, use_bert_model, lstm_timesteps


def prepare_training_data(feature_pool: Dict[str, pd.DataFrame], position: str, use_text: bool,
                          lstm_timesteps: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, Union[None, List[str]]]:
    """
    Prepares the training, validation, and test datasets.

    :param feature_pool: Dictionary containing the feature pool data.
    :param position: Player position being processed.
    :param use_text: Flag indicating whether to use text features.
    :param lstm_timesteps: Number of timesteps for LSTM model.
    :return: Tuple of datasets required for training and testing.
    """
    x_train, x_val, x_test, t_train, t_val, t_test, _ = prepare_data(feature_pool[position], concat_text=use_text)

    lstm_data_index = None
    if lstm_timesteps:
        # Convert data into timeseries format for LSTM
        x_train, t_train, _ = create_sequences(x_train, t_train, lstm_timesteps)
        x_test, t_test, _ = create_sequences(x_test, t_test, lstm_timesteps)
        x_val, t_val, _ = create_sequences(x_val, t_val, lstm_timesteps)
    else:
        # Ensure testing on the same data for non-LSTM models
        lstm_steps_to_compare_with = 2
        _, _, lstm_data_index = create_sequences(x_test, t_test, lstm_steps_to_compare_with)
        x_test = x_test.loc[lstm_data_index]
        t_test = t_test.loc[lstm_data_index]

    return x_train, x_val, x_test, t_train, t_val, t_test, lstm_data_index

def prepare_data(df: pd.DataFrame, concat_text: bool = False) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
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


def load_feature_pool() -> Dict[str, pd.DataFrame]:
    """

    :return:
    """
    fp = {}
    for position in positions:
        df = pd.read_csv(f"dataset/2024-03-12_comprehensive/{position}.csv", index_col="gameId")
        df.columns = df.columns.str.replace('<', '_under_')
        df = df[~df.index.to_series().astype(str).str.contains('2020|2019')]
        fp[position] = df
    return fp


def create_sequences(df: pd.DataFrame, series_target: pd.Series, timestep: int) -> Tuple[
    np.ndarray, np.ndarray, List[str]]:
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


def train_model1(args: argparse.Namespace, data_dim: int, x_train: pd.DataFrame, x_val: pd.DataFrame,
                 x_test: pd.DataFrame, t_train: pd.Series, t_val: pd.Series, t_test: pd.Series) -> None:
    """
    Trains model1 using the specified parameters and datasets.

    :param args: Command-line arguments.
    :param data_dim: Input data dimension.
    :param x_train: Training data features.
    :param x_val: Validation data features.
    :param x_test: Test data features.
    :param t_train: Training data targets.
    :param t_val: Validation data targets.
    :param t_test: Test data targets.
    """
    model1 = create_model1(args.lstm_timesteps, data_dim, args.hidden_dim, args.num_layers, args.dropout)
    model1.fit(x_train, t_train,
               epochs=200,
               batch_size=50,
               validation_data=(x_val, t_val),
               callbacks=[EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)])
    test_loss = model1.evaluate(x_test, t_test)
    print(f'Test loss (RMSE): {np.sqrt(test_loss)}')


def train_model2(args: argparse.Namespace, data_dim: int, lstm_data_index: Union[None, List[str]],
                 x_train: pd.DataFrame, x_val: pd.DataFrame, x_test: pd.DataFrame,
                 t_train: pd.Series, t_val: pd.Series, t_test: pd.Series, position: str) -> None:
    """
    Trains model2 using BERT and numerical inputs.

    :param args: Command-line arguments.
    :param data_dim: Input data dimension.
    :param lstm_data_index: LSTM data index if applicable.
    :param x_train: Training data features.
    :param x_val: Validation data features.
    :param x_test: Test data features.
    :param t_train: Training data targets.
    :param t_val: Validation data targets.
    :param t_test: Test data targets.
    :param position: Player position being processed.
    """
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

    train_inputs = [bert_train, att_mask_train, x_train]
    val_inputs = [bert_val, att_mask_val, x_val]

    # Limit test data if needed
    if lstm_data_index is not None:
        bert_test = bert_test.loc[lstm_data_index]
        att_mask_test = att_mask_test.loc[lstm_data_index]

    test_inputs = [bert_test, att_mask_test, x_test]

    model.fit(train_inputs, t_train,
              epochs=200,
              batch_size=20,
              validation_data=(val_inputs, t_val),
              callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

    model.save(f"model2_{position}.keras")
    test_loss = model.evaluate(test_inputs, t_test)
    print(f'Test loss (RMSE): {np.sqrt(test_loss)}')


def evaluate_and_print_results(x_test: pd.DataFrame, t_test: pd.Series, t_train: pd.Series) -> None:
    """
    Evaluates the test data and prints results.

    :param x_test: Test data features.
    :param t_test: Test data targets.
    :param t_train: Training data targets.
    """
    print("Test data size:", len(x_test))
    average_train_target = np.mean(t_train)
    baseline_predictions = np.full(shape=t_test.shape, fill_value=average_train_target)
    baseline_rmse = np.sqrt(np.mean((t_test - baseline_predictions) ** 2))
    print(f'Baseline RMSE (predicting average of train targets): {baseline_rmse}')


def main(args: argparse.Namespace) -> None:
    """
    Main function to execute the entire workflow.

    :param args: Command-line arguments.
    """
    game_stats, feature_pool = load_data()
    positions = ["Defender", "Midfielder", "Forward"]
    data_dim, use_text, use_bert_model, lstm_timesteps = configure_model_settings(args)

    for position in positions:
        x_train, x_val, x_test, t_train, t_val, t_test, lstm_data_index = prepare_training_data(
            feature_pool, position, use_text, lstm_timesteps)

        if not use_bert_model:
            train_model1(args, data_dim, x_train, x_val, x_test, t_train, t_val, t_test)
        else:
            train_model2(args, data_dim, lstm_data_index, x_train, x_val, x_test, t_train, t_val, t_test, position)

        evaluate_and_print_results(x_test, t_test, t_train)

if __name__ == "__main__":
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

    args = parser.parse_args()
    main(args)
