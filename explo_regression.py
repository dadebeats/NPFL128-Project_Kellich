import os
import copy
import json
from datetime import date
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from tqdm.auto import tqdm
from textblob import TextBlob
from transformers import DistilBertTokenizer, DistilBertModel
#import torch

#from utils import pick_best_name_candidate, get_season
from config import load_understat, numerics, load_gamestats, positions, load_league_stats, load_sofifa, load_simple
#from visualizations import plot_corr_data, plot_corr_matrix, plot_pca


def get_pca_top_components(former_df, trained_pca, top_n):
    """
    Funkce, která vrací jména n nejdůležitějších komponent po PCA zobrazení
    :param former_df: původní df, před PCA transformací
    :param trained_pca: nafitovaný pca model
    :param top_n: kolik komponent chceme vrátit
    :return: seznam názvů nejdůležitějších sloupců
    """
    top_components = []
    for i in range(top_n):
        max_index = np.argmax(np.abs(trained_pca.components_[i]))
        comp_name = former_df.columns[max_index]
        top_components.append(comp_name)
    return top_components


def pca_2D_visualization(df, y_kmeans, centers, n_clusters, n, plot=False, cut_outliers=False):
    pca = PCA(n_components=n)
    pca.fit(df)

    top_components = get_pca_top_components(df, pca, 2)

    X_pca = pca.transform(df)
    centers_pca = pca.transform(centers)
    if plot:
        plot_pca(X_pca, y_kmeans, centers_pca, n_clusters, top_components[0], top_components[1], cut_outliers)
    return X_pca


def get_teammates_at_gw(teams_games, gw):
    df = teams_games.sort_values(by=["gameWeek"], ascending=False)
    # gets teammates from the last 5 league matches
    avg_games_per_gw = 10
    last_gws_to_include = 10
    return df.head(avg_games_per_gw * last_gws_to_include).slug.unique()


def split_date(date):
    date_list = date.split("-")
    return date_list[2], date_list[1], date_list[0]


def get_source_data(row, source_df, teams, source_type):
    """
    Funkce, kterou voláme z FIFA a Understat zpracování dat. Tato funkce není zoptimalizovaná (caching) a zpomaluje zpracování pro fifu a understat.
    :param row:
    :param source_df:
    :param teams:
    :param source_type: string, buď f nebo u, podle FIFA nebo understat
    :return: seznam 3 statistik, kde každý prvek je dataframe pro 1) hráče 2) spoluhráče 3) protihráče
    """
    _, match_month, match_year = split_date(row.date)
    season = get_season(match_year, match_month, source_type)
    player_club = row.club

    if row["homeTeam"] == player_club:
        is_home = 1
    else:
        is_home = 0

    enemy_team = row["awayTeam"] if is_home else row["homeTeam"]

    teammates_games, enemies_games = teams.get_group(player_club), teams.get_group(enemy_team)
    teammates_slugs = get_teammates_at_gw(teammates_games, row.gameWeek)
    enemies_slugs = get_teammates_at_gw(enemies_games, row.gameWeek)

    def get_stats_for_players(player_slugs, player_games, entity, season):
        names = []
        for slug in player_slugs:
            pgames = player_games[player_games.slug == slug]
            if not pgames.empty:
                names.append(pgames.iloc[0][f"{source_type}Name"])

        players_dfs = [
            source_df[source_df["player_name" if source_type == "understat" else "long_name"] == name].select_dtypes(
                include='number')
            for name in names]
        try:
            stats = pd.concat(players_dfs, axis=0)
        except ValueError:
            return pd.DataFrame()

        if source_type == "fifa":
            stats = stats[stats.year <= season]
        else:
            stats = stats[stats.season <= season]

        stats.columns = [f"{source_type[0]}_{entity}_{cn}" for cn in stats.columns]
        return stats

    list_of_stats = [
        get_stats_for_players([row.slug], teammates_games, "player", season),
        get_stats_for_players(teammates_slugs, teammates_games, "teammates", season),
        get_stats_for_players(enemies_slugs, enemies_games, "enemies", season)
    ]

    return list_of_stats


class PerformanceExplorer:
    """
    Hlavní třída pro regresní úlohu.
    Vytvořený objekt se inicializuje pomocí game_stats, kde se nachází cílová veličina
    Následně chceme využít buď metod get_feature_pool, nebo load_feature_pool
    Tyto metody zajistí na objektu .feature_pool (1300 features), se kterým dál můžeme pracovat:
    Počítat selekční metody jako korelace, feature importance z RF - výsledky se uloží do atributů objektu
    Na základě těchto výsledků můžeme feature_pool filtrovat a vracet si jeho podmnožinu
    (.filter_correlations, .filter_rf_importances)

    """
    def __init__(self, g_stats):
        self.game_stats = g_stats
        # in feature pool, create all possible features with get_feature_pool, filter those later.
        self.feature_pool = {}
        self.feature_importances = {}
        self.correlations = {}


    #region Všechny metody, které užíváme privátně.
    # Zpracovávají datové zdroje a vytváří příznaky do feature_pool
    # Všechny metody zobrazují jeden game_stats řádek na nový řádek do feature_pool
    # Používáme je s df.apply, takže prvním argumentem je vždy row - řádek
    @staticmethod
    def _get_common_features(row, teams, players, n):
        """
        Metoda, která zpracovává data z game_stats
        :param row:  jeden řádek
        :param teams: df.GroupBy s týmy
        :param players: df.GroupBy s hráči
        :param n:
        :return: Získané informace k jednomu řádku z game_stats
        """
        player_club = row["club"]
        match_date = row["date"]
        player_slug = row.slug
        # binární indikátor is_home
        if row["homeTeam"] == player_club:
            is_home = 1
        else:
            is_home = 0

        enemy_team = row["awayTeam"] if is_home else row["homeTeam"]
        enemy_team_games = teams.get_group(enemy_team)

        def get_relevant_group_of_players(t_games, date, gw):
            teammates_slugs = get_teammates_at_gw(t_games, gw)
            teams_games = t_games[(t_games["date"] < date)]
            teams_games = teams_games[teams_games.slug.isin(teammates_slugs)]
            return teams_games.groupby("slug", dropna=True)

        teams_games = teams.get_group(player_club)
        teams_games = teams_games[teams_games.slug != player_slug]
        gs_teammates_games = get_relevant_group_of_players(teams_games, match_date, row.gameWeek)
        gs_enemy_players_games = get_relevant_group_of_players(enemy_team_games, match_date, row.gameWeek)

        player_games = players.get_group(player_slug).query("date < @match_date").head(n)
        if len(player_games):
            LN = player_games.totalScore.mean()
        else:
            # player has no previous matches
            LN = 44  # TODO: replace with median L40

        # vracíme si všechny statistiky z game_stats pro hráče, spoluhráče, protihráče
        return {
            f"L{n}": LN,
            "is_home": is_home,
            "gs_players_games": player_games,
            "gs_teammates_games": gs_teammates_games,
            "gs_enemy_players_games": gs_enemy_players_games,
        }

    @staticmethod
    def _get_common_league_stats(league_stats, comp, player_club, match_date, timeframes, enemy_club=None):
        ls = league_stats[comp]
        ls = ls[ls["Date"] <= match_date]
        ls_teams_games = ls[(ls["HomeTeam"] == player_club) | (ls["AwayTeam"] == player_club)]

        if not enemy_club:
            try:
                match_in_ls = ls_teams_games[ls_teams_games.Date == match_date]
                enemy_club = match_in_ls["AwayTeam"].item() if match_in_ls["HomeTeam"].item() == player_club \
                    else match_in_ls["HomeTeam"].item()
            except ValueError:
                print(f"Cant find enemy team in league stats for this team: {player_club}")
                return np.nan

        ls_enemy_team_games = ls[(ls["HomeTeam"] == enemy_club) | (ls["AwayTeam"] == enemy_club)]

        def get_team_stats(team_name, games, timeframes):
            numerics = games.select_dtypes(include=['number']).columns
            home_games = games[games["HomeTeam"] == team_name][numerics]
            away_games = games[games["AwayTeam"] == team_name][numerics]

            # keys: what stats I wanna include
            # values: how to find those
            column_map = {
                'goalsScored': ('FTHG', 'FTAG'),
                'goalsConceded': ('FTAG', 'FTHG'),
                'avgWinOdds': ('B365H', 'B365A'),
                'avgDrawOdds': ('B365D', 'B365D'),
                'maxWinOdds': ('MaxH', 'MaxA'),
                'maxDrawOdds': ('MaxD', 'MaxD'),
                'shotsOnTarget': ('HST', 'AST'),
                'shotsOnAgainst': ('AST', 'HST'),
                'shots': ('HS', 'AS'),
                'shotsAgainst': ('AS', 'HS'),
                'avgUnder2.5': ('Avg<2.5', 'Avg<2.5'),
                'avgOver2.5': ('Avg>2.5', 'Avg>2.5'),
                'maxUnder2.5': ('Max<2.5', 'Max<2.5'),
                'maxOver2.5': ('Max>2.5', 'Max>2.5'),
                'avgCAH': ('AvgCAHH', 'AvgCAHA'),
                'avgCAHAgainst': ('AvgCAHA', 'AvgCAHH'),
                'maxCAH': ('MaxCAHH', 'MaxCAHA'),
                'maxCAHAgainst': ('MaxCAHA', 'MaxCAHH')
            }

            stats = {}
            for timeframe in timeframes:
                home_stats = home_games.head(timeframe).mean().to_dict()
                away_stats = away_games.head(timeframe).mean().to_dict()

                for stat_key, (home_col, away_col) in column_map.items():
                    stats[f"LS_L{timeframe}_{stat_key}_all_mean"] = (home_stats[home_col] + away_stats[away_col]) / 2
                    stats[f"LS_L{timeframe}_{stat_key}_away_mean"] = away_stats[away_col]
                    stats[f"LS_L{timeframe}_{stat_key}_home_mean"] = home_stats[home_col]

            return stats

        pc_stats = get_team_stats(player_club, ls_teams_games, timeframes)
        ec_stats = {"enemy_" + k: v
                    for k, v in
                    get_team_stats(enemy_club, ls_enemy_team_games, timeframes).items()}

        return pc_stats | ec_stats

    @staticmethod
    def _get_common_understat(row, understats, teams, dict_template) -> dict:
        """
        Nejsložitější zpracování datového zdroje, data na vstupu v různých formátech
        :param row: řádek z game_stats
        :param understats: Tuple s daty z Understat
        :param teams:
        :param dict_template: Vzor pro slovník, který vyplňujeme ve funkci
        :return: Vracíme si slovník všech relevantních dat z understat
        """
        comp = row.competition
        player_club = row.club
        match_date = row.date
        _, match_month, match_year = split_date(match_date)

        u_players, u_league_tables, u_league_stats, u_all_team_stats = understats
        u_season = get_season(match_year, match_month, "understat")

        u_season -= 1
        # careful of data leak - need to use strictly before game data
        # -> can add data from this season up to this match

        u_league_table = u_league_tables[f"{u_season}_{comp}"]
        if player_club == row["homeTeam"]:
            enemy_club = row["awayTeam"]
        else:
            enemy_club = row["homeTeam"]

        alL_team_names = list(u_league_table["Team"])
        u_team = pick_best_name_candidate(player_club, alL_team_names)
        u_e_team = pick_best_name_candidate(enemy_club, alL_team_names)

        u_features_vals = []
        u_col_names = []

        # Získáme týmové statistiky z understat pro tým hráče:
        u_team_table_row = u_league_table[u_league_table["Team"] == u_team].drop(["Team"], axis=1)
        u_e_team_table_row = u_league_table[u_league_table["Team"] == u_e_team].drop(["Team"], axis=1)
        try:
            u_features_vals += (list(u_team_table_row.iloc[0]) + list(u_e_team_table_row.iloc[0]))
        except IndexError:
            # stava se kdyz je v game stats nejakej zapas napr. z baraze, kde ten tym nepatri do ligovy tabulky
            return np.nan
        u_col_names += ["u_team_" + coln for coln in list(u_team_table_row.columns)] + \
                       ["u_enemy_team" + coln for coln in list(u_e_team_table_row.columns)]

        # Získáme záznam o ofenzivnosti ligy
        u_current_league_stats = u_league_stats[str(u_season) + "_" + comp]
        u_features_vals += [u_current_league_stats["hxg"] + u_current_league_stats["axg"],
                            u_current_league_stats["h"] + u_current_league_stats["a"],
                            u_current_league_stats["hxg"], u_current_league_stats["axg"],
                            u_current_league_stats["h"], u_current_league_stats["a"],
                            ]
        u_col_names += ["u_xg_league", "u_g_league",
                        "u_xg_home_league", "u_xg_away_league",
                        "u_g_home_league", "u_g_away_league"]

        # Získáme statistiky hráče z přechozí sezóny
        list_of_stats = get_source_data(row, u_players, teams, "understat")

        for df in list_of_stats:
            if not df.empty:
                u_features_vals += list(df.mean())
            else:
                u_features_vals += [np.nan for _ in df.columns]

            u_col_names += list(df.columns)

        # Statistiky o chování týmu v zápase za určitých situací
        u_team_stats = u_all_team_stats[str(u_season) + "_" + u_team]
        u_e_team_stats = u_all_team_stats[str(u_season) + "_" + u_e_team]

        def traverse_nested_dict(rest_of_dict, key_path, forbidden_keys, result, prefixes):
            if isinstance(rest_of_dict, (int, float)):
                for prefix in prefixes:
                    result[prefix + key_path] = rest_of_dict
            elif isinstance(rest_of_dict, dict):
                for k, val in rest_of_dict.items():
                    if k in forbidden_keys:
                        continue
                    traverse_nested_dict(val, key_path + "_" + k, forbidden_keys, result, prefixes)

        x_descriptive_combined = copy.deepcopy(dict_template)
        traverse_nested_dict(u_team_stats, "", ["formation"], x_descriptive_combined, ["", "enemy_"])
        u_feature_pool = {col_name: value for col_name, value in zip(u_col_names, u_features_vals)}
        x_descriptive_combined = {"u_" + k: v for k, v in x_descriptive_combined.items()}
        u_feature_pool = u_feature_pool | x_descriptive_combined
        u_feature_pool["understat_team"] = u_team
        return u_feature_pool

    @staticmethod
    def _get_common_fifa(row, sofifa, teams):
        fifa_stats = get_source_data(row, sofifa, teams, "fifa")
        return fifa_stats

    @staticmethod
    def _xs_ys_from_game_stats(row, teams, players, n):
        """
        V této metodě pouze využijeme ty informace, které vrátila metoda _get_common_features a spočítáme na nich std a mean
        :param row:
        :param teams:
        :param players:
        :param n:
        :return: jeden řádek ve feature pool
        """
        common_features = PerformanceExplorer._get_common_features(row, teams, players, n)
        LN = common_features[f"L{n}"]
        is_home = common_features["is_home"]
        teammates_games = common_features["gs_teammates_games"]
        enemy_players_games = common_features["gs_enemy_players_games"]
        player_games = common_features["gs_players_games"]
        player_games = player_games.drop(columns=['totalScore'])

        stats_to_combine = [
            ('gs_ally', teammates_games.head(n).select_dtypes(include='number')),
            ('gs_enemy', enemy_players_games.head(n).select_dtypes(include='number')),
            ('gs_player', player_games.select_dtypes(include='number'))
        ]

        final = {"gameId": f"{row.slug}_{row.date}",
                 f"L{n}": LN, "is_home": is_home}

        for prefix, df in stats_to_combine:
            means = df.mean().add_prefix(f"{prefix}_mean_")
            stds = df.std().add_prefix(f"{prefix}_std_")
            final.update(means)
            final.update(stds)

        return pd.Series(final)

    @staticmethod
    def _xs_ys_from_league_stats(row, league_stats):
        comp = row.competition
        player_club = row.club
        match_date = row.date
        player_slug = row.slug

        ls_stats = PerformanceExplorer._get_common_league_stats(
            league_stats, comp, player_club, match_date, [5, 15, 40])

        final = {"gameId": f"{player_slug}_{match_date}"} | ls_stats
        return pd.Series(final)

    @staticmethod
    def _xs_ys_from_understat(row, understats: tuple, teams, team_stats_dict_template):
        u_feature_pool = PerformanceExplorer._get_common_understat(row, understats, teams, team_stats_dict_template)
        final = {"gameId": f"{row.slug}_{row.date}"} | u_feature_pool
        return pd.Series(final)

    @staticmethod
    def _xs_ys_from_sofifa(row, sofifa, teams):

        # TODO: zpracovat docela zajimavy kategoricky promenny jako player_tags, body_type, player_traits...
        # tady je to tezsi, pro korelaci by se muselo vyjadrit ciselne, ale primo do datasetu by nemuselo byt spatny
        dataframes = PerformanceExplorer._get_common_fifa(row, sofifa, teams)

        final = pd.concat([df.select_dtypes(include="number").mean() for df in dataframes], axis=0)
        final["gameId"] = f"{row.slug}_{row.date}"
        return final

    #endregion

    @staticmethod
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


    @staticmethod
    def get_most_correlated_pairs(corr_matrix, upper_threshold, lower_threshold):
        """
        Statická metoda, která vybírá a třídí vzájemně korelované příznaky mezi zadanými práhy
        :param corr_matrix: Korelační matice
        :param upper_threshold:
        :param lower_threshold:
        :return: Setříděný seznam trojic (korelace, pří1, pří2) korelací mezi práhy.
        """
        res = []
        for i in range(corr_matrix.shape[1]):
            for j in range(i + 1, corr_matrix.shape[1]):
                value = corr_matrix.iloc[i, j]
                if upper_threshold > value > lower_threshold:
                    res.append((value, corr_matrix.columns[i], corr_matrix.columns[j]))
        return sorted(res)

    def load_feature_pool(self, positions, folder_fp):
        for position in positions:
            df = pd.read_csv(f"{folder_fp}/{position}.csv", index_col="gameId")
            df.columns = df.columns.str.replace('<', '_under_')
            # needs to replace < because it interferes with some models.
            df = df[~df.index.to_series().astype(str).str.contains('2020|2019')]
            self.feature_pool[position] = df

    def get_feature_pool(self,
                         positions: List[str],
                         league_stats: Dict[str, pd.DataFrame],
                         understats: Tuple[pd.DataFrame, Dict[str,pd.DataFrame], dict, dict],
                         sofifa: pd.DataFrame,
                         handcraft=False) -> Dict[str, pd.DataFrame]:
        """
        Metoda vytvoří pro každou pozici feature_pool, přidá ho jako atribut objektu a zároveň ho vrátí
        :param positions: seznam stringů pozic, načítáme většinou z config.py
        :param league_stats: data z footballdata.co.uk
        :param understats: čtveřice dat z Understat, tam kde uvádím v anotaci pouze dict, je struktura komplikovaná
        :param sofifa: data z FIFA
        :param handcraft: chceme nechávat False
        :return: slovník, kde klíče jsou jednotlivé pozice a hodnoty pd.DataFrame (feature_pool)
        """

        n = 40 # jak velké časové okno uvažujeme pro statistiky z game_stats
        reddit_json = json.load(open('../data/reddit.json'))

        def extract_sentiment_features(corpus):
            analysis = TextBlob(corpus)
            polarity = analysis.sentiment.polarity
            subjectivity = analysis.sentiment.subjectivity
            return polarity, subjectivity

        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        """
        def encode_text(text):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = model(**inputs)
            # Use the mean pooling strategy to get a single vector for the entire text
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.detach().numpy()
        """
        sentiments = {team:{date:extract_sentiment_features(text) for date, text in team_d.items()}
                          for team, team_d in reddit_json.items()}

        for position in positions:
            print("Getting feature pool: " + position)
            gs_by_teams = self.game_stats.groupby("club")  # gets all players that ever played for club
            gs_by_players = self.game_stats.groupby("slug")
            df = self.game_stats[self.game_stats.position == position]

            with open("../data/understat/team_stats/col_names.json", "r") as file:
                tsdt = json.load(file)

            if handcraft:
                # prozatím opuštěná myšlenka
                final = df.apply(self.handcraft_rows, teams=gs_by_teams, players=gs_by_players,
                                 understats=understats,
                                 league_stats=league_stats, team_stats_dict_template=tsdt, sofifa=sofifa, n=n, axis=1)

                final = final.set_index("gameId")
                filter_nonnan = ["aa_against_this_team_mean", "decisives_against_this_team_mean"]
                final = final[final[filter_nonnan].notna().all(axis=1)].drop(columns=[0])
            else:
                gs_args = {'teams': gs_by_teams, 'players': gs_by_players, 'n': n, 'axis': 1}
                ls_args = {'league_stats': league_stats, 'axis': 1}
                us_args = {'understats': understats, 'teams': gs_by_teams, 'team_stats_dict_template': tsdt, 'axis': 1}
                sf_args = {'sofifa': sofifa, 'teams': gs_by_teams, 'axis': 1}
                text_args = {'sentiments': sentiments , 'axis': 1}


                tqdm.pandas()
                # zde zpracovávám datové zdroje postupně, nejvíce trvají dva poslední
                # ve FIFA a Understa se počítají spoluhráči daného hráče, což trvá a nestihl jsem zoptimalizovat
                # používáme .apply metodu z pandas.DataFrame, všechny funke xs_ys tedy zpracovávají řádek po řádku
                outputs = [
                    df.progress_apply(self._xs_ys_from_game_stats, **gs_args),
                    df.progress_apply(self._xs_ys_from_league_stats, **ls_args),
                    #df.progress_apply(self._xs_ys_from_understat, **us_args),
                    #df.progress_apply(self._xs_ys_from_sofifa, **sf_args),
                    df.progress_apply(self.xs_ys_from_text, **text_args)
                ]

                outputs = [x.set_index("gameId") for x in outputs]
                final = pd.concat(outputs, axis=1)

            ys = df.totalScore.copy()
            ys.index = final.index
            final["totalScore"] = ys
            final[f"devFromL{n}"] = (ys / final[f"L{n}"]).replace([np.inf, -np.inf], 1)
            numerical_cols = final.select_dtypes(include=numerics).columns
            # fill missing values with median on numerical columns
            final[numerical_cols] = final[numerical_cols].fillna(final[numerical_cols].median())
            self.feature_pool[position] = final

        return self.feature_pool

    def compute_correlations(self, positions, plot) -> dict:
        """
        Use this method, after get_features_pool has already been called.
        :param positions:
        :param plot: ano, pokud chceme vizualizovat vztah mezi totalScore a příznakem f.
        :return: slovník, kde klíče jsou pozice a hodnotami setřídený slovník s korelacemi
        """
        if len(self.feature_pool) < 1:
            print("Run method get_feature_pool first")
            return {}

        correlations = {}
        for position in positions:
            pool_df = self.feature_pool[position]
            print("Calculating correlations for: " + position)

            # Calculate the correlation between each feature and 'totalScore'
            corrs_with_totalScore = pool_df.drop(columns=["totalScore", "devFromL40"]).corrwith(pool_df["totalScore"])

            # Sort the correlations by absolute values in descending order
            sorted_corrs = corrs_with_totalScore.sort_values(key=lambda x: abs(x), ascending=False)
            correlations[position] = dict(sorted_corrs)

            if plot:
                plot_corr_matrix(pool_df.corr(), 30)
                for col_name, corr in correlations[position].items():
                    plot_corr_data(pool_df[col_name], pool_df["totalScore"], f"Correlation: {corr}", col_name, position)

        self.correlations = correlations
        return correlations

    def filter_correlations(self, threshhold_of_importance=None, top_n_corrs=None) -> dict:
        """
        Metoda, která filtruje korelace v self.correlations a vrací je (nemění stav v self)
        :param threshhold_of_importance: práh pro velikost korelací, které chceme nechat
        :param top_n_corrs: kolik nejlepších korelací chceme nechat
        :return: slovník s korelacemi
        """
        all_corrs = {}
        for position in positions:
            corrs = self.correlations[position]
            if threshhold_of_importance:
                filtered_corrs = {}
                for col_name, corr in corrs.items():
                    if abs(corr) >= threshhold_of_importance:
                        filtered_corrs[col_name] = corr
                corrs = filtered_corrs
            if top_n_corrs:
                sub_dict = corrs
                sorted_sub_dict = dict(sorted(sub_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n_corrs])
                corrs = sorted_sub_dict

            all_corrs[position] = corrs
        return all_corrs

    def compute_rf_feature_importance(self, positions, plot, savepath) -> dict:
        """
        :param positions:
        :param plot:
        :param savepath: kam chceme uložit
        :return: slovník, kde klíčem jsou pozice a hodnotami setříděný slovník s rf importances
        """
        if len(self.feature_pool) < 1:
            print("Run method get_feature_pool first")
            return {}

        feature_importances = {}
        for position in positions:
            pool_df = self.feature_pool[position]
            print("Calculating feature importances for: " + position)

            X = pool_df.drop(columns=["totalScore", "devFromL40"], axis=1)
            X_numerical = X.select_dtypes(include=numerics)
            X = X_numerical.fillna(X_numerical.median()).fillna(0)
            y = pool_df["totalScore"]

            rf = RandomForestRegressor(verbose=1, n_jobs=-1)

            rf.fit(X, y)

            importances = dict(zip(X.columns, rf.feature_importances_))
            sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
            feature_importances[position] = {}

            feature_importances[position] = sorted_importances
        self.feature_importances = feature_importances
        if savepath:
            with open(savepath + "/feature_importances.json", "w") as f:
                json.dump(feature_importances, f)

        return feature_importances

    def filter_rf_importances(self, positions, top_n=None, threshold_of_importance=None):
        result = {}
        for position in positions:
            sub_dict = self.feature_importances[position]
            sorted_sub_dict = sorted(sub_dict.items(), key=lambda x: x[1], reverse=True)

            if top_n:
                sorted_sub_dict = sorted_sub_dict[:top_n]

            if threshold_of_importance:
                sorted_sub_dict = [(k, v) for k, v in sorted_sub_dict if v >= threshold_of_importance]

            result[position] = dict(sorted_sub_dict)
        return result

    def load_rf_feature_importance(self, path, plot):
        with open(path) as f:
            importances = json.load(f)
            self.feature_importances = importances

    def save_datasets(self, save_filename, cols_only, **kwargs):
        """
        Metoda ukládající feature_pool do složky, můžeme filtrovat pomocí kwargs (korelace, rf)
        :param save_filename: název datasetu, soubory se pak uloží do 'dataset/{datum}_{save_filename} '
        :param cols_only: nastavit jako True, pokud chceme ukládat jen jména sloupců a ne celý dataset
        :param kwargs: můžeme přidat argumenty pro filtrování datasetu
        :return:
        """
        today = date.today()
        dirpath = f"dataset/{str(today)}_{save_filename}"
        os.mkdir(dirpath)
        os.mkdir(dirpath + "/models")
        os.mkdir(dirpath + "/plots")
        for pos, position_df in self.feature_pool.items():
            filepath = f"{dirpath}/{pos}.csv"
            if "corrs" in kwargs:
                # based on features I find the most interesting filter the dataset
                interesting_columns = ["L40"] + ["totalScore"] + ["devFromL40"] \
                                      + list(kwargs["corrs"][pos].keys())
                position_df = position_df[interesting_columns]
            if "rf_importance" in kwargs:
                interesting_columns = ["L40"] + ["totalScore"] + ["devFromL40"] \
                                      + list(kwargs["rf_importance"][pos].keys())
                position_df = position_df[interesting_columns]
            if cols_only:
                position_df.columns.to_series().to_csv(filepath, index=False)
            else:
                position_df.to_csv(filepath)


if __name__ == "__main__":
    is_clustering = False
    simple = load_simple()
    understats = load_understat()
    game_stats = load_gamestats(is_clustering=is_clustering)
    league_stats = load_league_stats("../data/league_stats")
    sofifa_stats = load_sofifa()

    def get_best_teams(game_stats):
        best = game_stats[(game_stats.date > "2023-07") &
                          (game_stats.position == "Forward")].groupby("club")["totalScore"].mean()
        return best

    def get_above_75_df(grouped_df):
        agg_df = grouped_df.agg(
            total_count=('totalScore', 'size'),
            above_75_count=('totalScore', lambda x: (x > 75).sum()),
            L15=('totalScore', lambda x: x.tail(15).mean())
        )
        agg_df = agg_df[agg_df.total_count > 15]
        # Calculate percentage
        agg_df['percentage_above_75'] = (agg_df['above_75_count'] / agg_df['total_count']) * 100
        agg_df['ratio'] = agg_df["percentage_above_75"] / agg_df["L15"]
        return agg_df

    def print_percentage_above_75(game_stats):
        # Aggregate to count total and those scores above 75 for each club
        best = {}
        for position in positions:
            best[position] = get_above_75_df(game_stats[(game_stats.date > "2023-07") &
                            (game_stats.position == position)].groupby("slug"))
        return best


    #bt = get_best_teams(game_stats)
    #pas = print_percentage_above_75(game_stats)
    team_subreddits = {
        "arsenal-london": "Gunners",
        "manchester-united-manchester": "reddevils",
        "liverpool-liverpool": "LiverpoolFC",
        "chelsea-london": "chelseafc",
        "tottenham-hotspur-london": "coys",
        "barcelona-barcelona": "Barca",
        "real-madrid-madrid": "realmadrid",
        "manchester-city-manchester": "MCFC",
        "bayern-munchen-munchen": "fcbayern",
        "borussia-dortmund-dortmund": "borussiadortmund",
        "everton-liverpool": "Everton",
        "newcastle-united-newcastle-upon-tyne": "NUFC",
        "milan-milano": "ACMilan",
        "juventus-torino": "Juve",
        "west-ham-united-london": "Hammers",
        "aston-villa-birmingham": "avfc",
        "roma-roma": "ASRoma",
        "psg-paris": "psg",
        "atletico-madrid-madrid": "atletico",
        "internazionale-milano": "FCInterMilan"
    }
    game_stats = game_stats[
        (game_stats.homeTeam.isin(list(team_subreddits.keys()))) |
        (game_stats.awayTeam.isin(list(team_subreddits.keys())))
    ]
    pe = PerformanceExplorer(game_stats)
    pe.get_feature_pool(positions, league_stats=league_stats, understats=understats, sofifa=sofifa_stats, handcraft=False)



    #pe.load_feature_pool(positions, f"dataset/2023-04-27_comprehensive_version")

    #corrs = pe.compute_correlations(positions, plot=False)



    #rf_importance = pe.compute_rf_feature_importance(positions, False, "dataset/2023-04-27_comprehensive_version",)
    #rf_importance = pe.load_rf_feature_importance("dataset/2023-04-27_comprehensive_version/feature_importances.json", plot=False)

    pe.save_datasets("comprehensive", False)





