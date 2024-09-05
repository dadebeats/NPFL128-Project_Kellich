import pandas as pd

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
positions = ["Forward", "Midfielder", "Defender", "Goalkeeper"]
rarities = ["limited", "rare", "super_rare", "unique"]

understat_comps = [
    "ligue-1-fr",
    "serie-a-it",
    "laliga-es",
    "premier-league-gb-eng",
    "bundesliga-de"
]


minimal_mins_played_clustering = -1
minimal_mins_played_performance = 59
fifa_init_year = 18
current_year = 23
common_cols_in_ls = {'PSH', 'IWH', 'PSCD', 'HTR', 'AST', 'VCD', 'AF', 'FTHG', 'VCA', 'HY', 'season', 'WHD', 'BWH', 'HR', 'AwayTeam', 'HS', 'B365D', 'VCH', 'HomeTeam', 'HTAG', 'PSA', 'BWD', 'Div', 'B365H', 'B365A', 'PSCH', 'IWD', 'IWA', 'BWA', 'PSCA', 'AR', 'WHH', 'WHA', 'HST', 'AS', 'Date', 'FTR', 'PSD', 'FTAG', 'HTHG', 'HF', 'AC', 'HC', 'AY'}


def load_simple() -> pd.DataFrame:
    simple = pd.read_csv(open("data/all_players_simple.csv", "r", encoding="utf-8"))

    return simple


def load_gamestats(is_clustering: bool) -> pd.DataFrame:
    game_stats = pd.read_csv("data/game_stats.csv")
    simple = load_simple()
    #TODO: a lot of fixing here, maybe move all this filtering to params
    game_stats = game_stats.merge(simple[["slug", "clubMemberships", "position", "country"]])
    #TODO: better fifa and understat name matching

    # take out this guy, causes exception in xs_ys_from_league_stats
    game_stats = game_stats[game_stats.slug != "iago-herrerin-buisan"]
    game_stats = game_stats.sort_values(by="date")
    minimal_mins_played = minimal_mins_played_clustering if is_clustering else minimal_mins_played_performance
    game_stats = game_stats[(game_stats.mins_played > 0)]

    if is_clustering:
        return game_stats

    game_stats = game_stats[game_stats["competition"].isin(understat_comps)]

    gameWeek_data_counts = dict(game_stats.gameWeek.value_counts())
    MIN_MATCHES_IN_GW = 50
    gws_to_include = {gw for gw, count in gameWeek_data_counts.items() if count > MIN_MATCHES_IN_GW}
    game_stats = game_stats[game_stats.gameWeek.isin(gws_to_include)]
    return game_stats
