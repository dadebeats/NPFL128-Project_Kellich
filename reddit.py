import praw
import datetime

reddit = praw.Reddit(

)


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

data = {tn: {} for tn in team_subreddits.keys()}
cutoff_date_lower = datetime.datetime(2020, 12, 31, 23, 59, 59)
cutoff_date_upper = datetime.datetime(2023, 4, 27, 23, 59, 59)

for team_slug, subredd_name in team_subreddits.items():
    # Search submissions in the subreddit that contain the query string in their title
    for submission in reddit.subreddit(subredd_name).search(f"title:'match'", syntax='lucene', time_filter='all'):
        dt_object = datetime.datetime.utcfromtimestamp(submission.created_utc)
        if not(cutoff_date_upper > dt_object > cutoff_date_lower):
            continue
        print(f"Title: {submission.title}")
        print(f"URL: {submission.url}")
        submission.comments.replace_more(limit=0)
        corpus = [comment.body for comment in submission.comments.list()]

        data[team_slug][(dt_object)] = "\n____\n".join(corpus)






