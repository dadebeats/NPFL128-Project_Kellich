import praw
import datetime


def fetch_top_comments(submission, limit=100):
    submission.comments.replace_more(limit=0)
    comments = submission.comments.list()
    comments.sort(key=lambda x: x.ups, reverse=True)
    return comments[:limit]


def fetch_replies(comment, limit=10):
    comment.replies.replace_more(limit=0)
    replies = comment.replies.list()
    replies.sort(key=lambda x: x.ups, reverse=True)
    return replies[:limit]


# Pro tyhle týmy jsem stahoval data do "data/reddit.json"
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

if __name__ == "__main__":
    reddit = praw.Reddit(
    # Need to add login to your reddit account
    )

    # Pro tyhle týmy jsem stahoval data do "data/reddit.json"


    data = {tn: {} for tn in team_subreddits.keys()}
    # Stahoval jsem data od začátku roku 2021 do 6/2024
    cutoff_date_lower = datetime.datetime(2021, 1, 1, 0, 0, 0)
    cutoff_date_upper = datetime.datetime(2024, 6, 27, 23, 59, 59)

    for team_slug, subredd_name in team_subreddits.items():
        # Hledáme thready, které mají v názvu string "match"
        # protože thready s potenciálně užitečným info se často jmenují: Match thread, Post-match thread, Pre-match thread etc...
        submission_list = reddit.subreddit(subredd_name).search(f"title:'match'", syntax='lucene', time_filter='all')
        for submission in submission_list:
            submission_datetime = datetime.datetime.utcfromtimestamp(submission.created_utc)
            if not(cutoff_date_upper > submission_datetime > cutoff_date_lower):
                continue
            print(f"Title: {submission.title}")
            print(f"URL: {submission.url}")

            top_comments = fetch_top_comments(submission, limit=100)
            corpus = ["Top comments"] + [com.body for com in top_comments] + ["Replies"]
            for comment in top_comments:
                replies = fetch_replies(comment, limit=10)
                corpus.extend(reply.body for reply in replies)

            data[team_slug][submission_datetime] = "\n____\n".join(corpus)







