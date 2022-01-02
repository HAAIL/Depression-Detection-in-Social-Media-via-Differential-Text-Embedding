import re
from ReplaceAbbr import ReplaceCommonAbbr
import ftfy

# ----------------------------------------------------
# A function to do the pre-processing of the tweets
# ----------------------------------------------------


def clean_text(tweets_list):
    clean_tweets = []

    # for every tweet, preprocessing will be apllied
    for tweet in tweets_list:
        # 1. Cast all tweets to lower case.
        tweet = tweet.lower()

        # 2. Replace common abbreviations.
        tweet = ReplaceCommonAbbr(tweet)

        # 3. Remove double spacing.
        tweet = re.sub('\s+', ' ', tweet)
        # tweet = re.sub('(\s*)unhappy(\s*)', ' ', tweet)
        # 4. Remove hatshtags, emojis, mentions and images
        if re.match("(\w+:\/\/\S+)", tweet) == None and len(tweet) > 10:
            tweet = ' '.join(re.sub(
                "(RT\s@[a-z]+[a-z0-9-_])+|(http\S+)|(@[a-z0-9]+)|(\#[a-z0-9]+)|(<Emoji:.*>)|(pic\.twitter\.com\/.*)",
                " ", tweet).split())
        emoj = re.compile("["
                          u"\U0001F600-\U0001F64F"  # emoticons
                          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                          u"\U0001F680-\U0001F6FF"  # transport & map symbols
                          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                          u"\U00002500-\U00002BEF"  # chinese char
                          u"\U00002702-\U000027B0"
                          u"\U00002702-\U000027B0"
                          u"\U000024C2-\U0001F251"
                          u"\U0001f926-\U0001f937"
                          u"\U00010000-\U0010ffff"
                          u"\u2640-\u2642"
                          u"\u2600-\u2B55"
                          u"\u200d"
                          u"\u23cf"
                          u"\u23e9"
                          u"\u231a"
                          u"\ufe0f"  # dingbats
                          u"\u3030"
                          "]+", re.UNICODE)
        tweet = re.sub(emoj, '', tweet)

        # 5. Remove punctuation.
        tweet = ' '.join(re.sub("([^0-9a-z \t])", " ", tweet).split())
        # 6. fix weirdly encoded texts
        tweet = ftfy.fix_text(tweet)

        clean_tweets.append(tweet)
    return clean_tweets

