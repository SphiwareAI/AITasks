import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import emoji
# Remove hashtags
def hashtags(text):
    hash_tags = re.findall(r"#(\w+)", text)
    return hash_tags
# Translate emoji
def preprocess_emojis(text):
    # Replace emoji shorthand representations with their names
    text = emoji.demojize(text)
    # Remove underscores and colons from emoji names
    text = text.replace("_", " ").replace(":", "")    
    return text
def non_ascii(s):
    return "".join(i for i in s if ord(i) < 128)
def lower(text):
    return text.lower()
def remove_stopwords(text):
    # Select English stopwords
    cached_stopwords = set(stopwords.words("english"))
    cached_stopwords.update(('and', 'I', 'A', 'http', 'And', 'So', 'arnt', 'This', 'When', 'It', 'many', 'Many', 'so', 'cant', 'Yes', 'yes', 'No', 'no', 'These', 'these', 'ayanna', 'like', 'face'))
    new_text = ' '.join([word for word in text.split() if word not in cached_stopwords])
    return new_text
def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    text = " ".join(tokens)
    return text
def remove_underscores(text):
    text = re.sub('([_]+)', " ", text)
    return text
def clean_text(text):
    # Apply cleaning functions
    text = preprocess_emojis(text)
    text = non_ascii(text)
    text = lower(text)
    text = remove_stopwords(text)
    text = remove_punctuation(text)
    text = remove_underscores(text)    
    return text

