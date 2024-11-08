# Sentiment Analysis of RateMyProfessor Sample Data



# `1. Project Setup`


## `1.1 Introduction`


## `1.2 Project Dependencies`


```python
!pip install pandas numpy transformers
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install ktrain
!pip install textblob

!pip install openai
```

    Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (1.5.3)
    Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (1.23.5)
    Requirement already satisfied: transformers in /usr/local/lib/python3.10/dist-packages (4.35.2)
    Requirement already satisfied: python-dateutil>=2.8.1 in /usr/local/lib/python3.10/dist-packages (from pandas) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas) (2023.3.post1)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from transformers) (3.13.1)
    Requirement already satisfied: huggingface-hub<1.0,>=0.16.4 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.19.4)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from transformers) (23.2)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (6.0.1)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers) (2023.6.3)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from transformers) (2.31.0)
    Requirement already satisfied: tokenizers<0.19,>=0.14 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.15.0)
    Requirement already satisfied: safetensors>=0.3.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.4.1)
    Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.10/dist-packages (from transformers) (4.66.1)
    Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.16.4->transformers) (2023.6.0)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.16.4->transformers) (4.5.0)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.1->pandas) (1.16.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2023.11.17)
    Looking in indexes: https://download.pytorch.org/whl/cu118
    Requirement already satisfied: torch in /usr/local/lib/python3.10/dist-packages (2.1.0+cu118)
    Requirement already satisfied: torchvision in /usr/local/lib/python3.10/dist-packages (0.16.0+cu118)
    Requirement already satisfied: torchaudio in /usr/local/lib/python3.10/dist-packages (2.1.0+cu118)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch) (3.13.1)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from torch) (4.5.0)
    Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch) (1.12)
    Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch) (3.2.1)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch) (3.1.2)
    Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch) (2023.6.0)
    Requirement already satisfied: triton==2.1.0 in /usr/local/lib/python3.10/dist-packages (from torch) (2.1.0)
    Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from torchvision) (1.23.5)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from torchvision) (2.31.0)
    Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /usr/local/lib/python3.10/dist-packages (from torchvision) (9.4.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch) (2.1.3)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision) (2023.11.17)
    Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch) (1.3.0)
    Requirement already satisfied: ktrain in /usr/local/lib/python3.10/dist-packages (0.39.0)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.10/dist-packages (from ktrain) (1.2.2)
    Requirement already satisfied: matplotlib>=3.0.0 in /usr/local/lib/python3.10/dist-packages (from ktrain) (3.7.1)
    Requirement already satisfied: pandas>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from ktrain) (1.5.3)
    Requirement already satisfied: fastprogress>=0.1.21 in /usr/local/lib/python3.10/dist-packages (from ktrain) (1.0.3)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from ktrain) (2.31.0)
    Requirement already satisfied: joblib in /usr/local/lib/python3.10/dist-packages (from ktrain) (1.3.2)
    Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from ktrain) (23.2)
    Requirement already satisfied: langdetect in /usr/local/lib/python3.10/dist-packages (from ktrain) (1.0.9)
    Requirement already satisfied: jieba in /usr/local/lib/python3.10/dist-packages (from ktrain) (0.42.1)
    Requirement already satisfied: charset-normalizer in /usr/local/lib/python3.10/dist-packages (from ktrain) (3.3.2)
    Requirement already satisfied: chardet in /usr/local/lib/python3.10/dist-packages (from ktrain) (5.2.0)
    Requirement already satisfied: syntok>1.3.3 in /usr/local/lib/python3.10/dist-packages (from ktrain) (1.4.4)
    Requirement already satisfied: tika in /usr/local/lib/python3.10/dist-packages (from ktrain) (2.6.0)
    Requirement already satisfied: transformers>=4.17.0 in /usr/local/lib/python3.10/dist-packages (from ktrain) (4.35.2)
    Requirement already satisfied: sentencepiece in /usr/local/lib/python3.10/dist-packages (from ktrain) (0.1.99)
    Requirement already satisfied: keras-bert>=0.86.0 in /usr/local/lib/python3.10/dist-packages (from ktrain) (0.89.0)
    Requirement already satisfied: whoosh in /usr/local/lib/python3.10/dist-packages (from ktrain) (2.7.4)
    Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from keras-bert>=0.86.0->ktrain) (1.23.5)
    Requirement already satisfied: keras-transformer==0.40.0 in /usr/local/lib/python3.10/dist-packages (from keras-bert>=0.86.0->ktrain) (0.40.0)
    Requirement already satisfied: keras-pos-embd==0.13.0 in /usr/local/lib/python3.10/dist-packages (from keras-transformer==0.40.0->keras-bert>=0.86.0->ktrain) (0.13.0)
    Requirement already satisfied: keras-multi-head==0.29.0 in /usr/local/lib/python3.10/dist-packages (from keras-transformer==0.40.0->keras-bert>=0.86.0->ktrain) (0.29.0)
    Requirement already satisfied: keras-layer-normalization==0.16.0 in /usr/local/lib/python3.10/dist-packages (from keras-transformer==0.40.0->keras-bert>=0.86.0->ktrain) (0.16.0)
    Requirement already satisfied: keras-position-wise-feed-forward==0.8.0 in /usr/local/lib/python3.10/dist-packages (from keras-transformer==0.40.0->keras-bert>=0.86.0->ktrain) (0.8.0)
    Requirement already satisfied: keras-embed-sim==0.10.0 in /usr/local/lib/python3.10/dist-packages (from keras-transformer==0.40.0->keras-bert>=0.86.0->ktrain) (0.10.0)
    Requirement already satisfied: keras-self-attention==0.51.0 in /usr/local/lib/python3.10/dist-packages (from keras-multi-head==0.29.0->keras-transformer==0.40.0->keras-bert>=0.86.0->ktrain) (0.51.0)
    Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.0.0->ktrain) (1.2.0)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.0.0->ktrain) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.0.0->ktrain) (4.45.1)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.0.0->ktrain) (1.4.5)
    Requirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.0.0->ktrain) (9.4.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.0.0->ktrain) (3.1.1)
    Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.0.0->ktrain) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.0.1->ktrain) (2023.3.post1)
    Requirement already satisfied: regex>2016 in /usr/local/lib/python3.10/dist-packages (from syntok>1.3.3->ktrain) (2023.6.3)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from transformers>=4.17.0->ktrain) (3.13.1)
    Requirement already satisfied: huggingface-hub<1.0,>=0.16.4 in /usr/local/lib/python3.10/dist-packages (from transformers>=4.17.0->ktrain) (0.19.4)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from transformers>=4.17.0->ktrain) (6.0.1)
    Requirement already satisfied: tokenizers<0.19,>=0.14 in /usr/local/lib/python3.10/dist-packages (from transformers>=4.17.0->ktrain) (0.15.0)
    Requirement already satisfied: safetensors>=0.3.1 in /usr/local/lib/python3.10/dist-packages (from transformers>=4.17.0->ktrain) (0.4.1)
    Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.10/dist-packages (from transformers>=4.17.0->ktrain) (4.66.1)
    Requirement already satisfied: six in /usr/local/lib/python3.10/dist-packages (from langdetect->ktrain) (1.16.0)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->ktrain) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->ktrain) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->ktrain) (2023.11.17)
    Requirement already satisfied: scipy>=1.3.2 in /usr/local/lib/python3.10/dist-packages (from scikit-learn->ktrain) (1.11.4)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn->ktrain) (3.2.0)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from tika->ktrain) (67.7.2)
    Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.16.4->transformers>=4.17.0->ktrain) (2023.6.0)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.16.4->transformers>=4.17.0->ktrain) (4.5.0)
    Requirement already satisfied: textblob in /usr/local/lib/python3.10/dist-packages (0.17.1)
    Requirement already satisfied: nltk>=3.1 in /usr/local/lib/python3.10/dist-packages (from textblob) (3.8.1)
    Requirement already satisfied: click in /usr/local/lib/python3.10/dist-packages (from nltk>=3.1->textblob) (8.1.7)
    Requirement already satisfied: joblib in /usr/local/lib/python3.10/dist-packages (from nltk>=3.1->textblob) (1.3.2)
    Requirement already satisfied: regex>=2021.8.3 in /usr/local/lib/python3.10/dist-packages (from nltk>=3.1->textblob) (2023.6.3)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from nltk>=3.1->textblob) (4.66.1)
    Requirement already satisfied: openai in /usr/local/lib/python3.10/dist-packages (1.3.8)
    Requirement already satisfied: anyio<5,>=3.5.0 in /usr/local/lib/python3.10/dist-packages (from openai) (3.7.1)
    Requirement already satisfied: distro<2,>=1.7.0 in /usr/lib/python3/dist-packages (from openai) (1.7.0)
    Requirement already satisfied: httpx<1,>=0.23.0 in /usr/local/lib/python3.10/dist-packages (from openai) (0.25.2)
    Requirement already satisfied: pydantic<3,>=1.9.0 in /usr/local/lib/python3.10/dist-packages (from openai) (1.10.13)
    Requirement already satisfied: sniffio in /usr/local/lib/python3.10/dist-packages (from openai) (1.3.0)
    Requirement already satisfied: tqdm>4 in /usr/local/lib/python3.10/dist-packages (from openai) (4.66.1)
    Requirement already satisfied: typing-extensions<5,>=4.5 in /usr/local/lib/python3.10/dist-packages (from openai) (4.5.0)
    Requirement already satisfied: idna>=2.8 in /usr/local/lib/python3.10/dist-packages (from anyio<5,>=3.5.0->openai) (3.6)
    Requirement already satisfied: exceptiongroup in /usr/local/lib/python3.10/dist-packages (from anyio<5,>=3.5.0->openai) (1.2.0)
    Requirement already satisfied: certifi in /usr/local/lib/python3.10/dist-packages (from httpx<1,>=0.23.0->openai) (2023.11.17)
    Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.10/dist-packages (from httpx<1,>=0.23.0->openai) (1.0.2)
    Requirement already satisfied: h11<0.15,>=0.13 in /usr/local/lib/python3.10/dist-packages (from httpcore==1.*->httpx<1,>=0.23.0->openai) (0.14.0)
    

We will need pandas to clean and work with our json review data, numpy for help with general mathematics operations. We are running various pre-trained sentiment models (which can be found at https://huggingface.co/) on the Transformers library, using PyTorch as our framework.


```python
import pandas as pd
import numpy as np
import torch
import json
import re
import nltk
from transformers import pipeline
from ktrain.text.sentiment.core import SentimentAnalyzer
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
```

    [nltk_data] Downloading package wordnet to /root/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    




    True



# `2. Dataset Overview`

This data was determined by the team to be a good source for our project due to the large volume of easily accessible review text that could be analyzed in a sentiment analysis project. In total, our scraper collected 21+ million reviews, 2+ million professor profiles, and nearly 8000 schools. The entire scope of this data includes information to filter by geographical region, single university, university department, or a unique professor. Collectively, reviews date from as early as 2001 all the way to present-day. Individual reviews contain data about what professor it’s associated with, class number, ratings data, reviewer comment, and date of review.

## `2.1 Data Collecition`



```python
# Load the professor profile data
higher_professors = pd.read_json('https://raw.githubusercontent.com/Will-Alger/csc425-sentiment-analysis/main/higher_professors.json')
lower_professors = pd.read_json('https://raw.githubusercontent.com/Will-Alger/csc425-sentiment-analysis/main/lower_professors.json')

# Load the reviews data
higher_reviews = pd.read_json('https://raw.githubusercontent.com/Will-Alger/csc425-sentiment-analysis/main/higher_professor_reviews.json')
lower_reviews = pd.read_json('https://raw.githubusercontent.com/Will-Alger/csc425-sentiment-analysis/main/lower_professor_reviews.json')

higher_reviews_url ="https://raw.githubusercontent.com/Will-Alger/csc425-sentiment-analysis/main/higher_professor_reviews.json"
lower_reviews_url = "https://raw.githubusercontent.com/Will-Alger/csc425-sentiment-analysis/main/lower_professor_reviews.json"


science_professors_url = "https://raw.githubusercontent.com/ssdtac/Professor-Reviews/master/science_professors_v2.json"
humanities_professors_url = "https://raw.githubusercontent.com/ssdtac/Professor-Reviews/master/humanities_professors_v2.json"

```

## `2.2 Dataset Description`

To test the waters with a pre-trained model, 10 professors were chosen from our database.
5 were selected with an overall lower avgRating, and 5 were selected with an overall higher avgRating.

Selection of higher rated professors:
```
select *
  from professors      
  where avgRating between 3.5 and 4
  and numRatings between 20 and 35
limit 5
```

Selection of lower rated professors:
```
select *
  from professors
  where avgRating <= 2.5
  and numRatings between 20 and 50
limit 5
```

The URLs for science and humanities professors from NKU will be used later.

# `3. Data Preprocessing and Cleaning`

The following preprocessing methods will be used throughout the project.


```python
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^b\s+', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    lemmatized_tokens = [token for token in lemmatized_tokens if len(token) > 3]
    return ' '.join(lemmatized_tokens)



def load_and_preprocess_data(url):
    data = pd.read_json(url)
    original_row_count = len(data)
    print(f"Rows before preprocessing: {original_row_count}")
    data = data[data['comment'].ne('No Comments')]
    preprocessed_row_count = len(data)
    print(f"Rows discarded for being 'No Comments': {original_row_count - preprocessed_row_count}")
    data['comment'] = data['comment'].apply(preprocess_text)
    data = data[data['comment'].notna() & data['comment'].str.strip().ne('')]
    filtered_row_count = len(data)
    print(f"Rows discarded for no rating: {preprocessed_row_count - filtered_row_count}")
    data = data[data['qualityRating'].between(1, 5, inclusive='both') & data['difficultyRating'].between(1, 5, inclusive='both')]
#    data['qualityRating'] = data['qualityRating'].apply(preprocess_ratings)
    print(f"Rows left after preprocessing: {len(data)}")
    return data
```

## `3.1 Defining our preprocessed data`

Its worth noting that there exists roughly twice the number of reviews for humanities as there are sciences. A more accurate approach might include balancing these two datasets.


```python
science_professors = load_and_preprocess_data(science_professors_url)
humanities_professors = load_and_preprocess_data(humanities_professors_url)
professors = pd.concat([science_professors, humanities_professors], ignore_index=True)
```

    Rows before preprocessing: 6412
    Rows discarded for being 'No Comments': 56
    Rows discarded for no rating: 5
    Rows left after preprocessing: 6338
    Rows before preprocessing: 11389
    Rows discarded for being 'No Comments': 178
    Rows discarded for no rating: 22
    Rows left after preprocessing: 11166
    

# `4. Sentiment Analysis Model Proof of Concept`

### `4.1 A Preliminary Model`

In the Transformers library, there is a pipeline abstraction. According to the documentation, pipelines serve as "objects that abstract most of the complex code from the library, offering a simple API dedicated to several tasks... [such as] Sentiment Analysis." To start the project, we have opted to use pipelines. This will serve to simplify the start of the project, allowing us to build iteratively.


```python
pipe = pipeline(task='sentiment-analysis', framework='pt', model='distilbert-base-uncased-finetuned-sst-2-english')
```

### `4.2 Clean and Preprocess data`


```python
higher_reviews_processed = load_and_preprocess_data(higher_reviews_url)
lower_reviews_processed  = load_and_preprocess_data(lower_reviews_url)
```

    Rows before preprocessing: 151
    Rows discarded for being 'No Comments': 12
    Rows discarded for no rating: 1
    Rows left after preprocessing: 138
    Rows before preprocessing: 177
    Rows discarded for being 'No Comments': 11
    Rows discarded for no rating: 1
    Rows left after preprocessing: 165
    

### `4.3 Calculate sentiment distribution for professors`


```python
positive_higher = 0
negative_higher = 0
positive_lower = 0
negative_lower = 0

# Analyze sentiment for higher-rated professors
for index, row in higher_reviews_processed.iterrows():
    sentiment = pipe(row['comment'])
    if sentiment[0]['label'] == 'POSITIVE':
        positive_higher += 1
    else:
        negative_higher += 1

# Analyze sentiment for lower-rated professors
for index, row in lower_reviews_processed.iterrows():
    sentiment = pipe(row['comment'])
    if sentiment[0]['label'] == 'POSITIVE':
        positive_lower += 1
    else:
        negative_lower += 1

average_quality_higher_professors = higher_professors['avgRating'].mean()
average_quality_lower_professors = lower_professors['avgRating'].mean()

# Average overall rating for higher and lower rated professors
print(f"Average overall rating for selected higher-rated professors: {average_quality_higher_professors}")
print(f"Average overall rating for selected lower-rated professors: {average_quality_lower_professors}")

# Frequencies
print(f"\nHigher-rated professors - Positive: {positive_higher}, Negative: {negative_higher}")
print(f"Lower-rated professors - Positive: {positive_lower}, Negative: {negative_lower}")
```

    Average overall rating for selected higher-rated professors: 3.72
    Average overall rating for selected lower-rated professors: 2.16
    
    Higher-rated professors - Positive: 69, Negative: 69
    Lower-rated professors - Positive: 48, Negative: 117
    

### `4.4 Display a distribution of professor review sentiment`


```python
import matplotlib.pyplot as plt
import numpy as np

categories = ['Higher-rated Professors', 'Lower-rated Professors']
higher_counts = [positive_higher, negative_higher]
lower_counts = [positive_lower, negative_lower]

pos = np.arange(len(categories))
bar_width = 0.35

fig, ax = plt.subplots()

bar_higher = ax.bar(pos, higher_counts, bar_width, label='Positive Reviews')
bar_lower = ax.bar(pos + bar_width, lower_counts, bar_width, label='Negative Reviews')

ax.set_xlabel('Sentiment')
ax.set_ylabel('Count')
ax.set_title('Sentiment Count Distribution by Professor Rating')
ax.set_xticks(pos + bar_width / 2)
ax.set_xticklabels(categories)
ax.legend()

plt.show()

```


    
![png](Main_Document_CSC425_RateMyProfessor_Sentiment_Analysis%20%282%29_files/Main_Document_CSC425_RateMyProfessor_Sentiment_Analysis%20%282%29_25_0.png)
    


# `5. Labeling Data for Training`

> Our trained model will only be as good as our data. Its common practice to use crowd-sourcing to label a significant amount of data, however, this is not within our reach.

> We could consider using an open source sentiment analysis dataset (such as one on movie reviews), but this can sometime lead to bias and inaccuracies during a domain shift.

> One possibilty would be to train the model based on the quality and difficulty scores that exist alongside the comment. This with be explored later.

> Another possibility would be to attempt to a few methods to automatically classify the reviews to train our model on. This will also be investigated. As discussed previously, however, our model will only be as good as our data.


## 5.1 Labeling using OpenAI API

In practice, OpenAI's latest model gpt-4-1106-preview is fairly effective at detecting sentiment for a given text. In one study, it was found that ChatGPT can outperform crowd-workers for text-annotation tasks (https://arxiv.org/abs/2303.15056)

Labeling the data, however, can take a fairly significant amount of time making calls to the API and awaiting responses. Nevertheless, we thought it would be worth a short exploration.

</br>
Note: if you're curious to test yourself, please enter an openai api key in the specified location


```python
!pip install tqdm

```


```python
import os
from openai import OpenAI
import json
```


```python
client = OpenAI(
    api_key="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"
)

assistant = client.beta.assistants.create(
    name="Sentiment Analyzer",
    instructions="Analyze the sentiment of the text provided and respond with either 'Negative', 'Neutral', or 'Positive'. Only respond with a single word representing the sentiment",
    model="gpt-4-1106-preview"
)

# Function to analyze sentiment
def analyze_sentiment(text):
    # Create a new thread for each analysis
    thread = client.beta.threads.create()

    # Add the user's message to the Thread
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=text
    )

    # Run the Assistant on the Thread
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    # Wait for the Run to complete and check the status
    while run.status != "completed":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )

    # Retrieve and return the Assistant's response
    messages = client.beta.threads.messages.list(
        thread_id=thread.id,
        order="asc"
    )

    for message in messages:
        if message.role == 'assistant':
            sentiment_result = message.content[0].text.value
            print(f"Analyzed Sentiment: {sentiment_result}")
            return sentiment_result
```

### `5.1a utilizing a sample of 500 reviews to label`

In one instance, we labeled 2000 reviews through this approach. It took several hours however, and didn't improve our model accuracy substantially.


```python
chatgpt_labeled_data = professors.sample(n=500)
analyze_sentiment("very good")
```

    Analyzed Sentiment: Positive
    




    'Positive'




```python
from tqdm import tqdm
sentiments = []

# Loop through each row in the DataFrame with a progress bar
for comment in tqdm(chatgpt_labeled_data['comment'], desc="Analyzing sentiments"):
    sentiment = analyze_sentiment(comment)
    sentiments.append(sentiment)

# Add the sentiments list as a new column to your DataFrame
chatgpt_labeled_data['sentiment'] = sentiments
```

    Analyzing sentiments:   0%|          | 1/500 [00:02<21:30,  2.59s/it]

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Neutral
    

    

    Analyzed Sentiment: Negative
    

    

    Analyzed Sentiment: Positive
    

    

    Analyzed Sentiment: Neutral
    

    Analyzing sentiments: 100%|██████████| 500/500 [20:44<00:00,  2.49s/it]

    Analyzed Sentiment: Neutral
    

    
    


```python

valid_sentiments = ['Neutral', 'Negative', 'Positive']
valid_check = chatgpt_labeled_data['sentiment'].isin(valid_sentiments)

# Check if there are any rows where the sentiment is not valid
if not valid_check.all():
    # Find the number of invalid entries
    num_invalid = (~valid_check).sum()
    print(f"There are {num_invalid} invalid sentiment entries.")

    # Optional: Display the rows with invalid sentiments
    invalid_rows = chatgpt_labeled_data[~valid_check]
    print("Invalid rows:")
    print(invalid_rows)
else:
    print("All sentiment entries are valid.")
```

    All sentiment entries are valid.
    


```python
sentiment_counts = chatgpt_labeled_data['sentiment'].value_counts()

# Print the counts for each sentiment
print("Sentiment counts:")
print(sentiment_counts)
```

    Sentiment counts:
    Positive    227
    Neutral     140
    Negative    133
    Name: sentiment, dtype: int64
    

## 5.2 Using convenience libraries and a Heuristic function
We decided to come up with a heuristic function to determine sentiment (positive, negative, or neutral) based off of the student's quality rating and difficulty rating for the professor. We will be comparing this heuristic function's output to that of a pre-trained sentiment analysis model that outputs the same classes.

### `5.2a Create the SentimentAnalyzer Object`
This pretrained model will classify a string of text as having a positive, negative, or neutral sentiment.


```python
model = SentimentAnalyzer()
# usage: model.predict(["text"])
```


    config.json:   0%|          | 0.00/747 [00:00<?, ?B/s]



    pytorch_model.bin:   0%|          | 0.00/499M [00:00<?, ?B/s]



    vocab.json:   0%|          | 0.00/899k [00:00<?, ?B/s]



    merges.txt:   0%|          | 0.00/456k [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/150 [00:00<?, ?B/s]


### `5.2b Make a Heuristic Function`
Our heuristic function attempts to make a guess as to the sentiment of the student's review based off of the quality rating and difficulty rating they gave. It is a naive function but serves for the purpose of comparison and demonstration.


```python
def sentiment_heuristic(review):
    quality = review['qualityRating']
    difficulty = review['difficultyRating']
    quality_threshold = 2
    difficulty_threshold = 3

    if quality > quality_threshold and difficulty < difficulty_threshold:
        return 'Positive'
    elif quality < quality_threshold and difficulty > difficulty_threshold:
        return 'Negative'
    else:
        return 'Neutral'


professors['sentiment'] = professors.apply(sentiment_heuristic, axis=1)
science_professors['sentiment'] = science_professors.apply(sentiment_heuristic, axis=1)
humanities_professors['sentiment'] = humanities_professors.apply(sentiment_heuristic, axis=1)
```

### `5.2c Apply Heuristic and Sentiment Analysis and` Compare
First we make a method to compare sentiment predictions from the two sources:


```python
def compare_sentiment(df):
  sentiment_predictions = [list(item)[0].lower() for item in model.predict(df['comment'].tolist())]

  heuristic_predictions = [item.lower() for item in df['sentiment']]

  same_count = 0
  diff_count = 0
  for i in range(len(sentiment_predictions)):
    if sentiment_predictions[i] == heuristic_predictions[i]:
      same_count += 1
    else: diff_count += 1

  return same_count, diff_count, heuristic_predictions, sentiment_predictions
```

Then we compare the models for both science and humanities professors at NKU. Science First:


```python
same_science, diff_science, science_h, science_model = compare_sentiment(science_professors)
```

    /usr/local/lib/python3.10/dist-packages/transformers/pipelines/base.py:1101: UserWarning: You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset
      warnings.warn(
    

And then for Humanities Professors:


```python
same_humanities, diff_humanities, humanities_h, humanities_model = compare_sentiment(humanities_professors)
```

Then we make a quick layout to see how they compare:


```python
def show_heatmap(sentiment, heuristic, label):
  rating_map = {'positive': 1, 'neutral': 0, 'negative': -1}
  # Convert ratings to numeric values
  heuristic_numeric = np.array([rating_map[rating] for rating in heuristic])
  sentiment_numeric = np.array([rating_map[rating] for rating in sentiment])
  # Create a 2D histogram (heatmap data)
  heatmap_data, xedges, yedges = np.histogram2d(heuristic_numeric, sentiment_numeric, bins=3)
  # Plotting the heatmap
  plt.imshow(heatmap_data, interpolation='nearest', cmap='Blues')
  plt.colorbar()
  # Setting the axis labels
  plt.xlabel('Heuristic Ratings')
  plt.ylabel('Model Ratings')

  # Adjust the ticks to match the categories
  plt.xticks(np.arange(3), ['Negative', 'Neutral', 'Positive'])
  plt.yticks(np.arange(3), ['Negative', 'Neutral', 'Positive'])
  # Display the heatmap
  plt.title(f'Comparison of Heuristic vs Sentiment Model Ratings For {label} Professors')
  plt.show()
```


```python
print("Total accuracy for Science Professors:", float(100*same_science/(same_science+diff_science)))
show_heatmap(science_model, science_h, "Science")
print("Total accuracy for Humanities Professors:", float(100*same_humanities/(same_humanities+diff_humanities)))
show_heatmap(humanities_model, humanities_h, "Humanities")
```

    Total accuracy for Science Professors: 43.168191858630486
    


    
![png](Main_Document_CSC425_RateMyProfessor_Sentiment_Analysis%20%282%29_files/Main_Document_CSC425_RateMyProfessor_Sentiment_Analysis%20%282%29_50_1.png)
    


    Total accuracy for Humanities Professors: 47.22371484864768
    


    
![png](Main_Document_CSC425_RateMyProfessor_Sentiment_Analysis%20%282%29_files/Main_Document_CSC425_RateMyProfessor_Sentiment_Analysis%20%282%29_50_3.png)
    


For this dataset, it looks like our heuristic and the pretrained Sentiment analysis model only line up about 45% of the time, which is not particularly encouraging. From this, we can tell that Neutral sentiment is most commonly misinterpreted as positive sentiment by our heuristic, at least according to the pretrained model.


Next, we have decided to attempt to train our own models for the purpose of analyzing professor review sentiment.

# `6. Training Our Own Models`


## `6.1 Training sentiment model on qualityRating scores`


```python
import ktrain
from ktrain import text
from sklearn.model_selection import train_test_split
```

### `6.1a Split Data`
First we split our data into a training dataset and a test dataset. 80% of the data will be used for training and 20% will be used for testing.


```python
def preprocess_ratings(rating):
  return int(rating - 1)
```


```python
professors['qualityRating'] = professors['qualityRating'].apply(preprocess_ratings)
train_df, test_df = train_test_split(professors, test_size=0.2, random_state=40)

train_size = train_df.shape[0]
test_size = test_df.shape[0]

print("Size of training set:", train_size,"\n"+"Size of test set:", test_size)

x_train = train_df['comment'].to_numpy()
y_train = train_df['qualityRating'].to_numpy().astype(int)

x_test = test_df['comment'].to_numpy()
y_test = test_df['qualityRating'].to_numpy().astype(int)
```

    Size of training set: 14003 
    Size of test set: 3501
    

### `6.1b Add base distilBERT model`
The ktrain library has a Transformer object for convenience. It changes the data format to be usable by TensorFlow models so that we can create a learner object that will accept and train off of our data. The "classes" argument takes a list which is meant to represent the possible ways that the text classification object has the ability to classify text. We have started by making the classes the same as the previous experiment, "positive", "negative", and "neutral".

Here we are using the bas distilBERT model that is designed specifically to be fine-tuned for a downstream task.

We will use our heuristic function to validate the model. This means that the model we are training will attempt to match the heuristic as much as possible, which isn't necessarily ideal but it is a place to start.


```python
# Create a Transformer model
t = text.Transformer('distilbert-base-uncased', maxlen=120, classes=[1, 2, 3, 4, 5])
```

    /usr/local/lib/python3.10/dist-packages/ktrain/text/preprocessor.py:382: UserWarning: The class_names argument is replacing the classes argument. Please update your code.
      warnings.warn(
    


```python
trn1, val1, preproc = text.texts_from_df(train_df=train_df, text_column='comment', random_state=42,
                                       label_columns=['qualityRating'],
                                          val_df=test_df, lang='en',
                                          preprocess_mode='distilbert',
                                          maxlen=120, verbose=True,)

trn = t.preprocess_train(x_train, y_train)
val = t.preprocess_test(x_test, y_test)
```

    ['qualityRating_0', 'qualityRating_1', 'qualityRating_2', 'qualityRating_3', 'qualityRating_4']
           qualityRating_0  qualityRating_1  qualityRating_2  qualityRating_3  \
    7645               0.0              0.0              0.0              0.0   
    14168              0.0              0.0              0.0              0.0   
    2698               0.0              0.0              0.0              0.0   
    11271              0.0              0.0              0.0              0.0   
    688                0.0              1.0              0.0              0.0   
    
           qualityRating_4  
    7645               1.0  
    14168              1.0  
    2698               1.0  
    11271              1.0  
    688                0.0  
    ['qualityRating_0', 'qualityRating_1', 'qualityRating_2', 'qualityRating_3', 'qualityRating_4']
          qualityRating_0  qualityRating_1  qualityRating_2  qualityRating_3  \
    6029              1.0              0.0              0.0              0.0   
    1156              0.0              0.0              0.0              0.0   
    9820              1.0              0.0              0.0              0.0   
    2936              0.0              0.0              0.0              1.0   
    1602              0.0              0.0              0.0              0.0   
    
          qualityRating_4  
    6029              0.0  
    1156              1.0  
    9820              0.0  
    2936              0.0  
    1602              1.0  
    preprocessing train...
    language: en
    train sequence lengths:
    	mean : 18
    	95percentile : 30
    	99percentile : 33
    



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>







    Is Multi-Label? False
    preprocessing test...
    language: en
    test sequence lengths:
    	mean : 18
    	95percentile : 30
    	99percentile : 33
    



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>







    preprocessing train...
    language: en
    train sequence lengths:
    	mean : 18
    	95percentile : 30
    	99percentile : 33
    



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>







    Is Multi-Label? False
    preprocessing test...
    language: en
    test sequence lengths:
    	mean : 18
    	95percentile : 30
    	99percentile : 33
    



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>







### `6.1c Create Model`


```python
model = t.get_classifier()
```

### `6.1d Test for different Learning Rates`
ktrain reccomends using the lr_find() method to determine the optimal learning rate for each specific use case, so we will do that here. First, we create the learner object, then we attempt to find a learning rate.


```python
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=64)
learner.lr_find(show_plot=True, max_epochs=7)
```

    simulating training for different learning rates... this may take a few moments...
    Epoch 1/7
    218/218 [==============================] - 65s 235ms/step - loss: 1.5454 - accuracy: 0.4176
    Epoch 2/7
    218/218 [==============================] - 54s 247ms/step - loss: 1.2292 - accuracy: 0.5379
    Epoch 3/7
    218/218 [==============================] - 54s 246ms/step - loss: 0.9797 - accuracy: 0.6021
    Epoch 4/7
    218/218 [==============================] - 53s 244ms/step - loss: 1.2007 - accuracy: 0.5511
    Epoch 5/7
    218/218 [==============================] - 52s 240ms/step - loss: 1.4650 - accuracy: 0.4992
    Epoch 6/7
    218/218 [==============================] - 51s 235ms/step - loss: 1.3789 - accuracy: 0.5048
    Epoch 7/7
    218/218 [==============================] - 51s 235ms/step - loss: 1.7272 - accuracy: 0.4437
    
    
    done.
    Visually inspect loss plot and select learning rate associated with falling loss
    


    
![png](Main_Document_CSC425_RateMyProfessor_Sentiment_Analysis%20%282%29_files/Main_Document_CSC425_RateMyProfessor_Sentiment_Analysis%20%282%29_64_1.png)
    


### `6.1e Train Data`
From the loss graph, we can see that a good maximum learning rate is 1e-4, or 0.0001

It is important to note that training in this way only aims to estimate the heuristic function, and not the actual sentiment of the comments.


```python
learner.autofit(lr=1e-4, epochs=7)
```

    
    
    begin training using triangular learning rate policy with max lr of 0.0001...
    Epoch 1/15
    219/219 [==============================] - 57s 248ms/step - loss: 0.1146 - accuracy: 0.9624 - val_loss: 2.0650 - val_accuracy: 0.5895
    Epoch 2/15
    219/219 [==============================] - 55s 248ms/step - loss: 0.0991 - accuracy: 0.9684 - val_loss: 2.1363 - val_accuracy: 0.5944
    Epoch 3/15
    219/219 [==============================] - 55s 249ms/step - loss: 0.0871 - accuracy: 0.9709 - val_loss: 2.1314 - val_accuracy: 0.5978
    Epoch 4/15
    219/219 [==============================] - 55s 248ms/step - loss: 0.0848 - accuracy: 0.9714 - val_loss: 2.2540 - val_accuracy: 0.5984
    Epoch 5/15
    219/219 [==============================] - 54s 248ms/step - loss: 0.0780 - accuracy: 0.9754 - val_loss: 2.2018 - val_accuracy: 0.5893
    Epoch 6/15
    219/219 [==============================] - 54s 248ms/step - loss: 0.0772 - accuracy: 0.9745 - val_loss: 2.2779 - val_accuracy: 0.5901
    Epoch 7/15
    219/219 [==============================] - 55s 249ms/step - loss: 0.0747 - accuracy: 0.9750 - val_loss: 2.2620 - val_accuracy: 0.5904
    Epoch 8/15
    219/219 [==============================] - 54s 248ms/step - loss: 0.0687 - accuracy: 0.9779 - val_loss: 2.2016 - val_accuracy: 0.5904
    Epoch 9/15
    219/219 [==============================] - 54s 248ms/step - loss: 0.0628 - accuracy: 0.9787 - val_loss: 2.2722 - val_accuracy: 0.5795
    Epoch 10/15
    219/219 [==============================] - 54s 248ms/step - loss: 0.0568 - accuracy: 0.9811 - val_loss: 2.3058 - val_accuracy: 0.5867
    Epoch 11/15
    219/219 [==============================] - 54s 248ms/step - loss: 0.0508 - accuracy: 0.9821 - val_loss: 2.5425 - val_accuracy: 0.5944
    Epoch 12/15
    219/219 [==============================] - 54s 248ms/step - loss: 0.0497 - accuracy: 0.9848 - val_loss: 2.4223 - val_accuracy: 0.5947
    Epoch 13/15
    219/219 [==============================] - 54s 247ms/step - loss: 0.0489 - accuracy: 0.9831 - val_loss: 2.5522 - val_accuracy: 0.5873
    Epoch 14/15
    219/219 [==============================] - 54s 247ms/step - loss: 0.0468 - accuracy: 0.9854 - val_loss: 2.4750 - val_accuracy: 0.5810
    Epoch 15/15
    219/219 [==============================] - 54s 248ms/step - loss: 0.0465 - accuracy: 0.9845 - val_loss: 2.6479 - val_accuracy: 0.5813
    




    <keras.src.callbacks.History at 0x7c678f50cbb0>



Though the internal accuracy is high, the validation accuracy remained around 60% for the duration of training, meaning it is only slightly better than guessing. Still it is important to keep in mind that this model only attempts to guess what the heuristic function will say about a comment's ratings based off of the comment, not the actual sentiment of the comment, but it will still give an output of either positive, negative, or neutral if exported to a predictor object:


```python
naive_predictor = ktrain.get_predictor(learner.model, preproc=preproc)
```

### `6.1f Example Usage`


```python
print(naive_predictor.predict('hello'))
print(naive_predictor.predict("i love this class"))
print(naive_predictor.predict("awful horrible"))
print(naive_predictor.predict("it was okay"))
print(naive_predictor.predict("i really hate this bad class"))
```

    qualityRating_4
    qualityRating_4
    qualityRating_0
    qualityRating_2
    qualityRating_0
    

## `6.2 Training sentiment model on ChatGPT labeled dataset`

### `6.2a Split Data`


```python
# using our chatgpt_labeled_data now
train_df_gpt, test_df_gpt = train_test_split(chatgpt_labeled_data, test_size=0.075, random_state=40)

train_size = train_df_gpt.shape[0]
test_size = test_df_gpt.shape[0]

print("Size of training set:", train_size,"\n"+"Size of test set:", test_size)

x_train_gpt = train_df_gpt['comment'].to_numpy()
y_train_gpt = train_df_gpt['sentiment'].to_numpy().astype(str)

x_test_gpt = test_df_gpt['comment'].to_numpy()
y_test_gpt = test_df_gpt['sentiment'].to_numpy().astype(str)
```

    Size of training set: 462 
    Size of test set: 38
    

### `6.2b Add base distilBERT model`


```python
# Create a Transformer model
t_gpt = text.Transformer('distilbert-base-uncased', maxlen=120, classes=['Negative', 'Neutral', 'Positive'])
```

    /usr/local/lib/python3.10/dist-packages/ktrain/text/preprocessor.py:382: UserWarning: The class_names argument is replacing the classes argument. Please update your code.
      warnings.warn(
    


```python
trn_gpt, val_gpt, preproc_gpt = text.texts_from_df(train_df=train_df_gpt, text_column='comment', random_state=42,
                                       label_columns=['sentiment'],
                                          val_df=test_df_gpt, lang='en',
                                          preprocess_mode='distilbert',
                                          maxlen=120, verbose=True,)

# trn = t_gpt.preprocess_train(x_train_gpt, y_train_gpt)
# val = t_gpt.preprocess_test(x_test_gpt, y_test_gpt)
```

    ['Negative', 'Neutral', 'Positive']
           Negative  Neutral  Positive
    5954        1.0      0.0       0.0
    12737       0.0      1.0       0.0
    3741        0.0      0.0       1.0
    122         1.0      0.0       0.0
    9506        0.0      0.0       1.0
    ['Negative', 'Neutral', 'Positive']
           Negative  Neutral  Positive
    6902        1.0      0.0       0.0
    7669        0.0      1.0       0.0
    16881       0.0      1.0       0.0
    11255       0.0      0.0       1.0
    16409       0.0      1.0       0.0
    preprocessing train...
    language: en
    train sequence lengths:
    	mean : 18
    	95percentile : 29
    	99percentile : 32
    



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>







    Is Multi-Label? False
    preprocessing test...
    language: en
    test sequence lengths:
    	mean : 21
    	95percentile : 31
    	99percentile : 32
    



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>







### `6.2c Create Model`


```python
model_gpt = t_gpt.get_classifier()
```

    /usr/local/lib/python3.10/dist-packages/ktrain/text/preprocessor.py:392: UserWarning: The method preprocess_train was never called. You can disable this warning by setting preprocess_train_called=True.
      warnings.warn(
    

### `6.2d Test for different learning rates`


```python
learner_gpt = ktrain.get_learner(model_gpt, train_data=trn_gpt, val_data=val_gpt, batch_size=32)
learner_gpt.lr_find(show_plot=True, max_epochs=15)
```

    simulating training for different learning rates... this may take a few moments...
    Epoch 1/15
    14/14 [==============================] - 9s 132ms/step - loss: 1.1057 - accuracy: 0.2790
    Epoch 2/15
    14/14 [==============================] - 2s 134ms/step - loss: 1.0933 - accuracy: 0.4070
    Epoch 3/15
    14/14 [==============================] - 2s 126ms/step - loss: 1.0718 - accuracy: 0.4349
    Epoch 4/15
    14/14 [==============================] - 2s 127ms/step - loss: 1.0014 - accuracy: 0.5000
    Epoch 5/15
    14/14 [==============================] - 2s 126ms/step - loss: 0.6660 - accuracy: 0.7581
    Epoch 6/15
    14/14 [==============================] - 2s 126ms/step - loss: 0.5035 - accuracy: 0.8349
    Epoch 7/15
    14/14 [==============================] - 2s 126ms/step - loss: 0.6917 - accuracy: 0.7837
    Epoch 8/15
    14/14 [==============================] - 2s 126ms/step - loss: 1.1883 - accuracy: 0.4605
    Epoch 9/15
    14/14 [==============================] - 2s 126ms/step - loss: 1.1741 - accuracy: 0.3930
    Epoch 10/15
    14/14 [==============================] - 2s 127ms/step - loss: 1.1743 - accuracy: 0.4093
    Epoch 11/15
    14/14 [==============================] - 2s 126ms/step - loss: 1.1709 - accuracy: 0.4535
    Epoch 12/15
    14/14 [==============================] - 1s 80ms/step - loss: 66.2753 - accuracy: 0.3472
    
    
    done.
    Visually inspect loss plot and select learning rate associated with falling loss
    


    
![png](Main_Document_CSC425_RateMyProfessor_Sentiment_Analysis%20%282%29_files/Main_Document_CSC425_RateMyProfessor_Sentiment_Analysis%20%282%29_80_1.png)
    


### `6.2e Train Data`


```python
learner_gpt.autofit(lr=1e-4, epochs=15)
```

    
    
    begin training using triangular learning rate policy with max lr of 0.0001...
    Epoch 1/15
    15/15 [==============================] - 12s 241ms/step - loss: 1.0594 - accuracy: 0.4286 - val_loss: 0.8311 - val_accuracy: 0.7105
    Epoch 2/15
    15/15 [==============================] - 2s 134ms/step - loss: 0.8182 - accuracy: 0.6818 - val_loss: 0.7270 - val_accuracy: 0.6842
    Epoch 3/15
    15/15 [==============================] - 2s 133ms/step - loss: 0.5879 - accuracy: 0.7922 - val_loss: 0.5525 - val_accuracy: 0.7632
    Epoch 4/15
    15/15 [==============================] - 2s 133ms/step - loss: 0.4146 - accuracy: 0.8636 - val_loss: 0.6581 - val_accuracy: 0.7105
    Epoch 5/15
    15/15 [==============================] - 2s 133ms/step - loss: 0.2764 - accuracy: 0.9026 - val_loss: 0.8222 - val_accuracy: 0.7105
    Epoch 6/15
    15/15 [==============================] - 2s 133ms/step - loss: 0.2199 - accuracy: 0.9221 - val_loss: 0.6864 - val_accuracy: 0.7368
    Epoch 7/15
    15/15 [==============================] - 2s 134ms/step - loss: 0.1335 - accuracy: 0.9610 - val_loss: 1.6246 - val_accuracy: 0.6316
    Epoch 8/15
    15/15 [==============================] - 2s 134ms/step - loss: 0.2084 - accuracy: 0.9459 - val_loss: 0.8581 - val_accuracy: 0.6842
    Epoch 9/15
    15/15 [==============================] - 2s 133ms/step - loss: 0.1294 - accuracy: 0.9740 - val_loss: 1.1992 - val_accuracy: 0.6316
    Epoch 10/15
    15/15 [==============================] - 2s 133ms/step - loss: 0.0713 - accuracy: 0.9827 - val_loss: 1.1163 - val_accuracy: 0.7105
    Epoch 11/15
    15/15 [==============================] - 2s 134ms/step - loss: 0.0515 - accuracy: 0.9870 - val_loss: 1.3096 - val_accuracy: 0.6579
    Epoch 12/15
    15/15 [==============================] - 2s 133ms/step - loss: 0.0709 - accuracy: 0.9848 - val_loss: 1.3796 - val_accuracy: 0.6579
    Epoch 13/15
    15/15 [==============================] - 2s 134ms/step - loss: 0.0587 - accuracy: 0.9892 - val_loss: 1.1023 - val_accuracy: 0.6842
    Epoch 14/15
    15/15 [==============================] - 2s 135ms/step - loss: 0.0340 - accuracy: 0.9935 - val_loss: 1.1571 - val_accuracy: 0.6842
    Epoch 15/15
    15/15 [==============================] - 2s 135ms/step - loss: 0.0253 - accuracy: 0.9957 - val_loss: 1.1387 - val_accuracy: 0.7368
    




    <keras.src.callbacks.History at 0x7c69536bb8e0>



### `6.2f Example usage `


```python
gpt_model = ktrain.get_predictor(learner_gpt.model, preproc=preproc_gpt)
```


```python
gpt_model.predict("it was okay")
```




    'Neutral'



### `6.2g Thoughts and discussion`

With only a subset of reviews, this modal obtained a peak accuracy at about 73-76% val_accuracy. We believe there is promise for using openAI api to auto label a dataset, but we simply didn't have enough time to comprise enough labeled data.

We believe with more time, we could possibly emulate a more accurate model from this type of data.

# `7. Analysis and Conclusions`

## `7.1 Comparing Pre-trained model, Custom trained model, and Naive Heuristic`

Here will use our three models and perform sentiment analysis on our science and humanities professors.


```python
# we can access science_professors and humanities_professors as dataframes
# each dataframe has a comment => str, qualityRating => int

# example usage:

# using pre-trained model
pretrained = SentimentAnalyzer()
print(list(pretrained.predict("she was ok"))[0].capitalize())

# using custom model
print(gpt_model.predict("she was ok"))

# using heuristic
review = {'qualityRating' : 5, 'difficultyRating' : 0}
result = sentiment_heuristic(review)
print(result)
```

    Positive
    Neutral
    Positive
    


```python
def apply_pretrained_model(comment):
    return list(pretrained.predict(comment))[0].capitalize()

def apply_custom_model(comment):
    return gpt_model.predict(comment)

def apply_heuristic(row):
    return sentiment_heuristic({'qualityRating': row['qualityRating'], 'difficultyRating': row['difficultyRating']})

```


```python
!pip install tqdm
```

    Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (4.66.1)
    


```python
from tqdm import tqdm

def analyze_sentiment(dataframe):
    dataframe['pretrained_sentiment'] = dataframe['comment'].apply(apply_pretrained_model)
    dataframe['custom_model_sentiment'] = dataframe['comment'].apply(apply_custom_model)
    dataframe['heuristic_sentiment'] = dataframe.apply(apply_heuristic, axis=1)

```


```python
analyze_sentiment(science_professors)
analyze_sentiment(humanities_professors)
```

    /usr/local/lib/python3.10/dist-packages/transformers/pipelines/base.py:1101: UserWarning: You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset
      warnings.warn(
    

    

    /usr/local/lib/python3.10/dist-packages/transformers/pipelines/base.py:1101: UserWarning: You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset
      warnings.warn(
    

    

## `7.2 Displaying results`

### `7.2a Show what percentage of time all three models agree `


```python
science_professors
```


```python
def sentiment_agreement(row):
    return row['pretrained_sentiment'] == row['custom_model_sentiment'] == row['heuristic_sentiment']
```


```python
science_professors['agreement'] = science_professors.apply(sentiment_agreement, axis=1)
humanities_professors['agreement'] = humanities_professors.apply(sentiment_agreement, axis=1)
```


```python
science_agreement_pct = science_professors['agreement'].mean() * 100
humanities_agreement_pct = humanities_professors['agreement'].mean() * 100
```


```python
import matplotlib.pyplot as plt

categories = ['Science Professors', 'Humanities Professors']
agreement_percentages = [science_agreement_pct, humanities_agreement_pct]

plt.bar(categories, agreement_percentages)
plt.title('Percentage of Sentiment Agreement Between Analyzers')
plt.ylabel('Agreement Percentage (%)')
plt.show()
```


    
![png](Main_Document_CSC425_RateMyProfessor_Sentiment_Analysis%20%282%29_files/Main_Document_CSC425_RateMyProfessor_Sentiment_Analysis%20%282%29_101_0.png)
    



```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sentiment_distribution(df, column, title):
    plt.figure(figsize=(8, 5))
    sns.countplot(x=column, data=df, order=['Positive', 'Neutral', 'Negative'])
    plt.title(title)
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

```

How much does our custom model agree with the pretrained model?


```python
science_agreement_pct = science_professors.apply(lambda row: row['pretrained_sentiment'] == row['custom_model_sentiment'], axis=1).mean() * 100
humanities_agreement_pct = humanities_professors.apply(lambda row: row['pretrained_sentiment'] == row['custom_model_sentiment'], axis=1).mean() * 100
```


```python
print(f"Agreement between pretrained and custom models for Science Professors: {science_agreement_pct:.2f}%")
print(f"Agreement between pretrained and custom models for Humanities Professors: {humanities_agreement_pct:.2f}%")

```

    Agreement between pretrained and custom models for Science Professors: 65.57%
    Agreement between pretrained and custom models for Humanities Professors: 70.16%
    

### `7.2b Visualize the sentiment distribution for each method`


```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_combined_sentiment_distribution(df, title):
    plt.figure(figsize=(10, 6))

    # Setting up the bar positions
    bar_width = 0.2
    r1 = np.arange(len(df['pretrained_sentiment'].value_counts()))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # Plotting the bars
    plt.bar(r1, df['pretrained_sentiment'].value_counts().sort_index(), width=bar_width, label='Pretrained Model')
    plt.bar(r2, df['custom_model_sentiment'].value_counts().sort_index(), width=bar_width, label='Custom Model')
    plt.bar(r3, df['heuristic_sentiment'].value_counts().sort_index(), width=bar_width, label='Heuristic')

    # Adding labels and title
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks([r + bar_width for r in range(len(df['pretrained_sentiment'].value_counts()))],
               df['pretrained_sentiment'].value_counts().sort_index().index)

    plt.legend()
    plt.show()

# Example usage (replace science_professors and humanities_professors with actual dataframes)
plot_combined_sentiment_distribution(science_professors, 'Combined Sentiment Distribution for Science Professors')
plot_combined_sentiment_distribution(humanities_professors, 'Combined Sentiment Distribution for Humanities Professors')

```


    
![png](Main_Document_CSC425_RateMyProfessor_Sentiment_Analysis%20%282%29_files/Main_Document_CSC425_RateMyProfessor_Sentiment_Analysis%20%282%29_107_0.png)
    



    
![png](Main_Document_CSC425_RateMyProfessor_Sentiment_Analysis%20%282%29_files/Main_Document_CSC425_RateMyProfessor_Sentiment_Analysis%20%282%29_107_1.png)
    


## `7.3 Discussion`

From these results, we can see that all three models agree with eachother about 30% of the time on science professors, and 34% of the time on humanities professors.

Our assumption is that the pretrained model is likely the most accurate model. In a broader context, it would be ideal to measure the performance of this pretrained model to provide more perspective.

We can see that our heuristic function was pretty confident at capturing positive reviews, but fell short when it came to neutral and negative reviews.

Our custom model aimed to be closer to the more accurate pre-trained model, but appeared to categorize some positive reviews as neutral, and some neutral as negative, in comparison to the pretrained model.



