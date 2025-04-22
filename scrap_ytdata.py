#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[12]:


import requests
import pandas as pd

API_KEY = 'AIzaSyAIQkrXK4Pxm57LnxIQx3AuDsx9-XGVyAQ' # 🔑 Replace with your actual YouTube Data API key
REGION_CODE = 'IN'       # 🇮🇳 Region code for India

url="your google api"
response = requests.get(url)
data = response.json()

videos = []

for item in data['items']:
    snippet = item['snippet']
    stats = item['statistics']
    
    videos.append({
        'title': snippet['title'],
        'channel': snippet['channelTitle'],
        'categoryId': snippet['categoryId'],
        'publishedAt': snippet['publishedAt'],
        'viewCount': stats.get('viewCount'),
        'likeCount': stats.get('likeCount'),
        'commentCount': stats.get('commentCount')
    })

df = pd.DataFrame(videos)

# Save to CSV
df.to_csv(r'D:\Quadratic\indian_trnding_yt.csv', index=False)

print("Top 50 trending videos in India saved to youtube_trending_india.csv ✅")


# In[ ]:




