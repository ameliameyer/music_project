# Spotify EDA of Personal Music Taste

from IPython.display import Audio
sound_file = 'gtr-jazz.wav'
Audio(sound_file, autoplay=True)

## Import Dependencies


```python
# import dependencies for authentication
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv 
```


```python
# import dependencies for eda
import numpy as np
import pandas as pd
import altair as alt

import seaborn as sns
import plotly.express as px 
import matplotlib.pyplot as plt
%matplotlib inline
    
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings("ignore")

```

## EDA Begins

FIXME: do I want to extend this to include december? or at least more of it?


```python
# read in liked songs from nov92022 and prior
liked_songs_clean = pd.read_csv('data/liked_songs_nov92022_sec.csv')
```

#### Check for Missing Values


```python
# check for the total number of na values
liked_songs_clean.isna().sum()
```




    id                  0
    name                0
    artists             0
    duration_s          0
    popularity          0
    added_at            0
    acousticness        0
    speechiness         0
    key                 0
    liveness            0
    instrumentalness    0
    energy              0
    tempo               0
    time_signature      0
    loudness            0
    danceability        0
    valence             0
    genre_list          0
    genre               0
    pitch_class         0
    dtype: int64



Luckily, we don't have any missing values in our dataset. But we should take note that we changed empty genre lists to `other` when it could also be `unknown` or `unestablished`. We can see what proportion of items are listed as `other`.


```python
# calculate the proportion of tracks with [] as the genre list
other_proportion = len(liked_songs_clean[liked_songs_clean.genre == 'other'])*100/len(liked_songs_clean)
print(str(round(other_proportion,2)) + "% of the data is listed as 'other'")
```

    6.87% of the data is listed as 'other'
    

That's not too high but we should keep it in mind when we do genre analysis.

#### Change the Datetime Format

That makes it easier to read. We can also use that format in the future if we want to extract the parts of the added_at variable: year, month, day, time


```python
# change the datetime format
liked_songs_clean.added_at = pd.to_datetime(liked_songs_clean.added_at)
```


```python
# print the first 3 rows in our dataset
liked_songs_clean.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
      <th>artists</th>
      <th>duration_s</th>
      <th>popularity</th>
      <th>added_at</th>
      <th>acousticness</th>
      <th>speechiness</th>
      <th>key</th>
      <th>liveness</th>
      <th>instrumentalness</th>
      <th>energy</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>loudness</th>
      <th>danceability</th>
      <th>valence</th>
      <th>genre_list</th>
      <th>genre</th>
      <th>pitch_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3XyMM4crvrhOjAbMGiBDck</td>
      <td>Can't Come Back</td>
      <td>Kevin Garrett</td>
      <td>249.400</td>
      <td>23</td>
      <td>2022-11-06 04:18:28+00:00</td>
      <td>0.7010</td>
      <td>0.0278</td>
      <td>5</td>
      <td>0.0940</td>
      <td>0.00471</td>
      <td>0.245</td>
      <td>135.802</td>
      <td>4</td>
      <td>-9.977</td>
      <td>0.531</td>
      <td>0.287</td>
      <td>['nyc pop', 'pittsburgh indie']</td>
      <td>nyc pop</td>
      <td>F/Fa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0JbSghVDghtFEurrSO8JrC</td>
      <td>Country Girl (Shake It For Me)</td>
      <td>Luke Bryan</td>
      <td>225.560</td>
      <td>76</td>
      <td>2022-11-04 03:06:05+00:00</td>
      <td>0.0293</td>
      <td>0.0462</td>
      <td>2</td>
      <td>0.0834</td>
      <td>0.00000</td>
      <td>0.904</td>
      <td>105.970</td>
      <td>4</td>
      <td>-4.532</td>
      <td>0.645</td>
      <td>0.671</td>
      <td>['contemporary country', 'country', 'country r...</td>
      <td>contemporary country</td>
      <td>D/Re</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0YeLqMwKztVZ1MIZ4oYW4e</td>
      <td>The Wolf</td>
      <td>The Spencer Lee Band</td>
      <td>174.653</td>
      <td>50</td>
      <td>2022-11-04 03:03:51+00:00</td>
      <td>0.0932</td>
      <td>0.1860</td>
      <td>9</td>
      <td>0.1430</td>
      <td>0.00000</td>
      <td>0.912</td>
      <td>100.941</td>
      <td>4</td>
      <td>-4.746</td>
      <td>0.777</td>
      <td>0.659</td>
      <td>['modern funk']</td>
      <td>modern funk</td>
      <td>A/La</td>
    </tr>
  </tbody>
</table>
</div>



Now that our dataset looks a little better, we'll start to explore the varibales and data provided.

### Basic Details


```python
liked_songs_clean.shape
```




    (4786, 20)



There are 4786 rows and 20 columns in this dataset. That's a lot of songs!   

This already brings up some questions:   
1. What do all of these columns mean?
2. What is the range of these columns
3. How many songs did I add per year?
4. Am I more likely to add songs at a particular time of day/on a particular day/month/year?
5. How many songs do I have per artist?
6. Which genre/artist do I listen to the most (have I added the most songs for)?
7. Do I the songs I listen to tend to be more popular or less?
8. How many songs do I listen to that have the same name?
9. Is there a pattern to the audio features?

We've got some gears turning here. Let's start with the first two. We can use `liked_songs_clean.describe` and the Spotify API dashboard to create a table of our columns names and a brief description.


```python
# get column count, mean, std, etc.
liked_songs_clean.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>duration_s</th>
      <th>popularity</th>
      <th>acousticness</th>
      <th>speechiness</th>
      <th>key</th>
      <th>liveness</th>
      <th>instrumentalness</th>
      <th>energy</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>loudness</th>
      <th>danceability</th>
      <th>valence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4786.000000</td>
      <td>4786.000000</td>
      <td>4786.000000</td>
      <td>4786.000000</td>
      <td>4786.000000</td>
      <td>4786.000000</td>
      <td>4786.000000</td>
      <td>4786.000000</td>
      <td>4786.000000</td>
      <td>4786.000000</td>
      <td>4786.000000</td>
      <td>4786.000000</td>
      <td>4786.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>212.116574</td>
      <td>44.450898</td>
      <td>0.391306</td>
      <td>0.075392</td>
      <td>5.055788</td>
      <td>0.170167</td>
      <td>0.031683</td>
      <td>0.528987</td>
      <td>119.553181</td>
      <td>3.892186</td>
      <td>-8.006901</td>
      <td>0.576894</td>
      <td>0.438362</td>
    </tr>
    <tr>
      <th>std</th>
      <td>53.094822</td>
      <td>23.913458</td>
      <td>0.337102</td>
      <td>0.084115</td>
      <td>3.518814</td>
      <td>0.128310</td>
      <td>0.146487</td>
      <td>0.234362</td>
      <td>30.518612</td>
      <td>0.411529</td>
      <td>4.130290</td>
      <td>0.149951</td>
      <td>0.227007</td>
    </tr>
    <tr>
      <th>min</th>
      <td>34.050000</td>
      <td>0.000000</td>
      <td>0.000006</td>
      <td>0.022500</td>
      <td>0.000000</td>
      <td>0.019300</td>
      <td>0.000000</td>
      <td>0.000280</td>
      <td>39.120000</td>
      <td>1.000000</td>
      <td>-39.995000</td>
      <td>0.067600</td>
      <td>0.029100</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>182.303500</td>
      <td>30.000000</td>
      <td>0.060400</td>
      <td>0.033800</td>
      <td>2.000000</td>
      <td>0.097300</td>
      <td>0.000000</td>
      <td>0.348000</td>
      <td>95.011500</td>
      <td>4.000000</td>
      <td>-9.728750</td>
      <td>0.479250</td>
      <td>0.255000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>207.819500</td>
      <td>48.000000</td>
      <td>0.309000</td>
      <td>0.044400</td>
      <td>5.000000</td>
      <td>0.118000</td>
      <td>0.000001</td>
      <td>0.531000</td>
      <td>118.079500</td>
      <td>4.000000</td>
      <td>-7.147500</td>
      <td>0.583000</td>
      <td>0.412000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>237.000000</td>
      <td>63.000000</td>
      <td>0.722750</td>
      <td>0.074550</td>
      <td>8.000000</td>
      <td>0.193000</td>
      <td>0.000151</td>
      <td>0.715000</td>
      <td>140.047500</td>
      <td>4.000000</td>
      <td>-5.285000</td>
      <td>0.683750</td>
      <td>0.607000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>780.439000</td>
      <td>100.000000</td>
      <td>0.996000</td>
      <td>0.929000</td>
      <td>11.000000</td>
      <td>0.976000</td>
      <td>0.973000</td>
      <td>0.986000</td>
      <td>218.365000</td>
      <td>5.000000</td>
      <td>-1.148000</td>
      <td>0.981000</td>
      <td>0.974000</td>
    </tr>
  </tbody>
</table>
</div>



We can make a few observations here:

The average popularity of my songs is 44.45. With a popularity score range of 0 to 100, that's not very high at all! We should compare this with the distribution of popularity and the median.  

My average song tempo is 119.55 which is classified as a moderate tempo. FIXME It's also considered [sexiest](https://dancingastronaut.com/2020/07/research-reveals-119-bpm-to-be-the-sexiest-of-all-tempos/) tempo of all time. Cool musing.


```python
# print the data tpe of each column
liked_songs_clean.dtypes
```




    id                               object
    name                             object
    artists                          object
    duration_s                      float64
    popularity                        int64
    added_at            datetime64[ns, UTC]
    acousticness                    float64
    speechiness                     float64
    key                               int64
    liveness                        float64
    instrumentalness                float64
    energy                          float64
    tempo                           float64
    time_signature                    int64
    loudness                        float64
    danceability                    float64
    valence                         float64
    genre_list                       object
    genre                            object
    pitch_class                      object
    dtype: object



FIXME: is there a way to transform the data types to string instead of object?

Let's find the range of my dataset:


```python
# print the first and last date in the dataset
print("The first date I added a song to my `liked` list was " + str(liked_songs_clean.added_at.min()))
print("The last date I added a song to my `liked` list was " + str(liked_songs_clean.added_at.max()))
```

    The first date I added a song to my `liked` list was 2017-03-03 19:50:46+00:00
    The last date I added a song to my `liked` list was 2022-11-06 04:18:28+00:00
    

### Column Descriptions

| variable | description | range |
| --- | --- | --- |
| acousticness | is track acoustic? | confidence measure, [0.0,1.0] |
| added_at | date added | datetime format, 3/3/2017-10/7/2022|   
| artists | name of artist associate with song | --- |   
| danceability | how suitable the track is for dancing | float, [0.0,1.0] |
| duration_s | duration of track in seconds | float, [0.0,1.0] |
| energy | perceptual measure of intensity and activity | float, [0.0,1.0] |
| genre | top genre associated with artist of song | --- |
| genre_list | list of genres associated with artist of song | --- |
| pitch_class | pitch classes we took from the keys | --- |
| id | id number associated with song | --- |
| instrumentalness | predicts likelihood track contains no vocal content | float, [0.0,1.0] |
| key | key of track | int, standard Pitch Class notation; [-1 when no value detected,11] |
| liveliness | presence of audience in recording | float, [0.0,1.0] |
| loudness | overall loudness of track in decibels | float, [-60,0] |
| name | name of songs | --- |
| popularity | how popular a song is | [0.0,100.0] |
| speechiness | presence of spoken words in a track | float, [0.0,1.0] |
| tempo | tempo of track in beats per minute | float, [0.0,...] |
| time_signature | estimated time signature | int, [3,7] indicating "3/4" to "7/4"|
| valence | musical positivesness conveyed by a track | float, [0.0,1.0] |

Let's add in new columns for year, month, and day for later analysis


```python
# Add new columns for year month and day taken from the datetime column
liked_songs_clean['year'] = pd.DatetimeIndex(liked_songs_clean['added_at']).year
liked_songs_clean['month'] = pd.DatetimeIndex(liked_songs_clean['added_at']).month
liked_songs_clean['day'] = pd.DatetimeIndex(liked_songs_clean['added_at']).day

# view the first five rows of the columns we just added
liked_songs_clean[['year','month','day']].head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022</td>
      <td>11</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022</td>
      <td>11</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022</td>
      <td>11</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022</td>
      <td>11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022</td>
      <td>11</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Now, let's find how many songs I've added to my Liked Songs per year.


```python
# how many songs per year? 
liked_songs_clean.year.value_counts()
```




    2020    2112
    2022     907
    2021     842
    2019     711
    2017     208
    2018       6
    Name: year, dtype: int64



Wow! That's interesting. 2020 is currently the year I added the most songs, followed by 2022 and 2021. I wonder how this would look if I added in my Liked Songs from the rest of Nov and Dec of 2022. 

It would be good to visualize this later on and see the distribution of songs I added in each month based on the year. Maybe there's a trend there!

Something else we should question is why there were only 6 songs added in 2018 after there were 208 songs added in 2017. 


```python
liked_songs_clean[liked_songs_clean.year == 2018]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
      <th>artists</th>
      <th>duration_s</th>
      <th>popularity</th>
      <th>added_at</th>
      <th>acousticness</th>
      <th>speechiness</th>
      <th>key</th>
      <th>liveness</th>
      <th>...</th>
      <th>time_signature</th>
      <th>loudness</th>
      <th>danceability</th>
      <th>valence</th>
      <th>genre_list</th>
      <th>genre</th>
      <th>pitch_class</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4572</th>
      <td>1M2RkUUJ9wXRsW3jN1MCHS</td>
      <td>4 Impromptus, Op.90, D.899: No.4 in A Flat Maj...</td>
      <td>Franz Schubert</td>
      <td>497.773</td>
      <td>0</td>
      <td>2018-12-12 21:32:21+00:00</td>
      <td>0.9870</td>
      <td>0.0398</td>
      <td>8</td>
      <td>0.1260</td>
      <td>...</td>
      <td>3</td>
      <td>-27.015</td>
      <td>0.343</td>
      <td>0.0393</td>
      <td>['classical', 'classical era', 'early romantic...</td>
      <td>classical</td>
      <td>G#/Sol sost.</td>
      <td>2018</td>
      <td>12</td>
      <td>12</td>
    </tr>
    <tr>
      <th>4573</th>
      <td>5Ck3e6z1bcSCes5gdo0Ddt</td>
      <td>Bed Peace</td>
      <td>Jhené Aiko</td>
      <td>256.079</td>
      <td>61</td>
      <td>2018-11-18 02:05:31+00:00</td>
      <td>0.2690</td>
      <td>0.0575</td>
      <td>6</td>
      <td>0.1350</td>
      <td>...</td>
      <td>4</td>
      <td>-7.811</td>
      <td>0.628</td>
      <td>0.2990</td>
      <td>['pop', 'r&amp;b', 'urban contemporary']</td>
      <td>pop</td>
      <td>F#/Fa sost.</td>
      <td>2018</td>
      <td>11</td>
      <td>18</td>
    </tr>
    <tr>
      <th>4574</th>
      <td>1boXOL0ua7N2iCOUVI1p9F</td>
      <td>Japanese Denim</td>
      <td>Daniel Caesar</td>
      <td>270.846</td>
      <td>77</td>
      <td>2018-11-18 02:05:28+00:00</td>
      <td>0.0905</td>
      <td>0.0379</td>
      <td>3</td>
      <td>0.0842</td>
      <td>...</td>
      <td>3</td>
      <td>-8.818</td>
      <td>0.707</td>
      <td>0.3450</td>
      <td>['canadian contemporary r&amp;b', 'pop', 'r&amp;b']</td>
      <td>canadian contemporary r&amp;b</td>
      <td>D/Re sost.</td>
      <td>2018</td>
      <td>11</td>
      <td>18</td>
    </tr>
    <tr>
      <th>4575</th>
      <td>5NijSs5dAwaIybq1GaRTIe</td>
      <td>Poison</td>
      <td>Brent Faiyaz</td>
      <td>212.992</td>
      <td>74</td>
      <td>2018-11-14 22:28:43+00:00</td>
      <td>0.7110</td>
      <td>0.0650</td>
      <td>2</td>
      <td>0.0859</td>
      <td>...</td>
      <td>3</td>
      <td>-10.102</td>
      <td>0.689</td>
      <td>0.3820</td>
      <td>['dmv rap', 'hip hop', 'pop', 'r&amp;b', 'rap']</td>
      <td>dmv rap</td>
      <td>D/Re</td>
      <td>2018</td>
      <td>11</td>
      <td>14</td>
    </tr>
    <tr>
      <th>4576</th>
      <td>5JcmJLvm65rjRbg06FIrlE</td>
      <td>Debussy: Arabesque</td>
      <td>Michael Charles Clark</td>
      <td>272.352</td>
      <td>2</td>
      <td>2018-05-25 20:38:45+00:00</td>
      <td>0.9880</td>
      <td>0.0328</td>
      <td>4</td>
      <td>0.1210</td>
      <td>...</td>
      <td>5</td>
      <td>-14.151</td>
      <td>0.335</td>
      <td>0.1030</td>
      <td>[]</td>
      <td>other</td>
      <td>E/Mi</td>
      <td>2018</td>
      <td>5</td>
      <td>25</td>
    </tr>
    <tr>
      <th>4577</th>
      <td>7tFiyTwD0nx5a1eklYtX2J</td>
      <td>Bohemian Rhapsody - Remastered 2011</td>
      <td>Queen</td>
      <td>354.320</td>
      <td>72</td>
      <td>2018-02-24 14:34:19+00:00</td>
      <td>0.2880</td>
      <td>0.0536</td>
      <td>0</td>
      <td>0.2430</td>
      <td>...</td>
      <td>4</td>
      <td>-9.961</td>
      <td>0.392</td>
      <td>0.2280</td>
      <td>['classic rock', 'glam rock', 'rock']</td>
      <td>classic rock</td>
      <td>C/Do</td>
      <td>2018</td>
      <td>2</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
<p>6 rows × 23 columns</p>
</div>



Hmmm that didn't tell me anything. But there's no way to know what actually went down during this year - maybe I didn't use spotify as much or wasn't adding music as often that year. 

Now we'll address which artists I listen to the most


```python
# my top ten added artists overall
x = liked_songs_clean.artists.value_counts()[:25] 
x = x.reset_index()
x.columns = ['artist', 'number_of_songs']

ax = sns.barplot(x = 'number_of_songs', y = 'artist', data = x)
# ax.tick_params(axis='x', rotation=90)
```


    
![png](output_files/output_38_0.png)
    


Taylor Swift and Ed Sheeran have the greatest number of songs in my Liked Songs dataset. It would be interesting to see if this is true for this year and to compare the proportion of songs I have in my dataset/the number of total songs by that artist


```python
# my top ten artists for 2022 (until nov 9th)
x =  liked_songs_clean[liked_songs_clean.year == 2022]['artists'].value_counts()[:25]
x = x.reset_index()
x.columns = ['artist', 'number_of_songs']

ax = sns.barplot(x = 'number_of_songs', y = 'artist', data = x)
# ax.tick_params(axis='x', rotation=90)
```


    
![png](output_files/output_40_0.png)
    


Overall, I've added the most Taylor Swift, but in 2022, I've added the most 5 Seconds of Summer. It's also interesting to note that Ed Sheeran isn't in my top 10 for 2022, but he is in my top 10 overall. 


```python
# count of genres / what genres I listen to most often
liked_songs_clean.genre.value_counts()[:20]
```




    dance pop                    749
    pop                          703
    alt z                        391
    other                        329
    boy band                     222
    acoustic pop                 146
    canadian pop                 123
    canadian contemporary r&b    117
    folk-pop                      93
    bedroom pop                   92
    modern rock                   73
    neo mellow                    66
    indie folk                    61
    art pop                       58
    indie pop                     58
    nyc pop                       56
    british soul                  48
    hollywood                     39
    classical                     37
    adult standards               35
    Name: genre, dtype: int64



There's a lot of pop in there - completely expected. I can piece some of the other genres together, but alt z? So let's take a look at what kinds of songs are deemed `alt z`


```python
liked_songs_clean[liked_songs_clean.genre == 'alt z'].head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
      <th>artists</th>
      <th>duration_s</th>
      <th>popularity</th>
      <th>added_at</th>
      <th>acousticness</th>
      <th>speechiness</th>
      <th>key</th>
      <th>liveness</th>
      <th>...</th>
      <th>time_signature</th>
      <th>loudness</th>
      <th>danceability</th>
      <th>valence</th>
      <th>genre_list</th>
      <th>genre</th>
      <th>pitch_class</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25</th>
      <td>0eQJy4VAW7AkhKIHzXx3jG</td>
      <td>The First One</td>
      <td>Astrid S</td>
      <td>188.649</td>
      <td>44</td>
      <td>2022-10-20 04:40:59+00:00</td>
      <td>0.390</td>
      <td>0.1600</td>
      <td>7</td>
      <td>0.1030</td>
      <td>...</td>
      <td>4</td>
      <td>-8.304</td>
      <td>0.538</td>
      <td>0.183</td>
      <td>['alt z', 'dance pop', 'norwegian pop', 'pop',...</td>
      <td>alt z</td>
      <td>G/Sol</td>
      <td>2022</td>
      <td>10</td>
      <td>20</td>
    </tr>
    <tr>
      <th>26</th>
      <td>46ydq5g3k17iLJs3qMDvO6</td>
      <td>Hurts So Good</td>
      <td>Astrid S</td>
      <td>208.728</td>
      <td>69</td>
      <td>2022-10-20 04:40:55+00:00</td>
      <td>0.084</td>
      <td>0.0586</td>
      <td>7</td>
      <td>0.0957</td>
      <td>...</td>
      <td>4</td>
      <td>-5.027</td>
      <td>0.675</td>
      <td>0.378</td>
      <td>['alt z', 'dance pop', 'norwegian pop', 'pop',...</td>
      <td>alt z</td>
      <td>G/Sol</td>
      <td>2022</td>
      <td>10</td>
      <td>20</td>
    </tr>
    <tr>
      <th>77</th>
      <td>5QhJorJmjqYfETfqgBtGg3</td>
      <td>Already Over</td>
      <td>Sabrina Carpenter</td>
      <td>170.827</td>
      <td>64</td>
      <td>2022-10-19 17:32:04+00:00</td>
      <td>0.434</td>
      <td>0.5430</td>
      <td>4</td>
      <td>0.1210</td>
      <td>...</td>
      <td>4</td>
      <td>-7.109</td>
      <td>0.546</td>
      <td>0.468</td>
      <td>['alt z', 'dance pop', 'pop', 'post-teen pop',...</td>
      <td>alt z</td>
      <td>E/Mi</td>
      <td>2022</td>
      <td>10</td>
      <td>19</td>
    </tr>
    <tr>
      <th>79</th>
      <td>6BgOYYhN3yzY3GzaUv3b7T</td>
      <td>Fun While It Lasted</td>
      <td>Ashe</td>
      <td>146.436</td>
      <td>50</td>
      <td>2022-10-19 06:27:33+00:00</td>
      <td>0.834</td>
      <td>0.0316</td>
      <td>9</td>
      <td>0.1460</td>
      <td>...</td>
      <td>4</td>
      <td>-7.590</td>
      <td>0.494</td>
      <td>0.219</td>
      <td>['alt z', 'electropop', 'pop', 'post-teen pop'...</td>
      <td>alt z</td>
      <td>A/La</td>
      <td>2022</td>
      <td>10</td>
      <td>19</td>
    </tr>
    <tr>
      <th>80</th>
      <td>7LDwVpYKLQBJWS6YO7tsJx</td>
      <td>Count On Me</td>
      <td>Ashe</td>
      <td>219.658</td>
      <td>50</td>
      <td>2022-10-19 06:27:32+00:00</td>
      <td>0.208</td>
      <td>0.0662</td>
      <td>9</td>
      <td>0.3310</td>
      <td>...</td>
      <td>4</td>
      <td>-7.197</td>
      <td>0.440</td>
      <td>0.357</td>
      <td>['alt z', 'electropop', 'pop', 'post-teen pop'...</td>
      <td>alt z</td>
      <td>A/La</td>
      <td>2022</td>
      <td>10</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



Let's check the popularity of the music I listen to. 


```python
liked_songs_clean[['added_at','name', 'artists', 'popularity', 'genre']].sort_values('popularity', ascending=False)[:20]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>added_at</th>
      <th>name</th>
      <th>artists</th>
      <th>popularity</th>
      <th>genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>187</th>
      <td>2022-09-23 05:42:14+00:00</td>
      <td>Unholy (feat. Kim Petras)</td>
      <td>Sam Smith</td>
      <td>100</td>
      <td>dance pop</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2022-10-21 16:15:23+00:00</td>
      <td>Anti-Hero</td>
      <td>Taylor Swift</td>
      <td>96</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>132</th>
      <td>2022-10-08 03:32:21+00:00</td>
      <td>I Ain't Worried</td>
      <td>OneRepublic</td>
      <td>95</td>
      <td>piano rock</td>
    </tr>
    <tr>
      <th>795</th>
      <td>2022-04-01 00:18:10+00:00</td>
      <td>As It Was</td>
      <td>Harry Styles</td>
      <td>93</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>2296</th>
      <td>2020-10-01 00:25:25+00:00</td>
      <td>Another Love</td>
      <td>Tom Odell</td>
      <td>92</td>
      <td>chill pop</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2022-10-21 17:21:03+00:00</td>
      <td>Lavender Haze</td>
      <td>Taylor Swift</td>
      <td>92</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>535</th>
      <td>2022-05-24 05:26:10+00:00</td>
      <td>Late Night Talking</td>
      <td>Harry Styles</td>
      <td>91</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>613</th>
      <td>2022-05-20 04:14:23+00:00</td>
      <td>As It Was</td>
      <td>Harry Styles</td>
      <td>91</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2022-10-22 04:39:33+00:00</td>
      <td>Snow On The Beach (feat. Lana Del Rey)</td>
      <td>Taylor Swift</td>
      <td>91</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>2397</th>
      <td>2020-09-10 00:43:39+00:00</td>
      <td>Blinding Lights</td>
      <td>The Weeknd</td>
      <td>90</td>
      <td>canadian contemporary r&amp;b</td>
    </tr>
    <tr>
      <th>358</th>
      <td>2022-07-17 01:45:35+00:00</td>
      <td>Bad Habit</td>
      <td>Steve Lacy</td>
      <td>90</td>
      <td>afrofuturism</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2022-10-22 04:40:42+00:00</td>
      <td>Midnight Rain</td>
      <td>Taylor Swift</td>
      <td>90</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2022-10-22 04:05:17+00:00</td>
      <td>Maroon</td>
      <td>Taylor Swift</td>
      <td>90</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2022-10-22 04:41:17+00:00</td>
      <td>Vigilante Shit</td>
      <td>Taylor Swift</td>
      <td>89</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>628</th>
      <td>2022-05-16 04:40:15+00:00</td>
      <td>Vegas (From the Original Motion Picture Soundt...</td>
      <td>Doja Cat</td>
      <td>88</td>
      <td>dance pop</td>
    </tr>
    <tr>
      <th>4694</th>
      <td>2017-03-19 21:59:23+00:00</td>
      <td>Yellow</td>
      <td>Coldplay</td>
      <td>88</td>
      <td>permanent wave</td>
    </tr>
    <tr>
      <th>3391</th>
      <td>2020-03-17 02:07:13+00:00</td>
      <td>Watermelon Sugar</td>
      <td>Harry Styles</td>
      <td>88</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2022-10-22 04:41:51+00:00</td>
      <td>Bejeweled</td>
      <td>Taylor Swift</td>
      <td>88</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>1355</th>
      <td>2021-07-14 20:52:11+00:00</td>
      <td>STAY (with Justin Bieber)</td>
      <td>The Kid LAROI</td>
      <td>87</td>
      <td>australian hip hop</td>
    </tr>
    <tr>
      <th>2787</th>
      <td>2020-07-03 02:33:03+00:00</td>
      <td>Why'd You Only Call Me When You're High?</td>
      <td>Arctic Monkeys</td>
      <td>87</td>
      <td>garage rock</td>
    </tr>
  </tbody>
</table>
</div>



A couple of names stand out here: Taylor Swift, Harry Styles, and Doja Cat because as far as general popularity as well as TikTok Popularity, these are pretty high up on the list. Most of the top 20 songs in my dataset based on popularity are within the pop genre as well. 

That goes for Sam Smith's song 'Unholy' as well which has a popularity score of 100. 


```python
len(liked_songs_clean[liked_songs_clean.popularity > 79])*100/len(liked_songs_clean)
```




    3.2177183451734224



Only 3.21% of the songs I listen to have a popularity above 79%. This could also be a good visualization - to see the distribution of the popularity of the songs I have saved. 


```python
sns.histplot(data=liked_songs_clean, x="popularity", color='lavender');
```


    
![png](output_files/output_50_0.png)
    


I was also curious about the number of songs I saved that have the same name. 


```python
sns.histplot(data=liked_songs_clean[liked_songs_clean.year == 2022], x="popularity", color = 'pink');
```


    
![png](output_files/output_52_0.png)
    


The distribution of my songs is very roughly normal, barring the songs with a 0 popularity score. Let's see how many songs have 0 popularity. 


```python
# zero popularity songs
zero_pop = liked_songs_clean[liked_songs_clean.popularity == 0]
zero_pop.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
      <th>artists</th>
      <th>duration_s</th>
      <th>popularity</th>
      <th>added_at</th>
      <th>acousticness</th>
      <th>speechiness</th>
      <th>key</th>
      <th>liveness</th>
      <th>...</th>
      <th>time_signature</th>
      <th>loudness</th>
      <th>danceability</th>
      <th>valence</th>
      <th>genre_list</th>
      <th>genre</th>
      <th>pitch_class</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>305</th>
      <td>6rbeWjEavBHvX2kr6lSogS</td>
      <td>Work Out</td>
      <td>J. Cole</td>
      <td>234.773</td>
      <td>0</td>
      <td>2022-07-17 03:16:52+00:00</td>
      <td>0.05310</td>
      <td>0.1060</td>
      <td>2</td>
      <td>0.3160</td>
      <td>...</td>
      <td>4</td>
      <td>-6.903</td>
      <td>0.831</td>
      <td>0.216</td>
      <td>['conscious hip hop', 'hip hop', 'north caroli...</td>
      <td>conscious hip hop</td>
      <td>D/Re</td>
      <td>2022</td>
      <td>7</td>
      <td>17</td>
    </tr>
    <tr>
      <th>307</th>
      <td>6HHytHLXLX8QfWTtGfFSvH</td>
      <td>Drop It Like It's Hot</td>
      <td>Snoop Dogg</td>
      <td>266.066</td>
      <td>0</td>
      <td>2022-07-17 03:16:43+00:00</td>
      <td>0.16900</td>
      <td>0.2160</td>
      <td>1</td>
      <td>0.1020</td>
      <td>...</td>
      <td>4</td>
      <td>-3.832</td>
      <td>0.892</td>
      <td>0.676</td>
      <td>['g funk', 'gangster rap', 'hip hop', 'pop rap...</td>
      <td>g funk</td>
      <td>C#/Do sost.</td>
      <td>2022</td>
      <td>7</td>
      <td>17</td>
    </tr>
    <tr>
      <th>308</th>
      <td>2mpFm3f7QmdsVtSAIICEB7</td>
      <td>Candy Shop</td>
      <td>50 Cent</td>
      <td>208.533</td>
      <td>0</td>
      <td>2022-07-17 03:16:41+00:00</td>
      <td>0.03050</td>
      <td>0.4810</td>
      <td>7</td>
      <td>0.3690</td>
      <td>...</td>
      <td>5</td>
      <td>-7.992</td>
      <td>0.609</td>
      <td>0.797</td>
      <td>['east coast hip hop', 'gangster rap', 'hip ho...</td>
      <td>east coast hip hop</td>
      <td>G/Sol</td>
      <td>2022</td>
      <td>7</td>
      <td>17</td>
    </tr>
    <tr>
      <th>316</th>
      <td>2ng5lhEsASy6jgbOLg877a</td>
      <td>Gas Pedal</td>
      <td>Sage The Gemini</td>
      <td>208.160</td>
      <td>0</td>
      <td>2022-07-17 03:15:53+00:00</td>
      <td>0.02720</td>
      <td>0.0576</td>
      <td>1</td>
      <td>0.1230</td>
      <td>...</td>
      <td>4</td>
      <td>-8.059</td>
      <td>0.846</td>
      <td>0.441</td>
      <td>['dance pop', 'hyphy', 'pop rap', 'rap', 'sout...</td>
      <td>dance pop</td>
      <td>C#/Do sost.</td>
      <td>2022</td>
      <td>7</td>
      <td>17</td>
    </tr>
    <tr>
      <th>326</th>
      <td>06iMqWThw4w8fTFyccvOwr</td>
      <td>Ride Wit Me</td>
      <td>Nelly</td>
      <td>291.781</td>
      <td>0</td>
      <td>2022-07-17 03:15:08+00:00</td>
      <td>0.06680</td>
      <td>0.0479</td>
      <td>7</td>
      <td>0.2470</td>
      <td>...</td>
      <td>4</td>
      <td>-6.625</td>
      <td>0.854</td>
      <td>0.753</td>
      <td>['canadian latin', 'canadian pop', 'dance pop'...</td>
      <td>canadian latin</td>
      <td>G/Sol</td>
      <td>2022</td>
      <td>7</td>
      <td>17</td>
    </tr>
    <tr>
      <th>336</th>
      <td>2IpGdrWvIZipmaxo1YRxw5</td>
      <td>Bottoms Up (feat. Nicki Minaj)</td>
      <td>Trey Songz</td>
      <td>242.013</td>
      <td>0</td>
      <td>2022-07-17 03:12:30+00:00</td>
      <td>0.02050</td>
      <td>0.1610</td>
      <td>1</td>
      <td>0.3850</td>
      <td>...</td>
      <td>4</td>
      <td>-5.283</td>
      <td>0.845</td>
      <td>0.329</td>
      <td>['dance pop', 'pop', 'r&amp;b', 'southern hip hop'...</td>
      <td>dance pop</td>
      <td>C#/Do sost.</td>
      <td>2022</td>
      <td>7</td>
      <td>17</td>
    </tr>
    <tr>
      <th>337</th>
      <td>6C7RJEIUDqKkJRZVWdkfkH</td>
      <td>Stronger</td>
      <td>Kanye West</td>
      <td>311.866</td>
      <td>0</td>
      <td>2022-07-17 03:12:21+00:00</td>
      <td>0.00728</td>
      <td>0.1550</td>
      <td>10</td>
      <td>0.3180</td>
      <td>...</td>
      <td>4</td>
      <td>-7.731</td>
      <td>0.625</td>
      <td>0.483</td>
      <td>['chicago rap', 'rap']</td>
      <td>chicago rap</td>
      <td>A#/La sost.</td>
      <td>2022</td>
      <td>7</td>
      <td>17</td>
    </tr>
    <tr>
      <th>394</th>
      <td>4gbC4p3lbOS04ZY0NMoNd8</td>
      <td>Love Will Never Be the Same</td>
      <td>Gabriel Bernini</td>
      <td>228.739</td>
      <td>0</td>
      <td>2022-07-04 07:14:44+00:00</td>
      <td>0.32200</td>
      <td>0.0383</td>
      <td>0</td>
      <td>0.2020</td>
      <td>...</td>
      <td>4</td>
      <td>-14.522</td>
      <td>0.673</td>
      <td>0.964</td>
      <td>[]</td>
      <td>other</td>
      <td>C/Do</td>
      <td>2022</td>
      <td>7</td>
      <td>4</td>
    </tr>
    <tr>
      <th>467</th>
      <td>51OWppRfn0dVhmWql6x0fl</td>
      <td>Is The World Really Out There</td>
      <td>Gabriel Bernini</td>
      <td>132.405</td>
      <td>0</td>
      <td>2022-06-02 01:35:05+00:00</td>
      <td>0.01560</td>
      <td>0.0532</td>
      <td>7</td>
      <td>0.0822</td>
      <td>...</td>
      <td>4</td>
      <td>-17.038</td>
      <td>0.898</td>
      <td>0.884</td>
      <td>[]</td>
      <td>other</td>
      <td>G/Sol</td>
      <td>2022</td>
      <td>6</td>
      <td>2</td>
    </tr>
    <tr>
      <th>469</th>
      <td>2bOZ552GfKen8B6Uq6ltyv</td>
      <td>Lifeguard</td>
      <td>Gabriel Bernini</td>
      <td>237.215</td>
      <td>0</td>
      <td>2022-06-02 01:31:16+00:00</td>
      <td>0.37200</td>
      <td>0.0303</td>
      <td>7</td>
      <td>0.0669</td>
      <td>...</td>
      <td>3</td>
      <td>-12.694</td>
      <td>0.739</td>
      <td>0.566</td>
      <td>[]</td>
      <td>other</td>
      <td>G/Sol</td>
      <td>2022</td>
      <td>6</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 23 columns</p>
</div>



These seem to be somewhat older songs in the dataset or what would be considered 'party' songs. 


```python
# looking at the name frequency of the songs I've added
liked_songs_clean.name.value_counts()[:20]
```




    Daylight         6
    She              6
    Boyfriend        5
    Hurricane        5
    Memories         5
    Trouble          5
    One              4
    I'm Yours        4
    Invisible        4
    Colors           4
    Waiting          4
    Mine             4
    Enchanted        4
    Haunted          4
    The City         4
    Shapeshifter     4
    Golden           4
    Runaway          4
    Somebody Else    4
    Yours            4
    Name: name, dtype: int64



Does this say anything about the types of songs I enjoy listening to?


```python
liked_songs_clean[liked_songs_clean.name == "She"]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
      <th>artists</th>
      <th>duration_s</th>
      <th>popularity</th>
      <th>added_at</th>
      <th>acousticness</th>
      <th>speechiness</th>
      <th>key</th>
      <th>liveness</th>
      <th>...</th>
      <th>time_signature</th>
      <th>loudness</th>
      <th>danceability</th>
      <th>valence</th>
      <th>genre_list</th>
      <th>genre</th>
      <th>pitch_class</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1524</th>
      <td>56XCgjSYaLg0TjxkVSpqNu</td>
      <td>She</td>
      <td>Ed Sheeran</td>
      <td>244.653</td>
      <td>36</td>
      <td>2021-03-31 03:24:22+00:00</td>
      <td>0.811000</td>
      <td>0.0361</td>
      <td>9</td>
      <td>0.1300</td>
      <td>...</td>
      <td>3</td>
      <td>-10.910</td>
      <td>0.573</td>
      <td>0.3770</td>
      <td>['pop', 'uk pop']</td>
      <td>pop</td>
      <td>A/La</td>
      <td>2021</td>
      <td>3</td>
      <td>31</td>
    </tr>
    <tr>
      <th>1972</th>
      <td>63o0CBXZ6LAnJoqz26oRj9</td>
      <td>She</td>
      <td>Meyru</td>
      <td>334.608</td>
      <td>0</td>
      <td>2020-11-04 04:17:04+00:00</td>
      <td>0.533000</td>
      <td>0.0348</td>
      <td>11</td>
      <td>0.3400</td>
      <td>...</td>
      <td>4</td>
      <td>-9.908</td>
      <td>0.569</td>
      <td>0.0408</td>
      <td>[]</td>
      <td>other</td>
      <td>B/Si</td>
      <td>2020</td>
      <td>11</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2521</th>
      <td>1z1ztKUrDr09ZSMDnN3QIG</td>
      <td>She</td>
      <td>Selena Gomez</td>
      <td>172.999</td>
      <td>55</td>
      <td>2020-08-28 02:39:22+00:00</td>
      <td>0.277000</td>
      <td>0.0481</td>
      <td>2</td>
      <td>0.0672</td>
      <td>...</td>
      <td>4</td>
      <td>-6.686</td>
      <td>0.783</td>
      <td>0.7230</td>
      <td>['dance pop', 'pop', 'post-teen pop']</td>
      <td>dance pop</td>
      <td>D/Re</td>
      <td>2020</td>
      <td>8</td>
      <td>28</td>
    </tr>
    <tr>
      <th>2923</th>
      <td>5hwzzutxeqeC5VMSpnfgul</td>
      <td>She</td>
      <td>Selena Gomez</td>
      <td>172.999</td>
      <td>26</td>
      <td>2020-05-06 00:27:17+00:00</td>
      <td>0.286000</td>
      <td>0.0485</td>
      <td>2</td>
      <td>0.0682</td>
      <td>...</td>
      <td>4</td>
      <td>-6.687</td>
      <td>0.784</td>
      <td>0.7280</td>
      <td>['dance pop', 'pop', 'post-teen pop']</td>
      <td>dance pop</td>
      <td>D/Re</td>
      <td>2020</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3348</th>
      <td>6SQLk9HSNketfgs2AyIiMs</td>
      <td>She</td>
      <td>Harry Styles</td>
      <td>362.653</td>
      <td>74</td>
      <td>2020-03-23 01:37:17+00:00</td>
      <td>0.000532</td>
      <td>0.0272</td>
      <td>0</td>
      <td>0.1900</td>
      <td>...</td>
      <td>3</td>
      <td>-5.942</td>
      <td>0.535</td>
      <td>0.4570</td>
      <td>['pop']</td>
      <td>pop</td>
      <td>C/Do</td>
      <td>2020</td>
      <td>3</td>
      <td>23</td>
    </tr>
    <tr>
      <th>4744</th>
      <td>3ICdPHubhqTJ4Lm9NEb2W3</td>
      <td>She</td>
      <td>Ed Sheeran</td>
      <td>244.653</td>
      <td>46</td>
      <td>2017-03-11 21:45:28+00:00</td>
      <td>0.811000</td>
      <td>0.0361</td>
      <td>9</td>
      <td>0.1300</td>
      <td>...</td>
      <td>3</td>
      <td>-10.910</td>
      <td>0.573</td>
      <td>0.3770</td>
      <td>['pop', 'uk pop']</td>
      <td>pop</td>
      <td>A/La</td>
      <td>2017</td>
      <td>3</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
<p>6 rows × 23 columns</p>
</div>




```python
# Let's remove these duplicates: FIXME this should actually go in preprocessesing
liked_songs_clean_nodup = liked_songs_clean.drop_duplicates(
  subset = ['name', 'artists'],
  keep = 'first').reset_index(drop = True)
```


```python
liked_songs_clean_nodup.shape
```




    (4376, 23)




```python
# I'm going to change it to liked_songs because it's long
liked_songs = liked_songs_clean_nodup
```

## Visualization

Now that we've done some initial exploring, we'll create some visuals. 

#### Visualization of all the audio features


```python
audio_features = liked_songs[[ 'acousticness', 'speechiness', 'key', 'liveness', 
                                          'instrumentalness', 'energy', 'tempo', 'time_signature', 
                                          'loudness', 'danceability','valence',
                                         ]]
```


```python
for col in audio_features:
        plt.figure(figsize=(20,2))
        sns.histplot(data=audio_features, x=col)
# change the color for each histplot?
```


    
![png](output_files/output_66_0.png)
    



    
![png](output_files/output_66_1.png)
    



    
![png](output_files/output_66_2.png)
    



    
![png](output_files/output_66_3.png)
    



    
![png](output_files/output_66_4.png)
    



    
![png](output_files/output_66_5.png)
    



    
![png](output_files/output_66_6.png)
    



    
![png](output_files/output_66_7.png)
    



    
![png](output_files/output_66_8.png)
    



    
![png](output_files/output_66_9.png)
    



    
![png](output_files/output_66_10.png)
    



```python
for col in audio_features:
        plt.figure(figsize=(17,1))
        sns.boxplot(data=audio_features, x=col)
```


    
![png](output_files/output_67_0.png)
    



    
![png](output_files/output_67_1.png)
    



    
![png](output_files/output_67_2.png)
    



    
![png](output_files/output_67_3.png)
    



    
![png](output_files/output_67_4.png)
    



    
![png](output_files/output_67_5.png)
    



    
![png](output_files/output_67_6.png)
    



    
![png](output_files/output_67_7.png)
    



    
![png](output_files/output_67_8.png)
    



    
![png](output_files/output_67_9.png)
    



    
![png](output_files/output_67_10.png)
    


# FIXME


Find top 6 genres. Only keep data in those 6 genres
boxplot for liked songs, but hue is genre


create a visualization that is the percentage of each genre in the dataset


```python
# find the top 6 genres in the dataset
liked_songs.genre.value_counts().head(6).axes
```




    [Index(['dance pop', 'pop', 'alt z', 'other', 'boy band', 'acoustic pop'], dtype='object')]




```python
# liked_songs[liked_songs_clean.genre in top_six_genres]

top_six_genres = liked_songs[liked_songs.genre.isin(['dance pop', 'pop', 'alt z', 'other', 'boy band', 'acoustic pop'])]
```


```python
sns.boxplot(data=top_six_genres, x="acousticness", y="genre");
```


    
![png](output_files/output_72_0.png)
    


Okay, we can loop through and do this for all audio features if we want


```python
audio_list = [ 'acousticness', 'speechiness', 'key', 'liveness', 
                                          'instrumentalness', 'energy', 'tempo', 'time_signature', 
                                          'loudness', 'danceability','valence',
                                         ]
```

https://www.python-graph-gallery.com/33-control-colors-of-boxplot-seaborn

https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html


```python
for col in audio_list:
    plt.figure(figsize=(21,2))
    sns.boxplot(data=top_six_genres, x=col, y="genre", palette="Spectral");
```


    
![png](output_files/output_76_0.png)
    



    
![png](output_files/output_76_1.png)
    



    
![png](output_files/output_76_2.png)
    



    
![png](output_files/output_76_3.png)
    



    
![png](output_files/output_76_4.png)
    



    
![png](output_files/output_76_5.png)
    



    
![png](output_files/output_76_6.png)
    



    
![png](output_files/output_76_7.png)
    



    
![png](output_files/output_76_8.png)
    



    
![png](output_files/output_76_9.png)
    



    
![png](output_files/output_76_10.png)
    


It could be cool to look at the outliers of the genres to see which songs dont fit. 


```python
# running the violin plot on top 6 genres
sns.violinplot(x=top_six_genres["genre"], y=top_six_genres["popularity"])
```




    <AxesSubplot:xlabel='genre', ylabel='popularity'>




    
![png](output_files/output_78_1.png)
    


FIXME

do a kde plot of some kind for funzies    
or this https://www.python-graph-gallery.com/web-circular-barplot-with-matplotlib

The map below is cool but idk what I would use it for here


```python
# venn diagram to see which songs overlap on my top and whatever 

# interactive python map
# Import the folium library
# !pip install folium
# import folium

# Build the default map for a specific location
# map = folium.Map(location=[43.61092, 3.87723])
# map
```

#### Heatmap for correlation between variables


```python
# create heatmap for correlation
plt.figure(figsize=(15, 10))
corr = liked_songs.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm").set_title('Pearson correlation matrix')
plt.show()
```


    
![png](output_files/output_84_0.png)
    



```python
# printing out correlation table
corr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>duration_s</th>
      <th>popularity</th>
      <th>acousticness</th>
      <th>speechiness</th>
      <th>key</th>
      <th>liveness</th>
      <th>instrumentalness</th>
      <th>energy</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>loudness</th>
      <th>danceability</th>
      <th>valence</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>duration_s</th>
      <td>1.000000</td>
      <td>0.028403</td>
      <td>0.035021</td>
      <td>-0.132520</td>
      <td>-0.002766</td>
      <td>0.005063</td>
      <td>0.061818</td>
      <td>-0.081065</td>
      <td>-0.010638</td>
      <td>-0.055605</td>
      <td>-0.081318</td>
      <td>-0.240059</td>
      <td>-0.274188</td>
      <td>-0.075578</td>
      <td>0.031932</td>
      <td>-0.016063</td>
    </tr>
    <tr>
      <th>popularity</th>
      <td>0.028403</td>
      <td>1.000000</td>
      <td>-0.169642</td>
      <td>0.034327</td>
      <td>-0.012989</td>
      <td>-0.048861</td>
      <td>-0.104133</td>
      <td>0.145625</td>
      <td>0.033724</td>
      <td>0.020395</td>
      <td>0.155523</td>
      <td>0.113177</td>
      <td>0.079760</td>
      <td>0.108687</td>
      <td>0.043526</td>
      <td>0.047021</td>
    </tr>
    <tr>
      <th>acousticness</th>
      <td>0.035021</td>
      <td>-0.169642</td>
      <td>1.000000</td>
      <td>-0.117723</td>
      <td>-0.013664</td>
      <td>-0.102452</td>
      <td>0.242222</td>
      <td>-0.801977</td>
      <td>-0.189693</td>
      <td>-0.191742</td>
      <td>-0.649149</td>
      <td>-0.311349</td>
      <td>-0.429403</td>
      <td>-0.077496</td>
      <td>-0.039835</td>
      <td>-0.095081</td>
    </tr>
    <tr>
      <th>speechiness</th>
      <td>-0.132520</td>
      <td>0.034327</td>
      <td>-0.117723</td>
      <td>1.000000</td>
      <td>0.044114</td>
      <td>0.111797</td>
      <td>-0.054642</td>
      <td>0.140177</td>
      <td>0.071819</td>
      <td>0.046067</td>
      <td>0.066205</td>
      <td>0.194882</td>
      <td>0.189814</td>
      <td>-0.062073</td>
      <td>0.104195</td>
      <td>-0.000731</td>
    </tr>
    <tr>
      <th>key</th>
      <td>-0.002766</td>
      <td>-0.012989</td>
      <td>-0.013664</td>
      <td>0.044114</td>
      <td>1.000000</td>
      <td>0.022846</td>
      <td>-0.031028</td>
      <td>0.015503</td>
      <td>0.018817</td>
      <td>-0.003355</td>
      <td>0.012716</td>
      <td>0.030146</td>
      <td>0.032204</td>
      <td>0.005291</td>
      <td>-0.010298</td>
      <td>-0.024351</td>
    </tr>
    <tr>
      <th>liveness</th>
      <td>0.005063</td>
      <td>-0.048861</td>
      <td>-0.102452</td>
      <td>0.111797</td>
      <td>0.022846</td>
      <td>1.000000</td>
      <td>-0.053786</td>
      <td>0.144278</td>
      <td>0.038109</td>
      <td>0.017394</td>
      <td>0.092113</td>
      <td>-0.013343</td>
      <td>0.103353</td>
      <td>-0.012320</td>
      <td>-0.002331</td>
      <td>0.010790</td>
    </tr>
    <tr>
      <th>instrumentalness</th>
      <td>0.061818</td>
      <td>-0.104133</td>
      <td>0.242222</td>
      <td>-0.054642</td>
      <td>-0.031028</td>
      <td>-0.053786</td>
      <td>1.000000</td>
      <td>-0.265126</td>
      <td>-0.073585</td>
      <td>-0.088373</td>
      <td>-0.550983</td>
      <td>-0.273835</td>
      <td>-0.169752</td>
      <td>-0.054589</td>
      <td>0.045286</td>
      <td>-0.095711</td>
    </tr>
    <tr>
      <th>energy</th>
      <td>-0.081065</td>
      <td>0.145625</td>
      <td>-0.801977</td>
      <td>0.140177</td>
      <td>0.015503</td>
      <td>0.144278</td>
      <td>-0.265126</td>
      <td>1.000000</td>
      <td>0.224640</td>
      <td>0.213132</td>
      <td>0.777858</td>
      <td>0.267695</td>
      <td>0.540358</td>
      <td>0.043078</td>
      <td>0.021360</td>
      <td>0.116024</td>
    </tr>
    <tr>
      <th>tempo</th>
      <td>-0.010638</td>
      <td>0.033724</td>
      <td>-0.189693</td>
      <td>0.071819</td>
      <td>0.018817</td>
      <td>0.038109</td>
      <td>-0.073585</td>
      <td>0.224640</td>
      <td>1.000000</td>
      <td>-0.015931</td>
      <td>0.178471</td>
      <td>-0.094183</td>
      <td>0.101617</td>
      <td>0.037764</td>
      <td>0.008002</td>
      <td>0.028665</td>
    </tr>
    <tr>
      <th>time_signature</th>
      <td>-0.055605</td>
      <td>0.020395</td>
      <td>-0.191742</td>
      <td>0.046067</td>
      <td>-0.003355</td>
      <td>0.017394</td>
      <td>-0.088373</td>
      <td>0.213132</td>
      <td>-0.015931</td>
      <td>1.000000</td>
      <td>0.186285</td>
      <td>0.196456</td>
      <td>0.161164</td>
      <td>-0.001255</td>
      <td>0.017714</td>
      <td>0.039199</td>
    </tr>
    <tr>
      <th>loudness</th>
      <td>-0.081318</td>
      <td>0.155523</td>
      <td>-0.649149</td>
      <td>0.066205</td>
      <td>0.012716</td>
      <td>0.092113</td>
      <td>-0.550983</td>
      <td>0.777858</td>
      <td>0.178471</td>
      <td>0.186285</td>
      <td>1.000000</td>
      <td>0.325509</td>
      <td>0.412435</td>
      <td>0.047762</td>
      <td>-0.001035</td>
      <td>0.122597</td>
    </tr>
    <tr>
      <th>danceability</th>
      <td>-0.240059</td>
      <td>0.113177</td>
      <td>-0.311349</td>
      <td>0.194882</td>
      <td>0.030146</td>
      <td>-0.013343</td>
      <td>-0.273835</td>
      <td>0.267695</td>
      <td>-0.094183</td>
      <td>0.196456</td>
      <td>0.325509</td>
      <td>1.000000</td>
      <td>0.502720</td>
      <td>0.037172</td>
      <td>0.034644</td>
      <td>0.016570</td>
    </tr>
    <tr>
      <th>valence</th>
      <td>-0.274188</td>
      <td>0.079760</td>
      <td>-0.429403</td>
      <td>0.189814</td>
      <td>0.032204</td>
      <td>0.103353</td>
      <td>-0.169752</td>
      <td>0.540358</td>
      <td>0.101617</td>
      <td>0.161164</td>
      <td>0.412435</td>
      <td>0.502720</td>
      <td>1.000000</td>
      <td>0.053245</td>
      <td>0.000848</td>
      <td>0.043331</td>
    </tr>
    <tr>
      <th>year</th>
      <td>-0.075578</td>
      <td>0.108687</td>
      <td>-0.077496</td>
      <td>-0.062073</td>
      <td>0.005291</td>
      <td>-0.012320</td>
      <td>-0.054589</td>
      <td>0.043078</td>
      <td>0.037764</td>
      <td>-0.001255</td>
      <td>0.047762</td>
      <td>0.037172</td>
      <td>0.053245</td>
      <td>1.000000</td>
      <td>-0.181159</td>
      <td>-0.075296</td>
    </tr>
    <tr>
      <th>month</th>
      <td>0.031932</td>
      <td>0.043526</td>
      <td>-0.039835</td>
      <td>0.104195</td>
      <td>-0.010298</td>
      <td>-0.002331</td>
      <td>0.045286</td>
      <td>0.021360</td>
      <td>0.008002</td>
      <td>0.017714</td>
      <td>-0.001035</td>
      <td>0.034644</td>
      <td>0.000848</td>
      <td>-0.181159</td>
      <td>1.000000</td>
      <td>0.103043</td>
    </tr>
    <tr>
      <th>day</th>
      <td>-0.016063</td>
      <td>0.047021</td>
      <td>-0.095081</td>
      <td>-0.000731</td>
      <td>-0.024351</td>
      <td>0.010790</td>
      <td>-0.095711</td>
      <td>0.116024</td>
      <td>0.028665</td>
      <td>0.039199</td>
      <td>0.122597</td>
      <td>0.016570</td>
      <td>0.043331</td>
      <td>-0.075296</td>
      <td>0.103043</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
corr.min()
```




    duration_s         -0.274188
    popularity         -0.169642
    acousticness       -0.801977
    speechiness        -0.132520
    key                -0.031028
    liveness           -0.102452
    instrumentalness   -0.550983
    energy             -0.801977
    tempo              -0.189693
    time_signature     -0.191742
    loudness           -0.649149
    danceability       -0.311349
    valence            -0.429403
    year               -0.181159
    month              -0.181159
    day                -0.095711
    dtype: float64




```python
corr.max()
```




    duration_s          1.0
    popularity          1.0
    acousticness        1.0
    speechiness         1.0
    key                 1.0
    liveness            1.0
    instrumentalness    1.0
    energy              1.0
    tempo               1.0
    time_signature      1.0
    loudness            1.0
    danceability        1.0
    valence             1.0
    year                1.0
    month               1.0
    day                 1.0
    dtype: float64



Correlation values range from -1 to 1. The closer a correlation value is to 1 (positive or negative), the stronger the correlation between the two variables. Variables with a strong positive correlation increase together, whereas variables with a strong negative correlation experience opposing polarization (as one goes up, the other goes down and vice versa). The closer the correlation is to 0, the weaker the correlation. 

- variables will always have a correlation of 1 with themselves
- we can see the strongest correlation is between `energy` and `acousticness` at -0.8. 
- the weakest correlation is between `speechiness` and `dat` at -0.00036

Overall, none of our variables are very strongly correlated. 


```python
# we can also create a pairplot of the correlations
sns.pairplot(liked_songs);
```


    
![png](output_files/output_89_0.png)
    


Audio Features


```python
audio_descp = liked_songs.copy()
audio_descp = audio_descp.drop(columns=['year', 'month', 'day'])
```

The pairplot above is really crowded because it was run on all of the columns in our dataset. It shows the pairwise relationship between each column - so it's similar to the heat map but instead of correlation, it's the data plots of x against y where x is the first column of interest and y is the second column of interest. 


```python
# we can also create a pairplot of the correlations
sns.pairplot(audio_descp)
```




    <seaborn.axisgrid.PairGrid at 0x136e14d0df0>




    
![png](output_files/output_93_1.png)
    



```python
# find the closest corr for each col
corr.columns
```




    Index(['duration_s', 'popularity', 'acousticness', 'speechiness', 'key',
           'liveness', 'instrumentalness', 'energy', 'tempo', 'time_signature',
           'loudness', 'danceability', 'valence', 'year', 'month', 'day'],
          dtype='object')




```python
corr.iloc[:,0] # everything for first col
```




    duration_s          1.000000
    popularity          0.028403
    acousticness        0.035021
    speechiness        -0.132520
    key                -0.002766
    liveness            0.005063
    instrumentalness    0.061818
    energy             -0.081065
    tempo              -0.010638
    time_signature     -0.055605
    loudness           -0.081318
    danceability       -0.240059
    valence            -0.274188
    year               -0.075578
    month               0.031932
    day                -0.016063
    Name: duration_s, dtype: float64




```python
# plan: get the max corr for each column
# plot the max corr
```


```python
# create a histogram of the popularity of the tracks
sns.histplot(data=liked_songs, x="popularity", bins=25);
```


    
![png](output_files/output_97_0.png)
    


This is just a visual confirmation of how popular my music tastes are. The majority of my music tastes fall in the mid-range in terms of popularity


```python
# mean popularity score value
liked_songs.popularity.mean()
```




    45.02399451553931



What about the popularity in 2022?


```python
liked_songs[liked_songs.year == 2022].popularity.mean()
```




    52.004683840749415



It'll be interesting to see how the popularity of my songs has changed over the years. 


```python
# histogram of duration in seconds
sns.histplot(liked_songs.duration_s, bins=25);
```


    
![png](output_files/output_103_0.png)
    



```python
(liked_songs.duration_s.mean())/60
```




    3.524856326173065



Most of my saved songs are around 3 minutes long. 


```python
# longest song in my saved
liked_songs.iloc[ liked_songs['duration_s'].idxmax() ][['artists','name']]
```




    artists    Garth Stevenson
    name                  Dawn
    Name: 4128, dtype: object




```python
# shortest song in my saved
liked_songs.iloc[liked_songs['duration_s'].idxmin() ][['artists','name']]
```




    artists       Bo Burnham
    name       Unpaid Intern
    Name: 1258, dtype: object



### K Means

K-means clustering

I will perform K-means clustering on this dataset and try to analyze the change in output as the number of clusters increases.

K-means clustering is a centroid-based algorithm. The main objective of the algorithm is to minimize the sum of distances between the points and their respective cluster centroid.

Firstly I’d bring all the variables to the same magnitude using scale function

# FIXME: change to liked_songs


```python
# liked_songs_clean.pitch_class.value_counts()
# sns.countplot(data = liked_songs_clean,
#               x = 'pitch_class',
#               order = liked_songs_clean.pitch_class.value_counts().index)

# plt.xticks(rotation=70);
```


```python
sns.countplot(data=liked_songs, y='artists', order=liked_songs_clean.artists.value_counts()[:20].index);

plt.xticks(rotation=85);
```


    
![png](output_files/output_112_0.png)
    


Okay, but what about if we group by year


```python
liked_songs_2022 = liked_songs_clean[liked_songs_clean.year > 2021]
print(str(len(liked_songs_2022)) + " songs have been added in 2022 so far")
```

    907 songs have been added in 2022 so far
    


```python
liked_songs_2022.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
      <th>artists</th>
      <th>duration_s</th>
      <th>popularity</th>
      <th>added_at</th>
      <th>acousticness</th>
      <th>speechiness</th>
      <th>key</th>
      <th>liveness</th>
      <th>...</th>
      <th>time_signature</th>
      <th>loudness</th>
      <th>danceability</th>
      <th>valence</th>
      <th>genre_list</th>
      <th>genre</th>
      <th>pitch_class</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3XyMM4crvrhOjAbMGiBDck</td>
      <td>Can't Come Back</td>
      <td>Kevin Garrett</td>
      <td>249.400</td>
      <td>23</td>
      <td>2022-11-06 04:18:28+00:00</td>
      <td>0.7010</td>
      <td>0.0278</td>
      <td>5</td>
      <td>0.0940</td>
      <td>...</td>
      <td>4</td>
      <td>-9.977</td>
      <td>0.531</td>
      <td>0.287</td>
      <td>['nyc pop', 'pittsburgh indie']</td>
      <td>nyc pop</td>
      <td>F/Fa</td>
      <td>2022</td>
      <td>11</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0JbSghVDghtFEurrSO8JrC</td>
      <td>Country Girl (Shake It For Me)</td>
      <td>Luke Bryan</td>
      <td>225.560</td>
      <td>76</td>
      <td>2022-11-04 03:06:05+00:00</td>
      <td>0.0293</td>
      <td>0.0462</td>
      <td>2</td>
      <td>0.0834</td>
      <td>...</td>
      <td>4</td>
      <td>-4.532</td>
      <td>0.645</td>
      <td>0.671</td>
      <td>['contemporary country', 'country', 'country r...</td>
      <td>contemporary country</td>
      <td>D/Re</td>
      <td>2022</td>
      <td>11</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0YeLqMwKztVZ1MIZ4oYW4e</td>
      <td>The Wolf</td>
      <td>The Spencer Lee Band</td>
      <td>174.653</td>
      <td>50</td>
      <td>2022-11-04 03:03:51+00:00</td>
      <td>0.0932</td>
      <td>0.1860</td>
      <td>9</td>
      <td>0.1430</td>
      <td>...</td>
      <td>4</td>
      <td>-4.746</td>
      <td>0.777</td>
      <td>0.659</td>
      <td>['modern funk']</td>
      <td>modern funk</td>
      <td>A/La</td>
      <td>2022</td>
      <td>11</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3UMrglJeju5yWyYIW6o99b</td>
      <td>The Great War</td>
      <td>Taylor Swift</td>
      <td>240.355</td>
      <td>87</td>
      <td>2022-11-01 05:13:52+00:00</td>
      <td>0.2190</td>
      <td>0.0353</td>
      <td>5</td>
      <td>0.0842</td>
      <td>...</td>
      <td>4</td>
      <td>-8.987</td>
      <td>0.573</td>
      <td>0.554</td>
      <td>['pop']</td>
      <td>pop</td>
      <td>F/Fa</td>
      <td>2022</td>
      <td>11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4thL2Pt5jylQJGNYf9FySA</td>
      <td>Close Your Eyes</td>
      <td>Jump, Little Children</td>
      <td>169.400</td>
      <td>20</td>
      <td>2022-11-01 05:08:59+00:00</td>
      <td>0.9140</td>
      <td>0.0303</td>
      <td>4</td>
      <td>0.1040</td>
      <td>...</td>
      <td>4</td>
      <td>-12.440</td>
      <td>0.508</td>
      <td>0.282</td>
      <td>['south carolina indie']</td>
      <td>south carolina indie</td>
      <td>E/Mi</td>
      <td>2022</td>
      <td>11</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
sns.scatterplot(data=liked_songs_2022, x='energy', y='acousticness');
```


    
![png](output_files/output_116_0.png)
    


Above is the visualization of we saw earlier in the heatmap. As energy increases, the amount of acousticness the song has tends to decrease. And we can fit this with linear regression to point out an even cleaner pattern.


```python
sns.regplot(data=liked_songs_2022, x='energy', y='acousticness');
```


    
![png](output_files/output_118_0.png)
    


How have the audio features of my song choices changed over time? We can write a time series function to plot the change of our data over timie


```python
def plot_time_series(col_name, title, rolling_window_days):
    daily_series = pd.Series(data=np.array(liked_songs_clean[col_name]), 
                                      name=col_name, 
                                      index=liked_songs_clean['added_at']).sort_index()

    (daily_series.rolling(window = rolling_window_days)
     .mean()
     .plot(figsize=(30, 10))
     .set(xlabel='date (by day)', ylabel=col_name, title=title))

    plt.show()    
```


```python
plot_time_series('popularity', 'Popularity over time (window = 30 days)', 30)
plot_time_series('duration_s', 'Duration (s) over time (window = 30 days)', 30)
plot_time_series('danceability', 'Danceability over time (window = 30 days)', 30)
plot_time_series('valence', 'Valence over time (window = 30 days)', 30)
plot_time_series('energy', 'Energy over time (window = 30 days)', 30)
plot_time_series('tempo', 'Tempo over time (window = 30 days)', 30)
```


    
![png](output_files/output_121_0.png)
    



    
![png](output_files/output_121_1.png)
    



    
![png](output_files/output_121_2.png)
    



    
![png](output_files/output_121_3.png)
    



    
![png](output_files/output_121_4.png)
    



    
![png](output_files/output_121_5.png)
    


In general, our plots above are fairly stationary from 2020 on. However, they could be a lot better. 

We can also look at the change in the number of songs added over time.


```python
songs_added_in_window = (liked_songs_clean.groupby([pd.Grouper(freq='30D', key='added_at'), 'id'])
                         .size().reset_index(name='count').groupby('added_at').count()['count'])

(songs_added_in_window
 .plot(figsize=(20, 10))
 .set(xlabel='date (by day)', ylabel='count', title='New songs added over time (window = 30 days)'))

plt.show() 
```


    
![png](output_files/output_124_0.png)
    



```python
# plt.figure(figsize=(15, 10))
sns.boxplot(x=liked_songs_clean['valence']).set_title('Tracks valence (1 = happy, 0 = sad)')
plt.show()
```


    
![png](output_files/output_125_0.png)
    



```python
# plt.figure(figsize=(15, 10))
sns.boxplot(x=liked_songs_clean['loudness']).set_title('Loudness')
plt.show()
```


    
![png](output_files/output_126_0.png)
    



```python
# plt.figure(figsize=(15, 10))
sns.boxplot(x=liked_songs_clean['tempo']).set_title('Tempo')
plt.show()
```


    
![png](output_files/output_127_0.png)
    



```python
# plt.figure(figsize=(15, 10))
sns.boxplot(x=liked_songs_clean['energy']).set_title('Energy')
plt.show()
```


    
![png](output_files/output_128_0.png)
    


https://towardsdatascience.com/reverse-engineering-spotify-wrapped-ai-using-python-452b58ad1a62


```python
# !pip install yellowbrick
from yellowbrick.target import FeatureCorrelation

# define columns to select
feature_names = ['acousticness', 'danceability', 'energy', 'instrumentalness',
       'liveness', 'loudness', 'speechiness', 'tempo', 'valence','duration_s','key','year']

X, y = liked_songs_2022[feature_names], liked_songs_2022['popularity']

# Create a list of the feature names
features = np.array(feature_names)

# Instantiate the visualizer
visualizer = FeatureCorrelation(labels=features)

plt.rcParams['figure.figsize']=(6,6)
visualizer.fit(X, y)     # Fit the data to the visualizer
visualizer.show();
```


    
![png](output_files/output_130_0.png)
    





---------------------------------------------------


```python
# liked_songs_clean = pd.read_csv('liked_songs_nov92022_sec.csv')
```

Make this a graph instead


| Year | Number of songs |  
| --- | --- |  
| 2020 | 2112 |
| 2022 | 907 |
| 2021 | 842 | 
| 2019 | 711 |
| 2017 | 208 |
| 2018 | 6 |

Taylor Swift is the artist I have added the greatest number of songs for over the years. How did my addition of Taylor Swift songs change throughout the years, or did it? Note: We'll want to pay attention to album releases since she took a brief hiatus and I will often add her songs immediately when an album is released It might also be interesting to see if I add the songs the same year the song is released or later on.


```python
liked_songs_clean['year'] = pd.DatetimeIndex(liked_songs_clean['added_at']).year
liked_songs_clean['month'] = pd.DatetimeIndex(liked_songs_clean['added_at']).month
liked_songs_clean['day'] = pd.DatetimeIndex(liked_songs_clean['added_at']).day
```


```python
taylor_swift = liked_songs_clean[liked_songs_clean.artists == 'Taylor Swift']
taylor_swift
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
      <th>artists</th>
      <th>duration_s</th>
      <th>popularity</th>
      <th>added_at</th>
      <th>acousticness</th>
      <th>speechiness</th>
      <th>key</th>
      <th>liveness</th>
      <th>...</th>
      <th>time_signature</th>
      <th>loudness</th>
      <th>danceability</th>
      <th>valence</th>
      <th>genre_list</th>
      <th>genre</th>
      <th>pitch_class</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>3UMrglJeju5yWyYIW6o99b</td>
      <td>The Great War</td>
      <td>Taylor Swift</td>
      <td>240.355</td>
      <td>87</td>
      <td>2022-11-01 05:13:52+00:00</td>
      <td>0.21900</td>
      <td>0.0353</td>
      <td>5</td>
      <td>0.0842</td>
      <td>...</td>
      <td>4</td>
      <td>-8.987</td>
      <td>0.573</td>
      <td>0.554</td>
      <td>['pop']</td>
      <td>pop</td>
      <td>F/Fa</td>
      <td>2022</td>
      <td>11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0A1JLUlkZkp2EFrosoNQi0</td>
      <td>Labyrinth</td>
      <td>Taylor Swift</td>
      <td>247.962</td>
      <td>86</td>
      <td>2022-10-22 04:42:40+00:00</td>
      <td>0.78500</td>
      <td>0.0517</td>
      <td>0</td>
      <td>0.1220</td>
      <td>...</td>
      <td>4</td>
      <td>-15.480</td>
      <td>0.406</td>
      <td>0.122</td>
      <td>['pop']</td>
      <td>pop</td>
      <td>C/Do</td>
      <td>2022</td>
      <td>10</td>
      <td>22</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3qoftcUZaUOncvIYjFSPdE</td>
      <td>Bejeweled</td>
      <td>Taylor Swift</td>
      <td>194.165</td>
      <td>88</td>
      <td>2022-10-22 04:41:51+00:00</td>
      <td>0.06180</td>
      <td>0.0693</td>
      <td>7</td>
      <td>0.0887</td>
      <td>...</td>
      <td>4</td>
      <td>-9.190</td>
      <td>0.696</td>
      <td>0.433</td>
      <td>['pop']</td>
      <td>pop</td>
      <td>G/Sol</td>
      <td>2022</td>
      <td>10</td>
      <td>22</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1xwAWUI6Dj0WGC3KiUPN0O</td>
      <td>Vigilante Shit</td>
      <td>Taylor Swift</td>
      <td>164.801</td>
      <td>89</td>
      <td>2022-10-22 04:41:17+00:00</td>
      <td>0.17300</td>
      <td>0.3900</td>
      <td>4</td>
      <td>0.1210</td>
      <td>...</td>
      <td>4</td>
      <td>-11.096</td>
      <td>0.798</td>
      <td>0.163</td>
      <td>['pop']</td>
      <td>pop</td>
      <td>E/Mi</td>
      <td>2022</td>
      <td>10</td>
      <td>22</td>
    </tr>
    <tr>
      <th>12</th>
      <td>3rWDp9tBPQR9z6U5YyRSK4</td>
      <td>Midnight Rain</td>
      <td>Taylor Swift</td>
      <td>174.782</td>
      <td>90</td>
      <td>2022-10-22 04:40:42+00:00</td>
      <td>0.69000</td>
      <td>0.0767</td>
      <td>0</td>
      <td>0.1150</td>
      <td>...</td>
      <td>4</td>
      <td>-11.738</td>
      <td>0.643</td>
      <td>0.230</td>
      <td>['pop']</td>
      <td>pop</td>
      <td>C/Do</td>
      <td>2022</td>
      <td>10</td>
      <td>22</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3959</th>
      <td>15DeqWWQB4dcEWzJg15VrN</td>
      <td>Our Song</td>
      <td>Taylor Swift</td>
      <td>201.106</td>
      <td>70</td>
      <td>2019-12-30 00:17:25+00:00</td>
      <td>0.11100</td>
      <td>0.0303</td>
      <td>2</td>
      <td>0.3290</td>
      <td>...</td>
      <td>4</td>
      <td>-4.931</td>
      <td>0.668</td>
      <td>0.539</td>
      <td>['pop']</td>
      <td>pop</td>
      <td>D/Re</td>
      <td>2019</td>
      <td>12</td>
      <td>30</td>
    </tr>
    <tr>
      <th>3960</th>
      <td>5OOd01o2YS1QFwdpVLds3r</td>
      <td>Invisible</td>
      <td>Taylor Swift</td>
      <td>203.226</td>
      <td>51</td>
      <td>2019-12-30 00:17:25+00:00</td>
      <td>0.63700</td>
      <td>0.0243</td>
      <td>7</td>
      <td>0.1470</td>
      <td>...</td>
      <td>4</td>
      <td>-5.723</td>
      <td>0.612</td>
      <td>0.233</td>
      <td>['pop']</td>
      <td>pop</td>
      <td>G/Sol</td>
      <td>2019</td>
      <td>12</td>
      <td>30</td>
    </tr>
    <tr>
      <th>3961</th>
      <td>4pJi1rVt9GNegU9kywjg4z</td>
      <td>Teardrops on My Guitar - Pop Version</td>
      <td>Taylor Swift</td>
      <td>179.066</td>
      <td>52</td>
      <td>2019-12-30 00:17:25+00:00</td>
      <td>0.04020</td>
      <td>0.0537</td>
      <td>10</td>
      <td>0.0863</td>
      <td>...</td>
      <td>4</td>
      <td>-3.827</td>
      <td>0.459</td>
      <td>0.483</td>
      <td>['pop']</td>
      <td>pop</td>
      <td>A#/La sost.</td>
      <td>2019</td>
      <td>12</td>
      <td>30</td>
    </tr>
    <tr>
      <th>4025</th>
      <td>1dGr1c8CrMLDpV6mPbImSI</td>
      <td>Lover</td>
      <td>Taylor Swift</td>
      <td>221.306</td>
      <td>84</td>
      <td>2019-12-28 00:19:55+00:00</td>
      <td>0.49200</td>
      <td>0.0919</td>
      <td>7</td>
      <td>0.1180</td>
      <td>...</td>
      <td>4</td>
      <td>-7.582</td>
      <td>0.359</td>
      <td>0.453</td>
      <td>['pop']</td>
      <td>pop</td>
      <td>G/Sol</td>
      <td>2019</td>
      <td>12</td>
      <td>28</td>
    </tr>
    <tr>
      <th>4212</th>
      <td>6RRNNciQGZEXnqk8SQ9yv5</td>
      <td>You Need To Calm Down</td>
      <td>Taylor Swift</td>
      <td>171.360</td>
      <td>76</td>
      <td>2019-12-27 23:52:36+00:00</td>
      <td>0.00929</td>
      <td>0.0553</td>
      <td>2</td>
      <td>0.0637</td>
      <td>...</td>
      <td>4</td>
      <td>-5.617</td>
      <td>0.771</td>
      <td>0.714</td>
      <td>['pop']</td>
      <td>pop</td>
      <td>D/Re</td>
      <td>2019</td>
      <td>12</td>
      <td>27</td>
    </tr>
  </tbody>
</table>
<p>281 rows × 23 columns</p>
</div>




```python
sns.countplot(data=taylor_swift, y="year",hue='pitch_class');
```


    
![png](output_files/output_138_0.png)
    



```python
sns.countplot(data=taylor_swift, y="year",hue='month');
```


    
![png](output_files/output_139_0.png)
    


__Do this a better way__


```python
sns.countplot(data=liked_songs_clean, x="month", hue='year');
```


    
![png](output_files/output_141_0.png)
    


This isn't what I want. I want to see how many songs per month, but also see that sectioned off into year

I want to see two things: 
- how the number of songs added each month changes over the years
- how many songs are added each year
- how many songs overall are added each month

- I want this to be a lineplot that has month on the x axis and count per month on the y axis. Then I want one line per year



```python
#df  = pd.DataFrame()
df = ((liked_songs_clean.groupby(['year', 'month']).size())).to_frame()
```


```python
df = df.reset_index().rename(columns={0:'count'})
```


```python
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>month</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017</td>
      <td>3</td>
      <td>158</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017</td>
      <td>4</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017</td>
      <td>11</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>




```python
# compares plot of the difference between the number of songs added each month compared by year
sns.lineplot(data=df, x="month", y="count", hue="year", palette = "Paired");
```


    
![png](output_files/output_147_0.png)
    


Maybe use Altair instead to put these side by side per year


```python
t_swizzle = liked_songs_clean[liked_songs_clean.artists == 'Taylor Swift']
```


```python
liked_songs_clean.columns
```




    Index(['id', 'name', 'artists', 'duration_s', 'popularity', 'added_at',
           'acousticness', 'speechiness', 'key', 'liveness', 'instrumentalness',
           'energy', 'tempo', 'time_signature', 'loudness', 'danceability',
           'valence', 'genre_list', 'genre', 'pitch_class', 'year', 'month',
           'day'],
          dtype='object')




```python
sns.scatterplot(data=t_swizzle, x='year', y='popularity');

plt.xticks(rotation=85);
```


    
![png](output_files/output_151_0.png)
    


Change the above to a floating boxplot


```python
sns.boxplot(data=t_swizzle, y='year', x='popularity');

plt.xticks(rotation=90);
```


    
![png](output_files/output_153_0.png)
    



```python
sns.boxplot(x=t_swizzle['popularity']).set_title('Popularity')
plt.show()
```


    
![png](output_files/output_154_0.png)
    



```python
sns.boxplot(x=liked_songs_clean['popularity']).set_title('Popularity')
plt.show()
```


    
![png](output_files/output_155_0.png)
    


## T-Swizzle


```python
taylor_swift = liked_songs[liked_songs.artists =='Taylor Swift']
```


```python
taylor_swift.columns
```




    Index(['id', 'name', 'artists', 'duration_s', 'popularity', 'added_at',
           'acousticness', 'speechiness', 'key', 'liveness', 'instrumentalness',
           'energy', 'tempo', 'time_signature', 'loudness', 'danceability',
           'valence', 'genre_list', 'genre', 'pitch_class', 'year', 'month',
           'day'],
          dtype='object')




```python
taylor_swift.key.unique()
```




    array([ 5,  0,  7,  4,  9, 10, 11,  8,  6,  3,  2,  1], dtype=int64)




```python
sns.histplot(data=taylor_swift, x="popularity", hue='key');
```


    
![png](output_files/output_160_0.png)
    


Should I compare the data for my top artist this year vs last year? 

Most of TSwizzle's songs fall in the popularity of around 65-70



I want to see what the most and least popular song (tswizzle and otherwise) per year that I listen to per year 

Further visualization:  
2. which months I tend to add the most songs in  
3. compare my songs to spotify's top songs chart
4. map each audio feature to a color scale and create a picture or hue based on my artist and track preferences <-- audio aura
- add release date year for each track

## Analysis from eda2 with 
This notebook is based on this [article](https://blog.devgenius.io/spotify-data-analysis-with-python-a727542beaa7) with [repo](https://github.com/Vice10/ds_notebooks/blob/main/spotify_analysis.ipynb)


```python
# getting the most popular tracks in the dataset
liked_songs.sort_values(by=['popularity'], ascending=False)[['name', 'artists']].head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>artists</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>174</th>
      <td>Unholy (feat. Kim Petras)</td>
      <td>Sam Smith</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Anti-Hero</td>
      <td>Taylor Swift</td>
    </tr>
    <tr>
      <th>129</th>
      <td>I Ain't Worried</td>
      <td>OneRepublic</td>
    </tr>
    <tr>
      <th>2145</th>
      <td>Another Love</td>
      <td>Tom Odell</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Lavender Haze</td>
      <td>Taylor Swift</td>
    </tr>
    <tr>
      <th>577</th>
      <td>As It Was</td>
      <td>Harry Styles</td>
    </tr>
    <tr>
      <th>503</th>
      <td>Late Night Talking</td>
      <td>Harry Styles</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Snow On The Beach (feat. Lana Del Rey)</td>
      <td>Taylor Swift</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Midnight Rain</td>
      <td>Taylor Swift</td>
    </tr>
    <tr>
      <th>2237</th>
      <td>Blinding Lights</td>
      <td>The Weeknd</td>
    </tr>
    <tr>
      <th>338</th>
      <td>Bad Habit</td>
      <td>Steve Lacy</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Maroon</td>
      <td>Taylor Swift</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Vigilante Shit</td>
      <td>Taylor Swift</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Bejeweled</td>
      <td>Taylor Swift</td>
    </tr>
    <tr>
      <th>4314</th>
      <td>Yellow</td>
      <td>Coldplay</td>
    </tr>
    <tr>
      <th>592</th>
      <td>Vegas (From the Original Motion Picture Soundt...</td>
      <td>Doja Cat</td>
    </tr>
    <tr>
      <th>3172</th>
      <td>Watermelon Sugar</td>
      <td>Harry Styles</td>
    </tr>
    <tr>
      <th>269</th>
      <td>Woman</td>
      <td>Doja Cat</td>
    </tr>
    <tr>
      <th>1568</th>
      <td>I'm Not The Only One</td>
      <td>Sam Smith</td>
    </tr>
    <tr>
      <th>1296</th>
      <td>traitor</td>
      <td>Olivia Rodrigo</td>
    </tr>
  </tbody>
</table>
</div>



He also adds in the artist's followers and popularity - that could be cool to view. But that I'll have to add in the preprocessing. CLEAN UP YOUR CODE AND ORGANIZE THINGS. IT'S TOO COMPLICATED


```python
# top 20 genres by count
top_20c = pd.DataFrame(liked_songs['genre'].value_counts().head(20)).reset_index()
top_20c.set_axis(['genre', 'count'], inplace=True, axis=1)
sns.barplot(data=top_20c, y='genre', x='count').set(title='Number of Tracks By Genre (Top 20)');
```


    
![png](output_files/output_168_0.png)
    



```python
# popularity of the songs
# get the average popularity of the genres
top_20 = liked_songs.groupby('genre').mean().sort_values(by='popularity', ascending=False).head(20).reset_index()
top_20
sns.barplot(data=top_20, y='genre', x='popularity').set(title='Popularity of Tracks By Genre (Top 20)');
```


    
![png](output_files/output_169_0.png)
    


Maybe choose a different graph because the one above doesn't give you too much information

### Bivariate KDE


```python
sns.set(rc = {'figure.figsize':(20,20)})
sns.jointplot(data=liked_songs, x="loudness", y="energy", kind="kde");
```


    
![png](output_files/output_172_0.png)
    


He also does a feature portrait (whatever that is) and then some recommendations. Which I want to do

https://blog.devgenius.io/spotify-data-analysis-with-python-a727542beaa7

How are the most popular tracks different from all the tracks in the dataset? Let’s find out by plotting a feature portrait of the corresponding sets, given mean values of selected features.

## I NEED TO FIGURE THIS ONE OUT:

https://plotly.com/python/radar-chart/

## Spotify Wrapped Dataset


```python
my_top_songs_2022 = pd.read_csv("data/your_top_songs_2022.csv")
```


```python
# shape of dataset
my_top_songs_2022.shape
```




    (101, 23)



There are 101 songs on my top 2022 playlist created by spotify


```python
# change duration ms to s
my_top_songs_2022['Duration (ms)'] = my_top_songs_2022['Duration (ms)'] / 1000.0
```


```python
# make all col names lowercase
my_top_songs_2022.columns= my_top_songs_2022.columns.str.lower()
```


```python
my_top_songs_2022.columns
```




    Index(['spotify id', 'artist ids', 'track name', 'album name',
           'artist name(s)', 'release date', 'duration (ms)', 'popularity',
           'added by', 'added at', 'genres', 'danceability', 'energy', 'key',
           'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
           'liveness', 'valence', 'tempo', 'time signature'],
          dtype='object')




```python
# rename cols
my_top_songs_2022 = my_top_songs_2022.rename(
    columns={'spotify id':'spotify_id', 'artist ids':'artist_id', 'track name':'name', 'album name':'album',
       'artist name(s)':'artists', 'release date':'release_date', 'duration (ms)':'duration_s',
       'added by':'added_by', 'added at':'added_at', 'time signature':'time_signature'})
```


```python
my_top_songs_2022.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>spotify_id</th>
      <th>artist_id</th>
      <th>name</th>
      <th>album</th>
      <th>artists</th>
      <th>release_date</th>
      <th>duration_s</th>
      <th>popularity</th>
      <th>added_by</th>
      <th>added_at</th>
      <th>...</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>time_signature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0WttB2xYS66VopJmwD1UzF</td>
      <td>00x1fYSGhdqScXBRpSj3DW</td>
      <td>What Am I Gonna Do On Sundays?</td>
      <td>What Am I Gonna Do On Sundays?</td>
      <td>Olivia Dean</td>
      <td>2020-12-04</td>
      <td>209.213</td>
      <td>51</td>
      <td>spotify:user:</td>
      <td>1970-01-01T00:00:00Z</td>
      <td>...</td>
      <td>10</td>
      <td>-8.817</td>
      <td>1</td>
      <td>0.0414</td>
      <td>0.266</td>
      <td>0.000001</td>
      <td>0.104</td>
      <td>0.177</td>
      <td>135.537</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4LRPiXqCikLlN15c3yImP7</td>
      <td>6KImCVD70vtIoJWnq6nGn3</td>
      <td>As It Was</td>
      <td>As It Was</td>
      <td>Harry Styles</td>
      <td>2022-03-31</td>
      <td>167.303</td>
      <td>93</td>
      <td>spotify:user:</td>
      <td>1970-01-01T00:00:00Z</td>
      <td>...</td>
      <td>6</td>
      <td>-5.338</td>
      <td>0</td>
      <td>0.0557</td>
      <td>0.342</td>
      <td>0.001010</td>
      <td>0.311</td>
      <td>0.662</td>
      <td>173.930</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3ONe6SKdO3uRrWLsZePF1p</td>
      <td>55fhWPvDiMpLnE4ZzNXZyW</td>
      <td>Mr. Percocet</td>
      <td>The Hardest Part</td>
      <td>Noah Cyrus</td>
      <td>2022-09-16</td>
      <td>193.973</td>
      <td>61</td>
      <td>spotify:user:</td>
      <td>1970-01-01T00:00:00Z</td>
      <td>...</td>
      <td>2</td>
      <td>-6.340</td>
      <td>1</td>
      <td>0.0425</td>
      <td>0.179</td>
      <td>0.000089</td>
      <td>0.154</td>
      <td>0.862</td>
      <td>133.062</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 23 columns</p>
</div>




```python
# popularity plot of top songs
sns.histplot(data=my_top_songs_2022, x="popularity");
sns.set(rc={'figure.figsize':(7,7)})
```


    
![png](output_files/output_187_0.png)
    


# maybe compare to 2022 overall vs top songs of 2022
for col in audio_list:
    plt.figure(figsize=(21,2))
    sns.boxplot(data=top_six_genres, x=col, y="genre", palette="Spectral");


```python
# what year did I originally add the song? month? day?
# Add new columns for year month and day taken from the datetime column
my_top_songs_2022['year'] = pd.DatetimeIndex(my_top_songs_2022['release_date']).year
my_top_songs_2022['month'] = pd.DatetimeIndex(my_top_songs_2022['release_date']).month
my_top_songs_2022['day'] = pd.DatetimeIndex(my_top_songs_2022['release_date']).day

# view the first five rows of the columns we just added
my_top_songs_2022[['year','month','day']].head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020</td>
      <td>12</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022</td>
      <td>3</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022</td>
      <td>9</td>
      <td>16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016</td>
      <td>3</td>
      <td>25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022</td>
      <td>7</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.countplot(data=my_top_songs_2022, x='year');
```


    
![png](output_files/output_190_0.png)
    


The distribution of data


```python
sns.boxplot(x=liked_songs_clean['energy']).set_title('Energy')
```




    Text(0.5, 1.0, 'Energy')




    
![png](output_files/output_192_1.png)
    



```python
# top genres, top popularity for top songs
```

#### More ideas to look at 

- https://www.brandonlu.com/spotify-data-project 
- https://towardsdatascience.com/tagged/spotify
- https://github.com/areevesman/spotify-wrapped/blob/main/code/01_Data_Visualization.ipynb; https://towardsdatascience.com/spotify-wrapped-data-visualization-and-machine-learning-on-your-top-songs-1d3f837a9b27'
- https://towardsdatascience.com/reverse-engineering-spotify-wrapped-ai-using-python-452b58ad1a62
- https://towardsdatascience.com/build-your-first-mood-based-music-recommendation-system-in-python-26a427308d96
- https://towardsdatascience.com/visualizing-spotify-data-with-python-tableau-687f2f528cdd
- https://towardsdatascience.com/spotify-api-audio-features-5d8bcbd780b2


```python
# according to my spotify wrapped, these are my top 5 songs for 2022
my_top_5 = my_top_songs_2022.head(5)
my_top_5
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>spotify_id</th>
      <th>artist_id</th>
      <th>name</th>
      <th>album</th>
      <th>artists</th>
      <th>release_date</th>
      <th>duration_s</th>
      <th>popularity</th>
      <th>added_by</th>
      <th>added_at</th>
      <th>...</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0WttB2xYS66VopJmwD1UzF</td>
      <td>00x1fYSGhdqScXBRpSj3DW</td>
      <td>What Am I Gonna Do On Sundays?</td>
      <td>What Am I Gonna Do On Sundays?</td>
      <td>Olivia Dean</td>
      <td>2020-12-04</td>
      <td>209.213</td>
      <td>51</td>
      <td>spotify:user:</td>
      <td>1970-01-01T00:00:00Z</td>
      <td>...</td>
      <td>0.0414</td>
      <td>0.266</td>
      <td>0.000001</td>
      <td>0.1040</td>
      <td>0.177</td>
      <td>135.537</td>
      <td>4</td>
      <td>2020</td>
      <td>12</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4LRPiXqCikLlN15c3yImP7</td>
      <td>6KImCVD70vtIoJWnq6nGn3</td>
      <td>As It Was</td>
      <td>As It Was</td>
      <td>Harry Styles</td>
      <td>2022-03-31</td>
      <td>167.303</td>
      <td>93</td>
      <td>spotify:user:</td>
      <td>1970-01-01T00:00:00Z</td>
      <td>...</td>
      <td>0.0557</td>
      <td>0.342</td>
      <td>0.001010</td>
      <td>0.3110</td>
      <td>0.662</td>
      <td>173.930</td>
      <td>4</td>
      <td>2022</td>
      <td>3</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3ONe6SKdO3uRrWLsZePF1p</td>
      <td>55fhWPvDiMpLnE4ZzNXZyW</td>
      <td>Mr. Percocet</td>
      <td>The Hardest Part</td>
      <td>Noah Cyrus</td>
      <td>2022-09-16</td>
      <td>193.973</td>
      <td>61</td>
      <td>spotify:user:</td>
      <td>1970-01-01T00:00:00Z</td>
      <td>...</td>
      <td>0.0425</td>
      <td>0.179</td>
      <td>0.000089</td>
      <td>0.1540</td>
      <td>0.862</td>
      <td>133.062</td>
      <td>4</td>
      <td>2022</td>
      <td>9</td>
      <td>16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6lvsJDZ7336YmpBzcNGhbe</td>
      <td>5ZsFI1h6hIdQRw2ti0hz81</td>
      <td>fOoL fOr YoU</td>
      <td>Mind Of Mine (Deluxe Edition)</td>
      <td>ZAYN</td>
      <td>2016-03-25</td>
      <td>202.213</td>
      <td>67</td>
      <td>spotify:user:</td>
      <td>1970-01-01T00:00:00Z</td>
      <td>...</td>
      <td>0.0256</td>
      <td>0.281</td>
      <td>0.000000</td>
      <td>0.0976</td>
      <td>0.170</td>
      <td>77.903</td>
      <td>4</td>
      <td>2016</td>
      <td>3</td>
      <td>25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2qLyo5FeWquE7HBUbcVnEy</td>
      <td>74KM79TiuVKeVCqs8QtB0B</td>
      <td>emails i can’t send</td>
      <td>emails i can't send</td>
      <td>Sabrina Carpenter</td>
      <td>2022-07-15</td>
      <td>104.408</td>
      <td>71</td>
      <td>spotify:user:</td>
      <td>1970-01-01T00:00:00Z</td>
      <td>...</td>
      <td>0.1920</td>
      <td>0.757</td>
      <td>0.000000</td>
      <td>0.6220</td>
      <td>0.408</td>
      <td>86.313</td>
      <td>3</td>
      <td>2022</td>
      <td>7</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
# Get the distribution of the audio features for the top songs in my my_2022 playlist and compare
# sns.displot(my_top_5, x="flipper_length_mm")

# sns.displot(penguins, x="flipper_length_mm", hue="species", kind="kde", fill=True)

# but also do a distribution comparison by genre
```


```python
# spider plot of the audio features comparing my spotify wrapped to my whole dataset to the top 2022 songs on spotify
```

#### Combining the data from both and analyzing


```python
# drop extra cols
my_top_songs_2022.columns
my_top = my_top_songs_2022.drop(columns=[
       'duration_s', 'popularity', 'added_by', 'added_at', 'genres',
       'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
       'time_signature'],inplace=False)
```


```python
# merge info about for both cols
# find all the rows in like_songs that match the top songs
# x = pd.merge(liked_songs, my_top, on=['artists', 'name'])
```


```python
# the entries that were missing
missing = ['I’m Tired (with Zendaya) - Bonus Track',
 'Killing Me - acoustic',
 'Coast (feat. Anderson .Paak)',
 'Midnight River (feat. 6LACK)',
 'Every Beginning Ends',
 'Dear August',
 'Middle Ground (feat. CHIKA)',
 'Love Is Letting Go (feat. Diane Keaton)']
```

Resources

-- https://mode.com/blog/python-data-visualization-libraries/  
-- https://venngage.com/blog/data-visualization/


# Plan

Okay, finish comparing the two datasets  
Add information and graphs to website but in a comprehensive way. What conclusions came up?  
What overall info can be gathered?  

After it's added, create a youtube walkthrough video where you basically present what you did in your project.
Show that you can communicate your findings to people who aren't technical, but also can break down what and why

Recommendation system  
Use Tableau??  
Maybe make an interactive module or something.

Finally be done with this.


```python
!jupyter nbconvert sp_eda.ipynb --to markdown --output output.md
```

    Traceback (most recent call last):
      File "c:\users\candy\appdata\local\programs\python\python39\lib\runpy.py", line 197, in _run_module_as_main
        return _run_code(code, main_globals, None,
      File "c:\users\candy\appdata\local\programs\python\python39\lib\runpy.py", line 87, in _run_code
        exec(code, run_globals)
      File "C:\Users\candy\AppData\Local\Programs\Python\Python39\Scripts\jupyter-nbconvert.EXE\__main__.py", line 4, in <module>
      File "c:\users\candy\appdata\local\programs\python\python39\lib\site-packages\nbconvert\__init__.py", line 4, in <module>
        from .exporters import *
      File "c:\users\candy\appdata\local\programs\python\python39\lib\site-packages\nbconvert\exporters\__init__.py", line 3, in <module>
        from .html import HTMLExporter
      File "c:\users\candy\appdata\local\programs\python\python39\lib\site-packages\nbconvert\exporters\html.py", line 14, in <module>
        from jinja2 import contextfilter
    ImportError: cannot import name 'contextfilter' from 'jinja2' (c:\users\candy\appdata\local\programs\python\python39\lib\site-packages\jinja2\__init__.py)
    


```python
!pip install nbconvert==6.4.3
```

    Collecting nbconvert==6.4.3
      Downloading nbconvert-6.4.3-py3-none-any.whl (560 kB)
         ------------------------------------ 560.4/560.4 kB 550.1 kB/s eta 0:00:00
    Requirement already satisfied: nbclient<0.6.0,>=0.5.0 in c:\users\candy\appdata\local\programs\python\python39\lib\site-packages (from nbconvert==6.4.3) (0.5.3)
    Requirement already satisfied: nbformat>=4.4 in c:\users\candy\appdata\local\programs\python\python39\lib\site-packages (from nbconvert==6.4.3) (5.1.2)
    Requirement already satisfied: bleach in c:\users\candy\appdata\local\programs\python\python39\lib\site-packages (from nbconvert==6.4.3) (3.3.0)
    Requirement already satisfied: pygments>=2.4.1 in c:\users\candy\appdata\local\programs\python\python39\lib\site-packages (from nbconvert==6.4.3) (2.8.1)
    Requirement already satisfied: traitlets>=5.0 in c:\users\candy\appdata\local\programs\python\python39\lib\site-packages (from nbconvert==6.4.3) (5.0.5)
    Requirement already satisfied: jupyter-core in c:\users\candy\appdata\local\programs\python\python39\lib\site-packages (from nbconvert==6.4.3) (4.7.1)
    Requirement already satisfied: jupyterlab-pygments in c:\users\candy\appdata\local\programs\python\python39\lib\site-packages (from nbconvert==6.4.3) (0.1.2)
    Requirement already satisfied: entrypoints>=0.2.2 in c:\users\candy\appdata\local\programs\python\python39\lib\site-packages (from nbconvert==6.4.3) (0.3)
    Requirement already satisfied: pandocfilters>=1.4.1 in c:\users\candy\appdata\local\programs\python\python39\lib\site-packages (from nbconvert==6.4.3) (1.4.3)
    Requirement already satisfied: testpath in c:\users\candy\appdata\local\programs\python\python39\lib\site-packages (from nbconvert==6.4.3) (0.4.4)
    Requirement already satisfied: jinja2>=2.4 in c:\users\candy\appdata\local\programs\python\python39\lib\site-packages (from nbconvert==6.4.3) (3.1.2)
    Requirement already satisfied: mistune<2,>=0.8.1 in c:\users\candy\appdata\local\programs\python\python39\lib\site-packages (from nbconvert==6.4.3) (0.8.4)
    Requirement already satisfied: defusedxml in c:\users\candy\appdata\local\programs\python\python39\lib\site-packages (from nbconvert==6.4.3) (0.7.1)
    Requirement already satisfied: MarkupSafe>=2.0 in c:\users\candy\appdata\local\programs\python\python39\lib\site-packages (from jinja2>=2.4->nbconvert==6.4.3) (2.1.1)
    Requirement already satisfied: nest-asyncio in c:\users\candy\appdata\local\programs\python\python39\lib\site-packages (from nbclient<0.6.0,>=0.5.0->nbconvert==6.4.3) (1.5.1)
    Requirement already satisfied: jupyter-client>=6.1.5 in c:\users\candy\appdata\local\programs\python\python39\lib\site-packages (from nbclient<0.6.0,>=0.5.0->nbconvert==6.4.3) (7.1.2)
    Requirement already satisfied: async-generator in c:\users\candy\appdata\local\programs\python\python39\lib\site-packages (from nbclient<0.6.0,>=0.5.0->nbconvert==6.4.3) (1.10)
    Requirement already satisfied: ipython-genutils in c:\users\candy\appdata\local\programs\python\python39\lib\site-packages (from nbformat>=4.4->nbconvert==6.4.3) (0.2.0)
    Requirement already satisfied: jsonschema!=2.5.0,>=2.4 in c:\users\candy\appdata\local\programs\python\python39\lib\site-packages (from nbformat>=4.4->nbconvert==6.4.3) (3.2.0)
    Requirement already satisfied: six>=1.9.0 in c:\users\candy\appdata\local\programs\python\python39\lib\site-packages (from bleach->nbconvert==6.4.3) (1.15.0)
    Requirement already satisfied: webencodings in c:\users\candy\appdata\local\programs\python\python39\lib\site-packages (from bleach->nbconvert==6.4.3) (0.5.1)
    Requirement already satisfied: packaging in c:\users\candy\appdata\local\programs\python\python39\lib\site-packages (from bleach->nbconvert==6.4.3) (20.9)
    Requirement already satisfied: pywin32>=1.0 in c:\users\candy\appdata\local\programs\python\python39\lib\site-packages (from jupyter-core->nbconvert==6.4.3) (300)
    Requirement already satisfied: setuptools in c:\users\candy\appdata\local\programs\python\python39\lib\site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.4->nbconvert==6.4.3) (49.2.1)
    Requirement already satisfied: attrs>=17.4.0 in c:\users\candy\appdata\local\programs\python\python39\lib\site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.4->nbconvert==6.4.3) (20.3.0)
    Requirement already satisfied: pyrsistent>=0.14.0 in c:\users\candy\appdata\local\programs\python\python39\lib\site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.4->nbconvert==6.4.3) (0.17.3)
    Requirement already satisfied: pyzmq>=13 in c:\users\candy\appdata\local\programs\python\python39\lib\site-packages (from jupyter-client>=6.1.5->nbclient<0.6.0,>=0.5.0->nbconvert==6.4.3) (22.0.3)
    Requirement already satisfied: tornado>=4.1 in c:\users\candy\appdata\local\programs\python\python39\lib\site-packages (from jupyter-client>=6.1.5->nbclient<0.6.0,>=0.5.0->nbconvert==6.4.3) (6.1)
    Requirement already satisfied: python-dateutil>=2.1 in c:\users\candy\appdata\local\programs\python\python39\lib\site-packages (from jupyter-client>=6.1.5->nbclient<0.6.0,>=0.5.0->nbconvert==6.4.3) (2.8.1)
    Requirement already satisfied: pyparsing>=2.0.2 in c:\users\candy\appdata\local\programs\python\python39\lib\site-packages (from packaging->bleach->nbconvert==6.4.3) (2.4.7)
    Installing collected packages: nbconvert
      Attempting uninstall: nbconvert
        Found existing installation: nbconvert 6.0.7
        Uninstalling nbconvert-6.0.7:
          Successfully uninstalled nbconvert-6.0.7
    Successfully installed nbconvert-6.4.3
    


```python

```
