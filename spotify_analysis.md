# Spotify Music Project

I extracted my Liked Songs from Spotify using [Spotipy](https://spotipy.readthedocs.io/en/2.21.0/), a Python library for the Spotify API, in order to perform analysis on my Spotify data. 

## Important Files:
- Pre-processing
- EDA 
- EDA Writeup and Visualizations

## Further Exploration

**this project is ongoing**

# Exploring My Liked Songs on Spotify

## This Project

In this project I pulled my 'Liked' songs from Spotify using the Python library Spotipy. 

I used the data to answer the following:
- Is there a relationship between different audio features of a song?
- Are the songs I listen to more or less popular?
- Which artists to add the most?
- Is there a pattern to when I add the most songs or a certain type of songs? - seasons, months, days of the week, etc
- Do I add more songs in the beginning or at the end of the month?
- Which artists/songs 
- How do the songs I've added compare to the most popular songs on Spotify in 2022, 2021, ...?

## Background on the Dataset

*Data preprocessing and cleaning done in separate file. See `preprocessing.ipynb`

This dataset has 4786 rows and 20. Each row represents a 'liked song'. The columns are defined in the table below:   

| variable | description | range |  
| --- | --- | --- |
| acousticness | is track acoustic? | confidence measure, [0.0,1.0] |
| added_at | date added | datetime format, 3/3/2017-10/7/2022|   
| artists | name of artist associate with song | object |   
| danceability | how suitable the track is for dancing | float, [0.0,1.0] |
| duration_s | duration of track in seconds | float, [0.0,1.0] |
| energy | perceptual measure of intensity and activity | float, [0.0,1.0] |
| genre | top genre associated with artist of song | object |
| genre_list | list of genres associated with artist of song | object |
| pitch_class | pitch classes we took from the keys | object |
| id | id number associated with song | object |
| instrumentalness | predicts likelihood track contains no vocal content | float, [0.0,1.0] |
| key | key of track | int, standard Pitch Class notation; [-1 when no value detected,11] |
| liveliness | presence of audience in recording | float, [0.0,1.0] |
| loudness | overall loudness of track in decibels | float, [-60,0] |
| name | name of songs | object |
| popularity | how popular a song is | [0.0,100.0] |
| speechiness | presence of spoken words in a track | float, [0.0,1.0] |
| tempo | tempo of track in beats per minute | float, [0.0,...] |
| time_signature | estimated time signature | int, [3,7] indicating "3/4" to "7/4"|
| valence | musical positivesness conveyed by a track | float, [0.0,1.0] |  

Dates I added songs to my 'Liked' songs range from March 3rd, 2017 to Novemeber 7th, 2022 (the day I pulled the data from Spotify).  

![Liked Songs Clean Dataset](/images/liked_songs_clean.png) 


![Liked Songs Clean Dataset Describe](/images/liked_songs_clean_describe.png)

## Initial Visualizations

#### Pearson Correlation Matrix Heatmap

Correlation values range from -1 to 1. The closer a correlation value is to 1 (positive or negative), the stronger the correlation between the two variables. Variables with a strong positive correlation increase together, whereas variables with a strong negative correlation experience opposing polarization (as one goes up, the other goes down and vice versa). The closer the correlation is to 0, the weaker the correlation. 


<img src="/images/heatmap.png" alt="Pearson Correlation Matrix Heatmap" width="800"/>

- variables will always have a correlation of 1 with themselves
- we can see the strongest positive correlation is between `energy` and `loudness` at 0.78. 
- the strongest negative correlation is between `energy` and `acousticness` at -0.8.
- the weakest positive correlation is between `speechiness` and `day` at 0.00025
- the weakest negative correlation is between `liveliness` and `duration_s` at -6.25e-05.

Some of these correlations make sense. You would consider songs that have more energy to be louder. You would also consider energy and acoustiness to be negativily correlation i.e. as energy goes up, acousticness goes down and vice versa.

#### Popularity of the songs I've added sorted by popularity in descending order 

<img src="/images/popularity.png" alt="Popularity Table" width="800"/>

#### Audio Features Over Time

![Popularity Over Time](/images/popularity_time.png)
![Duration Over Time](/images/duration_time.png)
![Danceability Over Time](/images/danceability_time.png)
![Valence Over Time](/images/valence_time.png)
![Energy Over Time](/images/energy_time.png)
![Tempo Over Time](/images/tempo_time.png)

## Findings and Conclusions
