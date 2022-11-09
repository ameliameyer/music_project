{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "minute-still",
   "metadata": {},
   "source": [
    "# Exploring My Liked Songs on Spotify"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "separate-narrative",
   "metadata": {},
   "source": [
    "## This Project"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "activated-material",
   "metadata": {},
   "source": [
    "In this project I pulled my 'Liked' songs from Spotify using the Python library Spotipy. \n",
    "\n",
    "I used the data to answer the following:\n",
    "- Is there a relationship between different audio features of a song?\n",
    "- Are the songs I listen to more or less popular?\n",
    "- Which artists to add the most?\n",
    "- Is there a pattern to when I add the most songs or a certain type of songs? - seasons, months, days of the week, etc\n",
    "- Do I add more songs in the beginning or at the end of the month?\n",
    "- Which artists/songs \n",
    "- How do the songs I've added compare to the most popular songs on Spotify in 2022, 2021, ...?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "brave-polls",
   "metadata": {},
   "source": [
    "## Background on the Dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "spiritual-finding",
   "metadata": {},
   "source": [
    "*Data preprocessing and cleaning done in separate file. See `preprocessing.ipynb`\n",
    "\n",
    "This dataset has 4786 rows and 20. Each row represents a 'liked song'. The columns are defined in the table below: "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "catholic-macro",
   "metadata": {},
   "source": [
    "| variable | description | range |\n",
    "| --- | --- | --- |\n",
    "| acousticness | is track acoustic? | confidence measure, [0.0,1.0] |\n",
    "| added_at | date added | datetime format, 3/3/2017-10/7/2022|   \n",
    "| artists | name of artist associate with song | object |   \n",
    "| danceability | how suitable the track is for dancing | float, [0.0,1.0] |\n",
    "| duration_s | duration of track in seconds | float, [0.0,1.0] |\n",
    "| energy | perceptual measure of intensity and activity | float, [0.0,1.0] |\n",
    "| genre | top genre associated with artist of song | object |\n",
    "| genre_list | list of genres associated with artist of song | object |\n",
    "| pitch_class | pitch classes we took from the keys | object |\n",
    "| id | id number associated with song | object |\n",
    "| instrumentalness | predicts likelihood track contains no vocal content | float, [0.0,1.0] |\n",
    "| key | key of track | int, standard Pitch Class notation; [-1 when no value detected,11] |\n",
    "| liveliness | presence of audience in recording | float, [0.0,1.0] |\n",
    "| loudness | overall loudness of track in decibels | float, [-60,0] |\n",
    "| name | name of songs | object |\n",
    "| popularity | how popular a song is | [0.0,100.0] |\n",
    "| speechiness | presence of spoken words in a track | float, [0.0,1.0] |\n",
    "| tempo | tempo of track in beats per minute | float, [0.0,...] |\n",
    "| time_signature | estimated time signature | int, [3,7] indicating \"3/4\" to \"7/4\"|\n",
    "| valence | musical positivesness conveyed by a track | float, [0.0,1.0] |"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "combined-singles",
   "metadata": {},
   "source": [
    "Dates I added songs to my 'Liked' songs range from March 3rd, 2017 to Novemeber 7th, 2022 (the day I pulled the data from Spotify)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "viral-applicant",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<img src=\"liked_songs_clean.png\" width=\"1200\" height=\"1000\"/>"
      ],
      "text/plain": [
       "<IPython.core.display.Image object>"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from IPython.display import Image\n",
    "Image(url= \"liked_songs_clean.png\", width=1200, height=1000)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "nonprofit-protest",
   "metadata": {},
   "source": [
    "![Liked Songs Clean Dataset](liked_songs_clean.png?raw=true \"Title\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "substantial-cliff",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<img src=\"liked_songs_clean_describe.png\" width=\"1000\" height=\"700\"/>"
      ],
      "text/plain": [
       "<IPython.core.display.Image object>"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from IPython.display import Image\n",
    "Image(url= \"liked_songs_clean_describe.png\", width=1000, height=700)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "hindu-stroke",
   "metadata": {},
   "source": [
    "## Initial Visualizations"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "compressed-vienna",
   "metadata": {},
   "source": [
    "#### Pearson Correlation Matrix Heatmap"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "further-barrier",
   "metadata": {},
   "source": [
    "Correlation values range from -1 to 1. The closer a correlation value is to 1 (positive or negative), the stronger the correlation between the two variables. Variables with a strong positive correlation increase together, whereas variables with a strong negative correlation experience opposing polarization (as one goes up, the other goes down and vice versa). The closer the correlation is to 0, the weaker the correlation. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "anonymous-sigma",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<img src=\"heatmap.png\" width=\"800\" height=\"600\"/>"
      ],
      "text/plain": [
       "<IPython.core.display.Image object>"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from IPython.display import Image\n",
    "Image(url= \"heatmap.png\", width=800, height=600)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dried-retirement",
   "metadata": {},
   "source": [
    "- variables will always have a correlation of 1 with themselves\n",
    "- we can see the strongest positive correlation is between `energy` and `loudness` at 0.78. \n",
    "- the strongest negative correlation is between `energy` and `acousticness` at -0.8.\n",
    "- the weakest positive correlation is between `speechiness` and `day` at 0.00025\n",
    "- the weakest negative correlation is between `liveliness` and `duration_s` at -6.25e-05."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "later-horror",
   "metadata": {},
   "source": [
    "Some of these correlations make sense. You would consider songs that have more energy to be louder. You would also consider energy and acoustiness to be negativily correlation i.e. as energy goes up, acousticness goes down and vice versa."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "textile-curve",
   "metadata": {},
   "source": [
    "#### Popularity of the songs I've added sorted by popularity in descending order "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "federal-saudi",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<img src=\"popularity.png\" width=\"800\" height=\"600\"/>"
      ],
      "text/plain": [
       "<IPython.core.display.Image object>"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from IPython.display import Image\n",
    "Image(url= \"popularity.png\", width=800, height=600)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "useful-elimination",
   "metadata": {},
   "source": [
    "#### Audio Features Over Time"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "similar-router",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<img src=\"popularity_time.png\" width=\"800\" height=\"600\"/>"
      ],
      "text/plain": [
       "<IPython.core.display.Image object>"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from IPython.display import Image\n",
    "Image(url= \"popularity_time.png\", width=800, height=600)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "wrong-repair",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<img src=\"duration_time.png\" width=\"800\" height=\"600\"/>"
      ],
      "text/plain": [
       "<IPython.core.display.Image object>"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from IPython.display import Image\n",
    "Image(url= \"duration_time.png\", width=800, height=600)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "phantom-avatar",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<img src=\"danceability_time.png\" width=\"800\" height=\"600\"/>"
      ],
      "text/plain": [
       "<IPython.core.display.Image object>"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from IPython.display import Image\n",
    "Image(url= \"danceability_time.png\", width=800, height=600)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "numerous-guidance",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<img src=\"valence_time.png\" width=\"800\" height=\"600\"/>"
      ],
      "text/plain": [
       "<IPython.core.display.Image object>"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from IPython.display import Image\n",
    "Image(url= \"valence_time.png\", width=800, height=600)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "brutal-growing",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<img src=\"energy_time.png\" width=\"800\" height=\"600\"/>"
      ],
      "text/plain": [
       "<IPython.core.display.Image object>"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from IPython.display import Image\n",
    "Image(url= \"energy_time.png\", width=800, height=600)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "interested-amsterdam",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<img src=\"tempo_time.png\" width=\"800\" height=\"600\"/>"
      ],
      "text/plain": [
       "<IPython.core.display.Image object>"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from IPython.display import Image\n",
    "Image(url= \"tempo_time.png\", width=800, height=600)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "endangered-distribution",
   "metadata": {},
   "source": [
    "## Findings and Conclusions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "floating-operator",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
