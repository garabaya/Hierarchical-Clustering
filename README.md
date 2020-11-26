# Hierarchical Clustering

**Hierarchical clustering** refers to a class of clustering methods that seek to build a **hierarchy** of clusters, in which some clusters contain others. In this assignment, we will explore a top-down approach, recursively bipartitioning the data using k-means.

**Note to Amazon EC2 users**: To conserve memory, make sure to stop all the other notebooks before running this notebook.

## Import packages


```python
from __future__ import print_function # to conform python 2.x print to python 3.x
import turicreate
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import time
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
%matplotlib inline
```

## Load the Wikipedia dataset


```python
wiki = turicreate.SFrame('people_wiki.sframe/')
```

As we did in previous assignments, let's extract the TF-IDF features:


```python
wiki['tf_idf'] = turicreate.text_analytics.tf_idf(wiki['text'])
```

To run k-means on this dataset, we should convert the data matrix into a sparse matrix.


```python
from em_utilities import sframe_to_scipy # converter

# This will take about a minute or two.
wiki = wiki.add_row_number()
tf_idf, map_word_to_index = sframe_to_scipy(wiki, 'tf_idf')
```

To be consistent with the k-means assignment, let's normalize all vectors to have unit norm.


```python
from sklearn.preprocessing import normalize
tf_idf = normalize(tf_idf)
```

## Bipartition the Wikipedia dataset using k-means

Recall our workflow for clustering text data with k-means:

1. Load the dataframe containing a dataset, such as the Wikipedia text dataset.
2. Extract the data matrix from the dataframe.
3. Run k-means on the data matrix with some value of k.
4. Visualize the clustering results using the centroids, cluster assignments, and the original dataframe. We keep the original dataframe around because the data matrix does not keep auxiliary information (in the case of the text dataset, the title of each article).

Let us modify the workflow to perform bipartitioning:

1. Load the dataframe containing a dataset, such as the Wikipedia text dataset.
2. Extract the data matrix from the dataframe.
3. Run k-means on the data matrix with k=2.
4. Divide the data matrix into two parts using the cluster assignments.
5. Divide the dataframe into two parts, again using the cluster assignments. This step is necessary to allow for visualization.
6. Visualize the bipartition of data.

We'd like to be able to repeat Steps 3-6 multiple times to produce a **hierarchy** of clusters such as the following:
```
                      (root)
                         |
            +------------+-------------+
            |                          |
         Cluster                    Cluster
     +------+-----+             +------+-----+
     |            |             |            |
   Cluster     Cluster       Cluster      Cluster
```
Each **parent cluster** is bipartitioned to produce two **child clusters**. At the very top is the **root cluster**, which consists of the entire dataset.

Now we write a wrapper function to bipartition a given cluster using k-means. There are three variables that together comprise the cluster:

* `dataframe`: a subset of the original dataframe that correspond to member rows of the cluster
* `matrix`: same set of rows, stored in sparse matrix format
* `centroid`: the centroid of the cluster (not applicable for the root cluster)

Rather than passing around the three variables separately, we package them into a Python dictionary. The wrapper function takes a single dictionary (representing a parent cluster) and returns two dictionaries (representing the child clusters).


```python
def bipartition(cluster, maxiter=400, num_runs=4, seed=None):
    '''cluster: should be a dictionary containing the following keys
                * dataframe: original dataframe
                * matrix:    same data, in matrix format
                * centroid:  centroid for this particular cluster'''
    
    data_matrix = cluster['matrix']
    dataframe   = cluster['dataframe']
    
    # Run k-means on the data matrix with k=2. We use scikit-learn here to simplify workflow.
    kmeans_model = KMeans(n_clusters=2, max_iter=maxiter, n_init=num_runs, random_state=seed, n_jobs=1)
    kmeans_model.fit(data_matrix)
    centroids, cluster_assignment = kmeans_model.cluster_centers_, kmeans_model.labels_
    
    # Divide the data matrix into two parts using the cluster assignments.
    data_matrix_left_child, data_matrix_right_child = data_matrix[cluster_assignment==0], \
                                                      data_matrix[cluster_assignment==1]
    
    # Divide the dataframe into two parts, again using the cluster assignments.
    cluster_assignment_sa = turicreate.SArray(cluster_assignment) # minor format conversion
    dataframe_left_child, dataframe_right_child     = dataframe[cluster_assignment_sa==0], \
                                                      dataframe[cluster_assignment_sa==1]
        
    
    # Package relevant variables for the child clusters
    cluster_left_child  = {'matrix': data_matrix_left_child,
                           'dataframe': dataframe_left_child,
                           'centroid': centroids[0]}
    cluster_right_child = {'matrix': data_matrix_right_child,
                           'dataframe': dataframe_right_child,
                           'centroid': centroids[1]}
    
    return (cluster_left_child, cluster_right_child)
```

The following cell performs bipartitioning of the Wikipedia dataset. Allow 2+ minutes to finish.

Note. For the purpose of the assignment, we set an explicit seed (`seed=1`) to produce identical outputs for every run. In pratical applications, you might want to use different random seeds for all runs.


```python
%%time
wiki_data = {'matrix': tf_idf, 'dataframe': wiki} # no 'centroid' for the root cluster
left_child, right_child = bipartition(wiki_data, maxiter=100, num_runs=1, seed=0)
```

    /home/ruben/venv/lib/python3.6/site-packages/sklearn/cluster/_kmeans.py:939: FutureWarning: 'n_jobs' was deprecated in version 0.23 and will be removed in 0.25.
      " removed in 0.25.", FutureWarning)


    CPU times: user 1.86 s, sys: 125 ms, total: 1.98 s
    Wall time: 1.99 s


Let's examine the contents of one of the two clusters, which we call the `left_child`, referring to the tree visualization above.


```python
left_child
```




    {'matrix': <47561x547979 sparse matrix of type '<class 'numpy.float64'>'
     	with 8493452 stored elements in Compressed Sparse Row format>,
     'dataframe': Columns:
     	id	int
     	URI	str
     	name	str
     	text	str
     	tf_idf	dict
     
     Rows: Unknown
     
     Data:
     +----+-------------------------------+---------------------+
     | id |              URI              |         name        |
     +----+-------------------------------+---------------------+
     | 1  | <http://dbpedia.org/resour... |    Alfred J. Lewy   |
     | 2  | <http://dbpedia.org/resour... |    Harpdog Brown    |
     | 3  | <http://dbpedia.org/resour... | Franz Rottensteiner |
     | 4  | <http://dbpedia.org/resour... |        G-Enka       |
     | 5  | <http://dbpedia.org/resour... |    Sam Henderson    |
     | 6  | <http://dbpedia.org/resour... |    Aaron LaCrate    |
     | 7  | <http://dbpedia.org/resour... |   Trevor Ferguson   |
     | 8  | <http://dbpedia.org/resour... |     Grant Nelson    |
     | 9  | <http://dbpedia.org/resour... |     Cathy Caruth    |
     | 10 | <http://dbpedia.org/resour... |     Sophie Crumb    |
     +----+-------------------------------+---------------------+
     +-------------------------------+-------------------------------+
     |              text             |             tf_idf            |
     +-------------------------------+-------------------------------+
     | alfred j lewy aka sandy le... | {'time': 1.325334207420049... |
     | harpdog brown is a singer ... | {'society': 2.444804726208... |
     | franz rottensteiner born i... | {'kurdlawitzpreis': 10.986... |
     | henry krvits born 30 decem... | {'curtis': 5.2995200328853... |
     | sam henderson born october... | {'asses': 9.60020102810530... |
     | aaron lacrate is an americ... | {'streamz': 10.98649538922... |
     | trevor ferguson aka john f... | {'concordia': 6.2502969408... |
     | grant nelson born 27 april... | {'heavies': 8.907053847545... |
     | cathy caruth born 1955 is ... | {'2002': 1.875312588782230... |
     | sophia violet sophie crumb... | {'who': 0.9098952189804214... |
     +-------------------------------+-------------------------------+
     [? rows x 5 columns]
     Note: Only the head of the SFrame is printed. This SFrame is lazily evaluated.
     You can use sf.materialize() to force materialization.,
     'centroid': array([5.99333810e-05, 1.39284997e-05, 6.85264647e-05, ...,
            2.03016100e-06, 6.24672345e-06, 2.31038436e-06])}



And here is the content of the other cluster we named `right_child`.


```python
right_child
```




    {'matrix': <11510x547979 sparse matrix of type '<class 'numpy.float64'>'
     	with 1885831 stored elements in Compressed Sparse Row format>,
     'dataframe': Columns:
     	id	int
     	URI	str
     	name	str
     	text	str
     	tf_idf	dict
     
     Rows: Unknown
     
     Data:
     +----+-------------------------------+-------------------------------+
     | id |              URI              |              name             |
     +----+-------------------------------+-------------------------------+
     | 0  | <http://dbpedia.org/resour... |         Digby Morrell         |
     | 17 | <http://dbpedia.org/resour... | Paddy Dunne (Gaelic footba... |
     | 21 | <http://dbpedia.org/resour... |         Ceiron Thomas         |
     | 22 | <http://dbpedia.org/resour... |          Adel Sellimi         |
     | 25 | <http://dbpedia.org/resour... |          Vic Stasiuk          |
     | 28 | <http://dbpedia.org/resour... |          Leon Hapgood         |
     | 30 | <http://dbpedia.org/resour... |           Dom Flora           |
     | 33 | <http://dbpedia.org/resour... |           Bob Reece           |
     | 41 | <http://dbpedia.org/resour... | Bob Adams (American football) |
     | 48 | <http://dbpedia.org/resour... |           Marc Logan          |
     +----+-------------------------------+-------------------------------+
     +-------------------------------+-------------------------------+
     |              text             |             tf_idf            |
     +-------------------------------+-------------------------------+
     | digby morrell born 10 octo... | {'melbourne': 3.8914310119... |
     | paddy dunne was a gaelic f... | {'hall': 2.635356782139039... |
     | ceiron thomas born 23 octo... | {'resulting': 4.6286531227... |
     | adel sellimi arabic was bo... | {'sport': 3.80871297302999... |
     | victor john stasiuk born m... | {'coaching4': 10.986495389... |
     | leon duane hapgood born 7 ... | {'where': 1.08907621209067... |
     | dominick a dom flora born ... | {'admission': 6.0097616468... |
     | robert scott reece born ja... | {'out': 1.8484031814566355... |
     | robert bruce bob adams bor... | {'officials': 4.4170139688... |
     | marc anthony logan born ma... | {'part': 1.919448187075487... |
     +-------------------------------+-------------------------------+
     [? rows x 5 columns]
     Note: Only the head of the SFrame is printed. This SFrame is lazily evaluated.
     You can use sf.materialize() to force materialization.,
     'centroid': array([0.00050277, 0.00038182, 0.00014018, ..., 0.        , 0.        ,
            0.        ])}



## Visualize the bipartition

We provide you with a modified version of the visualization function from the k-means assignment. For each cluster, we print the top 5 words with highest TF-IDF weights in the centroid and display excerpts for the 8 nearest neighbors of the centroid.


```python
def display_single_tf_idf_cluster(cluster, map_index_to_word):
    '''map_index_to_word: SFrame specifying the mapping betweeen words and column indices'''
    
    wiki_subset   = cluster['dataframe']
    tf_idf_subset = cluster['matrix']
    centroid      = cluster['centroid']
    
    # Print top 5 words with largest TF-IDF weights in the cluster
    idx = centroid.argsort()[::-1]
    for i in range(5):
        print('{0}:{1:.3f}'.format(map_index_to_word['category'], centroid[idx[i]])),
    print('')
    
    # Compute distances from the centroid to all data points in the cluster.
    distances = pairwise_distances(tf_idf_subset, [centroid], metric='euclidean').flatten()
    # compute nearest neighbors of the centroid within the cluster.
    nearest_neighbors = distances.argsort()
    # For 8 nearest neighbors, print the title as well as first 180 characters of text.
    # Wrap the text at 80-character mark.
    for i in range(8):
        text = ' '.join(wiki_subset[nearest_neighbors[i]]['text'].split(None, 25)[0:25])
        print('* {0:50s} {1:.5f}\n  {2:s}\n  {3:s}'.format(wiki_subset[nearest_neighbors[i]]['name'],
              distances[nearest_neighbors[i]], text[:90], text[90:180] if len(text) > 90 else ''))
    print('')
```

Let's visualize the two child clusters:


```python
display_single_tf_idf_cluster(left_child, map_word_to_index)
```

    113949:0.025
    113949:0.017
    113949:0.012
    113949:0.011
    113949:0.011
    
    * Anita Kunz                                         0.97401
      anita e kunz oc born 1956 is a canadianborn artist and illustratorkunz has lived in london
       new york and toronto contributing to magazines and working
    * Janet Jackson                                      0.97472
      janet damita jo jackson born may 16 1966 is an american singer songwriter and actress know
      n for a series of sonically innovative socially conscious and
    * Madonna (entertainer)                              0.97475
      madonna louise ciccone tkoni born august 16 1958 is an american singer songwriter actress 
      and businesswoman she achieved popularity by pushing the boundaries of lyrical
    * %C3%81ine Hyland                                   0.97536
      ine hyland ne donlon is emeritus professor of education and former vicepresident of univer
      sity college cork ireland she was born in 1942 in athboy co
    * Jane Fonda                                         0.97621
      jane fonda born lady jayne seymour fonda december 21 1937 is an american actress writer po
      litical activist former fashion model and fitness guru she is
    * Christine Robertson                                0.97643
      christine mary robertson born 5 october 1948 is an australian politician and former austra
      lian labor party member of the new south wales legislative council serving
    * Pat Studdy-Clift                                   0.97643
      pat studdyclift is an australian author specialising in historical fiction and nonfictionb
      orn in 1925 she lived in gunnedah until she was sent to a boarding
    * Alexandra Potter                                   0.97646
      alexandra potter born 1970 is a british author of romantic comediesborn in bradford yorksh
      ire england and educated at liverpool university gaining an honors degree in
    



```python
display_single_tf_idf_cluster(right_child, map_word_to_index)
```

    113949:0.040
    113949:0.036
    113949:0.029
    113949:0.029
    113949:0.028
    
    * Todd Williams                                      0.95468
      todd michael williams born february 13 1971 in syracuse new york is a former major league 
      baseball relief pitcher he attended east syracuseminoa high school
    * Gord Sherven                                       0.95622
      gordon r sherven born august 21 1963 in gravelbourg saskatchewan and raised in mankota sas
      katchewan is a retired canadian professional ice hockey forward who played
    * Justin Knoedler                                    0.95639
      justin joseph knoedler born july 17 1980 in springfield illinois is a former major league 
      baseball catcherknoedler was originally drafted by the st louis cardinals
    * Chris Day                                          0.95648
      christopher nicholas chris day born 28 july 1975 is an english professional footballer who
       plays as a goalkeeper for stevenageday started his career at tottenham
    * Tony Smith (footballer, born 1957)                 0.95653
      anthony tony smith born 20 february 1957 is a former footballer who played as a central de
      fender in the football league in the 1970s and
    * Ashley Prescott                                    0.95761
      ashley prescott born 11 september 1972 is a former australian rules footballer he played w
      ith the richmond and fremantle football clubs in the afl between
    * Leslie Lea                                         0.95802
      leslie lea born 5 october 1942 in manchester is an english former professional footballer 
      he played as a midfielderlea began his professional career with blackpool
    * Tommy Anderson (footballer)                        0.95818
      thomas cowan tommy anderson born 24 september 1934 in haddington is a scottish former prof
      essional footballer he played as a forward and was noted for
    


The right cluster consists of athletes and artists (singers and actors/actresses), whereas the left cluster consists of non-athletes and non-artists. So far, we have a single-level hierarchy consisting of two clusters, as follows:

```
                                           Wikipedia
                                               +
                                               |
                    +--------------------------+--------------------+
                    |                                               |
                    +                                               +
         Non-athletes/artists                                Athletes/artists
```

Is this hierarchy good enough? **When building a hierarchy of clusters, we must keep our particular application in mind.** For instance, we might want to build a **directory** for Wikipedia articles. A good directory would let you quickly narrow down your search to a small set of related articles. The categories of athletes and non-athletes are too general to facilitate efficient search. For this reason, we decide to build another level into our hierarchy of clusters with the goal of getting more specific cluster structure at the lower level. To that end, we subdivide both the `athletes/artists` and `non-athletes/artists` clusters.

## Perform recursive bipartitioning

### Cluster of athletes and artists

To help identify the clusters we've built so far, let's give them easy-to-read aliases:


```python
non_athletes_artists   = left_child
athletes_artists       = right_child
```

Using the bipartition function, we produce two child clusters of the athlete cluster:


```python
# Bipartition the cluster of athletes and artists
left_child_athletes_artists, right_child_athletes_artists = bipartition(athletes_artists, maxiter=100, num_runs=6, seed=1)
```

    /home/ruben/venv/lib/python3.6/site-packages/sklearn/cluster/_kmeans.py:939: FutureWarning: 'n_jobs' was deprecated in version 0.23 and will be removed in 0.25.
      " removed in 0.25.", FutureWarning)


The left child cluster mainly consists of athletes:


```python
display_single_tf_idf_cluster(left_child_athletes_artists, map_word_to_index)
```

    113949:0.054
    113949:0.043
    113949:0.038
    113949:0.035
    113949:0.030
    
    * Tony Smith (footballer, born 1957)                 0.94677
      anthony tony smith born 20 february 1957 is a former footballer who played as a central de
      fender in the football league in the 1970s and
    * Justin Knoedler                                    0.94746
      justin joseph knoedler born july 17 1980 in springfield illinois is a former major league 
      baseball catcherknoedler was originally drafted by the st louis cardinals
    * Chris Day                                          0.94849
      christopher nicholas chris day born 28 july 1975 is an english professional footballer who
       plays as a goalkeeper for stevenageday started his career at tottenham
    * Todd Williams                                      0.94882
      todd michael williams born february 13 1971 in syracuse new york is a former major league 
      baseball relief pitcher he attended east syracuseminoa high school
    * Todd Curley                                        0.95007
      todd curley born 14 january 1973 is a former australian rules footballer who played for co
      llingwood and the western bulldogs in the australian football league
    * Ashley Prescott                                    0.95015
      ashley prescott born 11 september 1972 is a former australian rules footballer he played w
      ith the richmond and fremantle football clubs in the afl between
    * Tommy Anderson (footballer)                        0.95037
      thomas cowan tommy anderson born 24 september 1934 in haddington is a scottish former prof
      essional footballer he played as a forward and was noted for
    * Leslie Lea                                         0.95065
      leslie lea born 5 october 1942 in manchester is an english former professional footballer 
      he played as a midfielderlea began his professional career with blackpool
    


On the other hand, the right child cluster consists mainly of artists (singers and actors/actresses):


```python
display_single_tf_idf_cluster(right_child_athletes_artists, map_word_to_index)
```

    113949:0.045
    113949:0.043
    113949:0.035
    113949:0.031
    113949:0.031
    
    * Alessandra Aguilar                                 0.93880
      alessandra aguilar born 1 july 1978 in lugo is a spanish longdistance runner who specialis
      es in marathon running she represented her country in the event
    * Heather Samuel                                     0.93999
      heather barbara samuel born 6 july 1970 is a retired sprinter from antigua and barbuda who
       specialized in the 100 and 200 metres in 1990
    * Viola Kibiwot                                      0.94037
      viola jelagat kibiwot born december 22 1983 in keiyo district is a runner from kenya who s
      pecialises in the 1500 metres kibiwot won her first
    * Ayelech Worku                                      0.94052
      ayelech worku born june 12 1979 is an ethiopian longdistance runner most known for winning
       two world championships bronze medals on the 5000 metres she
    * Krisztina Papp                                     0.94105
      krisztina papp born 17 december 1982 in eger is a hungarian long distance runner she is th
      e national indoor record holder over 5000 mpapp began
    * Petra Lammert                                      0.94230
      petra lammert born 3 march 1984 in freudenstadt badenwrttemberg is a former german shot pu
      tter and current bobsledder she was the 2009 european indoor champion
    * Morhad Amdouni                                     0.94231
      morhad amdouni born 21 january 1988 in portovecchio is a french middle and longdistance ru
      nner he was european junior champion in track and cross country
    * Brian Davis (golfer)                               0.94378
      brian lester davis born 2 august 1974 is an english professional golferdavis was born in l
      ondon he turned professional in 1994 and became a member
    


Our hierarchy of clusters now looks like this:
```
                                           Wikipedia
                                               +
                                               |
                    +--------------------------+--------------------+
                    |                                               |
                    +                                               +
         Non-athletes/artists                                Athletes/artists
                                                                    +
                                                                    |
                                                         +----------+----------+
                                                         |                     |
                                                         |                     |
                                                         +                     |
                                                     athletes               artists
```

Should we keep subdividing the clusters? If so, which cluster should we subdivide? To answer this question, we again think about our application. Since we organize our directory by topics, it would be nice to have topics that are about as coarse as each other. For instance, if one cluster is about baseball, we expect some other clusters about football, basketball, volleyball, and so forth. That is, **we would like to achieve similar level of granularity for all clusters.**

Both the athletes and artists node can be subdivided more, as each one can be divided into more descriptive professions (singer/actress/painter/director, or baseball/football/basketball, etc.). Let's explore subdividing the athletes cluster further to produce finer child clusters.

Let's give the clusters aliases as well:


```python
athletes    = left_child_athletes_artists
artists     = right_child_athletes_artists
```

### Cluster of athletes

In answering the following quiz question, take a look at the topics represented in the top documents (those closest to the centroid), as well as the list of words with highest TF-IDF weights.

Let us bipartition the cluster of athletes.


```python
left_child_athletes, right_child_athletes = bipartition(athletes, maxiter=100, num_runs=6, seed=1)
```

    /home/ruben/venv/lib/python3.6/site-packages/sklearn/cluster/_kmeans.py:939: FutureWarning: 'n_jobs' was deprecated in version 0.23 and will be removed in 0.25.
      " removed in 0.25.", FutureWarning)



```python
display_single_tf_idf_cluster(left_child_athletes, map_word_to_index)
display_single_tf_idf_cluster(right_child_athletes, map_word_to_index)
```

    113949:0.110
    113949:0.102
    113949:0.051
    113949:0.046
    113949:0.045
    
    * Steve Springer                                     0.89370
      steven michael springer born february 11 1961 is an american former professional baseball 
      player who appeared in major league baseball as a third baseman and
    * Dave Ford                                          0.89622
      david alan ford born december 29 1956 is a former major league baseball pitcher for the ba
      ltimore orioles born in cleveland ohio ford attended lincolnwest
    * Todd Williams                                      0.89829
      todd michael williams born february 13 1971 in syracuse new york is a former major league 
      baseball relief pitcher he attended east syracuseminoa high school
    * Justin Knoedler                                    0.90102
      justin joseph knoedler born july 17 1980 in springfield illinois is a former major league 
      baseball catcherknoedler was originally drafted by the st louis cardinals
    * Kevin Nicholson (baseball)                         0.90619
      kevin ronald nicholson born march 29 1976 is a canadian baseball shortstop he played part 
      of the 2000 season for the san diego padres of
    * Joe Strong                                         0.90658
      joseph benjamin strong born september 9 1962 in fairfield california is a former major lea
      gue baseball pitcher who played for the florida marlins from 2000
    * James Baldwin (baseball)                           0.90691
      james j baldwin jr born july 15 1971 is a former major league baseball pitcher he batted a
      nd threw righthanded in his 11season career he
    * James Garcia                                       0.90738
      james robert garcia born february 3 1980 is an american former professional baseball pitch
      er who played in the san francisco giants minor league system as
    
    113949:0.048
    113949:0.043
    113949:0.041
    113949:0.036
    113949:0.034
    
    * Todd Curley                                        0.94563
      todd curley born 14 january 1973 is a former australian rules footballer who played for co
      llingwood and the western bulldogs in the australian football league
    * Tony Smith (footballer, born 1957)                 0.94590
      anthony tony smith born 20 february 1957 is a former footballer who played as a central de
      fender in the football league in the 1970s and
    * Chris Day                                          0.94605
      christopher nicholas chris day born 28 july 1975 is an english professional footballer who
       plays as a goalkeeper for stevenageday started his career at tottenham
    * Jason Roberts (footballer)                         0.94617
      jason andre davis roberts mbe born 25 january 1978 is a former professional footballer and
       now a football punditborn in park royal london roberts was
    * Ashley Prescott                                    0.94618
      ashley prescott born 11 september 1972 is a former australian rules footballer he played w
      ith the richmond and fremantle football clubs in the afl between
    * David Hamilton (footballer)                        0.94910
      david hamilton born 7 november 1960 is an english former professional association football
       player who played as a midfielder he won caps for the england
    * Richard Ambrose                                    0.94924
      richard ambrose born 10 june 1972 is a former australian rules footballer who played with 
      the sydney swans in the australian football league afl he
    * Neil Grayson                                       0.94944
      neil grayson born 1 november 1964 in york is an english footballer who last played as a st
      riker for sutton towngraysons first club was local
    


**Quiz Question**. Which diagram best describes the hierarchy right after splitting the `athletes` cluster? Refer to the quiz form for the diagrams.

**Caution**. The granularity criteria is an imperfect heuristic and must be taken with a grain of salt. It takes a lot of manual intervention to obtain a good hierarchy of clusters.

* **If a cluster is highly mixed, the top articles and words may not convey the full picture of the cluster.** Thus, we may be misled if we judge the purity of clusters solely by their top documents and words. 
* **Many interesting topics are hidden somewhere inside the clusters but do not appear in the visualization.** We may need to subdivide further to discover new topics. For instance, subdividing the `ice_hockey_football` cluster led to the appearance of runners and golfers.

### Cluster of non-athletes

Now let us subdivide the cluster of non-athletes.


```python
%%time 
# Bipartition the cluster of non-athletes
left_child_non_athletes_artists, right_child_non_athletes_artists = bipartition(non_athletes_artists, maxiter=100, num_runs=3, seed=1)
```

    /home/ruben/venv/lib/python3.6/site-packages/sklearn/cluster/_kmeans.py:939: FutureWarning: 'n_jobs' was deprecated in version 0.23 and will be removed in 0.25.
      " removed in 0.25.", FutureWarning)


    CPU times: user 6.14 s, sys: 453 ms, total: 6.59 s
    Wall time: 6.6 s



```python
display_single_tf_idf_cluster(left_child_non_athletes_artists, map_word_to_index)
```

    113949:0.016
    113949:0.013
    113949:0.013
    113949:0.012
    113949:0.012
    
    * Barry Sullivan (lawyer)                            0.97227
      barry sullivan is a chicago lawyer and as of july 1 2009 the cooney conway chair in advoca
      cy at loyola university chicago school of law
    * Kayee Griffin                                      0.97444
      kayee frances griffin born 6 february 1950 is an australian politician and former australi
      an labor party member of the new south wales legislative council serving
    * Christine Robertson                                0.97450
      christine mary robertson born 5 october 1948 is an australian politician and former austra
      lian labor party member of the new south wales legislative council serving
    * James A. Joseph                                    0.97464
      james a joseph born 1935 is an american former diplomatjoseph is professor of the practice
       of public policy studies at duke university and founder of
    * David Anderson (British Columbia politician)       0.97492
      david a anderson pc oc born august 16 1937 in victoria british columbia is a former canadi
      an cabinet minister educated at victoria college in victoria
    * Mary Ellen Coster Williams                         0.97594
      mary ellen coster williams born april 3 1953 is a judge of the united states court of fede
      ral claims appointed to that court in 2003
    * Sven Erik Holmes                                   0.97600
      sven erik holmes is a former federal judge and currently the vice chairman legal risk and 
      regulatory and chief legal officer for kpmg llp a
    * Andrew Fois                                        0.97652
      andrew fois is an attorney living and working in washington dc as of april 9 2012 he will 
      be serving as the deputy attorney general
    



```python
display_single_tf_idf_cluster(right_child_non_athletes_artists, map_word_to_index)
```

    113949:0.039
    113949:0.030
    113949:0.023
    113949:0.021
    113949:0.015
    
    * Madonna (entertainer)                              0.96092
      madonna louise ciccone tkoni born august 16 1958 is an american singer songwriter actress 
      and businesswoman she achieved popularity by pushing the boundaries of lyrical
    * Janet Jackson                                      0.96153
      janet damita jo jackson born may 16 1966 is an american singer songwriter and actress know
      n for a series of sonically innovative socially conscious and
    * Cher                                               0.96540
      cher r born cherilyn sarkisian may 20 1946 is an american singer actress and television ho
      st described as embodying female autonomy in a maledominated industry
    * Laura Smith                                        0.96600
      laura smith is a canadian folk singersongwriter she is best known for her 1995 single shad
      e of your love one of the years biggest hits
    * Natashia Williams                                  0.96677
      natashia williamsblach born august 2 1978 is an american actress and former wonderbra camp
      aign model who is perhaps best known for her role as shane
    * Anita Kunz                                         0.96716
      anita e kunz oc born 1956 is a canadianborn artist and illustratorkunz has lived in london
       new york and toronto contributing to magazines and working
    * Maggie Smith                                       0.96747
      dame margaret natalie maggie smith ch dbe born 28 december 1934 is an english actress she 
      made her stage debut in 1952 and has had
    * Lizzie West                                        0.96752
      lizzie west born in brooklyn ny on july 21 1973 is a singersongwriter her music can be des
      cribed as a blend of many genres including
    


The clusters are not as clear, but the left cluster has a tendency to show important female figures, and the right one to show politicians and government officials.

Let's divide them further.


```python
female_figures = left_child_non_athletes_artists
politicians_etc = right_child_non_athletes_artists
```

**Quiz Question**. Let us bipartition the clusters `female_figures` and `politicians`. Which diagram best describes the resulting hierarchy of clusters for the non-athletes? Refer to the quiz for the diagrams.

**Note**. Use `maxiter=100, num_runs=6, seed=1` for consistency of output.


```python
left_child_female_figures, right_child_female_figures = bipartition(female_figures, maxiter=100, num_runs=6, seed=1)
left_child_politicians_etc, right_child_politicians_etc = bipartition(politicians_etc, maxiter=100, num_runs=6, seed=1)
```

    /home/ruben/venv/lib/python3.6/site-packages/sklearn/cluster/_kmeans.py:939: FutureWarning: 'n_jobs' was deprecated in version 0.23 and will be removed in 0.25.
      " removed in 0.25.", FutureWarning)
    /home/ruben/venv/lib/python3.6/site-packages/sklearn/cluster/_kmeans.py:939: FutureWarning: 'n_jobs' was deprecated in version 0.23 and will be removed in 0.25.
      " removed in 0.25.", FutureWarning)



```python
display_single_tf_idf_cluster(left_child_female_figures, map_word_to_index)
display_single_tf_idf_cluster(right_child_female_figures, map_word_to_index)
display_single_tf_idf_cluster(left_child_politicians_etc, map_word_to_index)
display_single_tf_idf_cluster(right_child_politicians_etc, map_word_to_index)
```

    113949:0.039
    113949:0.036
    113949:0.033
    113949:0.028
    113949:0.026
    
    * Kayee Griffin                                      0.95170
      kayee frances griffin born 6 february 1950 is an australian politician and former australi
      an labor party member of the new south wales legislative council serving
    * Marcelle Mersereau                                 0.95417
      marcelle mersereau born february 14 1942 in pointeverte new brunswick is a canadian politi
      cian a civil servant for most of her career she also served
    * Lucienne Robillard                                 0.95453
      lucienne robillard pc born june 16 1945 is a canadian politician and a member of the liber
      al party of canada she sat in the house
    * Maureen Lyster                                     0.95590
      maureen anne lyster born 10 september 1943 is an australian politician she was an australi
      an labor party member of the victorian legislative assembly from 1985
    * Liz Cunningham                                     0.95690
      elizabeth anne liz cunningham is an australian politician she was an independent member of
       the legislative assembly of queensland from 1995 to 2015 representing the
    * Carol Skelton                                      0.95780
      carol skelton pc born december 12 1945 in biggar saskatchewan is a canadian politician she
       is a member of the security intelligence review committee which
    * Stephen Harper                                     0.95816
      stephen joseph harper pc mp born april 30 1959 is a canadian politician who is the 22nd an
      d current prime minister of canada and the
    * Doug Lewis                                         0.95875
      douglas grinslade doug lewis pc qc born april 17 1938 is a former canadian politician a ch
      artered accountant and lawyer by training lewis entered the
    
    113949:0.018
    113949:0.015
    113949:0.013
    113949:0.013
    113949:0.010
    
    * Lawrence W. Green                                  0.97495
      lawrence w green is best known by health education researchers as the originator of the pr
      ecede model and codeveloper of the precedeproceed model which has
    * James A. Joseph                                    0.97506
      james a joseph born 1935 is an american former diplomatjoseph is professor of the practice
       of public policy studies at duke university and founder of
    * Timothy Luke                                       0.97584
      timothy w luke is university distinguished professor of political science in the college o
      f liberal arts and human sciences as well as program chair of
    * Archie Brown                                       0.97628
      archibald haworth brown cmg fba commonly known as archie brown born 10 may 1938 is a briti
      sh political scientist and historian in 2005 he became
    * Jerry L. Martin                                    0.97687
      jerry l martin is chairman emeritus of the american council of trustees and alumni he serv
      ed as president of acta from its founding in 1995
    * Ren%C3%A9e Fox                                     0.97713
      rene c fox a summa cum laude graduate of smith college in 1949 earned her phd in sociology
       in 1954 from radcliffe college harvard university
    * Robert Bates (political scientist)                 0.97732
      robert hinrichs bates born 1942 is an american political scientist he is eaton professor o
      f the science of government in the departments of government and
    * Ferdinand K. Levy                                  0.97737
      ferdinand k levy was a famous management scientist with several important contributions to
       system analysis he was a professor at georgia tech from 1972 until
    
    113949:0.027
    113949:0.023
    113949:0.017
    113949:0.016
    113949:0.015
    
    * Julian Knowles                                     0.96904
      julian knowles is an australian composer and performer specialising in new and emerging te
      chnologies his creative work spans the fields of composition for theatre dance
    * Peter Combe                                        0.97080
      peter combe born 20 october 1948 is an australian childrens entertainer and musicianmusica
      l genre childrens musiche has had 22 releases including seven gold albums two
    * Craig Pruess                                       0.97121
      craig pruess born 1950 is an american composer musician arranger and gold platinum record 
      producer who has been living in britain since 1973 his career
    * Ceiri Torjussen                                    0.97169
      ceiri torjussen born 1976 is a composer who has contributed music to dozens of film and te
      levision productions in the ushis music was described by
    * Brenton Broadstock                                 0.97192
      brenton broadstock ao born 1952 is an australian composerbroadstock was born in melbourne 
      he studied history politics and music at monash university and later composition
    * Michael Peter Smith                                0.97318
      michael peter smith born september 7 1941 is a chicagobased singersongwriter rolling stone
       magazine once called him the greatest songwriter in the english language he
    * Marc Hoffman                                       0.97356
      marc hoffman born april 16 1961 is a composer of concert music and music for film pianist 
      vocalist recording artist and music educator hoffman grew
    * Tom Bancroft                                       0.97378
      tom bancroft born 1967 london is a british jazz drummer and composer he began drumming age
      d seven and started off playing jazz with his father
    
    113949:0.124
    113949:0.092
    113949:0.015
    113949:0.015
    113949:0.014
    
    * Janet Jackson                                      0.93374
      janet damita jo jackson born may 16 1966 is an american singer songwriter and actress know
      n for a series of sonically innovative socially conscious and
    * Barbara Hershey                                    0.93507
      barbara hershey born barbara lynn herzstein february 5 1948 once known as barbara seagull 
      is an american actress in a career spanning nearly 50 years
    * Lauren Royal                                       0.93717
      lauren royal born march 3 circa 1965 is a book writer from california royal has written bo
      th historic and novelistic booksa selfproclaimed angels baseball fan
    * Alexandra Potter                                   0.93802
      alexandra potter born 1970 is a british author of romantic comediesborn in bradford yorksh
      ire england and educated at liverpool university gaining an honors degree in
    * Cher                                               0.93804
      cher r born cherilyn sarkisian may 20 1946 is an american singer actress and television ho
      st described as embodying female autonomy in a maledominated industry
    * Madonna (entertainer)                              0.93806
      madonna louise ciccone tkoni born august 16 1958 is an american singer songwriter actress 
      and businesswoman she achieved popularity by pushing the boundaries of lyrical
    * Jane Fonda                                         0.93836
      jane fonda born lady jayne seymour fonda december 21 1937 is an american actress writer po
      litical activist former fashion model and fitness guru she is
    * Ellina Graypel                                     0.93927
      ellina graypel born july 19 1972 is an awardwinning russian singersongwriter she was born 
      near the volga river in the heart of russia she spent
    



```python
female_politicians = left_child_female_figures
non_female_figures = right_child_female_figures
male_musicians = left_child_politicians_etc
female_musicians = right_child_politicians_etc
```


```python

```
