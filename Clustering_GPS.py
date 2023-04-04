# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:13:52 2019

@author: ss
"""




# Import required packages
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint

for root, dirs, files in os.walk('.'):
    for file in files:
        filepath = os.path.join(root, file)
        try:
            data = pd.read_csv(file , sep="," , index_col=False) #, encoding='latin-1'
            splitResult = file.split( "_" )
            Id = splitResult[1]
            data = data[data.travelstate == "stationary"]
            data.drop(data.columns[[0,1,2,3,6,7,8,9]], axis=1, inplace=True)
            ############        DBSCAN              ############################################################
            coords = data.as_matrix(columns=['latitude', 'longitude'])
            kms_per_radian = 6371.0088
            epsilon = 0.5 / kms_per_radian
            db = DBSCAN(eps=epsilon, min_samples=3, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
            cluster_labels = db.labels_
            num_clusters = len(set(cluster_labels))
            clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
            File1.write('{0} {1}\n'.format(Id, num_clusters)) # Id, "  " , num_clusters) 
            
            
            #print('Number of clusters: {}'.format(num_clusters))
            #df.to_csv('E:\Depression Prediction\Activity1.csv', sep=',' , header=None, mode = 'a', index = False)
            
            
            ############        Variance and Stad Dev
            
            lat_var = data.loc[:,"latitude"].var()
            lon_var = data.loc[:,"longitude"].var()
            lat_std = data.loc[:,"latitude"].std()
            lon_std = data.loc[:,"longitude"].std()
            File2.write('{0}, {1}, {2}, {3}, {4}\n'.format(Id, lat_var, lon_var, lat_std, lon_std))
            
            
        except (IndexError, FileNotFoundError, UnicodeDecodeError, ValueError) as e:
            pass

File1 = open("D:\OneDrive\Depression Prediction\Prediction Models\Results & Others\Workfile1","a")
File2 = open("D:\OneDrive\Depression Prediction\Prediction Models\Results & Others\Workfile2","a")

File1.close()
File2.close()





data = pd.read_csv('D:\Depression Prediction\dataset\sensing\gps\gps_u33.csv', index_col=False)

data.shape
data = data[data['travelstate'].notnull()]

data = data[data.travelstate == "stationary"]

data.drop(data.columns[[0,1,2,3,6,7,8,9]], axis=1, inplace=True)


data.head()
data.columns
 
mms = MinMaxScaler()
mms.fit(data)
data_transformed = mms.transform(data)


Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    Sum_of_squared_distances.append(km.inertia_)
    
    
    
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()






df = pd.read_csv('D:\Depression Prediction\dataset\sensing\gps\gps_u33.csv', index_col=False)
df = df[df.travelstate == "stationary"]
df.drop(df.columns[[0,1,2,3,6,7,8,9]], axis=1, inplace=True)
df.columns
df.shape
#pd.read_csv('summer-travel-gps-full.csv')
coords = df.as_matrix(columns=['latitude', 'longitude'])


kms_per_radian = 6371.0088
epsilon = 0.5 / kms_per_radian
db = DBSCAN(eps=epsilon, min_samples=3, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
cluster_labels = db.labels_
num_clusters = len(set(cluster_labels))
clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
print('Number of clusters: {}'.format(num_clusters))


def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)

centermost_points = clusters.map(get_centermost_point)

lats, lons = zip(*centermost_points)
rep_points = pd.DataFrame({'longitude':lons, 'latitude':lats})

fig, ax = plt.subplots(figsize=[10, 6])
rs_scatter = ax.scatter(rep_points['longitude'], rep_points['latitude'], c='#99cc99', edgecolor='None', alpha=0.7, s=120)
df_scatter = ax.scatter(df['longitude'], df['latitude'], c='k', alpha=0.9, s=3)
ax.set_title('Full data set vs DBSCAN reduced set')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend([df_scatter, rs_scatter], ['Full set', 'Reduced set'], loc='upper right')
plt.show()


rs = rep_points.apply(lambda row: df[(df['latitude']==row['latitude']) &amp;&amp; (df['longitude']==row['longitude'])].iloc[0], axis=1)

fig, ax = plt.subplots(figsize=[10, 6])
rs_scatter = ax.scatter(rs['lon'], rs['lat'], c='#99cc99', edgecolor='None', alpha=0.7, s=120)
df_scatter = ax.scatter(df['lon'], df['lat'], c='k', alpha=0.9, s=3)
ax.set_title('Full data set vs DBSCAN reduced set')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend([df_scatter, rs_scatter], ['Full set', 'Reduced set'], loc='upper right')
plt.show()