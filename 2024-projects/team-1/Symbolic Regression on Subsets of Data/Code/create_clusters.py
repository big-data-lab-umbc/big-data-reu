import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import BisectingKMeans
from sklearn.cluster import AgglomerativeClustering

def bkmeans_cluster(df, type, colname):
    bisect_means = BisectingKMeans(n_clusters=3, random_state=1)
    predict = bisect_means.fit_predict(type)
    df[colname] = predict

def kmeans_cluster(df, type, colname):
    kmeans = KMeans(n_clusters=3, random_state=1, n_init='auto')
    predict = kmeans.fit_predict(type)
    df[colname] = predict

def agg_cluster(df, type, colname):
    agg = AgglomerativeClustering(n_clusters=3)
    predict = agg.fit_predict(type)
    df[colname] = predict

if __name__ == "__main__":
    df = pd.read_csv('dataset.csv')
    radar = df.loc[:, ['Reflectivity_sc','Zdr_sc','Kdp_sc','Rhohv_sc']]
    rhozdr = df.loc[:, ['Zdr_sc','Rhohv_sc']]
    rain = df.loc[:, ['gauge_precipitation_matched_sc']]

    bkmeans_cluster(df, radar, "cluster_bkmeans_radar")
    bkmeans_cluster(df, rhozdr, "cluster_bkmeans_rhozdr")
    bkmeans_cluster(df, rain, "cluster_bkmeans_rain")

    kmeans_cluster(df, radar, "cluster_kmeans_radar")
    kmeans_cluster(df, rhozdr, "cluster_kmeans_rhozdr")
    kmeans_cluster(df, rain, "cluster_kmeans_rain")

    agg_cluster(df, radar, "cluster_agg_radar")
    agg_cluster(df, rhozdr, "cluster_agg_rhozdr")
    agg_cluster(df, rain, "cluster_agg_rain")

    df.to_csv('dataset_clustered.csv')