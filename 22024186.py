#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from autoviz.AutoViz_Class import AutoViz_Class
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[ ]:


df=pd.read_csv("E:\CO2_LifeExpec\CO2Emission_LifeExp.csv")


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[1]:


fig = px.choropleth(df,locations='Code',color='CO2Emissions',scope='world',color_continuous_scale=px.colors.sequential.GnBu,
                    range_color=(100,11000000000),title='CO2Emissions',height=800
    )
fig.show()


# In[2]:


fig = px.choropleth(df,locations='Code',color='Population',scope='world',color_continuous_scale=px.colors.sequential.GnBu,
                    range_color=(2500,1400000000),title='Population',height=800
    )
fig.show()


# In[ ]:


fig = px.choropleth(df,locations='Code',color='Percapita',scope='world',color_continuous_scale=px.colors.sequential.GnBu,
                    range_color=(0,40),title='CO2Emissions per capita',height=800
    )
fig.show()


# In[ ]:


fig = px.choropleth(df,locations='Code',color='LifeExpectancy',scope='world',color_continuous_scale=px.colors.sequential.GnBu,
                    range_color=(50,85),title='LifeExpectancy',height=800
    )
fig.show()


# In[ ]:


AV = AutoViz_Class()
target='LifeExpectancy'


# In[ ]:


dft = AV.AutoViz(depVar=target,dfte=df, header=0, verbose=0,
                lowess=False,chart_format='svg',max_rows_analyzed=1500000,max_cols_analyzed=300,filename='', sep=',' )


# In[ ]:


df.head()


# In[ ]:


relevant_cols = ["CO2Emissions", "Population", "LifeExpectancy"]

df = df[relevant_cols]


# In[ ]:


scaler = StandardScaler()

scaler.fit(df)

scaled_data = scaler.transform(df)


# In[ ]:


def find_best_clusters(df, maximum_K):
    
    clusters_centers = []
    k_values = []
    
    for k in range(1, maximum_K):
        
        kmeans_model = KMeans(n_clusters = k)
        kmeans_model.fit(df)
        
        clusters_centers.append(kmeans_model.inertia_)
        k_values.append(k)
        
    
    return clusters_centers, k_values


# In[ ]:


def generate_elbow_plot(clusters_centers, k_values):
    
    figure = plt.subplots(figsize = (12, 6))
    plt.plot(k_values, clusters_centers, 'o-', color = 'orange')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Cluster Inertia")
    plt.title("Elbow Plot of KMeans")
    plt.show()


# In[ ]:


clusters_centers, k_values = find_best_clusters(scaled_data, 12)

generate_elbow_plot(clusters_centers, k_values)


# In[ ]:


kmeans_model = KMeans(n_clusters = 5)

kmeans_model.fit(scaled_data)


# In[ ]:


df["clusters"] = kmeans_model.labels_

df.head()


# In[ ]:


plt.scatter(df["CO2Emissions"], 
            df["Population"], 
            c = df["clusters"])


# In[ ]:


plt.scatter(df["CO2Emissions"], 
            df["LifeExpectancy"], 
            c = df["clusters"])


# In[ ]:




