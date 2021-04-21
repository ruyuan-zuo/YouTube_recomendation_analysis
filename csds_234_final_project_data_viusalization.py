#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import sys
# !{sys.executable} -m pip install GraphViz


# In[2]:


# pip install GraphViz


# In[7]:


import networkx as nx
import numpy as np
import pandas as pd
import pylab as plt
from networkx.drawing.nx_agraph import graphviz_layout
import graphviz
import math
import operator


# In[5]:


nod_lis = pd.read_table('node.txt')
nod_lis


# In[106]:


# This is importing df 0 1 2 3 
# df = pd.read_table('0.txt')
df_1 = pd.read_excel('1.xlsx')  
new_list = pd.read_csv('1txt_cleaned_pairs_related.csv')  
# df_2 = pd.read_excel('2.xlsx')  
# df_3 = pd.read_excel('3.xlsx')  
new_list


# In[78]:


# This is concate together df0,1,2,3,
# frames = [df, df_1, df_2]
result_df = pd.concat(frames, ignore_index=True)
#k= result_df.iloc[[0]].values.tolist()
#m=k[0][9:]
#len(m)


# In[49]:


# df[df['videoID'] == '4F-dflWbP14'] \
result_df = df_1
column = result_df["age"]
k=column.max()
result_df[result_df['age'] == k  ]


# In[50]:


# df[df['videoID'] == '4F-dflWbP14'] \
result_df = df_1
column = result_df["length"]
k=column.max()
result_df[result_df['length'] == k  ]


# In[52]:


# df[df['videoID'] == '4F-dflWbP14'] \
result_df = df_1
column = result_df["rate"]
k=column.max()
result_df[result_df['rate'] == k  ]


# In[53]:


# df[df['videoID'] == '4F-dflWbP14'] \
result_df = df_1
column = result_df["ratings"]
k=column.max()
result_df[result_df['ratings'] == k  ]


# In[46]:


# df[df['videoID'] == '4F-dflWbP14'] \
result_df = df_1
column = result_df["rate"]
k=column.max()
result_df[result_df['age'] == k  ]


# In[54]:


# df[df['videoID'] == '4F-dflWbP14'] \
result_df = df_1
column = result_df["views"]
k=column.max()
result_df[result_df['views'] == k  ]


# In[56]:


result_df[result_df['videoID'] == 'yNhDgOzo9n8'  ]


# In[278]:


n = 1
new_list['videoID'].value_counts()[:n].index.tolist()


# In[87]:


### This is a section to conver realted ids to a single pair, very slow 
# new_list = []

# for i in range(len(result_df)):
   # # print('processing node ',i)

    # line = result_df.iloc[[i]].values.tolist()
    # line = line[0]

    # focal_node = line[0] #pick your node
    # for friend in line[9:]:
         # # if friend is validte strings, add the edges between them 
        # if ((type(friend) == str) and (friend != '#NAME?')):
           # # print("friend is ", friend)



            # df2_ata = { 'videoID':[line[0]],
                        # 'uploader':[line[1]],
                        # 'age':[line[2]] ,
                        # 'category':[line[3]] ,
                        # 'length':[line[4]] ,
                        # 'views':[line[5]] ,
                        # 'rate':[line[6]] ,
                        # 'ratings':[line[7]],
                        # 'comments':[line[8]] ,
                        # 'relatedID':[friend ]}
            # counter = counter +1;

            # df2=pd.DataFrame(df2_ata)


            # if len(new_list)==0:
                   # #list is empty, we use appeend
                # new_list=df2
            # # print("edge_list is ", edge_list)

            # else:
                # new_list=pd.concat([df2, new_list],  ignore_index=True)


# In[91]:


# This section saves the dataframe to csv file
#header = ["videoID","uploader","age","category","length","views","rate","ratings","comments","relatedID"]
#new_list.to_csv('1txt_cleaned_pairs_related.csv', index=None,columns = header)


# In[109]:



#k = nod_lis[nod_lis['node'] == 'mfeZibn3vmU']
#k['label'].to_numpy()


# # Statistic Analysis

# In[57]:


# category stats
result_df["category"].value_counts()


# In[58]:


# category stats
plt.figure();
result_df['category'].value_counts().plot(kind='bar')
plt.title('youtube categories data')
#plt.savefig('category_boxplot_1_txt.png')


# In[59]:


# length stats
temp = result_df["length"].copy()
temp.plot.hist(bins=120, alpha=0.5)
plt.xlim([0,2000])
plt.title('youtube length data')
# plt.savefig('length_1_txt.png')


# In[60]:


# length stats
temp = result_df["comments"].copy()
temp.plot.hist(bins=500, alpha=0.5)
plt.xlim([0,200])
 
plt.title('youtube comments data')
# plt.savefig('comments_1_txt.png')


# In[69]:


# length stats
temp = result_df["rate"].copy()
temp.plot.hist( alpha=0.5)
#plt.xlim([0,200])
 
plt.title('youtube rate data')
#plt.savefig('rates_1_txt.png')


# In[75]:


# length stats
temp = result_df["ratings"].copy()
temp.plot.hist( bins=120,alpha=0.5)
plt.xlim([0,700])
 
plt.title('youtube ratings data')
#plt.savefig('ratings_1_txt.png')


# In[90]:


#  view count stats
plt.figure();

temp = result_df["views"].copy()
#temp=temp.to_numpy()
temp.plot(kind='hist')
#temp.plot.hist(bins=12000, alpha=0.5)
#plt.xlim([0,500000])
 
plt.title('youtube views data')
#plt.savefig('views_1_txt.png')
#plt.xlim([0,500000])
#plt.savefig('vies_zoom.png')


# # Node Analysis

# In[238]:


nod_lis =  pd.read_excel('node_1.xlsx')  
nod_lis


# In[239]:



node_df = result_df.drop(['uploader','age','category','length','views','rate','ratings','comments'], axis=1).copy()
node_df


# In[244]:


#node_df.drop(node_df.index[0])
nod_lis.iloc[[20]].values.tolist()


# In[245]:


# This section prepare for gephi analysis 
counter =len(nod_lis)
counter


# In[246]:



edge_list = []
counter =len(nod_lis)
#print(counter)
for i in range(len(nod_lis)):
    line = node_df.iloc[[i]].values.tolist()
    line = line[0]
    
    focal_node = line[0]#pick the core node node
  #  print("focal_node is ", focal_node)

    # this focus node can not be nan or name?
    if ( (type(focal_node) == str) and (focal_node != '#NAME?')):
        f_obj = nod_lis[nod_lis['node'] == focal_node]
        node_f_num = f_obj['label'].to_numpy()
     #   print("node_f_num is ", node_f_num)

        for friend in line[1:]:#loop over the friends
        #friend = line[1] #back indent
            if (type(friend) == str): 

                if(friend != '#NAME?'):
                    # find core node index 
                        # find friend index 
                        # add this pair to edge list 
                    print("i is ", i)

                    fr_obj = nod_lis[nod_lis['node'] == friend]

                    if (len(fr_obj)==0):
                        #list is empty
                        counter=counter+1;
                      #  print("counter",counter)

                        df2_ata = {'label': [counter], 'node': [friend]}
                        df2=pd.DataFrame(df2_ata)
                       # print("df2 is ", df2)

                        nod_lis = nod_lis.append(df2, ignore_index=True)
                       # print("nod_lis is ", nod_lis)
                        fr_obj = nod_lis[nod_lis['node'] == friend]

                        node_fr_num = fr_obj['label'].to_numpy()
                        #print("fr_obj is ", fr_obj)

                        temp =np.hstack((node_f_num,node_fr_num))
                        #print("here temp is ", temp)
                        if len(edge_list)==0:
                            edge_list=np.append(edge_list,temp)
                           # print("edge_list is ", edge_list)

                        else:
                            edge_list=np.vstack((edge_list,temp))

                    else:
                        node_fr_num = fr_obj['label'].to_numpy()
                        #print(" here else fr_obj is ", fr_obj)

                        temp =np.hstack((node_f_num,node_fr_num))
                      #  print("temp is ", temp)
                        #edge_list=np.vstack((edge_list,temp))
                        if len(edge_list)==0:
                            edge_list=np.append(edge_list,temp)
                           # print("edge_list is ", edge_list)

                        else:
                            edge_list=np.vstack((edge_list,temp))
    else:
        print("index",i)
        node_df.drop(node_df.index[i])


# In[249]:


dff


# In[255]:


dff= pd.DataFrame(edge_list, columns={'node 1','node 2'})
header = ["node 1", "node 2"]
dff.to_csv('output_edge.csv', index=None,columns = header)
nod_lis
header = ["label 1", "node 2"]
#dff.to_csv('output_edge.csv', index=None,columns = header)
dff


# In[148]:


node_df.iloc[[294]]
line = node_df.iloc[[1]].values.tolist()
line = line[0]

focal_node = line[0]
focal_node


# # Network Analysis

# In[76]:


result_df


# In[77]:



node_df = result_df.drop(['uploader','age','category','length','views','rate','ratings','comments'], axis=1).copy()
node_df


# In[78]:


# node_df.iloc[[19812]]
# line = node_df.iloc[[23807]].values.tolist()
# line = line[0]
# temp__ = line[0]
# type(temp__) == str


# In[79]:


# Statr of graph analysis 

[len_df, col_df] = np.shape(node_df)
len_df


# In[80]:




# from graphviz import Digraph

g = nx.DiGraph()
for i in range(len_df):
      #  print('processing node ',i)
        line = node_df.iloc[[i]].values.tolist()# split the line up into a list - the first entry will be the node, the others his friends
        line = line[0]
        temp1 =  line[0]
        if (type(temp1) == str): # check if the focus node should be string to continue
            if line[0] not in g: # if focus node is not in graph
                g.add_node(line[0]) # add this node
                
                #in case the node has friends, loop over all the entries in the list
                focal_node = line[0] #pick your node
                for friend in line[1:]:#loop over the friends
                    # if friend is validte strings, add the edges between them 
                    if ((type(friend) == str) and (friend != '#NAME?')):
                        g.add_edge(focal_node,friend)#add each edge to the graph
            else:
                # this case focus node is in graph
                focal_node = line[0]#pick your node
                for friend in line[1:]:#loop over the friends
                    
                    # for all the valid friend, add edge between them 
                    if ((type(friend) == str) and (friend != '#NAME?')):
                        g.add_edge(focal_node,friend)#add each edge to the graph


# In[81]:



# study graph properties
N, K =  g.order(), g.size()
avg_deg = float(K) / N
print ("Nodes: ", N)
print ("Edges: ", K)
print ("Average degree: ", avg_deg)
print ("SCC: ", nx.number_strongly_connected_components(g))
print ("WCC: ", nx.number_weakly_connected_components(g))


# In[137]:


# study graph degree distribution 


# in_degrees = g.in_degree() # dictionary node:degree
#in_values = sorted(set(in_degrees.values()))
#in_hist = [in_degrees.values().count(x) for x in in_values]
# in_values = sorted(set(dict(in_degrees).values()))
# in_hist = [list(dict(in_degrees).values()).count(x) for x in in_values]


# In[138]:


# in_hist = [in_values.count(x) for x in in_values]
# in_values


# In[147]:


dict(in_degrees).values().to_numpy()


# In[85]:



in_degrees = g.in_degree() # dictionary node:degree
#in_values = sorted(set(in_degrees.values()))
#in_hist = [in_degrees.values().count(x) for x in in_values]
in_values = sorted(set(dict(in_degrees).values()))
in_hist = [list(dict(in_degrees).values()).count(x) for x in in_values]


out_degrees = g.out_degree() # dictionary node:degree
#in_values = sorted(set(in_degrees.values()))
#in_hist = [in_degrees.values().count(x) for x in in_values]
out_values = sorted(set(dict(out_degrees).values()))
out_hist = [list(dict(out_degrees).values()).count(x) for x in out_values]

plt.figure() # you need to first do 'import pylab as plt'
plt.grid(True)
plt.plot(in_values, in_hist, 'ro-') # in-degree
plt.plot(out_values, out_hist, 'bv-') # out-degree
plt.legend(['In-degree', 'Out-degree'])
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.title('network of  youtube videos')
plt.xlim([0, 10*1**2])
#plt.savefig('./output/cam_net_degree_distribution.pdf')
#plt.close()
plt.savefig('in and out degrees 1 txt.png')


# In[188]:


# clustering 
g_un = g.to_undirected()
# Clustering coefficient of node 0
#print (nx.clustering(g_un, 0))
# Clustering coefficient of all nodes (in a dictionary)
clust_coefficients = nx.clustering(g_un)
# Average clustering coefficient
avg_clust = sum(clust_coefficients.values()) / len(clust_coefficients)
#print (avg_clust)
print ("avg_clust: ", avg_clust)

# Or use directly the built-in method
#print (nx.average_clustering(g_un))
print ("average_clustering: ", nx.average_clustering(g_un))


# In[87]:


g_un = g.to_undirected()

# nx.draw_networkx(g_un) #un directed network 


# In[93]:


# node centralities 
# Connected components are sorted in descending order of their size
g_net_components = nx.connected_component_subgraphs(g_un)
g_net_components = list(g_net_components)
len(g_net_components)
g_net_mc = g_net_components[0]
g_net_mc


# In[ ]:


nx.betweenness_centrality(g_un)


# In[89]:


g_net_mc = g_net_components[0]
# Betweenness centrality
bet_cen = nx.betweenness_centrality(g_net_mc)
print ("bet_cen",bet_cen)
# Closeness centrality
clo_cen = nx.closeness_centrality(g_net_mc)
print ("clo_cen",clo_cen)
# Eigenvector centrality
eig_cen = nx.eigenvector_centrality(g_net_mc)

print ("eig_cen",eig_cen)


# In[97]:



top_bet_cen =  dict(sorted(bet_cen.items(), key=operator.itemgetter(1), reverse=True)[:5])

top_clo_cen =dict(sorted(clo_cen.items(), key=operator.itemgetter(1), reverse=True)[:5])
top_eig_cent =dict(sorted(eig_cen.items(), key=operator.itemgetter(1), reverse=True)[:5])


# In[99]:


top_bet_cen


# In[100]:


top_clo_cen


# In[101]:


top_eig_cent


# In[108]:


new_list[new_list['videoID'] == '1umiJrKfpdk' ]


# In[ ]:





# In[ ]:





# In[185]:


bet_cen


# In[183]:


top_clo_cen


# In[184]:


top_eig_cent


# In[186]:


# most central nodes
# def get_top_keys(dictionary, top):
#     items = dictionary.items()
#     items.sort(reverse=True, key=lambda x: x[1])
#     return map(lambda x: x[0], items[:top])
# top_bet_cen = get_top_keys(bet_cen,10)
# top_clo_cen = get_top_keys(clo_cen,10)
# top_eig_cent = get_top_keys(eig_cen,10)


# In[ ]:


# draw the graph using information about the nodes geographic position
#pos_dict = {}
#for node_id, node_info in node_data.items():
#    pos_dict[node_id] = (node_info[2], node_info[1])
#nx.draw(cam_net, pos=pos_dict, with_labels=False, node_size=25)
#plt.savefig('cam_net_graph.pdf')
#plt.close()


# # Page Rank
# Use PageRank or other graph-based algorithms over 
#  the Youtube network to compute the scores efficiently. Intuitively, a video 
#  with a high PageRank score means that the video is related to many videos in 
#  the graph, thus has a high influence. Effectively find top k most influence videos in Youtube network.  
#  Check the properties of these videos (# of views, # edges, category…).
#  What can we find out? Present your findings.

# In[82]:


def pagerank(G, alpha=0.85, personalization=None, 
             max_iter=100, tol=1.0e-6, nstart=None, weight='weight', 
             dangling=None): 
  
    # converstion to undi
    if len(G) == 0: 
        return {} 
  
    if not G.is_directed(): 
        D = G.to_directed() 
    else: 
        D = G 
  
    # Create a copy 
    W = nx.stochastic_graph(D, weight=weight) 
    N = W.number_of_nodes() 
  
    # Choose fixed starting  vec 
    if nstart is None: 
        x = dict.fromkeys(W, 1.0 / N) 
    else: 
        # Normalized  
        s = float(sum(nstart.values())) 
        x = dict((k, v / s) for k, v in nstart.items()) 
  
    if personalization is None: 
  
        # Assign uniform   vector if not given 
        p = dict.fromkeys(W, 1.0 / N) 
    else: 
        missing = set(G) - set(personalization) 
        if missing: 
            raise NetworkXError('error %s' % missing) 
        s = float(sum(personalization.values())) 
        p = dict((k, v / s) for k, v in personalization.items()) 
  
    if dangling is None: 
  
        # Use   vector if dangling vector  
        dangling_weights = p 
    else: 
        missing = set(G) - set(dangling) 
        if missing: 
            raise NetworkXError('Dangling node dictionary  %s' % missing) 
        s = float(sum(dangling.values())) 
        dangling_weights = dict((k, v/s) for k, v in dangling.items()) 
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0] 
  
    # main interation loop  
    for _ in range(max_iter): 
        xlast = x 
        x = dict.fromkeys(xlast.keys(), 0) 
        danglesum = alpha * sum(xlast[n] for n in dangling_nodes) 
        for n in x: 
  
            # matrix multiply 
            for nbr in W[n]: 
                x[nbr] += alpha * xlast[n] * W[n][nbr][weight] 
            x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n] 
  
        # check convergence
        err = sum([abs(x[n] - xlast[n]) for n in x]) 
        if err < N*tol: 
            return x 
    raise NetworkXError('pagerank: power iteration failed to converge '
                        'in %d iterations.' % max_iter) 


# In[83]:


k = pagerank(g, alpha=0.85, personalization=None, 
             max_iter=100, tol=1.0e-6, nstart=None, weight='weight', 
             dangling=None)


# In[84]:


k


# In[43]:


# stats = {'a':1000, 'b':3000, 'c': 100}
# max(stats.iteritems(), key=operator.itemgetter(1))[0]


# In[44]:


# max(k.items(), key=operator.itemgetter(1))[0]


# In[150]:


newA = dict(sorted(k.items(), key=operator.itemgetter(1), reverse=True)[:5])
newA


# In[162]:


new_list[new_list['videoID'] == '8240cBUuP']


# In[160]:


new_list[new_list['relatedID'] == 'Er3K59aVJmM']


# In[50]:


result_df[result_df['videoID'] == 'y0_XLRcKH_Y']


# In[51]:


result_df[result_df['relatedIDs'] == 'y0_XLRcKH_Y']


# In[ ]:




