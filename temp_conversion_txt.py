
edge_list = []
counter =len(nod_lis)
#print(counter)
for i in range(len(node_df)):
    line = node_df.iloc[[i]].values.tolist()
    line = line[0]
    
    focal_node = line[0]#pick the core node node
  #  print("focal_node is ", focal_node)

    # this focus node can not be nan or name?
    if ((focal_node != 'nan') and (focal_node != '#NAME?')):
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
                   # print("i is ", i)

                    fr_obj = nod_lis[nod_lis['node'] == friend]

                    if len(fr_obj)==0:
                        #list is empty
                        counter=counter+1;
                        #print("counter",counter)

                        df2_ata = {'label': [counter], 'node': [friend]}
                        df2=pd.DataFrame(df2_ata)
                       # print("df2 is ", df2)

                        nod_lis = nod_lis.append(df2, ignore_index=True)
                       # print("nod_lis is ", nod_lis)
                        fr_obj = nod_lis[nod_lis['node'] == friend]

                        node_fr_num = fr_obj['label'].to_numpy()
                        #print("fr_obj is ", fr_obj)

                        temp =np.hstack((node_f_num,node_fr_num))
                        #print("temp is ", temp)
                        if len(edge_list)==0:
                            edge_list=np.append(edge_list,temp)
                           # print("edge_list is ", edge_list)

                        else:
                            edge_list=np.vstack((edge_list,temp))

                    else:
                        node_fr_num = fr_obj['label'].to_numpy()
                      #  print("fr_obj is ", fr_obj)

                        temp =np.hstack((node_f_num,node_fr_num))
                      #  print("temp is ", temp)
                        edge_list=np.vstack((edge_list,temp))
    else:
        print("index",i)