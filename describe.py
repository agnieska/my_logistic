
# coding: utf-8

# In[1]:


get_ipython().system('pip3 install numpy')


# In[1]:


get_ipython().system('pip3 install pandas')


# In[43]:


import numpy as np
import pandas as pd
from pprint import pprint
import math


# In[3]:


data_train = pd.read_csv("resources/dataset_train.csv")
data_test = pd.read_csv("resources/dataset_test.csv")


# In[25]:


data_train.head(10)


# In[24]:


data_test.head(100)


# In[8]:


data_train['Index'].shape


# In[9]:


data_test['Index'].shape


# In[10]:


data_test.columns


# In[30]:


column_names_list = list(data_train.columns)
column_names_list


# In[150]:


measure_names_list = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']


# In[163]:


describe_dict = {}
for column_name in column_names_list[6:] :
    #values = list(train_dict[column_name].values())
    values = list(data_train[column_name])
    #print(type(values[4]))
    values = [x for x in values if (math.isnan(x) == False)]
    #pprint(values[0:5])
    values.sort()
    values = np.array(values)
    describe_dict[column_name]={}
    count = round(len(values),6)
    somme = round(sum(values),6)
    mean = round(somme/count,6)
    variance = round(sum((values - mean)**2)/count, 6)
    #print(variance)
    std = round(variance **0.5,6)
    #print(std)
    count = int(count)
    q25 = int(count*0.25)
    q50 = int(count*0.5)
    q75 = int(count*0.75)
    describe_dict[column_name]['count'] = count
    #describe_dict[column_name]['sum'] = somme
    describe_dict[column_name]['mean'] = mean
    describe_dict[column_name]['std'] = std
    describe_dict[column_name]['min'] = round(values[0],6)
    describe_dict[column_name]['25%'] = round(values[q25],6)
    describe_dict[column_name]['50%'] = round(values[q50],6)
    describe_dict[column_name]['75%'] = round(values[q75],6)
    describe_dict[column_name]['max'] = round(values[-1],6)
    


# In[164]:


describe_dict


# In[165]:


df = pd.DataFrame(describe_dict)


# In[166]:


df.head(10)


# In[167]:


describe_dict = df.transpose().to_dict()
describe_dict


# In[168]:


first_line = "{:<10s}".format(" ")
for name in column_names_list[6:11]:
    first_line = first_line +"{:>16s}".format(name[:7])
    #first_line = "{:>10s}".format(name[:10])
print(first_line)

first_line = "{:<10s}".format(" ")
for name in column_names_list[11:16]:
    first_line = first_line +"{:>16s}".format(name[:7])
    #first_line = "{:>10s}".format(name[:10])
print(first_line)

first_line = "{:<10s}".format(" ")
for name in column_names_list[16:]:
    first_line = first_line +"{:>16s}".format(name[:7])
    #first_line = "{:>10s}".format(name[:10])
print(first_line)


# In[169]:


first_line = "{:<10s}".format(" ")
for name in column_names_list[6:11]:
    first_line = first_line +"{:>16s}".format(name[:7])
    #first_line = "{:>10s}".format(name[:10])
print(first_line)
for measure in measure_names_list :
    line = "{:<10s}".format(str(measure))
    #print(s)
    for name in column_names_list[6:11]:
        value = describe_dict[measure][name]
        line = line + "{:16.6f}".format(value)
    print(line)


# In[170]:


first_line = "{:<10s}".format(" ")
for name in column_names_list[11:16]:
    first_line = first_line +"{:>16s}".format(name[:7])
    #first_line = "{:>10s}".format(name[:10])
print(first_line)
for measure in measure_names_list:
    line = "{:<10s}".format(str(measure))
    #print(s)
    for name in column_names_list[11:16]:
        value = describe_dict[measure][name]
        line = line + "{:16.6f}".format(value)
    print(line)


# In[171]:


first_line = "{:<10s}".format(" ")
for name in column_names_list[16:]:
    first_line = first_line +"{:>16s}".format(name[:7])
    #first_line = "{:>10s}".format(name[:10])
print(first_line)
for measure in measure_names_list :
    line = "{:<10s}".format(str(measure))
    #print(s)
    for name in column_names_list[16:]:
        value = describe_dict[measure][name]
        line = line + "{:16.6f}".format(value)
    print(line)

