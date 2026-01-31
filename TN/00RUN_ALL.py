#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
files = os.listdir(os.getcwd())
for nb in sorted([i for i in files if i.endswith('.ipynb')])[1:]:
    os.system(f'jupyter nbconvert --to notebook --inplace --execute {nb}')
os.system('shutdown')

