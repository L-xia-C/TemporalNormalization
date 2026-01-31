#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
results = pd.read_csv('results.csv', index_col=0)


# In[2]:


models = {
        'TNTS-SAN-F2W':{
            'title':'Proposed',
            'TN':True
        },
        'TS-SAN-F2W':{
            'title':'Proposed',
            'TN':False
        },
        'TSCAN':{
            'title':'TS-CAN',
            'TN':False
        },
        'TSCANtn':{
            'title':'TS-CAN',
            'TN':True
        },
        'PhysNet':{
            'title':'PhysNet',
            'TN':False
        },
        'PhysNetTN':{
            'title':'PhysNet',
            'TN':True
        },
        'EfficientPhys':{
            'title':'EFFPhys-C',
            'TN':False
        },
        'EfficientPhystn':{
            'title':'EFFPhys-C',
            'TN':True
        },
        'PhysFormer':{
            'title':'PhysFormer',
            'TN':False
        },
        'PhysFormertn':{
            'title':'PhysFormer',
            'TN':True
        }
    }


# In[3]:


from itertools import product

def get_metric(metric, model, training, testing):
    return f"{float(str(results.to_dict()[metric][f'{model}_{training}_{testing}.h5']).split('±')[0]):.3f}"[:5]

results_wotn = []

for model, inf in models.items():
    if inf['TN']:
        continue
    if not results_wotn:
        bg = rf'{inf["title"]} & \multirow{{{len(models)//2}}}*{{\ding{{55}}}} & '
    else:
        bg = rf'{inf["title"]} & & '
    data = [*[get_metric(met, model, 'RLAP', test) for test, met in product(('MMPD', 'COHFACE', 'PURE', 'UBFC'), ('MAE', 'RMSE', 'R'))]]
    results_wotn.append((bg, data))

results_wtn = []

for model, inf in models.items():
    if not inf['TN']:
        continue
    if not results_wtn:
        bg = rf'\rowcolor{{light-gray}} {inf["title"]} & \multirow{{{len(models)//2}}}*{{\checkmark}} & '
    else:
        bg = rf'{inf["title"]} & & '
    data = [*[get_metric(met, model, 'RLAP', test) for test, met in product(('MMPD', 'COHFACE', 'PURE', 'UBFC'), ('MAE', 'RMSE', 'R'))]]
    results_wtn.append((bg, data))

results_all = results_wotn + results_wtn

for j in range(4):
    bfidx = np.min([float(i[1][j*3+0]) for i in results_all]), np.min([float(i[1][j*3+1]) for i in results_all]), np.max([float(i[1][j*3+2]) for i in results_all])
    for i in np.where(bfidx[0]==[float(i[1][j*3+0]) for i in results_all])[0]:
        results_all[i][1][j*3+0] = rf'\bf {results_all[i][1][j*3+0]}'
    for i in np.where(bfidx[1]==[float(i[1][j*3+1]) for i in results_all])[0]:
        results_all[i][1][j*3+1] = rf'\bf {results_all[i][1][j*3+1]}'
    for i in np.where(bfidx[2]==[float(i[1][j*3+2]) for i in results_all])[0]:
        results_all[i][1][j*3+2] = rf'\bf {results_all[i][1][j*3+2]}'

results_wotn, results_wtn = results_all[:len(results_all)//2], results_all[len(results_all)//2:]

results_wotn = '\n'.join([i[0]+' &'.join(i[1])+'\\\\' for i in results_wotn])
results_wtn = '\n'.join([i[0]+' &'.join(i[1])+'\\\\' for i in results_wtn])


# In[4]:


table1 = rf'''
\begin{{table*}}
\caption{{Cross-testing on the MMPD, COHFACE, PURE, and UBFC. (trained on RLAP). \textbf{{Bold}}: The best result.}}
\label{{resultubfc}}
\centering
\scalebox{{1}}{{
\begin{{threeparttable}}
\begin{{tabular}}{{lccccccccccccc}}
  \toprule
  \multirow{{2}}*{{\bf{{Method}}}} & \multirow{{2}}*{{\bf{{w/TN}}}} & \multicolumn{{3}}{{c}}{{\bf{{MMPD}}}} & \multicolumn{{3}}{{c}}{{\bf{{COHFACE}}}} & \multicolumn{{3}}{{c}}{{\bf{{PURE}}}} & \multicolumn{{3}}{{c}}{{\bf{{UBFC}}}} \\
  \cmidrule(lr){{3-5}}\cmidrule(lr){{6-8}}\cmidrule(lr){{9-11}}\cmidrule(lr){{12-14}}
  ~ & ~ & MAE↓ & RMSE↓ & $\rho$↑ & MAE↓ & RMSE↓& $\rho$↑ & MAE↓ & RMSE↓& $\rho$↑ & MAE↓ & RMSE↓& $\rho$↑\\
  \midrule
  {results_wotn}
  \hline
  {results_wtn}
  \hline
\end{{tabular}}
     \begin{{tablenotes}}[flushleft]
     \item \textbf{{MAE}}: Mean Absolute Error, \textbf{{RMSE}}: Root Mean Square Error, \textbf{{$\rho$}}: Pearson correlation coefficient.
     \end{{tablenotes}}
\end{{threeparttable}}
}}
\end{{table*}}
'''


# In[5]:


results_wotn = []

for model, inf in models.items():
    if inf['TN']:
        continue
    if not results_wotn:
        bg = rf'{inf["title"]} & \multirow{{{len(models)//2}}}*{{\ding{{55}}}} & '
    else:
        bg = rf'{inf["title"]} & & '
    data = [*[get_metric(met, model, 'MMPD', test) for test, met in product(('RLAP', 'COHFACE', 'PURE', 'UBFC'), ('MAE', 'RMSE', 'R'))]]
    results_wotn.append((bg, data))

results_wtn = []

for model, inf in models.items():
    if not inf['TN']:
        continue
    if not results_wtn:
        bg = rf'\rowcolor{{light-gray}} {inf["title"]} & \multirow{{{len(models)//2}}}*{{\checkmark}} & '
    else:
        bg = rf'{inf["title"]} & & '
    data = [*[get_metric(met, model, 'MMPD', test) for test, met in product(('RLAP', 'COHFACE', 'PURE', 'UBFC'), ('MAE', 'RMSE', 'R'))]]
    results_wtn.append((bg, data))
    
results_all = results_wotn + results_wtn

for j in range(4):
    bfidx = np.min([float(i[1][j*3+0]) for i in results_all]), np.min([float(i[1][j*3+1]) for i in results_all]), np.max([float(i[1][j*3+2]) for i in results_all])
    for i in np.where(bfidx[0]==[float(i[1][j*3+0]) for i in results_all])[0]:
        results_all[i][1][j*3+0] = rf'\bf {results_all[i][1][j*3+0]}'
    for i in np.where(bfidx[1]==[float(i[1][j*3+1]) for i in results_all])[0]:
        results_all[i][1][j*3+1] = rf'\bf {results_all[i][1][j*3+1]}'
    for i in np.where(bfidx[2]==[float(i[1][j*3+2]) for i in results_all])[0]:
        results_all[i][1][j*3+2] = rf'\bf {results_all[i][1][j*3+2]}'

results_wotn, results_wtn = results_all[:len(results_all)//2], results_all[len(results_all)//2:]

results_wotn = '\n'.join([i[0]+' &'.join(i[1])+'\\\\' for i in results_wotn])
results_wtn = '\n'.join([i[0]+' &'.join(i[1])+'\\\\' for i in results_wtn])

table2 = rf'''
\begin{{table*}}
\caption{{Cross-testing on the RLAP, COHFACE, PURE, and UBFC. (trained on MMPD). \textbf{{Bold}}: The best result.}}
\label{{resultubfc}}
\centering
\scalebox{{1}}{{
\begin{{threeparttable}}
\begin{{tabular}}{{lccccccccccccc}}
  \toprule
  \multirow{{2}}*{{\bf{{Method}}}} & \multirow{{2}}*{{\bf{{w/TN}}}} & \multicolumn{{3}}{{c}}{{\bf{{RLAP}}}} & \multicolumn{{3}}{{c}}{{\bf{{COHFACE}}}} & \multicolumn{{3}}{{c}}{{\bf{{PURE}}}} & \multicolumn{{3}}{{c}}{{\bf{{UBFC}}}} \\
  \cmidrule(lr){{3-5}}\cmidrule(lr){{6-8}}\cmidrule(lr){{9-11}}\cmidrule(lr){{12-14}}
  ~ & ~ & MAE↓ & RMSE↓ & $\rho$↑ & MAE↓ & RMSE↓& $\rho$↑ & MAE↓ & RMSE↓& $\rho$↑ & MAE↓ & RMSE↓& $\rho$↑\\
  \midrule
  {results_wotn}
  \hline
  {results_wtn}
  \hline
\end{{tabular}}
     \begin{{tablenotes}}[flushleft]
     \item \textbf{{MAE}}: Mean Absolute Error, \textbf{{RMSE}}: Root Mean Square Error, \textbf{{$\rho$}}: Pearson correlation coefficient.
     \end{{tablenotes}}
\end{{threeparttable}}
}}
\end{{table*}}
'''


# In[6]:


print(table1)


# In[7]:


print(table2)

