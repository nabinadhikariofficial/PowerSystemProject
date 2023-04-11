import pickle
import pandas as pd

modal = pickle.load(open('final_hgr.pkl', 'rb'))

data = [[180.0, 40.0, 49.9, 50.01, 50.02, 1.02, 1.01, 0.99]]
columns = ['Total_MVA', 'Po_GFM_MVA', 'Fre_SG_Hz',
           'Fre_GFM_Hz', 'Fre_GFL_Hz', 'Vo_SG_PU',
           'Vo_GFM_PU', 'Vo_GFL_PU']
df = pd.DataFrame(data=data, columns=columns)

print(modal.predict(df))
