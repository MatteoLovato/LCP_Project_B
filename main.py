import mesa_web as mw
import pandas as pd

filename='MESA-Web_M07_Z00001/profile1.data'
data=mw.read_profile(filename)

profile1_df=pd.DataFrame(data)
column_filter = ['mass','radius', 'initial_mass', 'initial_z', 'star_age', 'logRho','logT','Teff','energy','photosphere_L', 'photosphere_r', 'star_mass','h1','he3','he4']

# Create a new DataFrame with only the selected columns
filtered_profile1_df = profile1_df[column_filter].copy()

print(filtered_profile1_df)

radius