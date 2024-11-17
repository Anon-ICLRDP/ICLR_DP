import pandas as pd
import numpy as np
import os

d1 = pd.read_csv('waterbird_complete95_forest2water2/metadata.csv')

d1 = d1.sort_values('split')
d1.to_csv('sorted.csv')
