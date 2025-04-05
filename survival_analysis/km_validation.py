from lifelines import KaplanMeierFitter
import pandas as pd

def validate():
    df = pd.DataFrame({'time': [5, 6, 6, 2, 4], 'event': [1, 0, 0, 1, 1]})
    kmf = KaplanMeierFitter()
    kmf.fit(df['time'], df['event'])
    kmf.plot()