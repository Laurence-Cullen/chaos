import pandas as pd
import matplotlib.pyplot as plt


eth_usd = pd.read_csv('eth_usd.csv')

plt.plot(eth_usd['UnixTimeStamp'], eth_usd['Value'])
plt.show()
