import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

x = df['title']
y = df['price'].str.replace(',', '').astype(float) 

explode = [0.1] * len(x) 

plt.pie(y, labels=x, radius=1.2, autopct='%0.01f%%', shadow=True, explode=explode)
plt.show()