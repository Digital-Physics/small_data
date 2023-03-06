import pandas as pd
import numpy as np
from itertools import product

companies = ['apple', 'microsoft', 'amazon', 'twitter', 'openai', 'facebook', 'alphabet', 'netflix', 'tesla', 'philadelphia philms']
# not including descriptive first column which we'll add later. just columns that needs data.
col_names = [f"{company}_period_{i}" for company, i in product(companies, range(25))]

input_name_rows = ["revenue", "expenses", "tweet count", "meme count", "investment income", "likes", "retweets",
                   "video games", "films", "free lunches", "executive compensation", "employees", "computer usage",
                   "debt to equity", "marketing spend", "bot articles written", "bots created", "bot impressions"]

target_rows = ["followers", "stock price", "rating"]

field_desc = input_name_rows + target_rows

# create a dictionary that maps input names to their row indices in the dataframe
name_idx = {name: i for i, name in enumerate(field_desc)}

# create 2D array of random numbers
num_rows = len(input_name_rows) + len(target_rows)
num_cols = len(col_names)
data = np.random.rand(num_rows, num_cols)

# create dataframe from 2D array
df = pd.DataFrame(data, columns=col_names)

# replace one of the three targets with a signal, a linear combination of two of the inputs
df.iloc[name_idx["stock price"], :] = 3*df.iloc[name_idx["likes"], :] - 2*df.iloc[name_idx["retweets"], :]

# Add a "field name" column to the dataframe
df.insert(0, "field name", field_desc)

df.to_excel(r'./noise_and_signal.xlsx', sheet_name="input_tab", index=False)