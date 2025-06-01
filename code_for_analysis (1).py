# Set Up Environment

Install necessary tools
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, stddev, count, lit, when, max as spark_max, min as spark_min
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

pip install pandas numpy matplotlib seaborn scikit-learn tensorflow dask

"""Install PySpark via pip"""

pip install pyspark

"""Verify Installation"""

from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder.master("local[*]").appName("PySparkTest").getOrCreate()

# Check Spark version
print("Spark version:", spark.version)

"""List Uploaded Files"""

import os

# List all files in the 'content' directory
uploaded_files = os.listdir('/content')
print("Uploaded Files:", uploaded_files)

"""Load All Files"""

df = spark.read.csv(f'/content/fh_5yrs.csv', header=True, inferSchema=True)

"""In order to analyse further, sample of data belong to 6 companies analyze further"""

from pyspark.sql import SparkSession

# Initialize SparkSession
spark = SparkSession.builder.master("local[*]").appName("StockDataAnalysis").getOrCreate()

# Load all CSV files from the content folder
combined_df = spark.read.csv('/content/*.csv', header=True, inferSchema=True)
combined_df.show(5)  # Display the first 5 rows
combined_df.printSchema()  # Check the structure of the dataset

"""Data Characterization

View the First Few Records
"""

df.show(5)

""" Schema (data types)"""

df.printSchema()

"""Number of Records and Features"""

#Count the Rows (Records)
num_records = df.count()
print(f"Number of records (rows): {num_records}")

#Count the Columns (Features)
num_features = len(df.columns)
print(f"Number of features (columns): {num_features}")

# Group columns by data type
string_columns = [col for col, dtype in df.dtypes if dtype == 'string']
numeric_columns = [col for col, dtype in df.dtypes if dtype in ['int', 'double']]

print("String Columns:", string_columns)
print("Numeric Columns:", numeric_columns)

"""Unique Symbols (Companies)"""

unique_symbols = df.select("symbol").distinct().count()
print(f"Number of unique companies: {unique_symbols}")

"""## Pre processing

Check Missing values
"""

from pyspark.sql.functions import col, sum

missing_values = df.select([sum(col(c).isNull().cast("int")).alias(c) for c in df.columns])
missing_values.show()

"""Ensure correct data types"""

df = df.withColumn("date", df["date"].cast("date"))
df = df.withColumn("close", df["close"].cast("double"))

"""Date Range"""

from pyspark.sql.functions import min, max

date_range = df.select(min("date").alias("Earliest Date"), max("date").alias("Latest Date"))
date_range.show()

"""Calculate Date Range for each company"""

from pyspark.sql.functions import min, max

# Group by symbol and calculate earliest and latest dates
date_range_df = df.groupBy("symbol").agg(
    min("date").alias("Earliest Date"),
    max("date").alias("Latest Date")
)

# Show results
date_range_df.show(truncate=False)

""" Top/Bottom Prices"""

from pyspark.sql.functions import max, min

# Group by symbol and calculate max and min close prices
price_summary = df.groupBy("symbol").agg(
    max("close").alias("Max Close Price"),
    min("close").alias("Min Close Price")
)

# Show the results
price_summary.show(truncate=False)

"""Filter High Performers"""

top_performers = price_summary.orderBy("Max Close Price", ascending=False)
top_performers.show(10)

"""Filter Low Performers"""

low_performers = price_summary.orderBy("Min Close Price", ascending=True)
low_performers.show(10)

"""Visualize Top and Bottom Prices"""

import matplotlib.pyplot as plt

# Convert PySpark DataFrame to Pandas
pandas_price_summary = price_summary.toPandas()

# Sort data for max and min close prices
top_10_max = pandas_price_summary.sort_values(by="Max Close Price", ascending=False).head(10)
bottom_10_min = pandas_price_summary.sort_values(by="Min Close Price", ascending=True).head(10)

# Plot Top 10 Max Close Prices
plt.figure(figsize=(12, 6))
plt.bar(top_10_max['symbol'], top_10_max['Max Close Price'], color='blue', alpha=0.7)
plt.title("Top 10 Companies by Max Close Price", fontsize=16)
plt.xlabel("Symbol", fontsize=12)
plt.ylabel("Max Close Price", fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot Bottom 10 Min Close Prices
plt.figure(figsize=(12, 6))
plt.bar(bottom_10_min['symbol'], bottom_10_min['Min Close Price'], color='orange', alpha=0.7)
plt.title("Bottom 10 Companies by Min Close Price", fontsize=16)
plt.xlabel("Symbol", fontsize=12)
plt.ylabel("Min Close Price", fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



""" Identify the Target Variable"""

if 'close' in df.columns:
    print("Target variable 'close' exists in the dataset.")
else:
    print("Target variable 'close' not found. Check column names:", df.columns)

"""Distribution of features"""

# Group by symbol and count the number of records for each
record_count_by_symbol = df.groupBy("symbol").count().orderBy("count", ascending=False)

# Show the top 10 companies with the highest record counts
record_count_by_symbol.show(10, truncate=False)

# Show the top 10 companies with the lowest record counts
record_count_by_symbol.orderBy("count").show(10, truncate=False)

"""Visualize Record Counts"""

# Convert to Pandas for visualization
pandas_record_count = record_count_by_symbol.toPandas()

import matplotlib.pyplot as plt

# Plot bar chart for record distribution (e.g., top 20 companies)
plt.figure(figsize=(12, 6))
plt.bar(pandas_record_count["symbol"][:20], pandas_record_count["count"][:20], color="blue")
plt.title("Record Distribution Across Top 20 Companies")
plt.xlabel("Symbol")
plt.ylabel("Record Count")
plt.xticks(rotation=45)
plt.show()

"""Summary Statistics"""

# Group by symbol and calculate basic statistics for close prices
summary_stats = df.groupBy("symbol").agg(
    {"close": "avg", "volume": "avg", "close": "stddev"}
).withColumnRenamed("avg(close)", "avg_close") \
 .withColumnRenamed("stddev(close)", "stddev_close") \
 .withColumnRenamed("avg(volume)", "avg_volume")

# Show summary statistics
summary_stats.show(10, truncate=False)

summary_stats.write.csv("summary_stats.csv", header=True)

import pandas as pd

# Load the uploaded file
file_path = '/content/summary_stats.csv/part-00000-e029183b-1e8a-4b83-9a06-4abbd094ef49-c000.csv'
data_distribution = pd.read_csv(file_path)

# Display the first few rows of the dataset and basic information
data_distribution.head(), data_distribution.info()

import matplotlib.pyplot as plt

data= data_distribution
# Summary statistics for numeric columns
summary_stats = data.describe()

# Visualize distributions with histograms and boxplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histogram for avg_volume
axes[0, 0].hist(data['avg_volume'], bins=30, edgecolor='k')
axes[0, 0].set_title('Histogram of avg_volume')
axes[0, 0].set_xlabel('avg_volume')
axes[0, 0].set_ylabel('Frequency')

# Boxplot for avg_volume
axes[0, 1].boxplot(data['avg_volume'], vert=False)
axes[0, 1].set_title('Boxplot of avg_volume')
axes[0, 1].set_xlabel('avg_volume')

# Histogram for stddev_close
axes[1, 0].hist(data['stddev_close'].dropna(), bins=30, edgecolor='k')
axes[1, 0].set_title('Histogram of stddev_close')
axes[1, 0].set_xlabel('stddev_close')
axes[1, 0].set_ylabel('Frequency')

# Boxplot for stddev_close
axes[1, 1].boxplot(data['stddev_close'].dropna(), vert=False)
axes[1, 1].set_title('Boxplot of stddev_close')
axes[1, 1].set_xlabel('stddev_close')

plt.tight_layout()
plt.show()

# Extract necessary statistics for CV and percentiles calculation
mean_avg_volume = data['avg_volume'].mean()
std_avg_volume = data['avg_volume'].std()

mean_stddev_close = data['stddev_close'].mean()
std_stddev_close = data['stddev_close'].std()

# Compute Coefficient of Variation (CV)
cv_avg_volume = std_avg_volume / mean_avg_volume
cv_stddev_close = std_stddev_close / mean_stddev_close

# Percentiles for avg_volume and stddev_close
percentiles_avg_volume = data['avg_volume'].quantile([0.25, 0.5, 0.75])
percentiles_stddev_close = data['stddev_close'].quantile([0.25, 0.5, 0.75])

# Prepare results
distribution_analysis = {
    "CV_avg_volume": cv_avg_volume,
    "CV_stddev_close": cv_stddev_close,
    "Percentiles_avg_volume": percentiles_avg_volume.to_dict(),
    "Percentiles_stddev_close": percentiles_stddev_close.to_dict()
}

distribution_analysis

"""Visualize price and volume"""

# Import necessary libraries
from pyspark.sql import functions as F
import matplotlib.pyplot as plt

# Group data by symbol and calculate summary statistics for each symbol
grouped_stats = df.groupBy("symbol").agg(
    F.mean("close").alias("mean_close"),
    F.stddev("close").alias("stddev_close"),
    F.mean("volume").alias("mean_volume"),
    F.stddev("volume").alias("stddev_volume"),
    F.count("close").alias("count_entries")
)

# Show the summary statistics for each symbol
grouped_stats.show()

# Convert grouped statistics to Pandas DataFrame for visualization
pandas_grouped_stats = grouped_stats.toPandas()


# 2. Bar Plot of Mean Volume for Each Symbol
plt.figure(figsize=(14, 7))
plt.bar(pandas_grouped_stats["symbol"], pandas_grouped_stats["mean_volume"], color="orange", alpha=0.7)
plt.title("Mean Trade Volumes by Symbol")
plt.xlabel("Symbol")
plt.ylabel("Mean Volume")
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

# 3. Line Chart of Closing Prices Over Time for Each Symbol
# Filter a specific symbol (e.g., AAPL) for demonstration
symbol_data = df.filter(F.col("symbol") == "AAPL").orderBy("date").toPandas()

plt.figure(figsize=(14, 7))
plt.plot(symbol_data["date"], symbol_data["close"], label="AAPL Close Price", color="green")
plt.title("Closing Prices Over Time for AAPL")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.legend()
plt.grid(True)
plt.show()

# 4. Scatterplot of Closing Prices vs. Volume for a Specific Symbol
plt.figure(figsize=(10, 6))
plt.scatter(symbol_data["close"], symbol_data["volume"], alpha=0.6, edgecolors="k", color="red")
plt.title("Scatterplot of Closing Prices vs Volume for AAPL")
plt.xlabel("Closing Price")
plt.ylabel("Volume")
plt.grid(True)
plt.show()

"""# Stock Price Changes and Daily Returns

Calculate Daily Price Changes
"""

from pyspark.sql.window import Window
from pyspark.sql.functions import lag, col

window_spec = Window.partitionBy("symbol").orderBy("date")
df = df.withColumn("price_change", col("close") - lag("close", 1).over(window_spec))
df.show(5)

"""Daily Returns Analysis"""

df = df.withColumn("daily_return", (col("close") - lag("close", 1).over(window_spec)) / lag("close", 1).over(window_spec))
df.show(5)

"""Daily Price Range"""

df = df.withColumn("daily_range", col("high") - col("low"))
daily_range_summary = df.groupBy("symbol").agg(avg("daily_range").alias("Avg Daily Range"))
daily_range_summary.show(10)

from pyspark.sql.functions import col, avg, max, min
import matplotlib.pyplot as plt


# Calculate daily price range (high - low)
df = df.withColumn("daily_range", col("high") - col("low"))

# Group by symbol and calculate average, max, and min daily range
daily_range_summary = df.groupBy("symbol").agg(
    avg("daily_range").alias("Avg Daily Range"),
    max("daily_range").alias("Max Daily Range"),
    min("daily_range").alias("Min Daily Range")
)

# Show the summary
daily_range_summary.show(truncate=False)

# Convert PySpark DataFrame to Pandas for visualization
pandas_daily_range = daily_range_summary.toPandas()

# Sort data for better visualization
pandas_daily_range = pandas_daily_range.sort_values(by="Avg Daily Range", ascending=False)

# Top 10 companies by Average Daily Range
top_10_daily_range = pandas_daily_range.head(10)

# Plot Top 10 Companies by Average Daily Range
plt.figure(figsize=(12, 6))
plt.bar(top_10_daily_range['symbol'], top_10_daily_range['Avg Daily Range'], color='blue', alpha=0.7)
plt.title("Top 10 Companies by Average Daily Price Range", fontsize=16)
plt.xlabel("Symbol", fontsize=12)
plt.ylabel("Average Daily Range", fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot Max and Min Daily Range for Top 10 Companies
plt.figure(figsize=(12, 6))
plt.bar(top_10_daily_range['symbol'], top_10_daily_range['Max Daily Range'], label="Max Daily Range", alpha=0.7, color='green')
plt.bar(top_10_daily_range['symbol'], top_10_daily_range['Min Daily Range'], label="Min Daily Range", alpha=0.7, color='orange')
plt.title("Max and Min Daily Price Range for Top 10 Companies", fontsize=16)
plt.xlabel("Symbol", fontsize=12)
plt.ylabel("Daily Price Range", fontsize=12)
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Save the summarized daily range data
daily_range_summary.write.csv("/content/daily_range_summary", header=True)

"""Cumulative Average Abnormal Return (CAAR)"""

from pyspark.sql.functions import avg, sum

df = df.withColumn("cumulative_return", sum("daily_return").over(window_spec))
df = df.withColumn("caar", avg("cumulative_return").over(Window.partitionBy("symbol")))
df.show(5)

"""Trends and Moving Averages"""

from pyspark.sql.functions import avg

df = df.withColumn("MA_20", avg("close").over(Window.partitionBy("symbol").orderBy("date").rowsBetween(-19, 0)))
df = df.withColumn("MA_50", avg("close").over(Window.partitionBy("symbol").orderBy("date").rowsBetween(-49, 0)))
df = df.withColumn("MA_200", avg("close").over(Window.partitionBy("symbol").orderBy("date").rowsBetween(-199, 0)))
df.show(5)

"""Correlation Analysis"""

import seaborn as sns
# Overall correlation
overall_corr = pandas_df[['open', 'high', 'low', 'close', 'volume']].corr()
print(overall_corr)

# Visualize overall correlation
sns.heatmap(overall_corr, annot=True, cmap='coolwarm')
plt.title("Overall Correlation Heatmap")
plt.show()

"""Normalize the numeric data"""

from pyspark.sql.functions import col, min, max

# Calculate min and max for the column (e.g., "close")
stats = df.agg(
    min("close").alias("min_close"),
    max("close").alias("max_close")
).collect()[0]

min_close = stats["min_close"]
max_close = stats["max_close"]

# Apply Min-Max normalization
df = df.withColumn("normalized_close", (col("close") - min_close) / (max_close - min_close))

# Show the results
df.select("symbol", "date", "close", "normalized_close").show(10)

# Check the schema of the dataset
df.printSchema()

# List all numeric columns
numeric_cols = [field.name for field in df.schema.fields if str(field.dataType) in ['DoubleType', 'IntegerType', 'FloatType']]
print("Numeric columns:", numeric_cols)

# Check the schema of the dataset
df.printSchema()

# Identify numeric columns
numeric_cols = [field.name for field in df.schema.fields if str(field.dataType) in ['DoubleType', 'IntegerType', 'FloatType']]
print("Numeric columns:", numeric_cols)

numeric_cols = ["volume", "open", "high", "low", "close", "adjclose", "daily_return", "cumulative_return", "caar"]
print("Numeric columns:", numeric_cols)

def compute_numeric_stats(data):
    """Compute descriptive statistics for numeric columns."""
    # Select numeric columns
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    # Compute statistics
    stats = numeric_data.describe()
    return stats
    stats

    # Compute descriptive statistics
numeric_stats = data.describe()

# Save the statistics to a CSV file
output_path = "/content/descriptive_statistics.csv"
stats.to_csv(output_path, index=True)

print(f"Descriptive statistics saved to {'output_path'}")

def compute_numeric_stats(data):
    """Compute descriptive statistics for numeric columns."""
    # Select numeric columns
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    # Compute statistics
    stats = numeric_data.describe()
    return stats


# Compute descriptive statistics
numeric_stats = data.describe()

# Save the statistics to a CSV file
output_path = "/content/descriptive_statistics.csv"
# Changed stats to numeric_stats and call to_csv on the Pandas DataFrame
numeric_stats.to_csv(output_path, index=True)

print(f"Descriptive statistics saved to {output_path}")

"""Data Convertion

## Analysis

## Descriptive summary

Summative Analysis
"""

df.select("close").describe().show()

df.describe().show()

"""Group by Company (Symbol)"""

from pyspark.sql.functions import avg, max, min

company_summary = df.groupBy("symbol").agg(
    avg("close").alias("avg_close"),
    max("close").alias("max_close"),
    min("close").alias("min_close"),
    avg("volume").alias("avg_volume")
)
company_summary.show()



"""## Analysis"""



"""## 1Explore Price Trends and Behaviors for Each Company Over Time

Filter Relevant Features for Analysis
"""

relevant_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
filtered_df = df.select(relevant_columns)
filtered_df.show(5)

"""Trend Analysis"""

from pyspark.sql.functions import window

trend_df = df.groupBy("symbol", window("date", "30 days")).agg(avg("close").alias("avg_close"))
trend_df.show()

pandas_df = df.toPandas()

import matplotlib.pyplot as plt
company = "AAPL"
firm_data = pandas_df[pandas_df["symbol"] == company]
plt.figure(figsize=(12, 6))
plt.plot(firm_data["date"], firm_data["close"], label="Close Price")
plt.plot(firm_data["date"], firm_data["MA_20"], label="20-Day MA")
plt.plot(firm_data["date"], firm_data["MA_50"], label="50-Day MA")
plt.plot(firm_data["date"], firm_data["MA_200"], label="200-Day MA")
plt.legend()
plt.title(f"{company} Close Price and Moving Averages")
plt.show()

import pandas as pd
import glob
import matplotlib.pyplot as plt

# Step 1: Read all CSV files from the `/content` directory
file_path = '/content/data/'  # Update this if the files are in another directory
all_files = glob.glob(file_path + "*.csv")

# Combine all files into a single Pandas DataFrame
dataframes = [pd.read_csv(f) for f in all_files]
pandas_df = pd.concat(dataframes, ignore_index=True)

# Ensure the date column is in datetime format
pandas_df['date'] = pd.to_datetime(pandas_df['date'])

# Step 2: Filter data for a specific company (e.g., "AAPL")
company = "AAPL"  # Replace with the desired company symbol
firm_data = pandas_df[pandas_df["symbol"] == company]

# Ensure the data is sorted by date for proper time-series plotting
firm_data = firm_data.sort_values(by="date")

# Step 3: Plot Close Price and Moving Averages
plt.figure(figsize=(12, 6))
plt.plot(firm_data["date"], firm_data["close"], label="Close Price")
plt.plot(firm_data["date"], firm_data["MA_20"], label="20-Day MA")
plt.plot(firm_data["date"], firm_data["MA_50"], label="50-Day MA")
plt.plot(firm_data["date"], firm_data["MA_200"], label="200-Day MA")
plt.legend()
plt.title(f"{company} Close Price and Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid()
plt.show()

# Convert PySpark DataFrame to Pandas
pandas_df = df.toPandas()

# Ensure `Date` column is in datetime format for better handling
pandas_df['date'] = pandas_df['date'].astype('datetime64[ns]')



from pyspark.sql.functions import corr # Import the corr function

df.select(corr("close", "volume").alias("correlation")).show()



"""## Predictive Analysis

Perform Stock Price Prediction
"""



import matplotlib.pyplot as plt

# Unique symbols in the dataset
symbols = pandas_df['symbol'].unique()

# Generate line plots for each symbol
for symbol in symbols:
    firm_data = pandas_df[pandas_df['symbol'] == symbol]
    plt.figure(figsize=(12, 6))
    plt.plot(firm_data['date'], firm_data['close'], label=f"{symbol} Close Price")
    plt.title(f"Close Price Over Time for {symbol}")
    plt.xlabel("date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.show()

len(pandas_df['symbol'].unique())

#

"""Correlation Analysis Across Companies:

Time Series Analysis with Moving Averages
"""

# Add moving averages to the dataset
pandas_df['MA_20'] = pandas_df['close'].rolling(window=20).mean()
pandas_df['MA_50'] = pandas_df['close'].rolling(window=50).mean()
pandas_df['MA_200'] = pandas_df['close'].rolling(window=200).mean()

import matplotlib.pyplot as plt

# Plot the close price and moving averages
plt.figure(figsize=(12, 6))
plt.plot(pandas_df['date'], pandas_df['close'], label='Close Price', linewidth=1)
plt.plot(pandas_df['date'], pandas_df['MA_20'], label='20-Day MA', linestyle='--')
plt.plot(pandas_df['date'], pandas_df['MA_50'], label='50-Day MA', linestyle='--')
plt.plot(pandas_df['date'], pandas_df['MA_200'], label='200-Day MA', linestyle='--')
plt.title("Close Price and Moving Averages")
plt.xlabel("Date")
plt.ylabel("date")
plt.legend()
plt.grid()
plt.show()

"""## Predictive Analysis"""

from pyspark.sql.window import Window
from pyspark.sql.functions import avg

window_spec_20 = Window.partitionBy("symbol").orderBy("date").rowsBetween(-19, 0)
df = df.withColumn("MA_20", avg("close").over(window_spec_20))

window_spec_50 = Window.partitionBy("symbol").orderBy("date").rowsBetween(-49, 0)
df = df.withColumn("MA_50", avg("close").over(window_spec_50))

window_spec_200 = Window.partitionBy("symbol").orderBy("date").rowsBetween(-199, 0)
df = df.withColumn("MA_200", avg("close").over(window_spec_200))

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=["MA_20", "MA_50", "MA_200", "volume"], outputCol="features")
ml_df = assembler.transform(df).select("features", "close")

train_data, test_data = ml_df.randomSplit([0.8, 0.2], seed=42)

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol="features", labelCol="close")
lr_model = lr.fit(train_data)

from pyspark.ml.evaluation import RegressionEvaluator

predictions = lr_model.transform(test_data)
evaluator = RegressionEvaluator(labelCol="close", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"RMSE: {rmse}")

import matplotlib.pyplot as plt

predictions_pd = predictions.select("prediction", "close").toPandas()
plt.figure(figsize=(10, 6))
plt.scatter(predictions_pd["close"], predictions_pd["prediction"], alpha=0.6, edgecolors="k")
plt.plot([predictions_pd["close"].min(), predictions_pd["close"].max()],
         [predictions_pd["close"].min(), predictions_pd["close"].max()],
         color="red", linestyle="--")
plt.title("Actual vs Predicted Close Prices")
plt.xlabel("Actual Close Price")
plt.ylabel("Predicted Close Price")
plt.grid(True)
plt.show()

"""model 2"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

combined_data=df.toPandas()

# Preprocessing: Handling missing values
combined_data.fillna(method='ffill', inplace=True)  # Forward fill for simplicity
combined_data.fillna(method='bfill', inplace=True)  # Backward fill for any remaining

# Convert date to datetime for feature engineering
combined_data['date'] = pd.to_datetime(combined_data['date'])

# Feature engineering: Extracting date-related features
combined_data['day_of_week'] = combined_data['date'].dt.dayofweek
combined_data['month'] = combined_data['date'].dt.month

# Feature engineering: Lag features for 'close' price
combined_data['lag_1_close'] = combined_data.groupby('symbol')['close'].shift(1)
combined_data['lag_2_close'] = combined_data.groupby('Company')['close'].shift(2)

# Drop rows with NaN values after creating lag features
combined_data.dropna(inplace=True)

# Prepare features and target
features = ['lag_1_close', 'lag_2_close', 'volume', 'open', 'high', 'low', 'day_of_week', 'month']
target = 'close'

X = combined_data[features]
y = combined_data[target]

# Scaling numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42)
}

# Train and evaluate models
results = []
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({
        "Model": model_name,
        "MSE": mse,
        "MAE": mae,
        "R²": r2
    })

df

from pyspark.sql import functions as F

# Add date-related features: day of the week and month
df = df.withColumn("day_of_week", F.dayofweek("date"))
df = df.withColumn("month", F.month("date"))

from pyspark.sql.window import Window

# Define a window partitioned by symbol and ordered by date
window_spec = Window.partitionBy("symbol").orderBy("date")

# Add lag features for 'close' price
df = df.withColumn("lag_1_close", F.lag("close", 1).over(window_spec))
df = df.withColumn("lag_2_close", F.lag("close", 2).over(window_spec))

# Drop rows with NaN values created by lag features
df = df.dropna()

from pyspark.ml.feature import VectorAssembler

# Assemble features into a single column
assembler = VectorAssembler(
    inputCols=["lag_1_close", "lag_2_close", "volume", "open", "high", "low", "day_of_week", "month"],
    outputCol="features"
)
df = assembler.transform(df)

from pyspark.ml.feature import StandardScaler

# Scale features to standardize their range
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(df)
df = scaler_model.transform(df)

# Split data into training and testing sets
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

from pyspark.ml.regression import RandomForestRegressor

# Initialize and train a Random Forest Regressor
rf = RandomForestRegressor(featuresCol="scaled_features", labelCol="close", predictionCol="prediction")
rf_model = rf.fit(train_df)

# Predict on the test data
predictions = rf_model.transform(test_df)

from pyspark.ml.evaluation import RegressionEvaluator

# Evaluate the model using RMSE, MAE, and R²
evaluator_rmse = RegressionEvaluator(labelCol="close", predictionCol="prediction", metricName="rmse")
rmse = evaluator_rmse.evaluate(predictions)

evaluator_mae = RegressionEvaluator(labelCol="close", predictionCol="prediction", metricName="mae")
mae = evaluator_mae.evaluate(predictions)

evaluator_r2 = RegressionEvaluator(labelCol="close", predictionCol="prediction", metricName="r2")
r2 = evaluator_r2.evaluate(predictions)

# Print evaluation results
evaluation_results = {
    "Root Mean Squared Error (RMSE)": rmse,
    "\nMean Absolute Error (MAE)": mae,
    "\nR² Score": r2
}
print(evaluation_results)

import matplotlib.pyplot as plt

# Convert predictions to Pandas for visualization
predictions_pd = predictions.select("close", "prediction").toPandas()

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(predictions_pd["close"], predictions_pd["prediction"], alpha=0.6, label="Predicted vs Actual")
plt.plot([predictions_pd["close"].min(), predictions_pd["close"].max()],
         [predictions_pd["close"].min(), predictions_pd["close"].max()],
         color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel("Actual Close Prices")
plt.ylabel("Predicted Close Prices")
plt.title("Actual vs Predicted Close Prices")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


# Load the uploaded file containing data from different companies
file_path = "/content/data.csv"
data = pd.read_csv(file_path)

# Inspect the structure of the data to prepare it for modeling
data.head(), data.info()



# Convert date to datetime format and sort data by date for each company
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(['Symbol', 'date'])

# Feature selection and target variable
features = ['volume', 'open', 'high', 'low']
target = 'close'

# Prepare a dictionary to store evaluation results for each company
evaluation_results = {}

# Train models for each company individually
unique_symbols = data['Symbol'].unique()
for symbol in unique_symbols:
    company_data = data[data['Symbol'] == symbol]

    # Define features (X) and target (y)
    X = company_data[features]
    y = company_data[target]

    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Define the model and hyperparameter grid
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10],
    }

    # Cross-validation with time series split
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Store evaluation results
    evaluation_results[symbol] = {
        'Best Parameters': grid_search.best_params_,
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
    }
