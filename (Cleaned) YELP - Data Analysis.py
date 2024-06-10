# Databricks notebook source
# MAGIC %md
# MAGIC #Install Required Libraries

# COMMAND ----------

# Install Langchain and its LLM support library
%pip install langchain transformers databricks-sql-connector databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ##Model Serving

# COMMAND ----------

# MAGIC %md The below code uses DBRX-Instruct model available within Databricks to provide a user recommendations on a location to visit based upon their request. The ideal use-case for this code is for suggesting recommmendations to the user for restaurants or other places to visit.
# MAGIC
# MAGIC Due to inability to provide data to the serving model via the dataframe_records body parameter, the data is being sources from the model itself. Attempts to provide the data to the serving endpoint were unsuccessful due to the API not accepting the parameter(s) mentioned in the docs.
# MAGIC https://docs.databricks.com/api/workspace/servingendpoints/query#dataframe_records

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole, QueryEndpointResponse
import json

w = WorkspaceClient()
serving_endpoint = w.serving_endpoints.list()[0]
messages=[
  ChatMessage(
      content = """You are an AI assistant who should provide 10 recommendations to locations based upon their reviews and business information and general customer sentiment. Your job is to clean and summarized the data to answer a user's questions about which business would be an ideal recommendation in json format. Response should only contains JSON with all lowercase keys and content for the top 10 recommendations. The user will provide you with inputs about the type of business they are looking for, and you should provide the top recommendations based upon their inputs. The location will be Moscone Center in San Francisco, CA. Provide the response in json format with the following fields: [Name, Rating, Accuracy, Price, Meets_Requests, Explanation]. The definition for each field is as follows. 
      
      For rating, it should be a cumulative rating based upon reviews and the user's expected experience. 
      
      Accuracy should represent how well the users' requests are met. If 4/5 of the requests are met, the accuracy should be 0.8.
      Price should indicate average price in $, $$, $$$, $$$$. 
      
      Meets_Requests should break down what the user asks for and provide a json response of each request and whether it was met with a true/false value. 
      Explanation should be one short sentence and include why the metrics provided were given.
      
      Provide the response in the following format. Below is an example. The response must be valid json and should contain the top 10 results. The response must not be truncated and must include 10 results. The response must be valid json without comments.
      [
        {
          "name": "",
          "rating": 0.0,
          "accuracy": 0.0,
          "price": "",
          "meets_requests": {},
          "explanation": ""
        }
      ]
      """,
      role = ChatMessageRole.SYSTEM
  ),
  ChatMessage(
      content = 'Find a restaurant within 5 miles with vegan options that is open until at least 11:30 with vegan options?',
      role = ChatMessageRole.USER
  )
]
query_endpoint_response: QueryEndpointResponse = w.serving_endpoints.query(serving_endpoint.name, messages=messages)
last_index = query_endpoint_response.choices[0].message.content.rfind(']') + 1
print(query_endpoint_response.choices[0].message.content[:last_index])
json_content = json.loads(query_endpoint_response.choices[0].message.content[:last_index])
recommendations_df = spark.createDataFrame(json_content)
recommendations_df.display()

# COMMAND ----------

import matplotlib.pyplot as plt

# Convert the DataFrame to Pandas DataFrame
recommendations_pd = recommendations_df.toPandas()

# Sort the recommendations by rating in descending order
recommendations_pd = recommendations_pd.sort_values('rating', ascending=False)

# Create a bar plot for the ratings with a different color for each bar
plt.bar(recommendations_pd['name'], recommendations_pd['rating'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
plt.xlabel('Business Name')
plt.ylabel('Rating')
plt.title('Top 10 Recommendations')
plt.xticks(rotation=90)
plt.show()

# COMMAND ----------

dbutils.notebook.exit('')

# COMMAND ----------

# MAGIC %md #Yelp Data

# COMMAND ----------

yelp_business_overview_df = spark.table('bright_data_yelp_businesses_overview_businesses_reviews_datasets.datasets.yelp_businesses_overview')
yelp_business_overview_df.display()

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import from_json, col

schema = StructType([
    StructField("City", StringType(), True),
    StructField("Country", StringType(), True),
    StructField("State", StringType(), True),
    StructField("zip_code", StringType(), True)
])
yelp_business_overview_df.withColumn("ADDRESS", from_json(col("ADDRESS"), schema)).display()

# COMMAND ----------

from pyspark.sql.functions import col, explode
from pyspark.sql.types import BooleanType, StructType, StructField, StringType
from pyspark.sql.functions import from_json
amenities_df = yelp_business_overview_df
amenities_df = amenities_df.withColumn("amenities", from_json("amenities", "array<struct<available:boolean, name:string>>"))
amenities_df = amenities_df.withColumn("amenities", explode("amenities"))
amenities_df = amenities_df.select("name", "amenities")
amenities_df = amenities_df.select("name", col("amenities.available").alias("available"), col("amenities.name").alias("name")).distinct()
amenities_df = amenities_df.orderBy("name")
amenities_df.show(truncate=False)

# COMMAND ----------

# MAGIC %md #Business Reviews

# COMMAND ----------

yelp_business_reviews_df = spark.table('bright_data_yelp_businesses_overview_businesses_reviews_datasets.datasets.yelp_businesses_reviews')
yelp_business_reviews_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #**Analysis**

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from bright_data_yelp_businesses_overview_businesses_reviews_datasets.datasets.yelp_businesses_overview
# MAGIC where REVIEWS_COUNT = 'null'
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC
