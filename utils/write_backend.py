# Run this file in your Databricks Notebook
# File location and type
file_location = "/FileStore/auto_iot_sensor_data.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = (
    spark.read.format(file_type)
    .option("inferSchema", infer_schema)
    .option("header", first_row_is_header)
    .option("sep", delimiter)
    .load(file_location)
)

display(df)

permanent_table_name = "main.resamplerdata.auto_iot_data_bronze_sensors"
df.write.format("delta").saveAsTable(permanent_table_name)
