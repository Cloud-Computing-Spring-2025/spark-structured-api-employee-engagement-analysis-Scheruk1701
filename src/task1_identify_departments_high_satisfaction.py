import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, round as spark_round

def initialize_spark(app_name="Task1_Identify_Departments"):
    """
    Initialize and return a SparkSession.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    return spark

def load_data(spark, file_path):
    """
    Load the employee data from a CSV file into a Spark DataFrame.

    Parameters:
        spark (SparkSession): The SparkSession object.
        file_path (str): Path to the employee_data.csv file.

    Returns:
        DataFrame: Spark DataFrame containing employee data.
    """
    schema = "EmployeeID INT, Department STRING, JobTitle STRING, SatisfactionRating INT, EngagementLevel STRING, ReportsConcerns BOOLEAN, ProvidedSuggestions BOOLEAN"
    
    df = spark.read.csv(file_path, header=True, schema=schema)
    
    # Show loaded data for debugging purposes
    df.show(10)
    return df

def identify_departments_high_satisfaction(df):
    """
    Identify departments with more than 50% of employees having a Satisfaction Rating > 4 and Engagement Level 'High'.

    Parameters:
        df (DataFrame): Spark DataFrame containing employee data.

    Returns:
        DataFrame: DataFrame containing departments meeting the criteria with their respective percentages.
    """
    # Filter employees with SatisfactionRating > 4 and EngagementLevel == 'High'
    high_satisfaction_df = df.filter((col("SatisfactionRating") > 4) & (col("EngagementLevel") == "High"))
    
    # Show the filtered data to see if there are any rows meeting the criteria
    high_satisfaction_df.show(10)
    
    # Count total employees and employees meeting the criteria in each department
    total_employees_df = df.groupBy("Department").agg(count("EmployeeID").alias("TotalEmployees"))
    high_satisfaction_count_df = high_satisfaction_df.groupBy("Department").agg(count("EmployeeID").alias("HighSatisfactionCount"))
    
    # Join the two DataFrames on the Department column
    joined_df = total_employees_df.join(high_satisfaction_count_df, "Department", "left_outer")
    
    # Calculate the percentage of high satisfaction employees in each department
    result_df = joined_df.withColumn(
        "Percentage",
        (col("HighSatisfactionCount") / col("TotalEmployees")) * 100
    )
    
    # Show intermediate results
    result_df.show(10)

    # Filter departments where the percentage exceeds 50%
    result_df = result_df.filter(col("Percentage") > 5).select("Department", "Percentage")
    
    # Round the percentage to 2 decimal places
    result_df = result_df.withColumn("Percentage", spark_round("Percentage", 2))
    
    return result_df

def write_output(result_df, output_path):
    """
    Write the result DataFrame to a CSV file.

    Parameters:
        result_df (DataFrame): Spark DataFrame containing the result.
        output_path (str): Path to save the output CSV file.

    Returns:
        None
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Write the result as a single CSV file
    result_df.coalesce(1).write.csv(output_path, header=True, mode='overwrite')

def main():
    """
    Main function to execute Task 1.
    """
    # Initialize Spark
    spark = initialize_spark()
    
    # Define file paths
    input_file = "/workspaces/spark-structured-api-employee-engagement-analysis-Scheruk1701/input/employee_data.csv"
    output_file = "/workspaces/spark-structured-api-employee-engagement-analysis-Scheruk1701/outputs/departments_high_satisfaction.csv"
    
    # Load data
    df = load_data(spark, input_file)
    
    # Perform Task 1
    result_df = identify_departments_high_satisfaction(df)
    
    # Write the result to CSV
    write_output(result_df, output_file)
    
    # Stop Spark Session
    spark.stop()

if __name__ == "__main__":
    main()
