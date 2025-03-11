# Employee Engagement Analysis Assignment

## **Prerequisites**

Before starting the assignment, ensure you have the following software installed and properly configured on your machine:

1. **Python 3.x**:
   - [Download and Install Python](https://www.python.org/downloads/)
   - Verify installation:
     ```bash
     python3 --version
     ```

2. **PySpark**:
   - Install using `pip`:
     ```bash
     pip install pyspark
     ```

3. **Apache Spark**:
   - Ensure Spark is installed. You can download it from the [Apache Spark Downloads](https://spark.apache.org/downloads.html) page.
   - Verify installation by running:
     ```bash
     spark-submit --version
     ```

4. **Docker & Docker Compose** (Optional):
   - If you prefer using Docker for setting up Spark, ensure Docker and Docker Compose are installed.
   - [Docker Installation Guide](https://docs.docker.com/get-docker/)
   - [Docker Compose Installation Guide](https://docs.docker.com/compose/install/)
5. **HDFS on Docker** (Optional):
   - [HDFS, Hive, Hue,Spark](https://github.com/Wittline/apache-spark-docker)
   - [HDFS on Spark](https://github.com/big-data-europe/docker-hadoop-spark-workbench)

## **Setup Instructions**

### **1. Project Structure**

Ensure your project directory follows the structure below:

```
EmployeeEngagementAnalysis/
├── input/
│   └── employee_data.csv
├── outputs/
│   ├── task1_departments_high_satisfaction.csv
│   ├── task2_valued_no_suggestions.csv
│   └── task3_engagement_levels_job_titles.csv
├── src/
│   ├── task1_identify_departments_high_satisfaction.py
│   ├── task2_valued_no_suggestions.py
│   └── task3_compare_engagement_levels.py
├── docker-compose.yml
└── README.md
```

- **input/**: Contains the `employee_data.csv` dataset.
- **outputs/**: Directory where the results of each task will be saved.
- **src/**: Contains the individual Python scripts for each task.
- **docker-compose.yml**: Docker Compose configuration file to set up Spark.
- **README.md**: Assignment instructions and guidelines.

### **2. Running the Analysis Tasks**

You can run the analysis tasks either locally or using Docker.

#### **a. Running Locally**

1. **Navigate to the Project Directory**:
   ```bash
   cd spark-structured-api-employee-engagement-analysis-Scheruk1701/
   ```

2. **Execute Each Task Using `spark-submit`**:
   ```bash
   spark-submit src/task1_identify_departments_high_satisfaction.py
   spark-submit src/task2_valued_no_suggestions.py
   spark-submit src/task3_compare_engagement_levels.py
   ```

3. **Verify the Outputs**:
   Check the `outputs/` directory for the resulting files:
   ```bash
   ls outputs/
   ```
   You should see:
   - `departments_high_satisfaction.csv`
   - `valued_no_suggestions.csv`
   - `engagement_levels_job_titles.csv`

#### **b. Running with Docker (Optional)**

1. **Start the Spark Cluster**:
   ```bash
   docker-compose up -d
   ```

2. **Access the Spark Master Container**:
   ```bash
   docker exec -it my-spark-master bash
   ```

3. **Navigate to the Spark Directory**:
   ```bash
   cd /opt/bitnami/spark/
   ```

4. **Run Your PySpark Scripts Using `spark-submit`**:
   ```bash
   spark-submit src/task1_identify_departments_high_satisfaction.py
   spark-submit src/task2_valued_no_suggestions.py
   spark-submit src/task3_compare_engagement_levels.py
   ```

5. **Exit the Container**:
   ```bash
   exit
   ```

6. **Verify the Outputs**:
   On your host machine, check the `outputs/` directory for the resulting files.

7. **Stop the Spark Cluster**:
   ```bash
   docker-compose down
   ```

## **Overview**

In this assignment, you will leverage Spark Structured APIs to analyze a dataset containing employee information from various departments within an organization. Your goal is to extract meaningful insights related to employee satisfaction, engagement, concerns, and job titles. This exercise is designed to enhance your data manipulation and analytical skills using Spark's powerful APIs.

## **Objectives**

By the end of this assignment, you should be able to:

1. **Data Loading and Preparation**: Import and preprocess data using Spark Structured APIs.
2. **Data Analysis**: Perform complex queries and transformations to address specific business questions.
3. **Insight Generation**: Derive actionable insights from the analyzed data.

## **Dataset**

### **Employee Data (`employee_data.csv`)**

You will work with a dataset containing information about 100 employees across various departments. The dataset includes the following columns:

| Column Name             | Data Type | Description                                           |
|-------------------------|-----------|-------------------------------------------------------|
| **EmployeeID**          | Integer   | Unique identifier for each employee                   |
| **Department**          | String    | Department where the employee works (e.g., Sales, IT) |
| **JobTitle**            | String    | Employee's job title (e.g., Manager, Executive)      |
| **SatisfactionRating**  | Integer   | Employee's satisfaction rating (1 to 5)               |
| **EngagementLevel**     | String    | Employee's engagement level (Low, Medium, High)       |
| **ReportsConcerns**     | Boolean   | Indicates if the employee has reported concerns       |
| **ProvidedSuggestions** | Boolean   | Indicates if the employee has provided suggestions    |

### **Sample Data**

Below is a snippet of the `employee_data.csv` to illustrate the data structure. Ensure your dataset contains at least 100 records for meaningful analysis.

```
EmployeeID,Department,JobTitle,SatisfactionRating,EngagementLevel,ReportsConcerns,ProvidedSuggestions
1,Sales,Manager,5,High,False,True
2,IT,Developer,3,Low,True,False
3,HR,Executive,4,High,False,True
4,Sales,Executive,2,Low,True,False
5,IT,Manager,5,High,False,True
...
```

## **Assignment Tasks**

You are required to complete the following three analysis tasks using Spark Structured APIs. Ensure that your analysis is well-documented, with clear explanations and any relevant visualizations or summaries.

### **1. Identify Departments with High Satisfaction and Engagement**

**Objective:**

Determine which departments have more than 50% of their employees with a Satisfaction Rating greater than 4 and an Engagement Level of 'High'.

**Tasks:**

- **Filter Employees**: Select employees who have a Satisfaction Rating greater than 4 and an Engagement Level of 'High'.
- **Analyze Percentages**: Calculate the percentage of such employees within each department.
- **Identify Departments**: List departments where this percentage exceeds 50%.

**Expected Outcome:**

A list of departments meeting the specified criteria, along with the corresponding percentages.

**Example Output:**

| Department | Percentage |
|------------|------------|
| Finance    | 60%        |
| Marketing  | 55%        |

---

### **2. Who Feels Valued but Didn’t Suggest Improvements?**

**Objective:**

Identify employees who feel valued (defined as having a Satisfaction Rating of 4 or higher) but have not provided suggestions. Assess the significance of this group within the organization and explore potential reasons for their behavior.

**Tasks:**

- **Identify Valued Employees**: Select employees with a Satisfaction Rating of 4 or higher.
- **Filter Non-Contributors**: Among these, identify those who have `ProvidedSuggestions` marked as `False`.
- **Calculate Proportion**: Determine the number and proportion of these employees relative to the entire workforce.

**Expected Outcome:**

Insights into the number and proportion of employees who feel valued but aren’t providing suggestions.

**Example Output:**

```
Number of Employees Feeling Valued without Suggestions: 25
Proportion: 25%
```

---

### **3. Compare Engagement Levels Across Job Titles**

**Objective:**

Examine how Engagement Levels vary across different Job Titles and identify which Job Title has the highest average Engagement Level.

**Tasks:**

- **Map Engagement Levels**: Convert categorical Engagement Levels ('Low', 'Medium', 'High') to numerical values to facilitate calculation.
- **Group and Calculate Averages**: Group employees by Job Title and compute the average Engagement Level for each group.
- **Identify Top Performer**: Determine which Job Title has the highest average Engagement Level.

**Expected Outcome:**

A comparative analysis showing average Engagement Levels across Job Titles, highlighting the top-performing Job Title.

**Example Output:**

| JobTitle    | AvgEngagementLevel |
|-------------|--------------------|
| Manager     | 4.5                |
| Executive   | 4.2                |
| Developer   | 3.8                |
| Analyst     | 3.5                |
| Coordinator | 3.0                |
| Support     | 2.8                |

---
