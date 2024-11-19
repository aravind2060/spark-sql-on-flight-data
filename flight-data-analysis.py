from pyspark.sql import SparkSession
from pyspark.sql.functions import unix_timestamp, abs, col,stddev, count,when, sum,hour, avg
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number


# Create a Spark session
spark = SparkSession.builder.appName("Advanced Flight Data Analysis").getOrCreate()

# Load datasets
flights_df = spark.read.csv("flights.csv", header=True, inferSchema=True)
airports_df = spark.read.csv("airports.csv", header=True, inferSchema=True)
carriers_df = spark.read.csv("carriers.csv", header=True, inferSchema=True)

# Define output paths
output_dir = "output/"
task1_output = output_dir + "task1_largest_discrepancy"
task2_output = output_dir + "task2_consistent_airlines"
task3_output = output_dir + "task3_canceled_routes"
task4_output = output_dir + "task4_carrier_performance_time_of_day"

# ------------------------
# Task 1: Flights with the Largest Discrepancy Between Scheduled and Actual Travel Time
# ------------------------
def task1_largest_discrepancy(flights_df, carriers_df):
    # Calculate travel times and discrepancy
    flights_with_discrepancy = flights_df.withColumn(
        "ScheduledTravelTime",
        (unix_timestamp("ScheduledArrival") - unix_timestamp("ScheduledDeparture")) / 60
    ).withColumn(
        "ActualTravelTime",
        (unix_timestamp("ActualArrival") - unix_timestamp("ActualDeparture")) / 60
    ).withColumn(
        "Discrepancy",
        abs(col("ActualTravelTime") - col("ScheduledTravelTime"))
    )

    # Rank flights by discrepancy within each carrier
    window_spec = Window.partitionBy("CarrierCode").orderBy(col("Discrepancy").desc())
    largest_discrepancy = flights_with_discrepancy.withColumn(
        "Rank", row_number().over(window_spec)
    ).filter(col("Rank") == 1)

    # Join with carrier names
    result = largest_discrepancy.join(carriers_df, "CarrierCode").select(
        "FlightNum","CarrierCode" ,"CarrierName", "Origin", "Destination", "ScheduledTravelTime", "ActualTravelTime", "Discrepancy", "Rank"
    )

    # Save the result
    result.write.csv(task1_output, header=True)

# ------------------------
# Task 2: Most Consistently On-Time Airlines Using Standard Deviation
# ------------------------
def task2_consistent_airlines(flights_df, carriers_df):
    # Calculate departure delay
    flights_with_delays = flights_df.withColumn(
        "DepartureDelay", (unix_timestamp("ActualDeparture") - unix_timestamp("ScheduledDeparture"))
    )

    # Group by CarrierCode and calculate standard deviation and count
    carrier_stats = flights_with_delays.groupBy("CarrierCode").agg(
        stddev("DepartureDelay").alias("StdDevDelay"),
        count("FlightNum").alias("FlightCount")
    ).filter(col("FlightCount") > 100)

    # Join with carriers to get carrier names
    consistent_airlines = carrier_stats.join(carriers_df, "CarrierCode").select(
        "CarrierName", "FlightCount", "StdDevDelay"
    ).orderBy("StdDevDelay")

    # Save the result
    consistent_airlines.write.csv(task2_output, header=True)

# ------------------------
# Task 3: Origin-Destination Pairs with the Highest Percentage of Canceled Flights
# ------------------------
def task3_canceled_routes(flights_df, airports_df):
    # Mark canceled flights
    flights_with_cancellation = flights_df.withColumn(
        "IsCanceled", when(col("ActualDeparture").isNull(), 1).otherwise(0)
    )

    # Calculate cancellation rate per route
    route_cancellation_stats = flights_with_cancellation.groupBy("Origin", "Destination").agg(
        sum("IsCanceled").alias("CanceledFlights"),
        count("FlightNum").alias("TotalFlights")
    ).withColumn(
        "CancellationRate", (col("CanceledFlights") / col("TotalFlights")) * 100
    ).orderBy(col("CancellationRate").desc())

    # Rename columns for airport names to avoid duplication
    origin_airports = airports_df.withColumnRenamed("AirportName", "OriginAirportName") \
                                 .withColumnRenamed("City", "OriginCity")
    
    destination_airports = airports_df.withColumnRenamed("AirportName", "DestinationAirportName") \
                                      .withColumnRenamed("City", "DestinationCity") \
                                      .withColumnRenamed("AirportCode", "DestinationAirportCode")

    # Join with airport details for human-readable names
    result = route_cancellation_stats.join(
        origin_airports, route_cancellation_stats["Origin"] == origin_airports["AirportCode"]
    ).join(
        destination_airports, route_cancellation_stats["Destination"] == destination_airports["DestinationAirportCode"]
    ).select(
        "OriginAirportName", "OriginCity",
        "DestinationAirportName", "DestinationCity",
        "CancellationRate"
    )

    # Save the result
    result.write.csv(task3_output, header=True)


# ------------------------
# Task 4: Carrier Performance Based on Time of Day
# ------------------------
def task4_carrier_performance_time_of_day(flights_df, carriers_df):
    # Categorize time of day
    flights_with_time_of_day = flights_df.withColumn(
        "TimeOfDay",
        when((hour("ScheduledDeparture") >= 6) & (hour("ScheduledDeparture") < 12), "Morning")
        .when((hour("ScheduledDeparture") >= 12) & (hour("ScheduledDeparture") < 18), "Afternoon")
        .when((hour("ScheduledDeparture") >= 18) & (hour("ScheduledDeparture") < 24), "Evening")
        .otherwise("Night")
    ).withColumn(
        "DepartureDelay", (unix_timestamp("ActualDeparture") - unix_timestamp("ScheduledDeparture"))
    )

    # Calculate average delay by carrier and time of day
    performance_stats = flights_with_time_of_day.groupBy("CarrierCode", "TimeOfDay").agg(
        avg("DepartureDelay").alias("AvgDepartureDelay")
    )

    # Join with carrier names
    result = performance_stats.join(carriers_df, "CarrierCode").select(
        "CarrierName", "TimeOfDay", "AvgDepartureDelay"
    ).orderBy("TimeOfDay", "AvgDepartureDelay")

    # Save the result
    result.write.csv(task4_output, header=True)

# ------------------------
# Call the functions for each task
# ------------------------
task1_largest_discrepancy(flights_df, carriers_df)
task2_consistent_airlines(flights_df, carriers_df)
task3_canceled_routes(flights_df, airports_df)
task4_carrier_performance_time_of_day(flights_df, carriers_df)

# Stop the Spark session
spark.stop()
