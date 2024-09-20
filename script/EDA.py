import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from dateutil.easter import easter
import datetime 
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class CustomerBehaviorAnalyzer:
    def missing_value(self,data):
        logging.info("Check missing values")
        missing_value=data.isnull().sum()
        return missing_value
    # def replace_outlier(self,df):
    def replace_outliers_with_mean(self,data, z_threshold=3):
        logging.info("replace outliers of Sales and Customer")
        # Iterate through each numeric column
        for col in data.select_dtypes(include=[np.number]).columns:
            if col not in ['Sales','Customers']:
                col_data = data[col].dropna()
                col_zscore = zscore(col_data)
                
                # Create a boolean mask for outliers
                outlier_mask = abs(col_zscore) > z_threshold
                
                # Calculate the mean of non-outlier values (excluding NaNs)
                mean_value = col_data[~outlier_mask].mean()
                
                # Ensure the column is of float type to avoid dtype incompatibility
                data[col] = data[col].astype(float)
                
                # Replace outliers in the original DataFrame Need to align the original index with the calculated z-scores
                data.loc[data[col].notna() & (abs(zscore(data[col].fillna(0))) > z_threshold), col] = mean_value
    def handle_missing_values(self,df):
        logging.info('handle missing values')
        # Fill missing CompetitionDistance with a new category 
        df.fillna({
            'CompetitionDistance': -1,  # Can also use -1 to mark missing distances
            'CompetitionOpenSinceMonth': 0,
            'CompetitionOpenSinceYear': 0,
            'Promo2SinceWeek': 0,
            'Promo2SinceYear': 0,
            'PromoInterval': 'None'
        }, inplace=True)

    def compare_promo_distribution(self,train_df, test_df):
        logging.info("Promo Distribution Comparison: Train vs Test......")
        # Get the promo distribution for both training and test sets
        train_promo_dist = train_df['Promo'].value_counts(normalize=True).sort_index()
        test_promo_dist = test_df['Promo'].value_counts(normalize=True).sort_index()
        # Combine the distributions into a single dataframe for easy comparison
        promo_dist_df = pd.DataFrame({
            'Train Promo Distribution': train_promo_dist,
            'Test Promo Distribution': test_promo_dist
        })

        # Plotting
        promo_dist_df.plot(kind='bar', figsize=(10, 6), width=0.8)
        plt.title('Promo Distribution Comparison: Train vs Test')
        plt.xlabel('Promo (0 = No Promo, 1 = Promo)')
        plt.ylabel('Proportion')
        plt.xticks(rotation=0)
        plt.show()
        
    def compare_sales_holiday_periods(self,df):
        logging.info("Compare sales behavior before, during, and after holidays.....")
        df_copy=df.copy()
        # Ensure 'Date' is in datetime format
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        
        # Sort by Date for easier comparison
        df_copy = df_copy.sort_values(by='Date')
        
        # Create a flag for holidays (state holidays or school holidays)
        df_copy['IsHoliday'] = ((df_copy['StateHoliday'] != '0') | (df_copy['SchoolHoliday'] == 1)).astype(int)
        
        # Create labels for 'before', 'during', and 'after' holidays
        df_copy['HolidayPeriod'] = 'Non-holiday'  # Default value
        
        # Identify the periods
        df_copy.loc[df_copy['IsHoliday'] == 1, 'HolidayPeriod'] = 'During Holiday'
        df_copy.loc[df_copy['IsHoliday'].shift(7) == 1, 'HolidayPeriod'] = 'Before Holiday'  # 7 days before
        df_copy.loc[df_copy['IsHoliday'].shift(-7) == 1, 'HolidayPeriod'] = 'After Holiday'   # 7 days after
        
        # Group by HolidayPeriod and calculate average sales
        sales_by_period = df_copy.groupby('HolidayPeriod')['Sales'].mean().reindex(['Before Holiday', 'During Holiday', 'After Holiday', 'Non-holiday'])
        
        # Plotting the results
        plt.figure(figsize=(10, 6))
        sales_by_period.plot(kind='bar', color=['#ffa07a', '#20b2aa', '#87cefa', '#778899'])
        plt.title('Average Sales Before, During, and After Holidays')
        plt.xlabel('Holiday Period')
        plt.ylabel('Average Sales')
        plt.xticks(rotation=0)
        plt.show()

    def analyze_seasonal_sales(self,df):
        logging.info("Analyze seasonal sales behaviors, particularly around major holidays like Christmas and Easter.")
        # Ensure 'Date' is in datetime format
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Create a copy to avoid modifying the original dataframe
        df_copy = df.copy()
        
        # Extract the year from the date to calculate Easter
        df_copy['Year'] = df_copy['Date'].dt.year
        
        # Define a function to determine if a date is within a specific holiday period
        def is_christmas(date):
            return (date.month == 12) & (date.day >= 18)  # Week leading to Christmas

        def is_easter(date):
            easter_date = pd.Timestamp(easter(date.year))  # Convert easter date to Timestamp
            return (date >= (easter_date - pd.Timedelta(days=3))) & (date <= (easter_date + pd.Timedelta(days=3)))  # 3 days before and after Easter

        def is_july_4th(date):
            return (date.month == 7) & (date.day == 4)  # July 4th

        def is_thanksgiving(date):
            # Thanksgiving: 4th Thursday of November
            if date.month == 11:
                # Calculate the 4th Thursday of November
                first_day = datetime.date(date.year, 11, 1)
                first_thursday = first_day + pd.DateOffset(days=(3 - first_day.weekday()) % 7)  # Nearest Thursday
                thanksgiving_day = first_thursday + pd.DateOffset(weeks=3)  # 4th Thursday
                thanksgiving_day = pd.Timestamp(thanksgiving_day)  # Convert to Timestamp
                return date == thanksgiving_day
            return False

        # Label Christmas, Easter, July 4th, and Thanksgiving periods
        df_copy['Season'] = 'Non-Holiday'
        df_copy.loc[df_copy['Date'].apply(is_christmas), 'Season'] = 'Christmas Season'
        df_copy.loc[df_copy['Date'].apply(is_easter), 'Season'] = 'Easter Season'
        df_copy.loc[df_copy['Date'].apply(is_july_4th), 'Season'] = 'July 4th Season'
        df_copy.loc[df_copy['Date'].apply(is_thanksgiving), 'Season'] = 'Thanksgiving Season'

        # Group by 'Season' and calculate average sales
        seasonal_sales = df_copy.groupby('Season')['Sales'].mean()

        # Plot the seasonal sales
        plt.figure(figsize=(10, 6))
        seasonal_sales.plot(kind='bar', color=['#87cefa', '#ffa07a', '#20b2aa', '#ffb6c1', '#90ee90'])
        plt.title('Average Sales During Seasonal Holidays')
        plt.xlabel('Holiday Season')
        plt.ylabel('Average Sales')
        plt.xticks(rotation=0)
        plt.show()
        
        # Line plot to show daily sales during Christmas, Easter, July 4th, and Thanksgiving seasons
        plt.figure(figsize=(10, 6))
        for season in ['Christmas Season', 'Easter Season', 'July 4th Season', 'Thanksgiving Season']:
            season_data = df_copy[df_copy['Season'] == season].groupby('Date')['Sales'].mean()
            plt.plot(season_data.index, season_data.values, label=season)
        
        plt.title('Daily Sales Trend During Seasonal Holidays')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def analyze_sales_customers_correlation(self,df):
        logging.info("Correlation between customer and sales")
        
        # Calculate the correlation coefficient
        correlation = df['Sales'].corr(df['Customers'])
        print(f"Correlation coefficient between Sales and Customers: {correlation:.4f}")
        
        # Plot the scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(df['Customers'], df['Sales'], alpha=0.5)
        plt.title('Sales vs. Number of Customers')
        plt.xlabel('Number of Customers')
        plt.ylabel('Sales')
        plt.grid(True)
        
        # Fit a regression line (optional)
        z = np.polyfit(df['Customers'], df['Sales'], 1)
        p = np.poly1d(z)
        plt.plot(df['Customers'], p(df['Customers']), color='red', linewidth=2, label='Trend Line')
        
        plt.legend(loc='upper left')
        plt.show()
        
    def analyze_promo_effect(self,df):
        logging.info("Promo effect on sales and customers")
        
        # Calculate average sales and customers during promo and non-promo periods
        promo_sales = df[df['Promo'] == 1]['Sales'].mean()
        non_promo_sales = df[df['Promo'] == 0]['Sales'].mean()
        
        promo_customers = df[df['Promo'] == 1]['Customers'].mean()
        non_promo_customers = df[df['Promo'] == 0]['Customers'].mean()
        
        print(f"Average Sales During Promo: {promo_sales:.2f}")
        print(f"Average Sales During Non-Promo: {non_promo_sales:.2f}")
        print(f"Average Customers During Promo: {promo_customers:.2f}")
        print(f"Average Customers During Non-Promo: {non_promo_customers:.2f}")
        
        # Bar plot for average sales and customers
        labels = ['Promo', 'Non-Promo']
        sales_means = [promo_sales, non_promo_sales]
        customers_means = [promo_customers, non_promo_customers]

        x = range(len(labels))
        
        plt.figure(figsize=(12, 6))
        
        # Plotting sales
        plt.subplot(1, 2, 1)
        plt.bar(x, sales_means, color=['#87cefa', '#ffa07a'])
        plt.title('Average Sales During Promo vs Non-Promo')
        plt.xticks(x, labels)
        plt.ylabel('Average Sales')
        
        # Plotting customers
        plt.subplot(1, 2, 2)
        plt.bar(x, customers_means, color=['#20b2aa', '#ffb6c1'])
        plt.title('Average Customers During Promo vs Non-Promo')
        plt.xticks(x, labels)
        plt.ylabel('Average Customers')

        plt.tight_layout()
        plt.show()
        
    def analyze_promo_effectiveness_by_store_type_assortment(self,df):
        logging.info(" effective ways promos can be deployed  ")
        
        # Group by StoreType, Assortment, and Promo status
        store_promo_analysis = df.groupby(['StoreType', 'Assortment', 'Promo']).agg({
            'Sales': 'mean',
            'Customers': 'mean'
        }).reset_index()
        
        # Pivot the table for easier comparison
        sales_pivot = store_promo_analysis.pivot(index=['StoreType', 'Assortment'], columns='Promo', values='Sales')
        customers_pivot = store_promo_analysis.pivot(index=['StoreType', 'Assortment'], columns='Promo', values='Customers')

        # Calculate the difference in sales and customers between promo and non-promo periods
        sales_pivot['Sales_Difference'] = sales_pivot[1] - sales_pivot[0]
        customers_pivot['Customers_Difference'] = customers_pivot[1] - customers_pivot[0]

        # Identify store types and assortments where promo increases sales
        effective_combinations = sales_pivot[sales_pivot['Sales_Difference'] > 0].index.tolist()

        print("StoreType and Assortment combinations where promos could be effectively deployed:")
        for store_type, assortment in effective_combinations:
            print(f"StoreType: {store_type}, Assortment: {assortment}")

        # Plotting sales and customers by StoreType and Assortment
        plt.figure(figsize=(14, 6))

        # Sales plot
        plt.subplot(1, 2, 1)
        sales_pivot[[0, 1]].plot(kind='bar', ax=plt.gca(), colormap='coolwarm')
        plt.title('Average Sales by StoreType, Assortment, and Promo Status')
        plt.xlabel('StoreType, Assortment')
        plt.ylabel('Average Sales')
        plt.xticks(rotation=45)

        # Customers plot
        plt.subplot(1, 2, 2)
        customers_pivot[[0, 1]].plot(kind='bar', ax=plt.gca(), colormap='coolwarm')
        plt.title('Average Customers by StoreType, Assortment, and Promo Status')
        plt.xlabel('StoreType, Assortment')
        plt.ylabel('Average Customers')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()
        
    def analyze_customer_behavior_open_close(self,df):
        logging.info("Trends of customer behavior during store opening and closing times")
        # Group by 'Open' status and calculate the mean number of customers
        open_close_analysis = df.groupby('Open').agg({
            'Customers': 'mean',
            'Date': 'count'
        }).rename(columns={'Date': 'Days'}).reset_index()

        print("Average number of customers during open and closed days:")
        print(open_close_analysis)
        
        # Plot the average number of customers during open and closed days
        plt.figure(figsize=(8, 5))
        plt.bar(['Closed', 'Open'], open_close_analysis['Customers'], color=['red', 'green'])
        plt.title('Average Customers During Store Open/Closed Times')
        plt.ylabel('Average Number of Customers')
        plt.show()

        # Trend of customer behavior over time when the store is open
        df_open = df[df['Open'] == 1].copy()

        # Group by Date to analyze daily customer trends
        daily_customers = df_open.groupby('Date').agg({
            'Customers': 'sum'
        }).reset_index()

        # Plot daily customer trends
        plt.figure(figsize=(14, 6))
        plt.plot(daily_customers['Date'], daily_customers['Customers'], label='Daily Customers', color='blue')
        plt.title('Customer Trends Over Time (Store Open Days)')
        plt.xlabel('Date')
        plt.ylabel('Number of Customers')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()
        
    def analyze_weekday_open_stores(self,df):
        logging.info('stores which are open on all weekdays')
        
        # Filter data for weekdays (Monday to Friday: DayOfWeek 1 to 5)
        weekdays_df = df[(df['DayOfWeek'] >= 1) & (df['DayOfWeek'] <= 5)]
        
        # Group by Store and check if they are open on all weekdays
        weekday_open_stores = weekdays_df.groupby('Store').agg({
            'Open': lambda x: (x == 1).all()  # Check if the store is open all weekdays
        }).reset_index()

        # Filter out stores that are open all weekdays
        stores_open_all_weekdays = weekday_open_stores[weekday_open_stores['Open'] == True]['Store'].tolist()

        # Filter data for weekends (Saturday and Sunday: DayOfWeek 6 and 7)
        weekends_df = df[(df['DayOfWeek'] >= 6) & (df['DayOfWeek'] <= 7)]

        # Split weekend data into two groups: stores open all weekdays and others
        weekend_sales_all_weekdays = weekends_df[weekends_df['Store'].isin(stores_open_all_weekdays)]
        weekend_sales_other_stores = weekends_df[~weekends_df['Store'].isin(stores_open_all_weekdays)]

        # Calculate average weekend sales for both groups
        avg_sales_all_weekdays = weekend_sales_all_weekdays['Sales'].mean()
        avg_sales_other_stores = weekend_sales_other_stores['Sales'].mean()

        # Print the results
        print(f"Average weekend sales for stores open all weekdays: {avg_sales_all_weekdays:.2f}")
        print(f"Average weekend sales for other stores: {avg_sales_other_stores:.2f}")

        # Plot the comparison
        plt.figure(figsize=(8, 5))
        plt.bar(['Stores Open All Weekdays', 'Other Stores'], [avg_sales_all_weekdays, avg_sales_other_stores], color=['blue', 'gray'])
        plt.title('Weekend Sales Comparison for Stores Open All Weekdays vs Other Stores')
        plt.ylabel('Average Sales')
        plt.show()

    def analyze_assortment_sales(self,df):
        logging.info("sales per assortment types")        
        # Mapping for the assortment labels
        assortment_mapping = {'a': 'Basic', 'b': 'Extra', 'c': 'Extended'}
        
        # Map the assortment column to the descriptive names
        df['Assortment'] = df['Assortment'].map(assortment_mapping)
        
        # Group by Assortment and calculate average sales
        assortment_sales = df.groupby('Assortment')['Sales'].mean().reset_index()

        # Print the average sales for each assortment type
        print("Average sales by Assortment type:")
        print(assortment_sales)
        
        # Plot the average sales by Assortment type
        plt.figure(figsize=(8, 5))
        plt.bar(assortment_sales['Assortment'], assortment_sales['Sales'], color=['blue', 'green', 'orange'])
        plt.title('Average Sales by Assortment Type')
        plt.xlabel('Assortment Type')
        plt.ylabel('Average Sales')
        plt.xticks(ticks=[0, 1, 2], labels=['Basic', 'Extra', 'Extended'])
        plt.show()
        
    def analyze_competition_distance_sales(self,df, city_center_threshold=1000):
        logging.info("Effect of competitor distance")
        
        # Handling missing or undefined competition distances by assigning a separate category
        df['CompetitionDistance'] = df['CompetitionDistance'].fillna(-1)  # -1 represents missing distances

        # Grouping by distance ranges
        df['DistanceRange'] = pd.cut(df['CompetitionDistance'], 
                                    bins=[-1, 500, 1000, 5000, 10000, float('inf')], 
                                    labels=['0-500m', '500-1000m', '1-5km', '5-10km', '10km+'])

        # Calculate average sales for each distance range
        distance_sales = df.groupby('DistanceRange',observed=True)['Sales'].mean().reset_index()

        # Print the average sales for each distance range
        print("Average sales by competition distance range:")
        print(distance_sales)

        # Plot the average sales by distance range
        plt.figure(figsize=(10, 6))
        plt.bar(distance_sales['DistanceRange'], distance_sales['Sales'], color='teal')
        plt.title('Average Sales by Competitor Distance')
        plt.xlabel('Competition Distance Range')
        plt.ylabel('Average Sales')
        plt.show()

        # City center analysis (stores within the city center threshold)
        city_center_stores = df[df['CompetitionDistance'] <= city_center_threshold]
        non_city_center_stores = df[df['CompetitionDistance'] > city_center_threshold]

        # Calculate and compare average sales in city center vs non-city center
        city_vs_non_city_sales = pd.DataFrame({
            'Location': ['City Center', 'Non-City Center'],
            'Average Sales': [city_center_stores['Sales'].mean(), non_city_center_stores['Sales'].mean()]
        })

        print("\nAverage Sales Comparison between City Center and Non-City Center Stores:")
        print(city_vs_non_city_sales)

        # Plot comparison between city center and non-city center stores
        plt.figure(figsize=(8, 5))
        plt.bar(city_vs_non_city_sales['Location'], city_vs_non_city_sales['Average Sales'], color=['blue', 'orange'])
        plt.title('Average Sales: City Center vs Non-City Center Stores')
        plt.xlabel('Location')
        plt.ylabel('Average Sales')
        plt.show()
        
    def analyze_new_competitor_impact(self,df):
        logging.info('new competitor impact')
        
        # Convert 'Date' column to datetime if not already
        df['Date'] = pd.to_datetime(df['Date'])

        # Sort by 'Store' and 'Date' for easier comparison
        df = df.sort_values(by=['Store', 'Date'])

        # Identify stores that initially had no competition (NA in CompetitionDistance) but later got valid values
        df['CompetitionInitiallyNA'] = df.groupby('Store')['CompetitionDistance'].transform(lambda x: x.ffill().isna())

        # Filter for the rows where CompetitionDistance was initially NA but later had valid values
        stores_with_new_competition = df.groupby('Store').filter(lambda x: x['CompetitionInitiallyNA'].any() and x['CompetitionDistance'].notna().any())

        # Split data into two periods: before and after competition
        before_competition = stores_with_new_competition[stores_with_new_competition['CompetitionInitiallyNA']]
        after_competition = stores_with_new_competition[~stores_with_new_competition['CompetitionInitiallyNA']]

        # Aggregate sales data for stores with new competitors
        avg_sales_before = before_competition.groupby('Store')['Sales'].mean()
        avg_sales_after = after_competition.groupby('Store')['Sales'].mean()

        # Compare average sales before and after competitor entry
        sales_comparison = pd.DataFrame({
            'AvgSalesBefore': avg_sales_before,
            'AvgSalesAfter': avg_sales_after
        }).dropna()

        # Plot the comparison
        plt.figure(figsize=(10, 6))
        plt.plot(sales_comparison.index, sales_comparison['AvgSalesBefore'], label='Before Competitor', color='blue', marker='o')
        plt.plot(sales_comparison.index, sales_comparison['AvgSalesAfter'], label='After Competitor', color='red', marker='o')
        plt.title('Average Sales Before and After Competitor Entry')
        plt.xlabel('Store')
        plt.ylabel('Average Sales')
        plt.xticks(rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Print the overall difference in sales
        sales_comparison['Difference'] = sales_comparison['AvgSalesAfter'] - sales_comparison['AvgSalesBefore']
        print("\nAverage sales comparison (Before vs After competitor entry):")
        print(sales_comparison)
        
    