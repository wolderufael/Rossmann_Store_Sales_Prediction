import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.easter import easter
import datetime 
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class CustomerBehaviorAnalyzer:
    def missing_value(self,data):
        logging.info("Loading data from file...")
        missing_value=data.isnull().sum()
        return missing_value
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