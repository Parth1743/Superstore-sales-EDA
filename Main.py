import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load CSV with encoding fix
df = pd.read_csv(r'C:\Users\Parth garg\Documents\GitHub\Superstore-sales-EDA\Superstore.csv', encoding='ISO-8859-1')

# Convert date columns
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)

# Set index for time-based resampling
df.set_index('Order Date', inplace=True)

# --- Feature Engineering ---

# 1. Extract Year, Month, Day of Week from 'Order Date'
df['Year'] = df.index.year
df['Month'] = df.index.month
df['Day of Week'] = df.index.dayofweek  # 0 = Monday, 6 = Sunday

# 2. Profit Margin
df['Profit Margin'] = (df['Profit'] / df['Sales']) * 100

# 3. Discount-to-Sales Ratio
df['Discount-to-Sales Ratio'] = df['Discount'] / df['Sales']

# 4. Year-over-Year Sales Growth (assuming monthly data)
df['YoY Growth'] = df['Sales'].pct_change(periods=12) * 100  # Change periods as needed

# 5. High-Value Customers: Top 10% based on total sales
high_value_threshold = df.groupby('Customer ID')['Sales'].sum().quantile(0.9)
df['High Value Customer'] = df.groupby('Customer ID')['Sales'].transform('sum') > high_value_threshold

# 6. High Profit Products based on Profit Margin threshold (median value)
high_profit_threshold = df['Profit Margin'].median()
df['High Profit Product'] = df['Profit Margin'] > high_profit_threshold

# --- Exploratory Data Analysis (EDA) ---

# Prepare Data for Plots
monthly_sales = df['Sales'].resample('M').sum()
category_sales = df.groupby('Category')['Sales'].sum()
region_profit = df.groupby('Region')['Profit'].sum()
corr_matrix = df[['Sales', 'Quantity', 'Discount', 'Profit']].corr()

# --- Plotting ---
fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # 2 rows x 2 columns grid

# Plot 1: Monthly Sales
axs[0, 0].plot(monthly_sales, color='blue')
axs[0, 0].set_title('Monthly Sales Over Time')
axs[0, 0].set_xlabel('Date')
axs[0, 0].set_ylabel('Sales')

# Plot 2: Sales by Category
category_sales.plot(kind='bar', ax=axs[0, 1], color='orange')
axs[0, 1].set_title('Sales by Category')
axs[0, 1].set_xlabel('Category')
axs[0, 1].set_ylabel('Total Sales')

# Plot 3: Profit by Region
region_profit.plot(kind='bar', ax=axs[1, 0], color='green')
axs[1, 0].set_title('Profit by Region')
axs[1, 0].set_xlabel('Region')
axs[1, 0].set_ylabel('Total Profit')

# Plot 4: Correlation Heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axs[1, 1])
axs[1, 1].set_title('Correlation Matrix')

# Auto layout to prevent overlap
plt.tight_layout()

# Show all plots together
plt.show()

# --- Additional Output for Insight --
# You can display a few rows of your updated DataFrame with engineered features
print(df.head())

# We'll use 'Month' and 'Day of Week' as our features to predict 'Sales'
X = df[['Month', 'Day of Week', 'Discount-to-Sales Ratio', 'Profit Margin']]  # Example features
y = df['Sales']  # Target variable (Sales)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)  # Time series data, so no random shuffle

# --- Step 2: Standardize Data (Optional, but often helps for models) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step 3: Train the Model ---

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# --- Step 4: Evaluate the Model ---

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")

# --- Step 5: Visualize Results ---

# Plot actual vs predicted sales
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual Sales', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Sales', color='red', linestyle='--')
plt.title('Actual vs Predicted Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()