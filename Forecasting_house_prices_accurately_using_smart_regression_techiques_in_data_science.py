import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load Dataset
df = pd.read_csv("C:/Users/PC/Documents/kc_house_data.csv")
print("Original Dataset Shape:", df.shape)

# Clean column names by removing spaces
df.columns = df.columns.str.strip()

# Step 3: Data Cleaning
# Drop columns with more than 30% missing values
missing_ratio = df.isnull().mean()
df = df.loc[:, missing_ratio < 0.3]

# Check if 'SalePrice' column exists
if 'SalePrice' not in df.columns:
    print("Error: 'SalePrice' column is missing!")
else:
    print("'SalePrice' column is available")

    # Drop rows with remaining missing values for 'SalePrice' only
    print("Missing values in 'SalePrice':", df['SalePrice'].isnull().sum())
    df.dropna(subset=['SalePrice'], inplace=True)

print("Cleaned Dataset Shape:", df.shape)

# Step 4: Convert Date Columns to Datetime (if applicable)
if 'date' in df.columns:
    print("First few unique values in 'date' column:")
    print(df['date'].unique()[:5])
    try:
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    except ValueError:
        try:
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
        except ValueError:
            try:
                df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            except ValueError:
                print("Warning: Could not infer a common date format. Parsing might be inconsistent.")
                df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Step 5: Feature Selection
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()['SalePrice'].sort_values(ascending=False)

# Step 6: Exploratory Data Analysis (EDA)
plt.figure(figsize=(10,6))
top_corr_features = corr.head(10)
sns.barplot(x=top_corr_features.values, y=top_corr_features.index)
plt.title("Top 10 Features Correlated with SalePrice")
plt.xlabel("Correlation Coefficient")
plt.tight_layout()
plt.show()

# Step 7: Prepare Features and Target for Modeling
X = numeric_df.drop(['SalePrice'], axis=1)
y = numeric_df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Step 9: Predictions and Evaluation
y_pred_lr = lr_model.predict(X_test)

print("\nLinear Regression Performance:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_lr))
print("R-squared Score:", r2_score(y_test, y_pred_lr))
