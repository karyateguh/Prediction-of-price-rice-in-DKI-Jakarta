# Report of Prediction of Price RIce in DKI Jakarta

Original file is located at
    https://colab.research.google.com/drive/1buSHtMxcATqnI-SPmMWQREp8r8jmzd-y?usp=sharing 

# Project Domain

As basic needs, rice having significant increment in few months. Economics from University of Gajahmada (UGM) in [this article](https://feb.ugm.ac.id/en/news/4510-rice-prices-exceed-het-highest-record-in-rice-history-in-indonesia), said, the price rice reached IDR 18.000 per kg in February 2024. And it becomes the country highest record in the history.

Those who will get the benefit from this work are:

1. The Government : This is serious economics problem. The goverment should take action regarding this phenomenon.
2. Rice supplier : To make a strategy whether add more stock or not.
3. The public : To set financial strategy.
4. Other business selling secondary needs : To help to adjust what they sell. Since people tend to spend their money on basic ones first.

To solve this problem, we will use time series. By its fuction to predict the future rice price, everything will be handled better.

# Business Understanding

Problem Statements:

1. Which models can predict the best
2. How is the price rice in the future

Goals

1. To find out which model can predict the best
2. To find out the price rice in the future

Solutions

1. Using Deep Learning algorithm. The Deep learning algoritm applied is RNN using LSTM. To get the best result, I use hyperparameter tunning. Also, I consider using different layer to optimize the result.
2. Using Statistical algorithm. It is ARIMA. In here, I apply Parameter Setup with Grid Search.

All of solution applied means to find the best model regarding the dataset condition.

# Data Understanding

The data belong to Badan Pangan Nasional (www.bpn.go.id). The price is based on DKI Jakarta region. And it is taken from traditional market.

The dataset consists of 1028 rows. It starts from 01-01-2022 to 10-24-2024. And there is no duplicate data.

I separate the dataset into 3 by its quality: Raw data, Clean data and Final data.

**Raw data**, which can be taken from [this site](https://panelharga.badanpangan.go.id/).

**Clean data**, Which can be downloaded from [this link](https://drive.google.com/file/d/1V7Ee6-_P6axh9PDdsYCZRO3NZa4lJt6X/view?usp=sharing). I use transpose technique and then change the string into integer. Here the variables of the data have:

1. index
2. Beras Premium
3. Beras Medium
4. Kedelai Biji Kering (impor)
5. Bawang Merah
6. Bawang Putih Bonggol
7. Cabai Merah Keriting
8. Cabai Rawit Merah
9. Daging Sapi Murni
10. Daging Ayam Ras
11. Telur Ayam Ras
12. Gula Konsumsi
13. Minyak Goreng Kemasan Sederhana
14. Tepung Terigu (curah)
15. Minyak Goreng Kemasan Premium
16. Minyak Goreng Curah
17. Jagung TK Peternak
18. Ikan Kembung
19. Ikan Tongkol
20. Ikan Bandeng
21. Garam Halus Beryodium
22. Tepung Teriugu Kemasan (Non Curah)

**Final Data**, which can be downloaded from [this link](https://drive.google.com/file/d/1f9N-A84c6sqYYCXS6xS_KFqlyO9nWbZh/view?usp=sharing). Final Data only consists of three columns: date(index), beras_premium, and beras_medium, after being deleted before. 37 of each missing data have been interpolated. This work uses this dataset to analyse.

# Data Preparation

## Data Cleaning

### Convert data to integer and transpose it 

The given code is essential for cleaning and transforming a transposed DataFrame, ensuring that all values are converted to integers while handling any non-numeric entries.

**Data Transposition:** The process begins with df_transposed = df.T, where .T transposes df, switching rows and columns. This transposition enables easier data handling, especially when certain operations require access to columns that were originally rows.

**Setting Column Names:** The line df_transposed.columns = df_transposed.iloc[0] assigns the first row of df_transposed as the new header. By doing this, we designate the initial row (iloc[0]) as column names. The line df_transposed = df_transposed[1:] then removes this now redundant row from the data, leaving the transposed DataFrame ready for further processing.

**Integer Conversion Process:** The function convert_to_int(df_transposed) performs several operations:

String Conversion: df.astype(str) converts each element in the DataFrame to a string, allowing easy manipulation using string functions.
Removing Periods: The lambda function x.str.replace('.', '', regex=False) eliminates periods from the strings, handling cases where they might represent thousands separators.
Numeric Conversion: pd.to_numeric(..., errors='coerce') then converts these strings to numeric values, marking non-convertible entries (e.g., words) as NaN.
Integer Conversion: Finally, .astype('Int64') casts the data to the Int64 type, a pandas integer format that can handle missing values (shown as <NA>).
Error Management and Data Integrity
Using errors='coerce' allows the conversion to proceed without issues if any non-numeric values are present. Entries that can’t be converted are set to <NA>, preserving data integrity and avoiding errors. This approach is particularly useful for datasets with formatting inconsistencies, common in financial or time series data where numeric values might have thousands separators or non-numeric elements.

The code performs essential data transformation and cleaning to standardize values for analysis, making the dataset more reliable by addressing common inconsistencies—an essential step in preprocessing for large datasets (Pandas Documentation, 2023).

### Drop unnecessary columns

The code snippet in here performs column removal on the df_transposed DataFrame by specifying a list of column names that should be dropped. Here’s a detailed breakdown of what each part does:

Define Columns to Drop: The columns_to_drop list includes the names of columns in df_transposed that are to be removed. These columns represent various commodity names like 'Kedelai Biji Kering (Impor)', 'Bawang Merah', 'Daging Sapi Murni', etc., possibly because they are not needed for further analysis or might be irrelevant to the study's focus.

Dropping Specified Columns:

df_transposed.drop(columns=columns_to_drop, inplace=True)
The drop() method removes the columns listed in columns_to_drop from df_transposed. The columns=columns_to_drop parameter specifically indicates that the drop operation targets columns (not rows). Setting inplace=True modifies df_transposed directly without creating a new DataFrame.

**Usage Context and Benefits**
By removing unnecessary or irrelevant columns, this code helps reduce memory usage and computational overhead, making the dataset easier to work with and more focused on relevant variables. This is especially useful in time series or forecasting tasks where including irrelevant features could introduce noise and negatively impact model performance or analysis quality.



### Interpolate the missing data

The code df_transposed = df_transposed.interpolate(method='linear') applies linear interpolation to fill missing values in the df_transposed DataFrame. Linear interpolation estimates missing values by drawing a straight line between the surrounding known data points.


Linear interpolation is used when the data is expected to change gradually and predictably over time. By filling in missing values with estimates based on surrounding data points, it ensures that the dataset remains continuous and does not lose valuable information. This is especially useful in time series data, where gaps in the data can affect analysis and predictions.

**Why Use This Method?**
Linear interpolation is simple and effective for data with gradual trends. It helps preserve the integrity of the data without introducing artificial patterns. However, it may not be suitable for data with sharp fluctuations or non-linear trends, where other interpolation methods might be more appropriate (Pandas Documentation, 2023).


### Rename column and adjust date

**Resetting the Index and Renaming Columns**

The reset_index() function is applied to df_transposed to convert the current index of the DataFrame into a regular column. This operation is useful when the index holds meaningful information, such as dates or categories, that should be part of the data itself rather than as the index. The rename(columns={...}) function then renames specific columns for clarity and consistency.

The index column, now created by reset_index(), is renamed to date.
The column 'Beras Premium' is renamed to beras_premium and 'Beras Medium' to beras_medium.
This step ensures that the DataFrame has clear and consistent column names, improving the ease of use during analysis or modeling tasks.

**Converting the Date Column to Datetime Format**

The line df_final['date'] = pd.to_datetime(df_final['date'], format='%d/%m/%Y') converts the date column, which is initially in string format, into a proper pandas datetime object. This conversion is essential for any time series analysis, as it allows the dataset to be used in time-based calculations or visualizations.

The format='%d/%m/%Y' argument specifies the exact format of the date string, which is day/month/year. By doing this, pandas ensures that the string is correctly interpreted as a date without errors.


## Explanatory Data Analysis [EDA]

### 1. Plot Line

![Plot Line](https://github.com/karyateguh/Prediction-of-price-rice-in-DKI-Jakarta/raw/master/1.%20Plot%20Line.png)


### 2. Histogram and KDE Plot

![Histogram and KDE Plot](https://github.com/karyateguh/Prediction-of-price-rice-in-DKI-Jakarta/raw/master/2.%20Histogram%20and%20KDE%20Plot.png)


Thing that we can note from above is, beras_premium and beras_medium have simillar plot. To optimize the model, from now on, we only use beras_premium to be analyzed and trained.

### 3. Box Plot for Outliers

![Box Plot For Outliers](https://github.com/karyateguh/Prediction-of-price-rice-in-DKI-Jakarta/raw/master/3.%20Box%20Plot%20For%20Outliers.png)


### 4. Decompose Time Series

![Decompose Time Series](https://github.com/karyateguh/Prediction-of-price-rice-in-DKI-Jakarta/raw/master/4.%20Decompose%20Time%20Series.png)


### 5. ACF and PACF Plots

![ACF and PACF Plot](https://github.com/karyateguh/Prediction-of-price-rice-in-DKI-Jakarta/raw/master/5.%20ACF%20and%20PACF%20Plot.png)


### 6. Moving Average and Rolling Statistics

![Moving Average and Rolling Statistics](https://github.com/karyateguh/Prediction-of-price-rice-in-DKI-Jakarta/raw/master/6.%20Moving%20Average%20and%20Rolling%20Statistics.png)

### 7. Stationarity Test (ADF Test)

>
ADF Test Statistic: -0.7297178103124442

p-value: 0.8389010757791311

Critical Value 1%: -3.436752117511071

Critical Value 5%: -2.864366633740962

Critical Value 10%: -2.5682750226211546

### 8. Seasonal Plot or Heatmap for Seasonality

![Seasonal Plot or Heatmap for Seasonality](https://github.com/karyateguh/Prediction-of-price-rice-in-DKI-Jakarta/raw/master/8.%20Seasonal%20Plot%20or%20Heatmap%20for%20Seasonality.png)


### 9. Lag Plot

![Seasonal Plot](https://github.com/karyateguh/Prediction-of-price-rice-in-DKI-Jakarta/raw/master/9.%20Seasonal%20Plot.png)


### 10. Change Point Detection

![Change Point Detection](https://github.com/karyateguh/Prediction-of-price-rice-in-DKI-Jakarta/raw/master/10.%20Change%20Point%20Detection.png)


# Data Preparation

## Normalization

Since neural networks, including LSTM models, perform better with normalized data, the code applies a MinMaxScaler to rescale the data between 0 and 1. This step helps stabilize the learning process by keeping the values within a bounded range, which in turn improves model convergence and reduces training time (Brownlee, 2017).

## Creating a Dataset for Supervised Learning

The create_dataset function is critical for transforming the univariate time series into a supervised learning format, which is required for LSTM models. In time series forecasting, the goal is to predict future values based on past observations. The function takes in two arguments: the scaled data and time_step, which determines the number of previous time steps the model will use to predict the next point in the sequence.

X: Each element in X represents a sequence of past observations of length time_step.
y: Each element in y is the value following the corresponding X sequence, effectively making it the target value the model needs to learn to predict.
By looping through the dataset, the function appends these sequences and the target values to X and y, respectively. This approach is a typical method for structuring time series data for LSTM models, converting a time series into an input-output pair format (Chollet, 2018).

## Reshaping the Data for LSTM Input Requirements

In this section, time_step is set to 60, meaning each input sequence in X will contain the last 60 time steps. This length is often selected based on prior knowledge or experimentation. After calling create_dataset, X and y are reshaped to fit the expected input format for an LSTM layer, which requires data in a 3D shape: (samples, time steps, features) (Brownlee, 2017).

## Split the data

The data is splited into 80 percent train and 20 percent test.
The train_test_split function from sklearn.model_selection is commonly used to split datasets for supervised learning tasks. It divides the data into a training set (for model learning) and a test set (for evaluation).

**Parameters Explained**

X, y: These are the features (input sequences) and target values (output sequences) generated in previous steps using the create_dataset function.
test_size=0.2: This specifies that 20% of the data will be set aside as the test set, while 80% will be used for training.

shuffle=False: For time series data, it is crucial to maintain the chronological order of observations. Shuffling would disrupt this order, potentially causing data leakage (where future data is inadvertently used to predict past values). Setting shuffle=False ensures that the training set consists of earlier data points and the test set consists of later points.

# Model 1 : Model Sequential using LSTM

## Modelling

**Sequential Model:** The Sequential model in Keras allows us to stack layers linearly, which is ideal for LSTM architectures in time series forecasting.

**First LSTM Layer:**

LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)): This layer has 50 units (or neurons), which is a common choice for LSTMs as it provides a good balance between model complexity and generalization.
return_sequences=True ensures that this layer outputs the full sequence (i.e., each time step’s output), which is necessary for stacking multiple LSTM layers.

**Dropout Layer:**

Dropout(0.2): Dropout is a regularization technique that randomly "drops" a fraction (20%) of neurons during training, reducing the risk of overfitting by forcing the model to rely on distributed representations rather than memorizing patterns (Srivastava et al., 2014).
Second LSTM Layer:

**LSTM(50, return_sequences=False):** The second LSTM layer uses 50 units, but with return_sequences=False, it outputs only the last time step’s output, which feeds into the next layer.
Dense Layer:

**Dense(1)**: This fully connected layer provides the final output, a single value predicting the next time step in the series. This layer is common in regression-based time series forecasting, where the goal is to predict a continuous value.

**Optimizer:** adam (adaptive moment estimation) is a popular optimizer that adjusts the learning rate dynamically, which often leads to faster convergence, especially for LSTM models handling complex time dependencies (Kingma & Ba, 2015).

**Loss Function:** mean_squared_error is used because it’s a standard loss function for regression tasks, penalizing large prediction errors and aiming to minimize the average of squared differences between actual and predicted values.

**EarlyStopping:** This callback monitors the val_loss (validation loss) and stops training if there’s no improvement for 10 consecutive epochs. It prevents overfitting by halting training once the model stops improving on the validation data.

**ReduceLROnPlateau:** This reduces the learning rate by a factor of 0.2 if there’s no improvement in val_loss for 5 epochs. Lowering the learning rate helps the model settle into a minimum more effectively, especially as it approaches convergence.

## Trainning

**Epochs:** Set to 100, which is the maximum number of complete passes through the training data. However, this may stop earlier if EarlyStopping is triggered.

**Batch Size:** This divides the data into batches of 32 samples each for each pass. A batch size of 32 is typical for balancing computational efficiency and memory usage.

**Validation Data:** (X_test, y_test) is used to evaluate the model on unseen data after each epoch, which helps monitor its generalization capability.

**Callbacks:** [early_stopping, reduce_lr] are added to optimize training by controlling the learning rate and monitoring for overfitting.

The structure and techniques used here are well-suited for time series forecasting because LSTM models are designed to handle sequential dependencies. By applying dropout and using callbacks like EarlyStopping and ReduceLROnPlateau, this setup encourages generalization while avoiding overfitting, which is often a challenge with time series data that can be inherently noisy (Chollet, 2018; Brownlee, 2017).

## Predicting

* model.predict(X_train) and model.predict(X_test) generate the model’s predictions on the training and test sets, respectively. These predictions are in the scaled range (usually between 0 and 1, due to the MinMaxScaler used earlier).

* scaler.inverse_transform() reverts the scaled predictions back to the original scale, undoing the scaling applied in the preprocessing step. This step is crucial for interpreting the predictions in their actual units, as it allows us to directly compare predicted and actual values in the original data range (e.g., prices).

* Here, y_train and y_test (original target values for training and test sets) are also transformed back to the original scale for comparison purposes.

* reshape(-1, 1) converts y_train and y_test to the correct shape, as inverse_transform expects a 2D array. This is necessary because y_train and y_test were previously stored as 1D arrays.

## Visualizing

![LSTM](https://github.com/karyateguh/Prediction-of-price-rice-in-DKI-Jakarta/raw/master/LSTM.png)


## Evaluating

This section of the code calculates and prints the Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) for both the training and testing predictions. Here’s an explanation of each metric and why it’s relevant in time series forecasting:

MAE measures the average magnitude of errors between predicted and actual values, without considering their direction (i.e., whether the prediction was above or below the actual value). The formula for MAE is:

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|
$$

where $y_i$ is the actual value and $\hat{y}_i$ is the predicted value.


In this code, mae_train and mae_test represent the average error (in the original unit, like Rp for prices) in the model’s predictions for the training and testing sets. Lower MAE values indicate more accurate predictions.

RMSE is similar to MAE but places more emphasis on larger errors, as it squares each error before averaging. The formula for RMSE is:

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

Squaring the errors magnifies larger discrepancies between actual and predicted values, making RMSE more sensitive to outliers than MAE.


rmse_train and rmse_test reflect the standard deviation of the prediction errors, providing insight into how spread out the errors are. Lower RMSE values also indicate better model performance, especially in scenarios where avoiding large errors is crucial.

**Result**

The model’s performance results are as follows:

> Train Mean Absolute Error (MAE): 99.52 Rp

> Test Mean Absolute Error (MAE): 151.54 Rp

> Train Root Mean Squared Error (RMSE): 179.06 Rp

> Test Root Mean Squared Error (RMSE): 220.34 Rp

The standard LSTM model shows reasonably good training accuracy but experiences a larger increase in error on the test data, indicating potential overfitting. The LSTM structure captures temporal dependencies but struggles to generalize well enough for future data points.

**Strengths:** LSTMs are highly capable of capturing sequential dependencies and handling nonlinear relationships in time series data, making them a good choice for non-seasonal industrial data.

**Weaknesses:** Standard LSTM only processes information in one direction (forward), which can limit its understanding of complex time dependencies. This model shows signs of overfitting, with a higher testing error than training error.


# Model 2 Using Bidirectional LSTM

## Modelling

* The model model2 consists of a Bidirectional LSTM layer with 100 units, set to return sequences (i.e., all hidden states) for additional layers to process.
* The second Bidirectional LSTM layer has return_sequences=False because it outputs the final hidden state, which is then connected to a fully connected layer with a single unit (for a single value prediction).
* Dropout layers are added between the LSTM layers to mitigate overfitting by randomly setting some units to zero during training.
* The lr_schedule function adjusts the learning rate after 20 epochs, reducing it by a factor of 0.1. This learning rate decay can help the model converge to a minimum with more stability after initial rapid progress.
* The LearningRateScheduler callback incorporates this schedule into the model’s training loop.
* EarlyStopping: Monitors val_loss and stops training if it does not improve for 10 epochs. This helps prevent overfitting and saves training time.
* ModelCheckpoint: Saves the model only when there’s an improvement in val_loss, ensuring that the best-performing model on validation data is saved.
* LearningRateScheduler: Dynamically adjusts the learning rate during training, which can help the model converge more effectively and avoid getting stuck in local minima.

## Trainning

* The model trains with 50 epochs and a batch size of 32, a common starting configuration for time series data.
* Validation data is used to evaluate model performance after each epoch, informing the EarlyStopping and ModelCheckpoint callbacks.

## Predicting

The predicting process is simiilar to model 1

## Visualizing

![Bidirectional](https://github.com/karyateguh/Prediction-of-price-rice-in-DKI-Jakarta/raw/master/Bidirectional.png)

## Evaluating

To evaluate the model, I use MSE and RMSE.

**Result**

The results show improvements in both the training and testing errors:

> Train Mean Absolute Error (MAE): 88.62 Rp

> Test Mean Absolute Error (MAE): 127.66 Rp

> Train Root Mean Squared Error (RMSE): 158.80 Rp

> Test Root Mean Squared Error (RMSE): 177.12 Rp

These values indicate that the bidirectional LSTM model, along with some adjustments, has enhanced the model's performance compared to earlier configurations.

The Bidirectional LSTM outperforms the standard LSTM, showing both lower training and testing errors. The train-test error gap is smaller than in Model 1, suggesting better generalization. This bidirectional model captures dependencies from both past and future contexts at each time step, leading to a richer representation of the data and improved predictive performance. 

**Strengths:** By processing information in both directions, this model captures forward and backward dependencies. This dual processing leads to improved accuracy, especially when complex temporal relationships exist. As evidenced by the lower errors, Bidirectional LSTM is better suited for industrial time series where both past and future patterns influence outcomes.

**Weaknesses:** Bidirectional LSTM models are computationally more intensive and require more resources. They may also be harder to interpret compared to simpler models like ARIMA.



# Model 3 Using ARIMA

## Parameter Setup with Grid Search:

p = range(0, 3), d = range(0, 2), and q = range(0, 3) define the search ranges for the ARIMA model’s parameters:

* p is the order of the autoregressive (AR) term.
* d is the order of differencing, which helps make the series stationary by removing trends.
* q is the order of the moving average (MA) term.

The line pdq = list(itertools.product(p, d, q)) creates a list of all possible combinations for these parameters using the Cartesian product, which enables a thorough exploration of potential configurations for the ARIMA model.

## Model Selection Using AIC:

The code initializes placeholders for tracking the best model based on the Akaike Information Criterion (AIC), a standard metric used to evaluate model quality in time series analysis. The AIC balances model complexity and fit; lower AIC values generally indicate a better model with respect to the dataset.

The loop iterates through each possible combination of ARIMA parameters. For each combination (param), the code attempts to fit an ARIMA model to the beras_premium data:

Model Fit and AIC Evaluation: For each parameter combination, the code fits the ARIMA model and computes the AIC. It then compares this AIC to the best (lowest) AIC found so far. If the current model has a lower AIC than the best seen so far, it updates best_aic, best_order, and best_model with the new values.
Error Handling: If the model fails to fit for any reason (e.g., a non-invertible model), the try-except block catches the exception and allows the loop to continue without interruption.

## Prediction with the Best Model:

After identifying the best model based on AIC, the code uses this model to make future predictions:

Date Range: The forecast is generated from '2023-01-01' to '2024-12-31', which specifies the period for which future values will be predicted.
Prediction: The predict method generates the forecasted values. The parameter typ='levels' ensures that the predictions are produced in their original scale, without any differencing applied.

**Importance of AIC in Model Selection**

The AIC criterion, as outlined by Burnham and Anderson (2002), is a valuable method for evaluating models in time series forecasting due to its ability to balance model accuracy and complexity. By minimizing AIC, the model avoids overfitting while still capturing key data patterns, a tradeoff essential in forecastingIMA in Time Series Forecasting ARIMA (AutoRegressive Integrated Moving Average) is a foundational model in time series analysis due to its flexibility and effectiveness. It captures patterns through autoregression and moving averages, while also making data stationary through differencing. This approach makes ARIMA particularly suited for forecasting datasets with trends and seasonality .

Thiserful framework for automatic ARIMA parameter tuning, ensuring an efficient balance between model performance and generalization by focusing on AIC values. By implementing such a method, we can optimize ARIMA models to enhance the accuracy of time series predictions.

## Visualisation

![ARIMA](https://github.com/karyateguh/Prediction-of-price-rice-in-DKI-Jakarta/raw/master/ARIMA.png)


## Evaluating

Here, the code finds the minimum length between the actual data (beras_premium prices) and the predicted values. This ensures that both actual_data and pred arrays have the same length for an accurate comparison.

data['beras_premium'][pred_start_date:pred_end_date]: Selects the actual values within the date range from pred_start_date to pred_end_date.
pred[:min_len]: Adjusts the predictions to have the same length as actual_data, ensuring a fair comparison.

The choice of MSE and RMSE as evaluation metrics is common in time series forecasting due to their sensitivity to large errors. RMSE, in particular, is useful for practical interpretation, as it reflects the magnitude of errors in the original unit (e.g., currency in this case) and thus provides insight into how closely the model’s predictions align with reality.

**Result**

> Mean Squared Error (MSE): 19849.31

> Root Mean Squared Error (RMSE): 140.89

The ARIMA model has a lower RMSE than Model 1's test RMSE but does not match the accuracy of the Bidirectional LSTM model. Although ARIMA is strong for linear, seasonal, or stationary data, its performance here is weaker, likely because the dataset’s industrial context includes nonlinear and complex temporal dependencies that neural networks like LSTM are better equipped to handle.

**Strengths:** By processing information in both directions, this model captures forward and backward dependencies. This dual processing leads to improved accuracy, especially when complex temporal relationships exist. As evidenced by the lower errors, Bidirectional LSTM is better suited for industrial time series where both past and future patterns influence outcomes.

**Weaknesses:** Bidirectional LSTM models are computationally more intensive and require more resources. They may also be harder to interpret compared to simpler models like ARIMA.


# Conclusion

1. For this time series task, **the Bidirectional LSTM model offers the best performance** and should be the model of choice for accurate rice price forecasting.

2. Price rice decreases in the future.

# References

Box, G. E. P., & Jenkins, G. M. (1976). Time Series Analysis: Forecasting and Control. Holden-Day.

Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: Principles and Practice (2nd ed.). OTexts.

Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.

Brownlee, J. (2017). Introduction to Time Series Forecasting with Python: How to Prepare Data and Develop Models to Predict the Future. Machine Learning Mastery.

Chollet, F. (2018). Deep Learning with Python. Manning Publications.

Zhang, G. P. (2003). Time series forecasting using a hybrid ARIMA and neural network model. Neurocomputing, 50, 159-175.

Sarem, S., & Ai, C. (2020). Improved Deep Learning Models for Stock Market Prediction. IEEE Transactions on Computational Social Systems, 7(5), 1176–1183.

Montgomery, D. C., Jennings, C. L., & Kulahci, M. (2015). Introduction to Time Series Analysis and Forecasting. John Wiley & Sons.

Jin, X., & Kim, S. (2015). Forecasting the future of deep learning: A bibliometric analysis. Springer.

