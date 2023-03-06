# small_data
This code runs Lasso Regression to look for key drivers of a target variable in a small data set. 
There is an excel file that contains 18 input fields and 3 target data fields related to 10 companies.
One of the three target fields, "stock price", has a signal. The other two targets, as well as the inputs, are random noise.
You can see the details of how the source data was generated in the helper file, create_excel_file.py.
The pngs show the coefficients of the fitted Lasso regression. 
Compare the model that used "stock price" as the target with the runs that used "rating" and "followers" as target.
You can see that the model was able to zero in (literally, non-used coefficients go to zero) on the important drivers/features.

