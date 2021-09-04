# DIVIDE BETWEEN TRAIN AND TEST
X = df[predictive_variables_list]
y = df['name_of_target_variable']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# TRAINING MATCHING TESTING COLUMNS ORDER
def add_predictions_column(df, model, predictor_variables, reindex_reference):
    x_onehot = x_onehot.reindex(columns=reindex_reference, fill_value=0)
Reindex_reference: list of columns to use to reindex the DataFrame after One Hot Encoding. 
This is done to make sure that the columns of the one-hot-encoded DF are exactly the same of the training set.
