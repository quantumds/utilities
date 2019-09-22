# DIVIDE BETWEEN TRAIN AND TEST
X = df[predictive_variables_list]
y = df['name_of_target_variable']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
