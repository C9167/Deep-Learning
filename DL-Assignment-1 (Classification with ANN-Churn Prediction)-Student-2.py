#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <p style="text-align: center;"><img src="https://docs.google.com/uc?id=1lY0Uj5R04yMY3-ZppPWxqCr5pvBLYPnV" class="img-fluid" alt="CLRSWY"></p>
# 
# ___

# <h1 style="text-align: center;">Deep Learning<br><br>Assignment-1 (ANN)<br><br>Churn Prediction for Bank Customer<br><h1>

# # Dataset Info

# In[ ]:





# # Improt Libraries & Data

# In[ ]:


get_ipython().system('pip install tensorflow')


# In[8]:


try:
    import jupyter_black

    jupyter_black.load()
except ImportError:
    pass


# In[9]:


import os

# Set TF log level to ignore INFOs.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


# In[10]:


# Check python version
get_ipython().system('python --version')


# In[11]:


# Check tensorflow version
import tensorflow as tf

tf.__version__


# In[14]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import tensorflow as tf

# Uncomment the following lines if you want to suppress warnings:
# import warnings
# warnings.filterwarnings("ignore")
# warnings.warn("this will not show")

plt.rcParams["figure.figsize"] = (10, 6)

sns.set_style("whitegrid")
pd.set_option("display.float_format", lambda x: "%.3f" % x)

# Set it None to display all rows in the dataframe
# pd.set_option('display.max_rows', None)

# Set it to None to display all columns in the dataframe
pd.set_option("display.max_columns", None)


# In[15]:


# Set the seed using keras.utils.set_random_seed. This will set:
# 1) `numpy` seed
# 2) `tensorflow` random seed
# 3) `python` random seed
seed = 42
keras.utils.set_random_seed(seed)

# This will make TensorFlow ops as deterministic as possible, but it will
# affect the overall performance, so it's not enabled by default.
# `enable_op_determinism()` is introduced in TensorFlow 2.9.
tf.config.experimental.enable_op_determinism()


# In[16]:


# Get python version
get_ipython().system('python --version')


# In[17]:


# Get tensorflow version
tf.__version__


# In[18]:


# List cuda-capable gpu's that are attached to this session
if tf.config.list_physical_devices("GPU"):
    print("GPU support is enabled for this session.")
else:
    print("CPU will be used for this session.")


# In[22]:


# Get more information about gpu (if available)
if tf.config.list_physical_devices("GPU"):
    get_ipython().system('nvidia-smi')


# In[ ]:





# In[ ]:





# # Exploratory Data Analysis and Visualization

# 1. Implement basic steps to see how is your data looks like
# 2. Check for missing values
# 3. Drop the features that not suitable for modelling
# 4. Implement basic visualization steps such as histogram, countplot, heatmap
# 5. Convert categorical variables to dummy variables

# In[23]:


df = pd.read_csv("Churn_Modelling.csv")


# In[24]:


df.head().T


# In[25]:


df.info()


# In[26]:


df.isna().sum()


# In[27]:


df.nunique()


# In[ ]:


uniq=df.nunique()
ord_cols=uniq[uniq==3].index
ord_cols


# In[28]:


df.drop(["CustomerId","Surname","RowNumber"], inplace = True, axis = 1)


# In[29]:


df.describe().T


# In[30]:


sns.pairplot(df)
plt.show()


# In[31]:


df.corr()


# In[32]:


plt.figure(figsize=(15, 8))
sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, cmap="coolwarm")
plt.show()


# In[33]:


df['Gender'].value_counts()


# In[34]:


ax = sns.countplot(x=df["Gender"], color="g")
ax.bar_label(ax.containers[0])
plt.show()


# In[37]:


df['Geography'].value_counts()


# In[38]:


ax = sns.countplot(x=df["Geography"], color="g")
ax.bar_label(ax.containers[0])
plt.show()


# In[39]:


ax = sns.countplot(x=df["Age"], color="g")
ax.bar_label(ax.containers[0])
plt.show()


# In[43]:


ax = sns.countplot(x=df["Exited"])
ax.bar_label(ax.containers[0]);


# In[44]:


sns.scatterplot(x="Age", y="Balance", data=df, hue="Exited", palette="coolwarm")
plt.show()


# In[45]:


sns.scatterplot(x="Age", y="Balance", data=df, hue="Tenure", palette="coolwarm")
plt.show()


# In[46]:


sns.boxplot(x="Tenure", y="Balance", data=df, palette="Accent")
plt.show()


# In[47]:


df['Tenure'].describe()


# In[48]:


df['Tenure'].value_counts().head(10)


# In[57]:


def Tenure(t):
    if t<=2:
        return 1
    elif t>2 and t<=4:
        return 2
    elif t>4 and t<=6:
        return 3
    elif t>6 and t<=8:
        return 4
    else:
        return 5

df["Tenure_Group"]=df["Tenure"].apply(lambda x: Tenure(x))


# In[58]:


df["Tenure_Group"].value_counts()


# In[59]:


ax = sns.countplot(x=df["Tenure_Group"], color="g")
ax.bar_label(ax.containers[0])
plt.show()


# In[ ]:





# In[ ]:





# # Preprocessing of Data
# - Train | Test Split, Scalling

# In[60]:


from sklearn.model_selection import train_test_split


# In[61]:


X = df[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
Y = df['Exited']

# Performing one-hot encoding
X = pd.get_dummies(X, columns=['Geography', 'Gender'], drop_first=True)

# Resulting X after one-hot encoding
print(X.head())


# In[62]:


from sklearn.preprocessing import LabelEncoder

# Assuming your dataset is in a DataFrame named 'df'
# Extracting features and target variable
X = df[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
y= df['Exited']

# Performing label encoding
label_encoder = LabelEncoder()
X['Geography'] = label_encoder.fit_transform(X['Geography'])
X['Gender'] = label_encoder.fit_transform(X['Gender'])

X


# In[63]:


y


# In[64]:


import os

seed = 101
# "TF_DETERMINISTIC_OPS" to "1". This is a TensorFlow feature that, when a random seed is set,
# makes the operations deterministic (i.e., less random). This can also help with reproducibility.
os.environ["TF_DETERMINISTIC_OPS"] = "1"
# 'tf.keras.utils.set_random_seed' is a function that sets the random seed for TensorFlow.
# Here, we're using the seed value we set earlier. This means that the randomness
# in any TensorFlow operations will be determined by this seed, and so will be reproducible.
tf.keras.utils.set_random_seed(seed)


# In[65]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=seed
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=seed
)


# In[66]:


from sklearn.preprocessing import MinMaxScaler  # RobustScaler()

# If there are too many outliers in the data, robust scaler should be used, otherwise minmax can be used.


# In[67]:


scaler = MinMaxScaler()


# In[68]:


X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# # Modelling & Model Performance

# ## without class_weigth

# ### Create The Model

# In[ ]:


get_ipython().system('pip install livelossplot')


# In[69]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import RocCurveDisplay, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, cross_validate
#from sklearn.model_selection import GridSearchCV


# In[71]:


X_train.shape


# In[72]:


X_test.shape


# In[73]:


model = Sequential()

model.add(Dense(15, activation="relu", input_dim=X_train.shape[1]))
model.add(Dense(10, activation="relu"))
model.add(Dense(5, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# In[74]:


model.fit(
    x=X_train, y=y_train, validation_split=0.1, batch_size=32, epochs=1000, verbose=1
)


# In[75]:


# if you want to see the summary of the architecture before the fit process, you must define the input_dim
model.summary()


# In[76]:


loss_df = pd.DataFrame(model.history.history)
loss_df.head()


# In[77]:


loss_df.plot()
plt.show()


# In[78]:


loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("loss: ", loss)
print("accuracy: ", accuracy)


# In[79]:


y_pred = model.predict(X_test) > 0.5
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[80]:


X_train1, X_val, y_train1, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.1, random_state=seed)


# In[81]:


model2 = Sequential()

model2.add(Dense(15, activation="relu", input_dim=X_train.shape[1]))
model2.add(Dense(10, activation="relu"))
model2.add(Dense(5, activation="relu"))
model2.add(Dense(1, activation="sigmoid"))

model2.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# In[82]:


model2.fit(x=X_train1, y=y_train1, validation_data=(X_val,y_val), batch_size=32, epochs=1000, verbose=1
)


# In[83]:


loss_df = pd.DataFrame(model2.history.history)
loss_df.plot()


# In[84]:


model2.evaluate(X_test, y_test, verbose=0)


# In[85]:


loss, accuracy = model2.evaluate(X_test, y_test, verbose=0)
print("loss: ", loss)
print("accuracy: ", accuracy)


# In[86]:


y_pred = model2.predict(X_test) > 0.5
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# # EarlyStopping

# In[87]:


from tensorflow.keras.callbacks import EarlyStopping


# In[88]:


model = Sequential()

model.add(Dense(15, activation="relu", input_dim=X_train.shape[1]))
model.add(Dense(10, activation="relu"))
model.add(Dense(5, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# In[ ]:


# The patience is often set somewhere between 10 and 100
# (10 or 25 is more common), but it really depends on your dataset and network.


# In[89]:


early_stop = EarlyStopping(
    monitor="val_loss", mode="min", verbose=1, patience=15, restore_best_weights=True
)


# In[90]:


model.fit(
    x=X_train,
    y=y_train,
    validation_split=0.1,
    batch_size=32,
    epochs=1000,
    verbose=1,
    callbacks=[early_stop],
)


# In[91]:


loss_df = pd.DataFrame(model.history.history)
loss_df.plot()
plt.show()


# In[92]:


loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("loss: ", loss)
print("accuracy: ", accuracy)


# In[93]:


y_pred = model.predict(X_test) > 0.5
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# # learning_rate¶

# In[95]:


from tensorflow.keras.optimizers import Adam


# In[96]:


model = Sequential()

model.add(Dense(15, activation="relu", input_dim=X_train.shape[1]))
model.add(Dense(10, activation="relu"))
model.add(Dense(5, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

opt = Adam(learning_rate=0.005)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])


# In[97]:


early_stop = EarlyStopping(
    monitor="val_loss", mode="min", verbose=1, patience=15, restore_best_weights=True
)


# In[98]:


model.fit(
    x=X_train,
    y=y_train,
    validation_split=0.1,
    batch_size=32,
    epochs=1000,
    verbose=1,
    callbacks=[early_stop],
)


# In[99]:


loss_df = pd.DataFrame(model.history.history)
loss_df.plot()


# In[100]:


loss_df.plot(subplots=[['loss','val_loss'],['accuracy','val_accuracy']],layout=(2,1),figsize=(15,10));


# In[101]:


loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("loss: ", loss)
print("accuracy: ", accuracy)


# In[102]:


y_pred = model.predict(X_test) > 0.5
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# # Dropout

# The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting.

# In[103]:


from tensorflow.keras.layers import Dropout


# In[104]:


model = Sequential()

model.add(Dense(15, activation="relu", input_dim=X_train.shape[1]))
model.add(Dropout(0.5))

model.add(Dense(10, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(5, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(1, activation="sigmoid"))

opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])


# In[105]:


model.fit(
    x=X_train, y=y_train, validation_split=0.1, batch_size=32, epochs=1000, verbose=1
)


# In[106]:


loss_df = pd.DataFrame(model.history.history)
loss_df.plot()


# In[108]:


loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("loss: ", loss)
print("accuracy: ", accuracy)


# In[109]:


y_pred = model.predict(X_test) > 0.5
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[110]:


model.save("model_churn_1.h5")


# # Cross Validation

# Keras models can be used in scikit-learn by wrapping them with the KerasClassifier or KerasRegressor class.
# To use these wrappers you must define a function that creates and returns your Keras sequential model, then pass this function to the build_fn argument when constructing the KerasClassifier class.
# The constructor for the KerasClassifier class can take default arguments that are passed on to the calls to model.fit(), such as the number of epochs and the batch size.

# In[111]:


get_ipython().system('pip install scikeras')


# In[112]:


from scikeras.wrappers import KerasClassifier
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


# In[133]:


def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=15, activation="relu", input_dim=X_train.shape[1]))
    classifier.add(Dense(units=10, activation="relu"))
    classifier.add(Dense(units=5, activation="relu"))
    classifier.add(Dense(units=1, activation="sigmoid"))
    classifier.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    return classifier


# In[134]:


X_train.shape


# In[135]:


y_train.shape


# In[136]:


classifier_model = KerasClassifier(
    model=build_classifier, batch_size=32, epochs=100, verbose=0
)

scores = cross_validate(
    estimator=classifier_model,
    X=X_train,
    y=y_train,
    scoring=["accuracy", "precision", "recall", "f1"],
    cv=10,
)

df_scores = pd.DataFrame(scores, index=range(1, 11)).iloc[:, 2:]

df_scores


# In[137]:


df_scores_summary = pd.DataFrame(
    {"score_mean": df_scores.mean().values, "score_std": df_scores.std().values},
    index=["acc", "pre", "rec", "f1"],
)

df_scores_summary


# # Hyperparameter search with Optuna

# In[138]:


get_ipython().system('pip install optuna')


# In[139]:


from tensorflow.keras.optimizers import Adam, Adadelta, RMSprop, Nadam
import optuna


# In[140]:


early_stop = EarlyStopping(
    monitor="val_loss", mode="min", verbose=1, patience=15, restore_best_weights=True
)


# In[141]:


trial_metric = "accuracy"
batch_size = 64


def create_model(trial):
    # Some hyperparameters we want to optimize
    n_units1 = trial.suggest_int("n_units1", 8, 128)
    n_units2 = trial.suggest_int("n_units2", 8, 128)
    optimizer = trial.suggest_categorical("optimizer", [Adam, Adadelta, RMSprop, Nadam])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1.3e-1)

    model = Sequential()
    model.add(Dense(n_units1, input_dim=X_train.shape[1], activation="relu"))
    model.add(Dense(n_units2, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer(learning_rate=learning_rate),
        metrics=[trial_metric],
    )
    return model


def objective(trial):
    model = create_model(trial)
    w0 = trial.suggest_loguniform("w0", 0.01, 5)
    w1 = trial.suggest_loguniform("w1", 0.01, 5)
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=100,
        callbacks=[early_stop],
        class_weight={0: w0, 1: w1},
        verbose=0,
    )
    score = model.evaluate(X_test, y_test, verbose=0)[1]
    return score


# In[ ]:


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)
study.best_params


# In[ ]:


# build model with optuna parameters
unit1, unit2, optimizer, lr, w0, w1 = (
    study.best_params["n_units1"],
    study.best_params["n_units2"],
    study.best_params["optimizer"],
    study.best_params["learning_rate"],
    study.best_params["w0"],
    study.best_params["w1"],
)

model = Sequential()
model.add(Dense(unit1, activation="relu"))
model.add(Dense(unit2, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
opt = optimizer(learning_rate=lr)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["Recall"])

# train model
model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    batch_size=64,
    epochs=100,
    callbacks=[early_stop],
    verbose=1,
)


# In[ ]:


history = model.history.history


# In[ ]:


y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# # ROC (Receiver Operating Curve) and AUC (Area Under Curve)

# In[ ]:


y_pred_proba = model.predict(X_test)
RocCurveDisplay.from_predictions(y_test, y_pred_proba)


# In[ ]:


roc_auc_score(y_test, y_pred_proba)


# # Saving Final Model and Scaler

# In[ ]:





# ### Increase The Learning Rate and Observe The Results

# In[ ]:





# ## Prediction

# In[ ]:





# # Comparison with ML¶

# # Logistic Regression

# In[129]:


from sklearn.linear_model import LogisticRegression


# In[130]:


log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# # Random Forest

# In[131]:


from sklearn.ensemble import RandomForestClassifier


# In[132]:


rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:





# ___
# 
# <p style="text-align: center;"><img src="https://docs.google.com/uc?id=1lY0Uj5R04yMY3-ZppPWxqCr5pvBLYPnV" class="img-fluid" alt="CLRSWY"></p>
# 
# ___
