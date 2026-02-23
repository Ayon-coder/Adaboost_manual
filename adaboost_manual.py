from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

class Adaboost:

    def __init__(self):
        
        self.model_obj = None
        self.input_col_name = None
        self.output = None
        self.df = None
        self.target_col_name = None
        self.alpha = None
        self.weight = None

    def copy_df(self, df):

        self.df = df
        self.weight = 1 / self.df.shape[0]
        self.df["weight"] = self.weight

    def model_weight(self, error):

        return .5 * np.log((1 - error) / (error + .0000001))

    def fit_data(self, X, y):

        self.target_col_name = y
        self.input_col_name = X
        X_train = self.df[X]
        y_train = self.df[y]

        self.model_obj = DecisionTreeClassifier(max_depth=1)
        return self.model_obj.fit(X_train, y_train)
    
    def update_row_weight(self, row):

        if row[self.target_col_name] == row["y_pred"]:
            return row["weight"] * np.exp(- self.alpha)
        else:
            return row["weight"] * np.exp(self.alpha)
    
    def normalized(self):

        self.df["normalized_weight"] = self.df["update_weight"] / self.df["update_weight"].sum()
        self.df["cumsum_upper"] = np.cumsum(self.df["normalized_weight"])
        self.df["cumsum_lower"] = self.df["cumsum_upper"] - self.df["normalized_weight"]

    def create_new_dataset(self):

        indices = []
        for i in range(self.df.shape[0]):
            a = np.random.random()
            for index, row in self.df.iterrows():

                if row["cumsum_upper"] > a and a > row["cumsum_lower"]:
                    indices.append(index)
        return indices
    
    def alpha_calculate(self):

        y = self.df[self.target_col_name]
        y_pred = self.df["y_pred"] = y_pred = self.model_obj.predict(self.df[self.input_col_name])
        misclassified = (y != y_pred)
        error = (self.df["weight"] * misclassified).sum()
        self.alpha = self.model_weight(error)

        self.df["update_weight"] = self.df.apply(self.update_row_weight, axis = 1)
        self.normalized()

        index_values = self.create_new_dataset()
        self.df = self.df.iloc[index_values, [0, 1, 2, 3]]
        self.df = self.df.reset_index(drop=True)
        self.weight = 1 / self.df.shape[0]
        self.df["weight"] = self.weight

        return self.alpha, self.df



df = pd.DataFrame()
df["x1"] = [1, 2, 3, 4, 5, 6, 6, 7, 7, 8, 9, 9, 9]
df["x2"] = [5, 5, 6, 8, 7, 2, 3, 4, 1, 1, 4, 6, 7]
df["label"] = [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1]
df["label"] = df["label"].replace({0: -1, 1: 1})

adaboost = Adaboost()
adaboost.copy_df(df)
X = ["x1", "x2"]
y = "label"

query = np.array([[7, 1]])

# adaboost.fit_data(X, y)
# alpha, new_Df = adaboost.alpha_calculate(weight)
# print(new_Df)

model_num = int(input("Enter number of moodels: "))
alphas = []
dts = []

for _ in range(model_num):

    dts.append(adaboost.fit_data(X, y))
    alpha, new_df= adaboost.alpha_calculate()
    alphas.append(alpha)
ans = 0
for i in range(len(dts)):

    model = dts[i]
    ans += alphas[i] * model.predict(query)
    
print(np.sign(ans))