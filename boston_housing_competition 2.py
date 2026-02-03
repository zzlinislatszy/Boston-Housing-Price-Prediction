"""
File: boston_housing_competition.py
Name: tsz
--------------------------------
This file demonstrates how to analyze boston
housing dataset. Students will upload their 
results to kaggle.com and compete with people
in class!

You are allowed to use pandas, sklearn, or build the
model from scratch! Go data scientists!
"""

import pandas as pd
from sklearn import ensemble, linear_model, metrics
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, RobustScaler
import xgboost
from xgboost import plot_importance
import matplotlib.pyplot as plt

TRAIN_PATH = 'boston_housing/train.csv'
TEST_PATH = 'boston_housing/test.csv'


def main():
    # Read file
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    # Data prep.
    train_df = train.drop(columns='ID')
    test_id = test.ID
    test_df = test.drop(columns='ID')

    # Feature
    features = [col for col in train_df.columns if col != 'medv']

    # x, y
    x = train_df[features]
    y = train_df.medv

    # Data split
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.25, random_state=42)

    # Model
    # Ridge -> L2
    ridge = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        linear_model.Ridge(alpha=0.0001, solver='lsqr')
    )

    # Lasso -> L1 + 弱特徵歸0
    lasso = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        linear_model.Lasso(alpha=0.0001, max_iter=1000000)
    )

    # SVR
    svr_pipeline = make_pipeline(
        StandardScaler(),
        SVR(
            C=300,                  # 調整誤差容忍度的權重
            epsilon=0.3,            # 預測誤差容忍範圍
            kernel='poly',          # 非線性核函數
            degree=2,
            gamma=2                 # kernel影響範圍
        )
    )

    # Random Forest
    rf = ensemble.RandomForestRegressor(
        n_estimators=300,
        max_depth=9,
        max_features='sqrt',        # 分裂時選幾個特徵 -> 隨機減少特徵數量
        ccp_alpha=0.01,             # cost-complexity pruning 剪枝強度 -> 減少複雜度
        random_state=42
    )

    # XGB
    xgb = xgboost.XGBRegressor(
        n_estimators=300,
        max_depth=9,
        min_child_weight=2,         # 該節點的總權重必須大於某值，才能分裂
        gamma=2,                    # min split gain threshold
        reg_alpha=1,                # L1
        reg_lambda=3,               # L2
        learning_rate=0.15,
        random_state=42
    )

    # Voting Regressor
    vote = VotingRegressor(
        estimators=[('rf', rf), ('xgb', xgb), ('svr', svr_pipeline), ('lasso', lasso), ('ridge', ridge)],
        weights=[2, 12, 0.5, 0.5, 0.5]
    )
    vote.fit(x_train, y_train)

    # Train RMS
    train_pred = vote.predict(x_train)
    train_rms = metrics.mean_squared_error(y_train, train_pred, squared=False)
    print(f'Train RMS: {train_rms}')

    # Val RMS
    val_pred = vote.predict(x_val)
    val_rms = metrics.mean_squared_error(y_val, val_pred, squared=False)
    print(f'Val RMS: {val_rms}')

    # Test
    test_pred = vote.predict(test_df)
    out_file(test_pred, test_id, 'voting_39.csv')

    # # drawing
    # xgb_trained = vote.named_estimators_['xgb']
    # """
    # .named_estimators_
    # {'ridge': Ridge().fit(X_train, y_train),
    # 'rf': RandomForestRegressor().fit(X_train, y_train),
    # 'xgb': XGBRegressor().fit(X_train, y_train)}
    # """
    # plt.figure(figsize=(10, 6))
    # plot_importance(xgb_trained, max_num_features=10, importance_type='gain')
    # plt.title("Top 10 Feature Importances (Gain)")
    # plt.show()


def out_file(predictions, test_id, filename):
    """
    : param predictions: numpy.array, a list-like data structure that stores 0's and 1's
    : param filename: str, the filename you would like to write the results to
    """
    print('\n===============================================')
    print(f'Writing predictions to --> {filename}')
    with open(filename, 'w') as out:
        out.write('ID,medv\n')
        for i in range(len(predictions)):
            out.write(f"{int(test_id.iloc[i])},{predictions[i]}\n")
    print('===============================================')


"""
note 🥞
* XGBoost 的 boosting 過程中，每一筆樣本都會被計算出殘差：
    1. 一階導數（Gradient）→ 用來判斷方向（是不是要加/減）
    2. 二階導數（Hessian）→ 用來衡量「有多陡、多不穩定」
* Boosting 是一種將多個weak learners串聯起來、逐步糾正前一個錯誤的策略
"""

if __name__ == '__main__':
    main()
