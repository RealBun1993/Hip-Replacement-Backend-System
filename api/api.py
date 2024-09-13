import json
import joblib
import pandas as pd
import numpy as np
from sklearn.utils import resample

# 加载模型和特征
kidney_model = joblib.load('./kidney_model.pkl')
kidney_feature_names = joblib.load('./kidney_feature_names.pkl')
kidney_categorical_features = joblib.load('./kidney_categorical_features.pkl')

blood_model = joblib.load('./blood_model.pkl')
blood_feature_names = joblib.load('./blood_feature_names.pkl')
blood_categorical_features = joblib.load('./blood_categorical_features.pkl')

# Vercel Serverless 函数的标准入口
def handler(req, res):
    # 检查请求路径以调用不同的函数
    path = req.path
    
    if path == "/api/predict_kidney":
        return predict_kidney(req, res)
    elif path == "/api/predict_blood":
        return predict_blood(req, res)
    else:
        return res.status(404).json({"error": "Endpoint not found"})

# 肾功能损伤预测函数
def predict_kidney(req, res):
    try:
        data = json.loads(req.body.decode())
        df = pd.DataFrame(data)
        df = df[kidney_feature_names]

        # 将类别列转换为类别类型
        for col in kidney_categorical_features:
            df[col] = df[col].astype('category')

        # 进行预测
        prob_predictions = kidney_model.predict_proba(df)[:, 1]

        # 计算置信区间
        conf_intervals = []
        n_iterations = 1000
        noise_std = 0.4
        np.random.seed(42)
        for p in prob_predictions:
            bootstrapped_preds = []
            for _ in range(n_iterations):
                boot_df = resample(df)
                noise = np.random.normal(0, noise_std, size=boot_df.shape)
                boot_df += noise
                boot_pred = kidney_model.predict_proba(boot_df)[:, 1].mean()
                bootstrapped_preds.append(boot_pred)

            lower = np.percentile(bootstrapped_preds, 2.5)
            upper = np.percentile(bootstrapped_preds, 97.5)
            conf_intervals.append((lower, upper))

        # 返回结果
        results = [{'probability': p, 'conf_interval': ci} for p, ci in zip(prob_predictions, conf_intervals)]
        return res.json(results)
    except Exception as e:
        return res.status(500).json({"error": str(e)})

# 输血预测函数
def predict_blood(req, res):
    try:
        data = json.loads(req.body.decode())
        df = pd.DataFrame(data)
        df = df[blood_feature_names]

        # 将类别列转换为类别类型
        for col in blood_categorical_features:
            df[col] = df[col].astype('category')

        # 进行预测
        prob_predictions = blood_model.predict_proba(df)[:, 1]

        # 计算置信区间
        conf_intervals = []
        n_iterations = 1000
        noise_std = 0.4
        np.random.seed(42)
        for p in prob_predictions:
            bootstrapped_preds = []
            for _ in range(n_iterations):
                boot_df = resample(df)
                noise = np.random.normal(0, noise_std, size=boot_df.shape)
                boot_df += noise
                boot_pred = blood_model.predict_proba(boot_df)[:, 1].mean()
                bootstrapped_preds.append(boot_pred)

            lower = np.percentile(bootstrapped_preds, 2.5)
            upper = np.percentile(bootstrapped_preds, 97.5)
            conf_intervals.append((lower, upper))

        # 返回结果
        results = [{'probability': p, 'conf_interval': ci} for p, ci in zip(prob_predictions, conf_intervals)]
        return res.json(results)
    except Exception as e:
        return res.status(500).json({"error": str(e)})
