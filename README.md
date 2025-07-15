#Heart Disease Prediction with Logistic Regression and Random Forest
本專案使用心臟病數據集，建立 Logistic Regression 與 Random Forest 預測模型，並透過多種視覺化工具深入分析模型效果。
📁 資料說明

檔案名稱：Heart_Disease_and_Hospitals.csv
主要欄位：
blood_pressure — 血壓
cholesterol — 膽固醇
bmi — 身體質量指數
glucose_level — 血糖水平
gender — 性別（經 One-Hot Encoding）
heart_disease — 是否有心臟病 (0 = 否, 1 = 是)

⚙️ 使用技術與套件

Python 3.x
pandas
numpy
scikit-learn
matplotlib
seaborn

🧩 模型建構流程

資料前處理
One-Hot Encoding 處理 gender
特徵選取：blood_pressure, cholesterol, bmi, glucose_level, gender_Male
Logistic Regression 使用標準化 (StandardScaler)
資料切分
使用 train_test_split 切分訓練與測試集（比例 70% / 30%）
模型訓練與評估
Logistic Regression
Random Forest Classifier
視覺化分析
✅ 混淆矩陣 (Confusion Matrix)
✅ ROC 曲線 (ROC Curve) 與 AUC 分數
✅ Random Forest 特徵重要性 (Feature Importance)
✅ 模型效能比較圖 (Accuracy & AUC)

📊 模型效果範例

Model	Accuracy	AUC
Logistic Regression	0.78	0.84
Random Forest	0.82	0.87
混淆矩陣：顯示模型預測的真陽性、假陽性等結果
ROC 曲線：評估模型區分能力
特徵重要性：了解哪些特徵對模型最有影響

執行方式

  確認安裝以下套件：
  pip install pandas numpy scikit-learn matplotlib seaborn
  將資料放置在指定路徑：
  C:/Users/user/Desktop/test_01/Heart_Disease_and_Hospitals.csv
  執行 Python 程式即可看到結果與圖表

視覺化範例

  混淆矩陣
  ROC 曲線
  特徵重要性長條圖

可擴充方向建議

  增加其他模型（如 XGBoost、SVM）
  套用交叉驗證 (Cross Validation)
  使用網格搜尋 (Grid Search) 調整參數
  整合報告生成工具 (如 classification_report)
