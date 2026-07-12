# E-commerce Customer Churn Analysis & Prediction
My project for Data Intelligence class

## Dataset dictionary 

| Variable                     | Description                                                   |
|------------------------------|---------------------------------------------------------------|
| CustomerID                   | Unique customer ID                                            |
| Churn                        | Churn Flag (0 → No, 1 → Yes)                                                   |
| Tenure                       | Length customer has been with the company (in months)                    |
| PreferredLoginDevice         | Preferred login device of customer                            |
| CityTier                     | City tier (1 → big, 2 → medium, 3 → small)                    |
| WarehouseToHome              | Distance between warehouse and customer's home (in km)               |
| PreferredPaymentMode         | Preferred payment method of customer                          |
| Gender                       | Gender of customer                                            |
| HourSpendOnApp               | Number of hours spent on mobile application or website        |
| NumberOfDeviceRegistered     | Total number of devices registered for a particular customer  |
| PreferedOrderCat             | Type of products the customer orders most last month (Laptop & Accessory/Mobile Phone/Fashion/Mobile/Grocery)         |
| SatisfactionScore            | Satisfaction score of customer on service                     |
| MaritalStatus                | Marital status of customer                                    |
| NumberOfAddress              | Total number of addresses added for a particular customer     |
| Complain                     | Any complaint raised in last month                            |
| OrderAmountHikeFromlastYear  | Percentage increase in customer's spending amount from last year            |
| CouponUsed                   | Total number of coupons used last month                    |
| OrderCount                   | Total number of orders placed last month                   |
| DaySinceLastOrder            | Days since customer's last order                             |
| CashbackAmount               | Average cashback in last month                                |  

## Model Versions

This repo contains three iterations of the same e-commerce churn prediction pipeline (data loading → cleaning → feature engineering → feature selection → XGBoost tuning → evaluation), each built on the previous one.

### v1: e_comm_full_pipeline.ipynb - Baseline (F1-optimized)
Initial working pipeline: median imputation, IQR-based outlier handling (Winsorization + dropping), one-hot encoding, engineered ratio features, and RandomForest-based feature selection (median importance cutoff). Two competing approaches to class imbalance (`scale_pos_weight` vs. SMOTE-in-pipeline) were tuned side by side, and the decision threshold was chosen to maximize F1-score on the training set.
- **Test F1: 0.7563 | Precision: 0.7849 | Recall: 0.7297 | ROC AUC: 0.9209 | PR-AUC: 0.7887**

### v2: e_comm_full_pipeline_revised.ipynb - Cleaned up (still F1-optimized)
Same objective as v1, but fixes two methodology issues: `CustomerID` is dropped immediately after the train/test split instead of lingering through feature selection (where it was previously scoring high enough on RF importance to nearly get selected as a feature), and the pipeline is standardized on `scale_pos_weight` only, removing the SMOTE variant that was silently overwriting it despite scoring lower on cross-validation. The feature-selection importance threshold is also now chosen by comparing candidate thresholds (25th/50th/75th percentile, mean) via CV F1 instead of assuming the median is best.
- **Test F1: 0.7357 | Precision: 0.7418 | Recall: 0.7297 | ROC AUC: 0.9357 | PR-AUC: 0.8089**

### v3: e_comm_full_pipeline_revised_v3.ipynb - Recall-optimized (F2)
Same clean pipeline as v2, but retargeted end-to-end for recall: feature selection, hyperparameter tuning, and decision threshold selection all optimize for F2-score (recall weighted 2x precision) instead of F1, on the reasoning that missing a churner is costlier than a false alarm. The `scale_pos_weight` search grid was also widened to give the tuner more room to push toward higher recall.
- **Test F2: 0.7849 | Precision: 0.5548 | Recall: 0.8757 | F1: 0.6792 | ROC AUC: 0.9420 | PR-AUC: 0.8227**

**Note:** ROC AUC and PR-AUC — the two threshold-independent metrics - improve monotonically from v1 → v3, confirming that the underlying model quality genuinely improved at each step; the F1 dip in v3 reflects a deliberate shift in *where* on the precision-recall curve the threshold sits, not a worse model.
