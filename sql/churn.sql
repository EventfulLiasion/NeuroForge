-- JD: SQL. Churn analytics on SQLite (mirrors BigQuery patterns).
-- Q1: churn rate by contract
SELECT Contract, COUNT(*) AS n,
  ROUND(100.0*SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END)/COUNT(*),1) AS churn_pct
FROM churn GROUP BY Contract ORDER BY churn_pct DESC;
-- Q2: avg bill churned vs stayed
SELECT Churn, ROUND(AVG(MonthlyCharges),2) AS avg_bill, ROUND(AVG(tenure),1) AS avg_tenure
FROM churn GROUP BY Churn;
-- Q3: high-risk segment (interview demo)
SELECT tenure, MonthlyCharges, Contract FROM churn
WHERE Contract='Month-to-month' AND MonthlyCharges>70 AND tenure<12 LIMIT 20;
