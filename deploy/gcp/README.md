# GCP path (JD: GCP, Vertex AI, BigQuery, Cloud Functions) — deploy when ready.
# 1. Build: gcloud builds submit --tag gcr.io/PROJECT/neuroforge
# 2. Run: gcloud run deploy neuroforge --image gcr.io/PROJECT/neuroforge --platform managed --allow-unauthenticated
# 3. Vertex: swap llm_client.generate() Groq branch for vertexai.generative_models.GenerativeModel("gemini-1.5-flash")
# 4. BigQuery mirror of sql/churn.sql:
#    SELECT Contract, COUNT(*) n, AVG(MonthlyCharges) FROM `PROJECT.ds.churn` GROUP BY Contract
# 5. Cloud Function: wrap POST /ml/predict as GCF entry point (see deploy/gcp/function_main.py)
