"""GCF entry: forwards to same predict logic (JD: Cloud Functions)."""
def predict(request):
    from src.ml.predict import predict_rows
    body = request.get_json(silent=True) or {}
    return {"results": predict_rows(body.get("rows", []))}, 200
