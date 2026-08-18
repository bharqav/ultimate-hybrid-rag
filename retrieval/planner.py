class QueryPlanner:
    def classify(self, query: str):
        tokens = query.split()
        has_digits = any(t.isdigit() for t in tokens)
        is_short = len(tokens) <= 4
        if has_digits and is_short:
            return {"bm25": 0.4, "vector": 0.1, "splade": 0.3, "colbert": 0.2}
        if len(tokens) > 20:
            return {"bm25": 0.1, "vector": 0.4, "splade": 0.2, "colbert": 0.3}
        return {"bm25": 0.25, "vector": 0.3, "splade": 0.2, "colbert": 0.25}
