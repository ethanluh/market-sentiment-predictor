# Market Sentiment Predictor — Project Plan

## Objective
Build a pipeline that predicts stock price movement by analyzing market sentiment from multiple data sources, applying sentiment analysis to key indicators, and modeling effects using graph theory.

## Milestones & Steps

### 1. Project Setup
- [x] Set up Python environment and dependencies
- [x] Organize project structure (src/, tests/, data/, docs/)
- [ ] Version control with Git

### 2. Data Ingestion
- [ ] Implement data fetchers for:
    - News articles
    - Social media posts
    - Financial filings
    - General company info
- [ ] Store raw data in data/raw/

### 3. Indicator Extraction
- [ ] Define indicator schema (performance, management, ESG, etc.)
- [ ] Write extraction logic for each indicator type
- [ ] Validate extraction on sample data

### 4. Sentiment Analysis
- [ ] Integrate FinBERT or similar NLP model
- [ ] Map sentiment to five-point scale (very good → very bad)
- [ ] Test sentiment labeling on indicators

### 5. Graph Construction
- [ ] Build sector correlation graph (see src/graph/)
- [ ] Implement sentiment diffusion algorithm
- [ ] Validate graph propagation with test cases

### 6. Prediction Module
- [ ] Design quantile regression or similar model
- [ ] Integrate graph-informed features
- [ ] Backtest predictions on historical data

### 7. Evaluation & Reporting
- [ ] Define evaluation metrics (accuracy, coverage, etc.)
- [ ] Generate reports and visualizations
- [ ] Document findings and limitations

### 8. Documentation & Review
- [ ] Update architecture and data source docs
- [ ] Write usage and API documentation
- [ ] Peer review and code cleanup

## Timeline
- Weeks 1–2: Setup, ingestion, indicator extraction
- Weeks 3–4: Sentiment analysis, graph modeling
- Weeks 5–6: Prediction, evaluation, documentation

---

This plan is a living document. Adjust steps and priorities as the project evolves.