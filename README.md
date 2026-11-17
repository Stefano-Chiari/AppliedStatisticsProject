# NYC Airbnb – Applied Statistics Project

Team project for *Applied Statistics*, Politecnico di Milano (2026),
with Teresa Barrella, Diana Bonalumi and Lorenzo Pettenuzzo.

**Dataset:** New York City Airbnb Open Data (~49,000 listings: prices, locations, reviews).

**Questions:**
- **Clustering:** can we identify housing submarkets from geography and price alone?
  Hierarchical clustering on KMeans-downsampled data (3,000 centroids).
- **Regression:** what drives price, and can we predict it?
  Linear models with residual and heteroscedasticity diagnostics vs a cross-validated Random Forest.
- **Classification:** can we predict the most interesting listings (e.g. by number of reviews)?

**Results (regression):** the Random Forest outperforms the linear models on every metric;
predictions are accurate in the $0–200 range and degrade on luxury listings.

**Slides:** [presentation](docs/slides.pdf)

---
# Applied Statistics Project

Repository used for the data analysis project of the **Applied Statistics** exam.

The dataset used is the [New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data).

## Questions
1) Can we identify distinct "types" of listings based solely on their amenities and geographic features? (PCA, t-SNE, UMAP, Clustering)
2) What are the primary drivers of price, and can we predict it accurately? (Linear models, Ensembles, Neural Networks)
3) Can we predict which are the most interesting apartments (based for example on review number)? (Logistic regression, Neural Networks)
