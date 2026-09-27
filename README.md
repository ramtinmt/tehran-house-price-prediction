# House Price Prediction (Tehran)

A linear regression project predicting apartment prices in Tehran from basic listing information.

## Data

`data.csv` contains 3,479 apartment listings with these columns:

| Column | Description |
|---|---|
| Area | Size in m² |
| Room | Number of bedrooms |
| Parking, Warehouse, Elevator | Yes/no features |
| Address | Neighbourhood (192 unique) |
| Price | Price in Toman (target) |
| Price(USD) | Price converted to USD. Not used, since it's the target in another currency |

## Cleaning

- Converted `Area` to numbers and removed rows where a price had been typed into the Area column
- Dropped listings with no neighbourhood
- Removed clear data-entry errors (e.g. a 160 m² apartment listed for 3.6 million Toman)
- Removed listings whose price per m² was far off from the rest of their neighbourhood
- **Kept only homes under 300 m².** Most listings are small apartments, and the few very large properties behaved like a different market and hurt the model
- Grouped neighbourhoods with fewer than 5 listings into `Other`, then one-hot encoded neighbourhoods

About 3,370 listings remain after cleaning.

## Models

Both models are scikit-learn `LinearRegression`, trained on 80% of the data and tested on the other 20%. Model choices were made with 5-fold cross-validation on the training set only; the test set was used once at the end.

1. **Linear model**: predicts price directly
2. **Log model**: predicts log(price), then converts back to price

## Results (test set)

| Model | R² |
|---|---|
| Linear | 0.795 |
| Log | 0.787 |

**I chose the log model.** The scores are about the same, but the linear model predicts negative prices for some cheap houses, while the log model can't. The log model also has lower typical errors.

### Example prediction

150 m², 3 rooms, parking, warehouse, elevator, in Elahieh:

- Predicted price: **13.3 billion Toman**
- 80% range: **10.0 – 18.3 billion Toman**

The range comes from how far off the model was on the test houses. It's fairly wide because the data doesn't include things that strongly affect price (see below).

## Limitations

- Only valid for homes **under 300 m²**
- The data has no building age, floor, condition or view, all of which move prices a lot
- Some duplicate listings remain in the data
- Outliers were removed before the train/test split, so the test score is probably slightly optimistic

## Next steps

- Add more features (building age, floor)
- Remove duplicate listings
- Try tree-based models such as random forest

## Running it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Then open `model.ipynb` and run all cells.
