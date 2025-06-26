import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions

st.set_page_config(layout="wide")
st.title("🔧 Service Request Forecasting")
st.subheader("Forecast weekly volume for a selected product and region")

@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df["week"] = pd.to_datetime(df["week"])
    return df

df = load_data()

region = st.selectbox("Choose Region", df["region"].unique())
category = st.selectbox("Choose Product Category", df["category"].unique())

filtered_df = df[(df["region"] == region) & (df["category"] == category)].sort_values("week")

st.write(f"Data from {filtered_df['week'].min().date()} to {filtered_df['week'].max().date()}")
st.line_chart(filtered_df.set_index("week")["requests"])

series = [{
    "start": filtered_df["week"].iloc[0],
    "target": filtered_df["requests"].values.astype(float)
}]
train_ds = ListDataset(series, freq="W")

with st.spinner("Training model..."):
    estimator = DeepAREstimator(
        prediction_length=4,
        context_length=12,
        freq="W",
        trainer=Trainer(epochs=3)
    )
    predictor = estimator.train(training_data=train_ds)

forecast_it, ts_it = make_evaluation_predictions(
    dataset=train_ds, predictor=predictor, num_samples=100
)
forecast = next(forecast_it)
ts = next(ts_it)

fig, ax = plt.subplots(figsize=(10, 4))
ts.plot(ax=ax, label="Actual")
forecast.plot(ax=ax, color="orange")
plt.title(f"Forecast for {region} | {category}")
plt.legend()
plt.grid()
st.pyplot(fig)
