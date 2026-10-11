import streamlit as st, pandas as pd, numpy as np, plotly.express as px
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent/"src"))
from forecasting import clean_sales, make_features, compare, recursive_forecast

st.set_page_config(page_title="Sales Forecasting Studio",page_icon="📈",layout="wide")
st.title("📈 Time Series Sales Forecasting System")
st.caption("Trend, seasonality, lag/rolling features, chronological validation and future forecasting.")

uploaded=st.file_uploader("Upload historical sales CSV (or use demo data)",type=["csv"])
raw=pd.read_csv(uploaded) if uploaded else pd.read_csv("data/sales_demo.csv")
try:
    df=clean_sales(raw)
except Exception as e:
    st.error(str(e)); st.stop()

c1,c2,c3,c4=st.columns(4)
c1.metric("Periods",len(df))
c2.metric("Start",df.Date.min().strftime("%Y-%m"))
c3.metric("End",df.Date.max().strftime("%Y-%m"))
c4.metric("Average sales",f"₹{df.Sales.mean():,.0f}")

tab1,tab2,tab3,tab4=st.tabs(["EDA","Model Comparison","Forecast","Diagnostics"])
with tab1:
    st.subheader("Historical sales")
    st.plotly_chart(px.line(df,x="Date",y="Sales",markers=True,title="Sales trend"),use_container_width=True)
    monthly=df.assign(Month=df.Date.dt.month).groupby("Month",as_index=False).Sales.mean()
    st.plotly_chart(px.bar(monthly,x="Month",y="Sales",title="Average sales by month"),use_container_width=True)
    st.write("Unusual observations can be investigated using rolling statistics and residuals after model fitting.")
    st.dataframe(df.tail(25),use_container_width=True)

with tab2:
    results,preds=compare(df)
    st.dataframe(results.style.format({"MAE":"{:.2f}","RMSE":"{:.2f}","MAPE_%":"{:.2f}"}),use_container_width=True)
    st.plotly_chart(px.bar(results,x="Model",y="RMSE",title="RMSE comparison"),use_container_width=True)
    best=results.iloc[0].Model
    st.success(f"Best validation model by RMSE: **{best}**")

with tab3:
    results,preds=compare(df)
    best=results.iloc[0].Model
    model,te,p,features=preds[best]
    hist=df.copy()
    future_months=st.slider("Future months",1,12,6)
    fc=recursive_forecast(hist,model,features,future_months)
    hist_plot=pd.concat([hist.tail(36).assign(Type="Actual")[["Date","Sales"]].rename(columns={"Sales":"Value"}),
                         fc.assign(Type="Forecast").rename(columns={"Forecast":"Value"})[["Date","Value","Type"]]],ignore_index=True)
    st.plotly_chart(px.line(hist_plot,x="Date",y="Value",color="Type",markers=True,title="Actual history + future forecast"),
                    use_container_width=True)
    st.dataframe(fc,use_container_width=True)
    st.download_button("Download forecast",fc.to_csv(index=False),"sales_forecast.csv","text/csv")

with tab4:
    results,preds=compare(df)
    best=results.iloc[0].Model
    model,te,p,features=preds[best]
    d=te[["Date","Sales"]].copy()
    d["Predicted"]=p
    d["Residual"]=d.Sales-d.Predicted
    st.plotly_chart(px.line(d,x="Date",y=["Sales","Predicted"],title=f"Actual vs predicted — {best}"),
                    use_container_width=True)
    st.plotly_chart(px.bar(d,x="Date",y="Residual",title="Validation residuals"),use_container_width=True)
    st.info("Use residual spikes to investigate promotions, stock-outs, holidays, data errors or other external events.")
