import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

def clean_sales(df):
    df=df.copy()
    df.columns=[str(c).strip().replace(" ","_") for c in df.columns]
    date_col=next((c for c in ["Date","date","Order_Date","OrderDate"] if c in df.columns),df.columns[0])
    sales_col=next((c for c in ["Sales","sales","Revenue","Amount"] if c in df.columns),None)
    if sales_col is None: raise ValueError("Could not identify a Sales/Revenue column.")
    df[date_col]=pd.to_datetime(df[date_col],errors="coerce")
    df[sales_col]=pd.to_numeric(df[sales_col],errors="coerce")
    df=df.dropna(subset=[date_col,sales_col]).groupby(date_col,as_index=False)[sales_col].sum()
    return df.rename(columns={date_col:"Date",sales_col:"Sales"}).sort_values("Date")

def make_features(df,lags=(1,2,3,6,12),windows=(3,6,12)):
    x=df.copy()
    x["year"]=x.Date.dt.year
    x["month"]=x.Date.dt.month
    x["quarter"]=x.Date.dt.quarter
    x["month_sin"]=np.sin(2*np.pi*x.month/12)
    x["month_cos"]=np.cos(2*np.pi*x.month/12)
    for l in lags: x[f"lag_{l}"]=x.Sales.shift(l)
    for w in windows:
        x[f"roll_mean_{w}"]=x.Sales.shift(1).rolling(w).mean()
        x[f"roll_std_{w}"]=x.Sales.shift(1).rolling(w).std()
    return x.dropna()

def metrics(y,p):
    mae=mean_absolute_error(y,p)
    rmse=np.sqrt(mean_squared_error(y,p))
    mape=np.mean(np.abs((y-p)/np.where(y==0,1,y)))*100
    return {"MAE":mae,"RMSE":rmse,"MAPE_%":mape}

def compare(df,test_size=0.2):
    f=make_features(df)
    n=max(3,int(len(f)*test_size))
    tr,te=f.iloc[:-n],f.iloc[-n:]
    features=[c for c in f.columns if c not in ["Date","Sales"]]
    models={
        "Ridge":Ridge(alpha=10),
        "Random Forest":RandomForestRegressor(n_estimators=400,min_samples_leaf=2,random_state=42,n_jobs=-1),
        "Gradient Boosting":HistGradientBoostingRegressor(max_iter=300,learning_rate=.06,l2_regularization=1.0,random_state=42)
    }
    rows=[]; preds={}
    for name,m in models.items():
        m.fit(tr[features],tr.Sales)
        p=m.predict(te[features])
        mm=metrics(te.Sales,p)
        rows.append({"Model":name,**mm})
        preds[name]=(m,te,p,features)
    return pd.DataFrame(rows).sort_values("RMSE"),preds

def recursive_forecast(df,model,features,horizon):
    hist=df.copy().sort_values("Date")
    out=[]
    for _ in range(horizon):
        next_date=hist.Date.max()+pd.offsets.MonthBegin(1)
        temp=pd.concat([hist,pd.DataFrame({"Date":[next_date],"Sales":[np.nan]})],ignore_index=True)
        fx=make_features(temp)
        row=fx.iloc[[-1]][features]
        pred=float(model.predict(row)[0])
        hist.loc[len(hist)]=[next_date,pred]
        out.append([next_date,pred])
    return pd.DataFrame(out,columns=["Date","Forecast"])
