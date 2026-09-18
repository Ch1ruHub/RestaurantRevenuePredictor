"""Reproducible quality checks, multiple linear regression, and diagnostics."""
from __future__ import annotations
import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from restaurant_pipeline import RestaurantRevenuePipeline

ROOT = Path(__file__).resolve().parent; DATA = ROOT / "restaurant.csv"; FIG = ROOT / "figures"; TARGET = "Monthly_Revenue"
NUM = ["Average_Daily_Customers", "Average_Bill_Value", "Delivery_Order_Percentage", "Seating_Capacity", "Employee_Count", "Promotion_Spend", "Customer_Rating"]
CAT = ["Cuisine_Type", "Location_Type"]
RANGES = {"Average_Daily_Customers": (0,None), "Average_Bill_Value": (0,None), "Delivery_Order_Percentage": (0,100), "Seating_Capacity": (0,None), "Employee_Count": (0,None), "Promotion_Spend": (0,None), "Customer_Rating": (1,5), TARGET: (0,None)}

def r2(y,p): return 1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)
def svg(path,x,y,title,xlab,ylab,zero=False):
    w,h,m=760,460,60; x,y=np.asarray(x,float),np.asarray(y,float); xmin,xmax=x.min(),x.max(); ymin,ymax=y.min(),y.max()
    sx=lambda z: m+(z-xmin)/(xmax-xmin or 1)*(w-2*m); sy=lambda z: h-m-(z-ymin)/(ymax-ymin or 1)*(h-2*m)
    points=' '.join(f'{sx(a):.1f},{sy(b):.1f}' for a,b in zip(x,y)); zeroline=f'<line x1="{m}" y1="{sy(0):.1f}" x2="{w-m}" y2="{sy(0):.1f}" stroke="#c33"/>' if zero and ymin<=0<=ymax else ''
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><rect width="100%" height="100%" fill="white"/><text x="60" y="30" font-size="20">{title}</text><line x1="{m}" y1="{h-m}" x2="{w-m}" y2="{h-m}" stroke="black"/><line x1="{m}" y1="{m}" x2="{m}" y2="{h-m}" stroke="black"/>{zeroline}<polyline points="{points}" fill="none" stroke="#2463a6" stroke-opacity=".45" stroke-width="3"/><text x="300" y="445">{xlab}</text><text x="12" y="220" transform="rotate(-90 12 220)">{ylab}</text></svg>',encoding="utf-8")

def main():
    FIG.mkdir(exist_ok=True); raw=pd.read_csv(DATA); quality=[]
    for c in raw.columns:
        lo,hi=RANGES.get(c,(None,None)); bad=pd.Series(False,index=raw.index)
        if lo is not None: bad|=raw[c]<=lo
        if hi is not None: bad|=raw[c]>hi
        quality.append({"column":c,"missing":int(raw[c].isna().sum()),"unique_values":int(raw[c].nunique()),"impossible_values":int(bad.sum())})
    quality.append({"column":"duplicate_rows","missing":0,"unique_values":int(raw.duplicated().sum()),"impossible_values":0}); pd.DataFrame(quality).to_csv(ROOT/"data_quality_report.csv",index=False)
    # Five ratings are above the valid 1-5 scale. Cap only those values at 5 and document it.
    # Missing values or impossible values in any other field remain a hard stop.
    other_issues=sum(x["missing"]+x["impossible_values"] for x in quality if x["column"] != "Customer_Rating")
    if other_issues: raise ValueError("Fix data-quality issues before fitting.")
    df=raw.drop_duplicates().copy(); corrected=int((df["Customer_Rating"] > 5).sum()); df["Customer_Rating"]=df["Customer_Rating"].clip(1,5)
    (ROOT/"data_corrections.json").write_text(json.dumps({"Customer_Rating":f"Capped {corrected} values above 5 at 5.0; no rows removed.","Outlet_ID":"Excluded from predictors because it is an identifier."},indent=2),encoding="utf-8")
    rng=np.random.default_rng(42); test_idx=rng.choice(df.index,size=round(.2*len(df)),replace=False); train,test=df.loc[~df.index.isin(test_idx)],df.loc[test_idx]
    model=RestaurantRevenuePipeline(NUM, CAT).fit(train[NUM+CAT],train[TARGET]); train_pred=model.predict(train[NUM+CAT]); test_pred=model.predict(test[NUM+CAT]); resid=train[TARGET].to_numpy(float)-train_pred
    design,_=model._design(train[NUM+CAT]); n,p=design.shape; leverage=np.einsum('ij,jk,ik->i',design,np.linalg.pinv(design.T@design),design); mse=np.sum(resid**2)/(n-p); cooks=(resid**2/(p*mse))*leverage/(1-leverage)**2; std=resid/np.sqrt(mse*(1-leverage))
    vifs=[]
    for i,name in enumerate(model.feature_names[1:],1):
        z=design[:,i]; other=np.delete(design,np.s_[0,i],axis=1); zp=other@np.linalg.lstsq(other,z,rcond=None)[0]; vifs.append({"feature":name,"VIF":float(1/(1-r2(z,zp)))})
    pd.DataFrame(vifs).to_csv(ROOT/"vif_report.csv",index=False); coeff=pd.DataFrame({"feature":model.feature_names,"coefficient_in_INR":model.coefficients}); coeff["absolute_coefficient"]=coeff.coefficient_in_INR.abs(); coeff.sort_values("absolute_coefficient",ascending=False).to_csv(ROOT/"model_coefficients.csv",index=False)
    svg(FIG/"residuals_vs_fitted.svg",train_pred,resid,"Residuals versus fitted values","Fitted monthly revenue (INR)","Residual (INR)",True); svg(FIG/"cooks_distance.svg",np.arange(n),cooks,"Cook's distance","Training observation","Cook's distance"); svg(FIG/"linearity_daily_customers.svg",df.Average_Daily_Customers,df[TARGET],"Linearity check: customers and revenue","Average daily customers","Monthly revenue (INR)")
    order=np.sort(resid); reference=np.linspace(-3,3,n); svg(FIG/"qq_plot_residuals.svg",reference,order,"Q-Q residual diagnostic","Normal-score reference","Ordered residual (INR)")
    train_r2=r2(train[TARGET].to_numpy(float),train_pred); test_r2=r2(test[TARGET].to_numpy(float),test_pred); dw=float(np.sum(np.diff(resid)**2)/np.sum(resid**2))
    metrics={"data_rows":int(len(df)),"train_rows":int(len(train)),"test_rows":int(len(test)),"split":"80% training / 20% testing, seed=42","target":TARGET,"excluded_identifier":["Outlet_ID"],"test_r_squared":round(float(test_r2),4),"adjusted_r_squared_training":round(float(1-(1-train_r2)*(n-1)/(n-p)),4),"test_mae_in_INR":round(float(np.mean(np.abs(test[TARGET]-test_pred))),2),"test_mse":round(float(np.mean((test[TARGET]-test_pred)**2)),2),"test_rmse_in_INR":round(float(np.sqrt(np.mean((test[TARGET]-test_pred)**2))),2),"durbin_watson":round(dw,4),"max_vif":round(float(max(v["VIF"] for v in vifs)),4),"standardized_residuals_over_3":int(np.sum(np.abs(std)>3)),"cooks_distance_over_4_over_n":int(np.sum(cooks>4/n)),"outlier_decision":"Valid generated observations were retained; flags are reported rather than silently deleted."}
    (ROOT/"model_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    # Keep the pickle importable whether this file was executed as a script or as a notebook module.
    with open(ROOT/"model.pkl","wb") as f: pickle.dump(model,f)
    print(json.dumps(metrics,indent=2))
if __name__ == "__main__": main()
