import json, pathlib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,brier_score_loss,confusion_matrix
ROOT=pathlib.Path(__file__).resolve().parent
DATA=ROOT/'Reverb_30_Day_Sales_Master_Dataset.csv'
if not DATA.exists(): DATA=pathlib.Path('/mnt/data/files/Reverb_30_Day_Sales_Master_Dataset.csv')
df=pd.read_csv(DATA); df['published']=pd.to_datetime(df['listing_published_at_utc'],utc=True,errors='coerce')
num=['exported_listing_price','exported_shipping_price']; cat=['condition','product_type','offers_enabled','local_pickup','new_listing']
cols=num+cat
valid=df[(df.label_status=='determined')&df.sold_within_30_days.notna()&df.published.notna()].copy().sort_values(['published','listing_id'])
assert valid.listing_id.is_unique
n=len(valid); i=int(n*.7); j=int(n*.85)
sets={'train':valid.iloc[:i], 'validation':valid.iloc[i:j], 'test':valid.iloc[j:]}
pre=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat)])
models={'majority':DummyClassifier(strategy='most_frequent'),'logistic':LogisticRegression(max_iter=2000,random_state=42,class_weight=None),'random_forest':RandomForestClassifier(n_estimators=300,min_samples_leaf=10,random_state=42,n_jobs=-1)}
rows=[]; fitted={}
for name,model in models.items():
 pipe=Pipeline([('pre',pre),('model',model)]);pipe.fit(sets['train'][cols],sets['train'].sold_within_30_days.astype(int));fitted[name]=pipe
 for split in ['validation','test']:
  dd=sets[split];y=dd.sold_within_30_days.astype(int);prob=pipe.predict_proba(dd[cols])[:,1];pred=(prob>=.5).astype(int)
  rows.append(dict(model=name,split=split,n=len(y),prevalence=round(y.mean(),4),accuracy=round(accuracy_score(y,pred),4),balanced_accuracy=round(balanced_accuracy_score(y,pred),4),precision=round(precision_score(y,pred,zero_division=0),4),recall=round(recall_score(y,pred,zero_division=0),4),f1=round(f1_score(y,pred,zero_division=0),4),roc_auc=round(roc_auc_score(y,prob),4),average_precision=round(average_precision_score(y,prob),4),brier=round(brier_score_loss(y,prob),4),tn=int(confusion_matrix(y,pred,labels=[0,1])[0,0]),fp=int(confusion_matrix(y,pred,labels=[0,1])[0,1]),fn=int(confusion_matrix(y,pred,labels=[0,1])[1,0]),tp=int(confusion_matrix(y,pred,labels=[0,1])[1,1])))
results=pd.DataFrame(rows);results.to_csv(ROOT/'model_metrics.csv',index=False)
audit={'rows':len(df),'columns':len(df.columns)-1,'unique_ids':int(df.listing_id.nunique()),'duplicates':int(df.listing_id.duplicated().sum()),'determined':n,'positives':int(valid.sold_within_30_days.sum()),'negatives':int((valid.sold_within_30_days==0).sum()),'unresolved':int(len(df)-n),'price_at_publication_verified':int(df.price_at_publication_verified.fillna(False).sum()),'missing_features':{c:int(valid[c].isna().sum()) for c in cols},'splits':{k:{'n':len(v),'positive':int(v.sold_within_30_days.sum()),'start':str(v.published.min().date()),'end':str(v.published.max().date())} for k,v in sets.items()},'category_counts':valid.product_type.value_counts().head(8).to_dict()}
(ROOT/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2));print(results.to_string(index=False))
