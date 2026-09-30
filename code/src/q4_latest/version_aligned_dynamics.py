"""Quality-weighted auxiliary cohorts with nested lineage validation and explicit clocks."""
from pathlib import Path
import json,zipfile,io,datetime,hashlib,re
import numpy as np,pandas as pd
from scipy.special import expit
from scipy.optimize import least_squares
R=Path(__file__).resolve().parents[2];O=R/'/Users/lucasliao/Documents/Codex/2026-09-25/pao/outputs/第四问方案重跑_20260926_162328/版本时间统一模型';O.mkdir(exist_ok=True)
S=R/'outputs/第四问自定义开源口径_2026-09-26/O口径主样本.csv';Z=R/'outputs/第四问交接包_联合校准版_2026-09-26/data/第四问复跑附件子集.zip';J=R/'outputs/第四问继续建模_2026-09-26/q4_c2_official_sha_rowlink.json';V=R/'outputs/第四问版本日期补证_2026-09-26/audit.json'
T=['IFEval','BBH','MATH Lvl 5','GPQA','MUSR','MMLU-PRO']
with zipfile.ZipFile(Z) as z:raw=z.read('FΘóÿ/real_attachments/C_efficiency_evolution/leaderboard_enhanced.csv');lb=pd.read_csv(io.BytesIO(raw));lb['c2_row']=lb.index
rowlink=pd.DataFrame(json.loads(J.read_text())['rows']);d=lb.merge(rowlink[['C2_row_index','row_identity_verified','model_sha','official_upload_date']],left_on='c2_row',right_on='C2_row_index')
d['model']=d.Model;d['publisher']=d.model.str.split('/').str[0];d['track']=d.Type.map(lambda v:'pretrained' if 'pretrained' in str(v) else 'chat_finetuned' if ('chat models' in str(v) or 'fine-tuned' in str(v)) else 'excluded')
VA=R/'outputs/第四问版本日期补证_2026-09-26/auxiliary_audit.json'
version_records=json.loads(V.read_text())['records']+json.loads(VA.read_text())['records']
version_dates={};latest_dates={}
for record in version_records:
 if not record.get('sha_match') or not record.get('weight_timestamps_complete'):continue
 by_format={}
 for f in record.get('weight_file_dates',[]):
  date=(f.get('commit') or {}).get('date')
  if date:by_format.setdefault(Path(f['path']).suffix,[]).append(date)
 if by_format:
  version_dates[(record['model'],record['revision'])]=min(max(vals) for vals in by_format.values())
  latest_dates[(record['model'],record['revision'])]=max(max(vals) for vals in by_format.values())
d['version_date_evidence']=[version_dates.get((r.model,r.model_sha)) for r in d.itertuples()]
d['date']=pd.to_datetime(d.version_date_evidence,errors='coerce',utc=True).dt.tz_localize(None).dt.normalize();d['submission']=pd.to_datetime(d['Submission Date'],errors='coerce');d['logN']=np.log(pd.to_numeric(d['#Params (B)'],errors='coerce').where(lambda x:x>0));d['chat']=d.track.eq('chat_finetuned').astype(float);d['score']=d[T].mean(axis=1)
def lineage(n):
 n=n.lower()
 for tag in ['qwen','llama','pythia','gpt-neo','gpt-j','smollm','yi-','phi-','mistral','gemma','deepseek','olmo','falcon','bloom','opt-','stablelm','k2']:
  if tag in n:return tag
 return 'publisher:'+n.split('/')[0]
d['lineage']=d.model.map(lineage)
d['usable']=(d.Epoch_AI_Open_Weights.eq('Yes') & d.track.ne('excluded') & d.row_identity_verified & d.model_sha.str.len().eq(40) & d.date.notna() & d.submission.notna() & d.date.le(d.submission) & d.submission.le(pd.Timestamp('2025-03-13')) & np.isfinite(d.logN) & np.isfinite(d[T]).all(axis=1) & d[T].ge(0).all(axis=1) & d[T].le(100).all(axis=1) & d.model.ne('microsoft/phi-4'))
original=pd.read_csv(S);ids=set(original.c2_row);d['is_O']=d.c2_row.isin(ids);d['source_aux']=(~d.is_O).astype(float)
d=d[d.usable].sort_values(['model','model_sha','c2_row']).drop_duplicates(['model','model_sha']).copy();d['time']=(d.date-pd.Timestamp('2020-01-01')).dt.days/365.25
# O protocol offsets remain traceable; auxiliary protocol is explicitly unknown.
proto=original.set_index('c2_row').apply(lambda x:str(x.protocol_fingerprint)+'/'+str(x.harness_hash),axis=1).to_dict();d['protocol']=d.c2_row.map(proto).fillna('AUX_UNKNOWN')
core=d[d.is_O].reset_index(drop=True);aux=d[~d.is_O & ~d.model.isin(original.model)].reset_index(drop=True)
core.to_csv(O/'O主样本_版本制品时间.csv',index=False,encoding='utf-8-sig');aux.to_csv(O/'辅助样本_非O资格.csv',index=False,encoding='utf-8-sig')
# Independent fixed-revision artifact clocks (never silently called release dates).
vr={x['model']:x for x in json.loads(V.read_text())['records']};audit=[]
for row in original.itertuples():
 v=vr[row.model];formats={}
 for f in v.get('weight_file_dates',[]):
  ext=Path(f['path']).suffix;dates=formats.setdefault(ext,[]);date=(f.get('commit') or {}).get('date')
  if date:dates.append(date)
 readiness={ext:max(dates) for ext,dates in formats.items() if dates};first=min(readiness.values()) if readiness else None
 audit.append({'c2_row':row.c2_row,'model':row.model,'repository_created':v.get('repository_created'),'first_format_ready':first,'latest_format_update':v.get('latest_weight_file_commit'),'revision_commit':v.get('revision_commit_date'),'all_weight_dates_present':v.get('weight_timestamps_complete'),'same_content_across_formats_verified':False,'formats':json.dumps(readiness),'conversion_commit_seen':any('safetensors' in (f.get('commit') or {}).get('title','').lower() for f in v.get('weight_file_dates',[]))})
audit=pd.DataFrame(audit);audit.to_csv(O/'固定版本时间口径对照.csv',index=False,encoding='utf-8-sig')
GRID=[(a,b,w) for a in [1.,10.,100.] for b in [10.,100.,1000.] for w in [0.,.25,1.]]
contract={'created_before_fit':datetime.datetime.now(datetime.timezone.utc).isoformat(),'primary_clock':'earliest ready weight-format timestamp at fixed evaluation revision, both primary and auxiliary; artifact availability proxy, not certified first public release','main':'100*sigmoid(latent response), score-space weighted squared error with training-mean slope-scaled ridge penalties; auxiliary, type, time and protocol controls','auxiliary_not_O':True,'aux_total_weight':'w * number of core training rows; evenly across auxiliary lineage groups then within group','grid':GRID,'same_family_exclusion':'outer lineage folds exclude held lineage from BOTH core and auxiliaries; publisher folds purge held publisher and held lineages from auxiliaries; inner folds always exclude lineage','chronology':'for chronological holdout, auxiliaries must have both upload and submission dates earlier than cutoff; rolling inner splits obey same','cv_choice':'lineage-balanced MAE inside outer training; latest-cohort outer uses rolling-origin inner selection if >=2 folds available','no_score_clipping':True,'success_criteria':'compare with identical core-only nested baseline and constant on same heldout rows; report all designs, do not pick clock by score','clock_sensitivity':['official hub upload','first listed weight format ready','latest weight format update'],'claims':'exploratory repeated data; no causal/long-horizon coverage certification'}
(O/'contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2))
# Cached matrix construction for fast grid search.
def prepare(c,a,e,temporal=True):
 cols=['logN','chat']+(['time'] if temporal else [])
 mu=c[cols].mean().to_numpy();sd=c[cols].std(ddof=0).to_numpy().copy();sd[sd<1e-8]=1
 pc=sorted(c.protocol.unique());freq=c.protocol.value_counts(normalize=True).reindex(pc).to_numpy()
 def x(b):
  num=(b[cols].to_numpy()-mu)/sd
  mat=np.array([[float(v==k) for k in pc] for v in b.protocol],dtype=float).reshape(len(b),len(pc))-freq
  return np.c_[np.ones(len(b)),num,mat,b.source_aux.to_numpy()]
 xc=x(c);xa=x(a);xe=x(e);y=c.score.to_numpy();ya=a.score.to_numpy()
 if len(a):
  counts=a.lineage.value_counts();w0=a.lineage.map(lambda g:len(c)/(len(counts)*counts[g])).to_numpy()
 else:w0=np.zeros(0)
 return {'cxx':xc.T@xc,'cxy':xc.T@y,'axx':xa.T@(w0[:,None]*xa),'axy':xa.T@(w0*ya),'xe':xe,'cols':cols,'sd':sd,'mu':mu,'pc':pc,'freq':freq,'x':x,'xc':xc,'xa':xa,'yc':y,'ya':ya,'wa':w0}
def solve(p,pa,pt,w):
 penalty=np.array([0]+[pt if c=='time' else pa for c in p['cols']]+[100*pa]*len(p['pc'])+[100*pa])
 linear=np.linalg.solve(p['cxx']+w*p['axx']+np.diag(penalty),p['cxy']+w*p['axy'])
 mean=np.clip(p['yc'].mean()/100,1e-5,1-1e-5);scale=100*mean*(1-mean)
 weights=np.r_[np.ones(len(p['yc'])),w*p['wa']];x=np.vstack([p['xc'],p['xa']]);y=np.r_[p['yc'],p['ya']];sqrtw=np.sqrt(weights)
 reg=scale*np.sqrt(penalty)
 def residual(co):return np.r_[sqrtw*(100*expit(x@co)-y),reg*co]
 def jac(co):
  prob=expit(x@co);return np.vstack([(sqrtw*100*prob*(1-prob))[:,None]*x,np.diag(reg)])
 initial=linear/scale;initial[0]=np.log(mean/(1-mean))+(linear[0]-100*mean)/scale
 fit=least_squares(residual,initial,jac=jac,max_nfev=300,ftol=1e-8,xtol=1e-8,gtol=1e-8)
 if not fit.success:raise RuntimeError('bounded fit convergence failure')
 return fit.x
def candidates(kind):return GRID if kind=='aux_dynamic' else [(a,b,0.) for a in [1.,10.,100.] for b in [10.,100.,1000.]] if kind=='core_dynamic' else [(a,0.,0.) for a in [1.,10.,100.]]
def select(c,a,kind,rolling=False):
 temporal=kind!='core_scale';cache=[]
 if rolling:
  dates=np.sort(c.date.unique())
  for i in range(1,len(dates)):
   train=c[c.date<dates[i]];test=c[c.date==dates[i]]
   if len(train)<8:continue
   aa=a[(a.date<dates[i])&(a.submission<dates[i])]
   cache.append((prepare(train,aa,test,temporal),test.score.to_numpy()))
 if not cache:
  for g in sorted(c.lineage.unique()):
   train=c[c.lineage!=g];test=c[c.lineage==g];aa=a[a.lineage!=g]
   if len(train)<4:continue
   cache.append((prepare(train,aa,test,temporal),test.score.to_numpy()))
 out=[]
 for pa,pt,w in candidates(kind):
  err=[np.mean(abs(100*expit(p['xe']@solve(p,pa,pt,w))-y)) for p,y in cache]
  out.append((float(np.mean(err)),pa,pt,w))
 return min(out,key=lambda z:(z[0],z[3],-z[2])),out,len(cache)
folds=[]
for design,col in [('lineage','lineage'),('publisher','publisher')]:
 for g in sorted(core[col].unique()):
  train=core[core[col]!=g];test=core[core[col]==g]
  aa=aux[(aux[col]!=g)&~aux.lineage.isin(test.lineage.unique())]
  folds.append((design,g,train,aa,test,False))
cut=core.date.sort_values().iloc[int(.75*len(core))];folds.append(('latest_cohort',str(cut.date()),core[core.date<cut],aux[(aux.date<cut)&(aux.submission<cut)],core[core.date>=cut],True))
# Several historical cutoffs: hold all models from each of last 3 years/month regimes, no invented 12-month validation.
for ct in [pd.Timestamp('2024-06-01'),pd.Timestamp('2024-09-01')]:
 train=core[core.date<ct];test=core[core.date>=ct]
 if len(train)>=8 and len(test):folds.append(('chronology_stress',str(ct.date()),train,aux[(aux.date<ct)&(aux.submission<ct)],test,True))
rows=[];params=[]
for design,held,c,a,e,rolling in folds:
 for kind in ['aux_dynamic','core_dynamic','core_scale']:
  (err,pa,pt,w),_,nf=select(c,a,kind,rolling);prep=prepare(c,a,e,kind!='core_scale');coef=solve(prep,pa,pt,w);pr=100*expit(prep['xe']@coef)
  params.append({'design':design,'held':held,'model':kind,'pa':pa,'pt':pt,'aux_weight':w,'core_train_n':len(c),'aux_available_n':len(a),'aux_used_n':len(a) if w else 0,'inner_folds':nf,'inner_error':err})
  for (_,r),v in zip(e.iterrows(),pr):rows.append({'design':design,'held':held,'model':kind,'name':r.model,'track':r.track,'observed':r.score,'prediction':float(v),'constant':float(c.score.mean())})
 print(design,held,flush=True)
pred=pd.DataFrame(rows);pred.to_csv(O/'分组与时间留出预测.csv',index=False,encoding='utf-8-sig');pd.DataFrame(params).to_csv(O/'训练折选参与辅助权重.csv',index=False,encoding='utf-8-sig')
metrics=[]
for (design,kind),g in pred.groupby(['design','model']):
 err=abs(g.prediction-g.observed);metrics.append({'design':design,'model':kind,'n_predictions':len(g),'unique_models':g.name.nunique(),'MAE':err.mean(),'group_balanced_MAE':g.assign(err=err).groupby('held').err.mean().mean(),'constant_MAE':abs(g.constant-g.observed).mean(),'range_violations':int(((g.prediction<0)|(g.prediction>100)).sum())})
pd.DataFrame(metrics).to_csv(O/'模型对比.csv',index=False,encoding='utf-8-sig')
# Fixed reference population contrast; disclose composition/unexplained part.
best,search,_=select(core,aux,'aux_dynamic');_,pa,pt,w=best;prep=prepare(core,aux,core,True);coef=solve(prep,pa,pt,w)
rawcoef={c:float(coef[i+1]/prep['sd'][i]) for i,c in enumerate(prep['cols'])};mid=(core.time.min()+core.time.max())/2;lo=core[core.time<=mid];hi=core[core.time>mid];dn=hi.logN.mean()-lo.logN.mean();dt=hi.time.mean()-lo.time.mean();observed=hi.score.mean()-lo.score.mean()
def attr(co):
 def state(scale_values,t):
  copies=[]
  for val in scale_values:
   dd=core.copy();dd['logN']=val;dd['time']=t;copies.append(dd)
  xx=prep['x'](pd.concat(copies,ignore_index=True));return float(np.mean(100*expit(xx@co)))
 v00=state(lo.logN,lo.time.mean());v10=state(hi.logN,lo.time.mean());v01=state(lo.logN,hi.time.mean());v11=state(hi.logN,hi.time.mean())
 n=.5*((v10-v00)+(v11-v01));t=.5*((v01-v00)+(v11-v10));total=v11-v00
 return {'scale_points':float(n),'time_points':float(t),'model_gain':float(total),'time_share':float(t/total) if abs(total)>.001 else None,'observed_gap':float(observed),'composition_residual':float(observed-total)}
att=attr(coef);sens=[]
for err,aa,bb,ww in search:sens.append({'inner_error':err,'pa':aa,'pt':bb,'aux_weight':ww,**attr(solve(prep,aa,bb,ww))})
pd.DataFrame(sens).to_csv(O/'贡献规格敏感性.csv',index=False,encoding='utf-8-sig')
# Clock sensitivity core-only: auxiliary observations lack fixed-revision file dates.
clockmetrics=[]
for clock in ['first_format_ready','latest_format_update']:
 a=core.merge(audit[['model',clock]],on='model',validate='one_to_one');a['date']=pd.to_datetime(a[clock],utc=True).dt.tz_localize(None).dt.normalize();a=a[a.date.notna()&a.date.le(a.submission)];a['time']=(a.date-pd.Timestamp('2020-01-01')).dt.days/365.25
 ccut=a.date.sort_values().iloc[int(.75*len(a))]
 for design in ['lineage','latest_cohort']:
  ff=[(a[a.lineage!=g],a[a.lineage==g],False) for g in sorted(a.lineage.unique())] if design=='lineage' else [(a[a.date<ccut],a[a.date>=ccut],True)]
  pp=[];yy=[];cc=[]
  for c,e,rolling in ff:
   (er,aa,bb,ww),_,_=select(c,aux.iloc[:0],'core_dynamic',rolling);pr=prepare(c,aux.iloc[:0],e);co=solve(pr,aa,bb,0);pp+=list(100*expit(pr['xe']@co));yy+=list(e.score);cc +=[c.score.mean()]*len(e)
  clockmetrics.append({'clock':clock,'design':design,'n':len(yy),'MAE':float(np.mean(abs(np.array(pp)-yy))),'constant_MAE':float(np.mean(abs(np.array(cc)-yy))),'t0':str(a.date.max().date()),'plus12':str((a.date.max()+pd.DateOffset(months=12)).date()),'plus24':str((a.date.max()+pd.DateOffset(months=24)).date()),'range_violations':int(((np.array(pp)<0)|(np.array(pp)>100)).sum())})
pd.DataFrame(clockmetrics).to_csv(O/'版本时钟敏感性.csv',index=False,encoding='utf-8-sig')
summary={'status':'version_aligned_bounded_auxiliary_dynamics_executed','core_O_n':len(core),'auxiliary_W_not_O_n':len(aux),'core_tracks':core.track.value_counts().to_dict(),'aux_tracks':aux.track.value_counts().to_dict(),'core_publishers':core.publisher.nunique(),'core_lineage_proxies':core.lineage.nunique(),'aux_lineage_proxies':aux.lineage.nunique(),'primary_clock':'fixed-revision earliest listed weight-format timestamp; both core and auxiliaries','primary_window':[str(core.date.min().date()),str(core.date.max().date())],'primary_event_targets':[str((core.date.max()+pd.DateOffset(months=h)).date()) for h in [12,24]],'common_snapshot_forecast_anchor':'2025-03-13','snapshot_targets':['2026-03-13','2027-03-13'],'last_cohort_cutoff':str(cut.date()),'metrics':metrics,'clock_sensitivity':clockmetrics,'full_fit_penalties':{'scale':pa,'time':pt,'aux_weight':w},'coefficients_on_latent_logit_scale':rawcoef,'attribution':att,'time_share_grid_range':[min(v['time_share'] for v in sens if v['time_share'] is not None),max(v['time_share'] for v in sens if v['time_share'] is not None)],'final_choice_frozen':False,'new_long_horizon_forecast':False,'industry_causal_share':None,'inputs':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [S,J,V,VA]},'C2_sha256':hashlib.sha256(raw).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(O/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=lambda v:int(v) if isinstance(v,np.integer) else v));print(json.dumps(summary,ensure_ascii=False,indent=2,default=lambda v:int(v) if isinstance(v,np.integer) else v))
