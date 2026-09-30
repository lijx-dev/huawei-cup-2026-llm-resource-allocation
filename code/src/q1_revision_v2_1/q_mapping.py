"""六域可靠映射覆盖率、公平 Ridge 对照和固定置换 placebo。"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from q1_revision_v2.mixture import macro_nrmse
from .common import paths, save_json
from .mixture_validation import load_data, SPLITS


METHODS = ("Q_hierarchical_balanced", "Q_entropy_redundancy")


def mapping_columns(root, xfields):
    table = pd.read_csv(root / "data/real_attachments/A_data_value/domain_mapping_guide.csv", keep_default_na=False)
    selected = table[table.mapping_type.isin(("direct","near_direct"))]
    if len(selected) != 6 or set(selected.mapping_type) != {"direct","near_direct"}:
        raise RuntimeError("A16 可靠映射不再是 3 direct + 3 near_direct")
    cols = [xfields.index("train_the_pile_"+domain) for domain in selected.mixture_domain]
    return selected.reset_index(drop=True), cols


def mapped_q(x, cols, domain_values):
    coverage = x[:,cols].sum(axis=1)
    numerator = x[:,cols] @ np.asarray(domain_values,dtype=float)
    q = np.divide(numerator,coverage,out=np.full(len(x),np.nan),where=coverage>0)
    return q,coverage


def domain_vectors(out, mapping):
    new=pd.read_csv(out / "quality/domain_quality_summary.csv")
    new=new[new.view=="union_unique"].set_index("domain")
    vals={name:new.loc[mapping.quality_domain,name].to_numpy(dtype=float) for name in METHODS}
    if any(not np.isfinite(v).all() for v in vals.values()):
        raise RuntimeError("可靠映射的领域质量分不完整")
    return vals


def cv_ridge(x,y,folds,alphas):
    """只用训练样本与给定 folds，按每目标折均 MSE 选固定候选 λ。"""
    prediction=np.full((len(alphas),len(y),y.shape[1]),np.nan)
    fold_mse=np.empty((len(alphas),len(folds),y.shape[1]))
    for f,(train,valid) in enumerate(folds):
        scaler=StandardScaler().fit(x[train])
        xt,xv=scaler.transform(x[train]),scaler.transform(x[valid])
        for a,alpha in enumerate(alphas):
            pred=Ridge(alpha=alpha).fit(xt,y[train]).predict(xv)
            prediction[a,valid]=pred
            fold_mse[a,f]=np.mean((y[valid]-pred)**2,axis=0)
    choice=fold_mse.mean(axis=1).argmin(axis=0)
    selected=np.column_stack([prediction[choice[t],:,t] for t in range(y.shape[1])])
    if not np.isfinite(selected).all():
        raise RuntimeError("CV 未覆盖全部训练行")
    pooled=1-np.sum((y-selected)**2)/np.sum((y-y.mean(axis=0))**2)
    train_std=y.std(axis=0,ddof=1)
    return {"pooled_r2":float(pooled),"macro_nrmse":macro_nrmse(y,selected,train_std),
            "target_rmse":np.sqrt(np.mean((y-selected)**2,axis=0)),
            "selected_alphas":[alphas[i] for i in choice],"oof_predictions":selected}


def full_ridge_predict(x,y,valid_x,alphas):
    scaler=StandardScaler().fit(x)
    xt,xv=scaler.transform(x),scaler.transform(valid_x)
    return np.column_stack([Ridge(alpha=alphas[t]).fit(xt,y[:,t]).predict(xv)
                            for t in range(y.shape[1])])


def coverage_table(data,cols,vectors,dest):
    rows=[]
    summary=[]
    for split in SPLITS:
        ids,x,_=data[split]
        _,coverage=mapped_q(x,cols,np.ones(len(cols)))
        if np.min(coverage)<-1e-12 or np.max(coverage)>1+1e-10:
            raise RuntimeError("Qmapped coverage 越出 [0,1]")
        derived={name:mapped_q(x,cols,values)[0] for name,values in vectors.items()}
        rows.extend({"split":split,"index":rid,"coverage":float(mass),
                     "Q_hierarchical_balanced_mapped":float(derived[METHODS[0]][i]) if mass>0 else np.nan,
                     "Q_entropy_redundancy_mapped":float(derived[METHODS[1]][i]) if mass>0 else np.nan,
                     "mapping_scope":"six_direct_or_near_direct_only"}
                    for i,(rid,mass) in enumerate(zip(ids,coverage)))
        summary.append({"split":split,"n":len(ids),"n_with_q":int(np.count_nonzero(coverage>0)),
                        "mean":np.mean(coverage),"median":np.median(coverage),
                        "q10":np.quantile(coverage,.1),"q25":np.quantile(coverage,.25),
                        "q75":np.quantile(coverage,.75),"q90":np.quantile(coverage,.9),
                        "minimum":np.min(coverage),"maximum":np.max(coverage)})
    pd.DataFrame(rows).to_csv(dest / "qmapped_coverage.csv",index=False)
    pd.DataFrame(summary).to_csv(dest / "qmapped_coverage_summary.csv",index=False)
    return pd.DataFrame(rows)


def coverage_sensitivity(data,cols,vectors,config,dest):
    train_ids,train_x,train_y=data["train_1m"]
    test_ids,test_x,test_y=data["test_1m"]
    train_mass=train_x[:,cols].sum(axis=1)
    test_mass=test_x[:,cols].sum(axis=1)
    alphas=config["q_ridge_alphas"]
    rows=[]
    for threshold in (0.,*config["coverage_thresholds"]):
        train_keep=(train_mass>0)&(train_mass>=threshold)
        test_keep=(test_mass>0)&(test_mass>=threshold)
        ntrain,ntest=int(train_keep.sum()),int(test_keep.sum())
        for method,values in vectors.items():
            base={"method":method,"coverage_threshold":threshold,
                  "n_train":ntrain,"n_test":ntest,"fold_seed":config["seed"]}
            if ntrain<config["coverage_min_train"] or ntest<config["coverage_min_test"]:
                rows.append({**base,"status":"insufficient_samples"})
                continue
            tx,ty=train_x[train_keep],train_y[train_keep]
            vx,vy=test_x[test_keep],test_y[test_keep]
            qtrain,_=mapped_q(tx,cols,values)
            qtest,_=mapped_q(vx,cols,values)
            folds=list(KFold(n_splits=config["q_cv_folds"],shuffle=True,random_state=config["seed"]).split(tx))
            p_base,p_aug=tx[:,:-1],np.column_stack([tx[:,:-1],qtrain])
            cv_a=cv_ridge(p_base,ty,folds,alphas)
            cv_b=cv_ridge(p_aug,ty,folds,alphas)
            pred_a=full_ridge_predict(p_base,ty,vx[:,:-1],cv_a["selected_alphas"])
            pred_b=full_ridge_predict(p_aug,ty,np.column_stack([vx[:,:-1],qtest]),cv_b["selected_alphas"])
            denominator=np.sum((vy-vy.mean(axis=0))**2)
            r2_a=1-np.sum((vy-pred_a)**2)/denominator
            r2_b=1-np.sum((vy-pred_b)**2)/denominator
            std=ty.std(axis=0,ddof=1)
            rmse_a=np.sqrt(np.mean((vy-pred_a)**2,axis=0))
            rmse_b=np.sqrt(np.mean((vy-pred_b)**2,axis=0))
            rows.append({**base,"status":"computed","cv_macro_nrmse_p":cv_a["macro_nrmse"],
                         "cv_macro_nrmse_p_q":cv_b["macro_nrmse"],
                         "cv_pooled_r2_p":cv_a["pooled_r2"],"cv_pooled_r2_p_q":cv_b["pooled_r2"],
                         "cv_target_wins":int((cv_b["target_rmse"]<cv_a["target_rmse"]).sum()),
                         "test_pooled_r2_p":r2_a,"test_pooled_r2_p_q":r2_b,
                         "test_macro_nrmse_p":np.mean(rmse_a/std),
                         "test_macro_nrmse_p_q":np.mean(rmse_b/std),
                         "test_target_wins":int((rmse_b<rmse_a).sum())})
    pd.DataFrame(rows).to_csv(dest / "qmapped_coverage_sensitivity.csv",index=False)
    return rows


def nonidentity_permutations(seed,count,width=6):
    rng=np.random.default_rng(seed)
    identity=np.arange(width)
    values=[]
    while len(values)<count:
        p=rng.permutation(width)
        if not np.array_equal(p,identity):
            values.append(p)
    return np.asarray(values,dtype=int)


def placebo(data,cols,vectors,config,dest):
    """只接受 train_1m；从而没有读取测试集的模型选择通道。"""
    _,x,y=data["train_1m"]
    mass=x[:,cols].sum(axis=1)
    keep=mass>0
    x,y,mass=x[keep],y[keep],mass[keep]
    base=x[:,:-1]
    alphas=config["q_ridge_alphas"]
    folds=list(KFold(n_splits=config["q_cv_folds"],shuffle=True,random_state=config["seed"]).split(x))
    baseline=cv_ridge(base,y,folds,alphas)
    permutations=nonidentity_permutations(config["seed"],config["placebo_repetitions"],len(cols))
    pd.DataFrame(permutations,columns=[f"source_position_{i}" for i in range(len(cols))]).assign(draw=np.arange(len(permutations))).to_csv(dest / "qmapped_placebo_permutations.csv",index=False)
    records=[]
    summaries=[]
    for method,values in vectors.items():
        true_q=x[:,cols]@values/mass
        true=cv_ridge(np.column_stack([base,true_q]),y,folds,alphas)
        true_r2=true["pooled_r2"]-baseline["pooled_r2"]
        true_nrmse=baseline["macro_nrmse"]-true["macro_nrmse"]
        for draw,p in enumerate(permutations):
            q=x[:,cols]@values[p]/mass
            fit=cv_ridge(np.column_stack([base,q]),y,folds,alphas)
            records.append({"quality_method":method,"draw":draw,"seed":config["seed"],
                            "permutation":"-".join(map(str,p)),
                            "delta_pooled_r2_cv":fit["pooled_r2"]-baseline["pooled_r2"],
                            "delta_macro_nrmse_cv":baseline["macro_nrmse"]-fit["macro_nrmse"]})
        values_r2=np.array([r["delta_pooled_r2_cv"] for r in records if r["quality_method"]==method])
        values_nrmse=np.array([r["delta_macro_nrmse_cv"] for r in records if r["quality_method"]==method])
        summaries.append({"quality_method":method,"n_train":len(x),"repetitions":len(permutations),
                          "unique_nonidentity_permutations":len({tuple(p) for p in permutations}),
                          "base_cv_pooled_r2":baseline["pooled_r2"],
                          "real_cv_pooled_r2":true["pooled_r2"],
                          "real_delta_pooled_r2_cv":true_r2,
                          "real_delta_macro_nrmse_cv":true_nrmse,
                          "placebo_delta_r2_mean":float(values_r2.mean()),
                          "placebo_delta_r2_q025":float(np.quantile(values_r2,.025)),
                          "placebo_delta_r2_q975":float(np.quantile(values_r2,.975)),
                          "placebo_percentile":float(np.mean(values_r2<=true_r2)),
                          "one_sided_empirical_p":float((1+np.count_nonzero(values_r2>=true_r2))/(len(values_r2)+1)),
                          "placebo_delta_nrmse_mean":float(values_nrmse.mean())})
        print(f"Placebo {method}: {len(permutations)} 次完成",flush=True)
    pd.DataFrame(records).to_csv(dest / "qmapped_placebo.csv",index=False)
    pd.DataFrame(summaries).to_csv(dest / "qmapped_placebo_summary.csv",index=False)
    return summaries


def run(root,config,old_config):
    root,out,old=paths(root)
    dest=out / "q_mapping"
    dest.mkdir(parents=True,exist_ok=True)
    data,xfields,yfields=load_data(root,old,old_config)
    mapping,cols=mapping_columns(root,xfields)
    mapping.to_csv(dest / "reliable_mapping_used.csv",index=False)
    vectors=domain_vectors(out,mapping)
    pd.DataFrame({"mixture_domain":mapping.mixture_domain,"quality_domain":mapping.quality_domain,
                  **{method:values for method,values in vectors.items()}}).to_csv(dest / "mapped_domain_quality.csv",index=False)
    coverage_table(data,cols,vectors,dest)
    sensitivity=coverage_sensitivity(data,cols,vectors,config,dest)
    result=placebo(data,cols,vectors,config,dest)
    return {"mapping_fields":mapping.mixture_domain.tolist(),"mapped_columns":cols,
            "coverage_sensitivity_rows":len(sensitivity),"placebo_summaries":result}
