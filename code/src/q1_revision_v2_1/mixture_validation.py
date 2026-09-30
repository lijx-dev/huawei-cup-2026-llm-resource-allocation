"""冻结 v2 模型后的配方距离、距离分层与配方簇留出验证。"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from q1.mixture.dataset import load_pair
from q1_revision_v2.mixture import model_lgb, ridge_fit, ridge_predict, predict
from .common import paths, save_json, sha256


SPLITS = ("train_1m", "test_1m", "test_60m", "test_1b", "est_10b", "est_70b")


def load_data(root, old, old_config):
    audit = json.loads((old / "audit/audit_summary.json").read_text())
    xfields, yfields = audit["tables"]["mixture_fields"], audit["tables"]["loss_fields"]
    data = {name: load_pair(root, name, xfields, yfields, old_config["mixture_sum_tolerance"])[:3]
            for name in SPLITS}
    return data, xfields, yfields


def verify_frozen_model(root, old, config, old_config, data, xfields, yfields, audited_sha):
    path = old / "mixture/fitted_models.joblib"
    digest = sha256(path)
    if digest != audited_sha:
        raise RuntimeError("v2 模型文件在本次审计后发生变化")
    old_meta = json.loads((old / "metadata.json").read_text())
    anchored = old_meta.get("model_file_sha256")
    if anchored is not None and anchored != digest:
        raise RuntimeError("v2 模型文件 SHA 与 v2 metadata 不一致")
    artifact = joblib.load(path)
    selected = json.loads((old / "mixture/model_selection.json").read_text())
    if (artifact["selection"] != selected or artifact["seed"] != config["seed"] or
        artifact["mix_fields"] != xfields or artifact["loss_fields"] != yfields):
        raise RuntimeError("冻结模型配置、字段顺序或随机种子不一致")
    if selected["main_model"] != "LightGBM":
        raise RuntimeError("v2 主模型并非 LightGBM，停止 v2.1 的指定验证")
    expected_settings = {"random_state": config["seed"], "bagging_seed": config["seed"],
                         "feature_fraction_seed": config["seed"], "data_random_seed": config["seed"],
                         "deterministic": True, "force_col_wise": True}
    for model in artifact["models"]["LightGBM"]:
        params = model.get_params()
        if any(params.get(k) != v for k,v in expected_settings.items()):
            raise RuntimeError("冻结 LightGBM 随机或确定性参数不一致")
    ids, x, y = data["test_1m"]
    old_predictions = pd.read_csv(old / "mixture/predictions_test_1m.csv", dtype={"index":str})
    if list(old_predictions["index"]) != list(ids):
        raise RuntimeError("v2 1M 预测行的 index 顺序不一致")
    differences = {}
    for kind in ("Ridge","LightGBM"):
        actual = predict(artifact["models"][kind], x, kind)
        expected = old_predictions[[f"predicted_{kind}/{field}" for field in yfields]].to_numpy()
        max_error = float(np.max(np.abs(actual-expected)))
        differences[kind] = max_error
        if not np.allclose(actual, expected, rtol=0, atol=1e-10):
            raise RuntimeError(f"冻结 {kind} 模型重载后未复现 v2 1M 预测")
    return artifact, {"model_file_sha256": digest,
                      "model_hash_anchor_in_v2_metadata": anchored is not None,
                      "verification_status": "anchored_and_prediction_verified" if anchored else "prediction_verified_hash_unanchored",
                      "prediction_max_abs_diff": differences,
                      "frozen_params": selected["lightgbm_params"],
                      "random_settings": expected_settings,
                      "source": "inherited_from_q1-revision-v2"}


def nearest_rows(data):
    ids_train, train, _ = data["train_1m"]
    rows = []
    matrix = cdist(train, train, metric="euclidean")
    np.fill_diagonal(matrix, np.inf)
    nearest_index = matrix.argmin(axis=1)
    distance = matrix[np.arange(len(train)), nearest_index]
    rows.extend({"split":"train_1m", "index": rid, "nearest_train_index":ids_train[j],
                 "nearest_distance":float(d), "distance_rule":"euclidean_17d_leave_self_out"}
                for rid,j,d in zip(ids_train,nearest_index,distance))
    for split in SPLITS[1:]:
        ids, x, _ = data[split]
        matrix = cdist(x,train,metric="euclidean")
        nearest_index = matrix.argmin(axis=1)
        distance = matrix[np.arange(len(x)),nearest_index]
        rows.extend({"split":split,"index":rid,"nearest_train_index":ids_train[j],
                     "nearest_distance":float(d),"distance_rule":"euclidean_17d_to_train"}
                    for rid,j,d in zip(ids,nearest_index,distance))
    return pd.DataFrame(rows)


def safe_corr(func, x, y):
    if len(x)<3 or np.std(x)<1e-12 or np.std(y)<1e-12:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(func(x,y).statistic)


def target_metrics(y, pred, yfields):
    rows=[]
    for t,field in enumerate(yfields):
        rows.append({"target":field,"n":len(y),
                     "rmse":float(np.sqrt(mean_squared_error(y[:,t],pred[:,t]))),
                     "mae":float(mean_absolute_error(y[:,t],pred[:,t])),
                     "r2":float(r2_score(y[:,t],pred[:,t])),
                     "pearson":safe_corr(pearsonr,y[:,t],pred[:,t]),
                     "spearman":safe_corr(spearmanr,y[:,t],pred[:,t])})
    return rows


def distance_strata(root, old, dest, data, nearest, artifact, yfields):
    ids,x,y = data["test_1m"]
    subset = nearest[nearest.split=="test_1m"].copy()
    if list(subset["index"].astype(str)) != list(ids):
        raise RuntimeError("距离分层与 test_1m index 顺序不一致")
    subset["distance_quartile"] = pd.qcut(subset.nearest_distance, 4, labels=["Q1_nearest","Q2","Q3","Q4_farthest"], duplicates="raise")
    subset.to_csv(dest / "test_1m_distance_quartiles.csv", index=False)
    train_std = data["train_1m"][2].std(axis=0,ddof=1)
    detail,summary = [],[]
    for model_name,models in artifact["models"].items():
        pred = predict(models,x,model_name)
        for label in ["Q1_nearest","Q2","Q3","Q4_farthest"]:
            mask = (subset.distance_quartile==label).to_numpy()
            yy,pp = y[mask],pred[mask]
            rows = target_metrics(yy,pp,yfields)
            for row in rows:
                row.update({"model":model_name,"distance_quartile":label,
                            "distance_mean":float(subset.loc[mask,"nearest_distance"].mean())})
                detail.append(row)
            pooled = 1-((yy-pp)**2).sum()/((yy-yy.mean(axis=0))**2).sum()
            summary.append({"model":model_name,"distance_quartile":label,"n":int(mask.sum()),
                            "mean_distance":float(subset.loc[mask,"nearest_distance"].mean()),
                            "pooled_r2":float(pooled),
                            "macro_nrmse_train_scale":float(np.mean(np.sqrt(np.mean((yy-pp)**2,axis=0))/train_std)),
                            "mean_target_pearson":float(np.nanmean([r["pearson"] for r in rows])),
                            "mean_target_spearman":float(np.nanmean([r["spearman"] for r in rows]))})
    pd.DataFrame(detail).to_csv(dest / "distance_stratified_metrics.csv", index=False)
    pd.DataFrame(summary).to_csv(dest / "distance_stratified_summary.csv", index=False)
    return summary


def recipe_clusters(x, seed, n_clusters, n_init):
    """函数只接收配比 X，无法读取 Loss；用于固定配方簇。"""
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init)
    return model.fit_predict(x), model


def cluster_cv(data, xfields, yfields, selection, config, dest, old):
    ids,x,y = data["train_1m"]
    labels,kmeans = recipe_clusters(x,config["seed"],config["recipe_kmeans_clusters"],config["recipe_kmeans_n_init"])
    pd.DataFrame({"index":ids,"cluster":labels}).to_csv(dest / "cluster_assignments.csv",index=False)
    folds = list(GroupKFold(n_splits=config["recipe_group_cv_folds"]).split(x,y,labels))
    fold_rows=[]
    details=[]
    summaries=[]
    global_std=y.std(axis=0,ddof=1)
    for fold,(tr,va) in enumerate(folds):
        if set(labels[tr]) & set(labels[va]):
            raise RuntimeError("配方簇泄漏到同一 GroupKFold 的训练和验证侧")
        fold_rows.extend({"fold":fold,"row":int(i),"index":ids[i],"cluster":int(labels[i]),"role":role}
                         for role,values in (("train",tr),("validation",va)) for i in values)
        preds={}
        preds["Ridge"]=np.column_stack([ridge_predict(ridge_fit(x[tr],y[tr,t],selection["ridge_alphas_by_target"][t]),x[va])
                                          for t in range(y.shape[1])])
        preds["LightGBM"]=np.column_stack([model_lgb(selection["lightgbm_params"],config["seed"]).fit(x[tr],y[tr,t]).predict(x[va])
                                             for t in range(y.shape[1])])
        for name,pred in preds.items():
            rows=target_metrics(y[va],pred,yfields)
            for row in rows:
                row.update({"fold":fold,"model":name,"n_train":len(tr),"n_validation":len(va),
                            "alpha_or_params":"frozen_v2"})
                details.append(row)
            pooled=1-((y[va]-pred)**2).sum()/((y[va]-y[va].mean(axis=0))**2).sum()
            summaries.append({"fold":fold,"model":name,"n_train":len(tr),"n_validation":len(va),
                              "macro_nrmse_train_scale":float(np.mean(np.sqrt(np.mean((y[va]-pred)**2,axis=0))/global_std)),
                              "pooled_r2":float(pooled),"mean_target_pearson":float(np.nanmean([r["pearson"] for r in rows]))})
    pd.DataFrame(fold_rows).to_csv(dest / "cluster_cv_fold_membership.csv",index=False)
    pd.DataFrame(details).to_csv(dest / "cluster_cv_metrics.csv",index=False)
    pd.DataFrame(summaries).to_csv(dest / "cluster_cv_fold_summary.csv",index=False)
    old_cv=pd.read_csv(old / "mixture/cv_fold_metrics.csv")
    random_summary=[]
    for name,key in (("Ridge","ridge_rmse"),("LightGBM","lightgbm_rmse")):
        mean_by_target=old_cv.groupby("target_index")[key].apply(lambda s:np.sqrt(np.mean(s**2))).to_numpy()
        random_summary.append({"evaluation":"random_repeated_cv_v2","model":name,
                               "macro_nrmse_train_scale":float(np.mean(mean_by_target/global_std)),
                               "n_folds":int(old_cv.fold.nunique())})
    cluster_by_model=pd.DataFrame(summaries).groupby("model",as_index=False).macro_nrmse_train_scale.mean()
    for row in cluster_by_model.itertuples():
        random_summary.append({"evaluation":"cluster_group_cv_v2_1","model":row.model,
                               "macro_nrmse_train_scale":float(row.macro_nrmse_train_scale),"n_folds":len(folds)})
    pd.DataFrame(random_summary).to_csv(dest / "random_vs_cluster_cv.csv",index=False)
    return {"cluster_count":len(set(labels)),"fold_count":len(folds),
            "frozen_lightgbm_params":selection["lightgbm_params"],"frozen_ridge_alphas":selection["ridge_alphas_by_target"]}


def run(root, config, old_config):
    root,out,old=paths(root)
    dest=out / "mixture"
    dest.mkdir(parents=True,exist_ok=True)
    audit=json.loads((out / "audit/audit_summary.json").read_text())
    data,xfields,yfields=load_data(root,old,old_config)
    selection=json.loads((old / "mixture/model_selection.json").read_text())
    artifact,reference=verify_frozen_model(root,old,config,old_config,data,xfields,yfields,
                                            audit["v2_model_file_sha256"])
    save_json(dest / "inherited_v2_model_reference.json",reference)
    nearest=nearest_rows(data)
    nearest.to_csv(dest / "nearest_train_distance.csv",index=False)
    stats=nearest.groupby("split").nearest_distance.agg(
        n="count",mean="mean",median="median",q05=lambda a:a.quantile(.05),
        q25=lambda a:a.quantile(.25),q75=lambda a:a.quantile(.75),
        q95=lambda a:a.quantile(.95),maximum="max").reset_index()
    stats.to_csv(dest / "nearest_train_distance_summary.csv",index=False)
    strata=distance_strata(root,old,dest,data,nearest,artifact,yfields)
    cluster=cluster_cv(data,xfields,yfields,selection,config,dest,old)
    return {"model_reference":reference,"distance_summary":stats.to_dict(orient="records"),
            "cluster_cv":cluster}
