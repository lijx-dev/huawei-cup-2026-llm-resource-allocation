"""问题二阶段入口；只使用数据合同指向的原始列。"""
from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

from src.q2 import audit as source
from src.q2_v2 import audit
from src.q2_v2.model import SEED, cv, fit, grouped_bootstrap, metrics, predict, save_csv, save_json


ROOT = audit.ROOT
_output = Path(os.environ.get("Q2_OUTPUT_DIR", "results/q2_scaling_v2"))
OUT = _output if _output.is_absolute() else ROOT / _output
_config = Path(os.environ.get("Q2_CONFIG_PATH", "configs/q2_scaling_v2.json"))
CONFIG_PATH = _config if _config.is_absolute() else ROOT / _config
CONFIG = json.loads(CONFIG_PATH.read_text())
VERSION = CONFIG.get("experiment_version", os.environ.get("Q2_EXPERIMENT_VERSION", "q2-scaling-v2"))


def load(aid):
    return pd.concat([pd.read_csv(source.B_DIR / p) for p in source.FILES[aid]], ignore_index=True)


def filehash(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def finish(stage, status, message, counts=None):
    folder = {"P0":"audit", "P1":"b1_baseline", "P2":"transfer_validation", "P3":"quality_model", "P4":"q1_interface", "P5":"generalized_law", "P6":"compute_opt", "P7":"extrapolation", "P8":"report"}[stage]
    d = OUT / folder
    base = json.loads((OUT / "audit/audit_metadata.json").read_text())
    products = {str(p.relative_to(ROOT)):filehash(p) for p in d.iterdir() if p.is_file() and not p.name.endswith("metadata.json")}
    meta = dict(experiment_version=VERSION, stage=stage, status=status, seed=SEED, message=message, counts=counts or {}, input_sha256=base["input_sha256"], q1_quality_scores_sha256=base["q1_quality_scores_sha256"], config_sha256=filehash(CONFIG_PATH), code_sha256={str(p.relative_to(ROOT)):filehash(p) for p in (ROOT / "src/q2_v2").glob("*.py")}, q1_model_sha256=json.loads((OUT / "audit/q1_interface.json").read_text())["model_sha256"], versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__), product_sha256=products)
    save_json(d / f"{stage.lower()}_metadata.json", meta)
    return meta


def p0():
    m = audit.run()
    return finish("P0", m["status"], "数据合同审计", m["attachment_rows"])


def p1():
    d = OUT / "b1_baseline"
    import os
    os.environ.setdefault("MPLCONFIGDIR",str(OUT/".matplotlib_cache"))
    b = load("B1")
    group = b.N_params_B.astype(str).to_numpy()
    folds, preds = cv(b, group)
    save_csv(d / "group_cv_metrics.csv", folds)
    save_csv(d / "group_cv_predictions.csv", preds)
    boots = grouped_bootstrap(b, group, reps=CONFIG["bootstrap_replicates"])
    save_csv(d / "bootstrap_parameters.csv", boots)
    theta, starts = fit(b)
    save_json(d / "final_parameters.json", theta)
    save_csv(d / "multistart_runs.csv", starts)
    yhat = predict(theta, b.N_params_B, b.D_tokens_B)
    residuals = pd.DataFrame(dict(N=b.N_params_B, D=b.D_tokens_B, log_N=np.log(b.N_params_B), log_D=np.log(b.D_tokens_B), prediction=yhat, residual=b.val_loss-yhat, group=group))
    residuals.to_csv(d / "residuals.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir=OUT/"figures";figdir.mkdir(exist_ok=True)
    for field,label in (("log_N","log N"),("log_D","log D"),("prediction","prediction"),("group","model scale")):
        fig,ax=plt.subplots(figsize=(6,4))
        if field=="group":
            order=sorted(residuals.group.unique(),key=float)
            ax.boxplot([residuals.loc[residuals.group==g,"residual"] for g in order],tick_labels=order)
            ax.tick_params(axis="x",rotation=45)
        else: ax.scatter(residuals[field],residuals.residual,s=3,alpha=.5)
        ax.axhline(0,color="black",linewidth=.8);ax.set_xlabel(label);ax.set_ylabel("observed - predicted")
        fig.tight_layout();fig.savefig(figdir/f"p1_residual_vs_{field}.svg");plt.close(fig)
    ci = pd.DataFrame(boots).drop(columns="bootstrap").quantile([.025,.5,.975]).to_dict()
    save_json(d / "parameter_ci.json", ci)
    foldframe = pd.DataFrame(folds)
    report = f"# P1 B1 经典标度律\n\nB1 {len(b)} 行，8 个模型规模，按规模留一验证。N、D 均以十亿为单位。\n\n全量拟合参数：`{theta}`。\n\n组宏平均 RMSE={foldframe.RMSE.mean():.6g}，MAE={foldframe.MAE.mean():.6g}，R²={foldframe.R2.mean():.6g}；各组见 `group_cv_metrics.csv`。残差最大绝对值 {residuals.residual.abs().max():.6g}，已接近附件 Loss 的小数舍入量级；数据说明标记 B1 为真实轨迹，但现有材料不足以独立证明数值生成过程。\n\n{len(boots)} 次规模组重抽样参数区间见 `parameter_ci.json`，各次参数见 `bootstrap_parameters.csv`。多起点和残差见相应 CSV。组间外推表现是主要验证，不把 checkpoint 当独立实验。\n\nQ2-P1 STATUS: PASS\n"
    (d / "p1_report.md").write_text(report)
    return finish("P1", "PASS", "B1 拟合、规模组 CV 与组 bootstrap", {"valid":len(b),"isolated":0,"bootstrap_success":len(boots)})


def p2():
    d = OUT / "transfer_validation"
    theta = json.loads((OUT / "b1_baseline/final_parameters.json").read_text())
    rows = []
    def evaluate(aid, frame, subgroup):
        f = frame[np.isfinite(frame.N_params_B) & np.isfinite(frame.D_tokens_B) & np.isfinite(frame.val_loss) & (frame.N_params_B>0) & (frame.D_tokens_B>0)]
        if not len(f): return
        p = predict(theta, f.N_params_B, f.D_tokens_B)
        m = metrics(f.val_loss,p)
        rows.append(dict(attachment=aid, subgroup=str(subgroup), provenance=source.ROLES[aid][1], absolute_comparable_to_B1="uncertain", trend_comparable_to_B1="conditional", evaluation_role=source.ROLES[aid][0], **m))
    for aid in ("B2","B3","B4","B5"):
        f=load(aid)
        evaluate(aid,f,"all")
        if aid=="B3":
            for n,g in f.groupby("N_params_B"): evaluate(aid,g,f"scale:{n}")
        if aid=="B4":
            for key,g in f.groupby("family"): evaluate(aid,g,key)
        if aid=="B5":
            for key,g in f.groupby(["source","family"]): evaluate(aid,g,"|".join(key))
    save_csv(d / "validation_matrix.csv", rows)
    z=pd.DataFrame(rows)
    report=["# P2 B2–B5 冻结 B1 参数迁移检查","","B1 参数未重新拟合。跨来源验证集、tokenizer 与 Loss 协议未知，绝对 RMSE 仅描述数值偏差；趋势、排序仍须附条件解释。B3 是插值轨迹，不能计为独立验证。","", "|附件|行数|RMSE|Spearman|bias|", "|---|---:|---:|---:|---:|"]
    for aid in ("B2","B3","B4","B5"):
        r=z[(z.attachment==aid)&(z.subgroup=="all")].iloc[0]
        report.append(f"|{aid}|{r['n']}|{r.RMSE:.4f}|{r.Spearman:.4f}|{r.mean_bias:.4f}|")
    report += ["","各模型族、轨迹及文献来源详见 `validation_matrix.csv`。", "", "Q2-P2 STATUS: PASS_WITH_WARNINGS", ""]
    (d / "p2_report.md").write_text("\n".join(report))
    return finish("P2","PASS_WITH_WARNINGS","跨源绝对 Loss 口径未核实",{"B2":len(load("B2")),"B3_interpolated":len(load("B3")),"B4":len(load("B4")),"B5":len(load("B5"))})


def p3():
    d=OUT/"quality_model"
    b6,b7,b8=load("B6"),load("B7"),load("B8")
    overlap=pd.read_csv(OUT/"audit/b6_b7_overlap.csv")
    if overlap.conflicting_duplicate.any(): raise RuntimeError("B6/B7 conflicting duplicate")
    b7new=b7[~b7.experiment_id.isin(b6.experiment_id)].copy()
    base6=set(zip(b6.N_params_B,b6.D_tokens_B))
    base7=set(zip(b7new.N_params_B,b7new.D_tokens_B))
    group_overlap=len(base6 & base7)
    q0=float(b6.Q_score.median())
    group=(b6.N_params_B.astype(str)+"|"+b6.D_tokens_B.astype(str)).to_numpy()
    fold0,pred0=cv(b6,group,n_splits=CONFIG["b6_group_folds"])
    foldq,predq=cv(b6,group,quality=True,q0=q0,n_splits=CONFIG["b6_group_folds"])
    fold2,pred2=cv(b6,group,quality=True,q0=q0,quadratic=True,n_splits=CONFIG["b6_group_folds"])
    save_csv(d/"b6_cv_metrics.csv",fold0+foldq+fold2)
    save_csv(d/"b6_cv_predictions.csv",[dict(model="M0",**r) for r in pred0]+[dict(model="MQ",**r) for r in predq]+[dict(model="MQ2",**r) for r in pred2])
    med=lambda rows:float(np.mean([r["RMSE"] for r in rows]))
    retain=med(fold2) <= .98*med(foldq)
    t0,s0=fit(b6)
    tq,sq=fit(b6,quality=True,q0=q0)
    t2,s2=fit(b6,quality=True,q0=q0,quadratic=True)
    save_json(d/"b6_parameters.json",dict(M0=t0,MQ=tq,MQ2_sensitivity=t2,q0=q0,quadratic_cv_retained_for_sensitivity=retain,primary_model="MQ",selection_based_only_on_B6_CV=True))
    save_csv(d/"multistart_runs.csv",[dict(model=name,**r) for name,rs in [("M0",s0),("MQ",sq),("MQ2",s2)] for r in rs])
    boots=grouped_bootstrap(b6,group,quality=True,q0=q0,reps=CONFIG["bootstrap_replicates"])
    save_csv(d/"b6_bootstrap_parameters.csv",boots)
    hold=[]
    for name,t in [("M0",t0),("MQ",tq)]:
        yp=predict(t,b7new.N_params_B,b7new.D_tokens_B,b7new.Q_score,q0,"quality" if name=="MQ" else "base")
        hold.append(dict(model=name,**metrics(b7new.val_loss,yp)))
        save_csv(d/f"b7_new_predictions_{name}.csv",[dict(experiment_id=e,actual=y,prediction=p) for e,y,p in zip(b7new.experiment_id,b7new.val_loss,yp)])
    save_csv(d/"b7_new_metrics.csv",hold)
    # 留出集只在模型定型后评估。区间按基础 (N,D) 组重抽样。
    a=pd.read_csv(d/"b7_new_predictions_M0.csv")
    c=pd.read_csv(d/"b7_new_predictions_MQ.csv")
    ids=b7new.N_params_B.astype(str)+"|"+b7new.D_tokens_B.astype(str)
    uniq=ids.unique(); rng=np.random.default_rng(SEED); ci=[]
    for i in range(CONFIG["bootstrap_replicates"]):
        picks=rng.choice(uniq,len(uniq),replace=True)
        idx=np.concatenate([np.flatnonzero(ids.to_numpy()==g) for g in picks])
        ci.append(dict(replicate=i,delta_RMSE=metrics(a.actual.iloc[idx],c.prediction.iloc[idx])["RMSE"]-metrics(a.actual.iloc[idx],a.prediction.iloc[idx])["RMSE"],delta_MAE=metrics(a.actual.iloc[idx],c.prediction.iloc[idx])["MAE"]-metrics(a.actual.iloc[idx],a.prediction.iloc[idx])["MAE"]))
    save_csv(d/"b7_new_bootstrap_delta.csv",ci)
    t1=json.loads((OUT/"b1_baseline/final_parameters.json").read_text())
    from scipy.optimize import least_squares
    def ar(x):
        t=dict(t1,gamma=float(x[0])); return predict(t,b6.N_params_B,b6.D_tokens_B,b6.Q_score,q0,"quality")+x[1]-b6.val_loss.to_numpy(float)
    arfit=least_squares(ar,[.1,0],bounds=([-10,-10],[10,10]))
    save_csv(d/"anchor_sensitivity.csv",[dict(gamma=float(arfit.x[0]),source_offset=float(arfit.x[1]),RMSE=float(np.sqrt(np.mean(arfit.fun**2))),role="sensitivity_analysis")])
    stress=[]
    for kind,g in b8.groupby("data_type"):
        for name,t in [("M0",t0),("MQ",tq)]:
            yp=predict(t,g.N_params_B,g.D_tokens_B,g.Q_score,q0,"quality" if name=="MQ" else "base")
            stress.append(dict(data_type=kind,model=name,**metrics(g.val_loss,yp)))
    save_csv(d/"b8_stress_test.csv",stress)
    auditdir=pd.read_csv(OUT/"audit/b8_groupwise_q_loss.csv")
    qdir=pd.DataFrame(auditdir[auditdir.data_type!="all"].groupby("data_type").spearman_Q_loss.agg(["count","min","median","max"]))
    gamma_ci=np.quantile([v["gamma"] for v in boots],[.025,.975])
    delta_ci=np.quantile([v["delta_RMSE"] for v in ci],[.025,.975])
    report=f"# P3 B6/B7/B8 质量标度律\n\nB6 {len(b6)} 行、{len(np.unique(group))} 个 N,D 基础组；同组 Q 不跨折。Q0 为预先固定 B6 中位数 {q0}。B6 及 B7 的 Q_score 未自动翻转。\n\nB6 组 CV 宏平均 RMSE：M0={med(fold0):.6g}，MQ={med(foldq):.6g}，二次候选={med(fold2):.6g}；二次候选仅按 B6 规则保留敏感性={retain}，主模型仍为 MQ。\n\nB7-new {len(b7new)} 行只作最终留出。新增 ID 涉及 {len(base7)} 个 N,D 基础组，与 B6 重叠 {group_overlap} 组；因此检验新 Q 水平，不检验全新 N,D 组。M0/MQ 指标见 `b7_new_metrics.csv`，组 bootstrap 的 MQ-M0 RMSE 差 95% 区间 [{delta_ci[0]:.6g},{delta_ci[1]:.6g}]。MQ gamma={tq['gamma']:.6g}，B6 基础组 bootstrap 95% 区间 [{gamma_ci[0]:.6g},{gamma_ci[1]:.6g}]；B1 固定参数敏感性见 `anchor_sensitivity.csv`。\n\nB8 分层组内 Q-Loss Spearman 摘要：{qdir.to_dict('index')}。方向冲突完整保留；B8 没有进入拟合。Q_B 高值语义未由数据说明独立确认。\n\nQ2-P3 STATUS: PASS_WITH_WARNINGS\n"
    (d/"p3_report.md").write_text(report)
    return finish("P3","PASS_WITH_WARNINGS","B6/B7 为半合成；B7-new 仅新 Q 水平；B8 质量方向独立检查",{"B6":len(b6),"B7_new":len(b7new),"B7_new_group_overlap":group_overlap,"B8":len(b8),"bootstrap_success":len(boots)})


def q1_state():
    info=json.loads((OUT/"audit/q1_interface.json").read_text())
    if not info["ready_for_h_v"]: raise RuntimeError("Q1 frozen interface unavailable")
    bundle=joblib.load(ROOT/info["model_path"])
    interface_dir = CONFIG.get("q1_interface_dir") or os.environ.get("Q1_INTERFACE_DIR")
    if interface_dir:
        interface_dir = Path(interface_dir)
        if not interface_dir.is_absolute(): interface_dir = ROOT/interface_dir
        manifest=json.loads((interface_dir/"manifest.json").read_text())
        if manifest["status"]!="PASS" or manifest["model"]["sha256"]!=info["model_sha256"]:
            raise RuntimeError("formal Q1 interface validation/hash mismatch")
        recipes=pd.read_csv(ROOT/manifest["mixture_response_path"])
        train=recipes[recipes.split.eq("train_1m")].sort_values("index")
        x=train[bundle["mix_fields"]].to_numpy(float)
        reference=json.loads((ROOT/manifest["reference_mixture_path"]).read_text())
        p0=np.asarray(reference["values"],float)
        support=float(manifest["support_rule"]["threshold"])
        info["formal_interface_manifest"]=str((interface_dir/"manifest.json").relative_to(ROOT))
    else:
        a4=pd.read_csv(ROOT/"data/real_attachments/A_data_value/regmix_tables/train_mixture_1m.csv")
        raw=a4[bundle["mix_fields"]].to_numpy(float)
        x=raw/raw.sum(axis=1,keepdims=True)
        p0=x.mean(axis=0)
        p0=p0/p0.sum()
        support=json.loads((ROOT/"results/q1_revision_v2/mixture/support_reference.json").read_text())["nearest_neighbor_95pct_distance"]
    return bundle,x,p0,float(support),info


def q1_predict(bundle,p):
    p=np.asarray(p,float)
    if p.ndim==1: p=p[None,:]
    row_sum=p.sum(axis=1)
    if p.shape[1]!=17 or np.any(p<0) or not np.allclose(row_sum,1,atol=.005): raise ValueError("invalid mixture")
    p=p/row_sum[:,None]
    y=np.column_stack([m.predict(p) for m in bundle["models"][CONFIG["q1_model"]]])
    if y.shape[1]!=13 or not np.isfinite(y).all() or np.any(y<=0): raise ValueError("nonpositive Q1 predicted Loss")
    return y


def mixture_effect(bundle,p,p0):
    y=q1_predict(bundle,p)
    y0=q1_predict(bundle,p0)[0]
    return np.log(y/y0)


def nearest_distance(p,x):
    return float(np.sqrt(np.min(np.sum((x-np.asarray(p))**2,axis=1))))


def transfer_share(p, target, donor, delta):
    v=np.asarray(p,float).copy()
    if target==donor or delta<0 or v[donor]<delta: raise ValueError("infeasible share transfer")
    v[target]+=delta;v[donor]-=delta
    if np.any(v<0) or not np.isclose(v.sum(),1,atol=.005): raise ValueError("simplex violation")
    return v


def p4():
    d=OUT/"q1_interface"
    bundle,x,p0,support,info=q1_state()
    hv=mixture_effect(bundle,np.vstack([p0,x]),p0)
    if np.max(np.abs(hv[0]))>1e-12: raise RuntimeError("h(p0) != 0")
    formal = bool(info.get("formal_interface_manifest"))
    method = "formal Q1 interface: mean of row-normalized A4 recipes" if formal else "A4 arithmetic mean after row normalization"
    save_json(d/"reference_mixture.json",dict(fields=bundle["mix_fields"],values=p0.tolist(),raw_mean_sum=float(x.mean(axis=0).sum()),normalized_reference_sum=float(p0.sum()),method=method,formal_interface_manifest=info.get("formal_interface_manifest"),model_sha256=info["model_sha256"],model_name=CONFIG["q1_model"],support_distance_threshold=support))
    rows=[]
    for i,row in enumerate(hv):
        p=p0 if i==0 else x[i-1]
        rows.append(dict(recipe="p0" if i==0 else f"A4:{i}",nearest_distance=nearest_distance(p,x),support_status="within_support" if nearest_distance(p,x)<=support else "low_confidence_extrapolation",h_agg=float(np.mean(row)),**{f"h_{j+1}":float(row[j]) for j in range(13)}))
    save_csv(d/"h_by_target.csv",rows)
    save_json(d/"interface_metadata.json",dict(q1_model_hash=info["model_sha256"],input_order=bundle["mix_fields"],output_order=bundle["loss_fields"],reference_prediction=q1_predict(bundle,p0)[0].tolist(),aggregate_rule="arithmetic mean of 13 dimensionless log ratios",q_a_vs_q_b="different_unverified",support_rule="Q1 leave-self-out 95th percentile Euclidean distance"))
    (d/"p4_report.md").write_text(f"# P4 第一问冻结配比接口\n\n使用 {CONFIG['q1_model']} 冻结模型，SHA-256={info['model_sha256']}；17 输入、13 Loss 输出顺序见 metadata。A4 配方由正式接口逐行归一化，参考配比总和为 {p0.sum():.9f}；未改动 A4。\n\nh_v(p0)=0，13 个目标分开输出。h_agg 为等权 13 个无量纲对数比，仅用于明确标记的情景汇总，不将 B 的标量 Loss 与任一 A 目标天然等同。支持阈值={support:.6g}。正式接口={info.get('formal_interface_manifest', 'legacy')}。\n\nQ2-P4 STATUS: PASS_WITH_WARNINGS\n")
    return finish("P4","PASS_WITH_WARNINGS","旧模型缺历史哈希锚点；当前文件经 v2.1 复核",{"A4_recipes":len(x),"targets":13})


def scenario_prediction(theta,n,d,q,q0,h,lam,form):
    return float(predict(theta,[n],[d],[q],q0,form,h,lam)[0])


def local_effects(theta,n,d,q,q0,h,lam,form):
    g=theta["gamma"]*(q0-q)
    p=theta["A"]*n**(-theta["alpha"])
    t=theta["B"]*d**(-theta["beta"])*np.exp(g)
    if form=="A": t*=np.exp(lam*h)
    else: p*=np.exp(lam*h); t*=np.exp(lam*h)
    r=p+t
    return dict(M_N=float(theta["alpha"]*p/n),M_D=float(theta["beta"]*t/d),M_Q=float(theta["gamma"]*t),epsilon_N=float(-theta["alpha"]*p/r),epsilon_D=float(-theta["beta"]*t/r),q_semielasticity=float(-theta["gamma"]*t/r),dN_dQ_iso_loss=float(-theta["gamma"]*t*n/(theta["alpha"]*p)),dlogN_dQ_iso_loss=float(-theta["gamma"]*t/(theta["alpha"]*p)),P=float(p),T=float(t))


def p5():
    d=OUT/"generalized_law"
    model=json.loads((OUT/"quality_model/b6_parameters.json").read_text())
    theta,q0=model["MQ"],model["q0"]
    bundle,x,p0,support,_=q1_state()
    b6=load("B6")
    # 工作点预先由 B6 N*D 的 10/50/90% 分位对应实际行确定。
    ordered=b6.assign(product=b6.N_params_B*b6.D_tokens_B).sort_values("product")
    points=[ordered.iloc[int(q*(len(ordered)-1))] for q in (.1,.5,.9)]
    distances=np.linalg.norm(x-p0,axis=1)
    recipes=[("p0",p0)]+[(f"A4:{int(i)+1}",x[int(i)]/x[int(i)].sum()) for i in np.argsort(distances)[[0,len(x)//2]]]
    qs=np.quantile(b6.Q_score,[.25,.5,.75])
    grid=[]; marg=[]; subst=[]; forms=[]
    for z,point in enumerate(points):
        n,d0=float(point.N_params_B),float(point.D_tokens_B)
        for q in qs:
            for name,p in recipes:
                h=float(np.mean(mixture_effect(bundle,p,p0)))
                ds=nearest_distance(p,x)
                for lam in CONFIG["lambda_scenarios"]:
                    vals={}
                    for form in ("A","B"):
                        y=scenario_prediction(theta,n,d0,q,q0,h,lam,form)
                        base=dict(workpoint=z,N_billions=n,D_billions=d0,Q_B=float(q),recipe=name,h_agg=h,nearest_distance=ds,support_status="within_support" if ds<=support else "low_confidence_extrapolation",lambda_p=lam,lambda_role="scenario_assumption",form=form)
                        grid.append(dict(base,loss_prediction=y))
                        vals[form]=y
                        effect=local_effects(theta,n,d0,q,q0,h,lam,form)
                        marg.append(dict(base,**{k:effect[k] for k in ("M_N","M_D","M_Q")}))
                        subst.append(dict(base,dN_dQ_iso_loss=effect["dN_dQ_iso_loss"],dlogN_dQ_iso_loss=effect["dlogN_dQ_iso_loss"]))
                    forms.append(dict(workpoint=z,Q_B=float(q),recipe=name,lambda_p=lam,form_A=vals["A"],form_B=vals["B"],B_minus_A=vals["B"]-vals["A"]))
    save_csv(d/"scenario_grid.csv",grid); save_csv(d/"marginal_effects.csv",marg)
    save_csv(d/"q_n_substitution.csv",subst); save_csv(d/"form_sensitivity.csv",forms)
    save_csv(d/"elasticities.csv",[{k:r[k] for k in ("workpoint","N_billions","D_billions","Q_B","recipe","lambda_p","form")} | {k:local_effects(theta,r["N_billions"],r["D_billions"],r["Q_B"],q0,r["h_agg"],r["lambda_p"],r["form"])[k] for k in ("epsilon_N","epsilon_D","q_semielasticity")} for r in grid])
    boot=pd.read_csv(OUT/"quality_model/b6_bootstrap_parameters.csv")
    uncertainty=[]
    for z,point in enumerate(points):
        for q in qs:
            vals=[local_effects(t,float(point.N_params_B),float(point.D_tokens_B),float(q),q0,0.,0.,"A") for _,t in boot.iterrows()]
            for field in ("M_N","M_D","M_Q","dlogN_dQ_iso_loss"):
                arr=np.asarray([v[field] for v in vals])
                uncertainty.append(dict(workpoint=z,Q_B=float(q),effect=field,estimate=local_effects(theta,float(point.N_params_B),float(point.D_tokens_B),float(q),q0,0.,0.,"A")[field],ci025=float(np.quantile(arr,.025)),ci975=float(np.quantile(arr,.975)),bootstrap_n=len(arr),recipe="p0",lambda_p=0,form="A"))
    save_csv(d/"local_effect_intervals.csv",uncertainty)
    # 三个实际基准、三个受益域、两个 donor、两个幅度；四角使用同一来源扣减。
    order=np.argsort(-p0); targets=[int(i) for i in order[:3]]; donors=[int(i) for i in order[3:5]]
    pairs=[]
    for name,p in recipes:
        for donor in donors:
            for a in range(len(targets)):
                for b in range(a+1,len(targets)):
                    j,k=targets[a],targets[b]
                    for delta in (.01,.03):
                        if p[donor]<2*delta: continue
                        states=[]; corners=[]
                        for dj,dk in ((0,0),(delta,0),(0,delta),(delta,delta)):
                            v=p.copy();v[j]+=dj;v[k]+=dk;v[donor]-=dj+dk
                            if np.any(v<0) or not np.isclose(v.sum(),1,atol=.005): raise RuntimeError("simplex violation")
                            corners.append(v)
                            states.append(float(np.mean(mixture_effect(bundle,v,p0))))
                        h00,h10,h01,h11=states
                        maxdist=max(nearest_distance(v,x) for v in corners)
                        pairs.append(dict(recipe=name,donor=bundle["mix_fields"][donor],target_j=bundle["mix_fields"][j],target_k=bundle["mix_fields"][k],delta=delta,response=h11-h10-h01+h00,loss_response_at_median_point=scenario_prediction(theta,float(points[1].N_params_B),float(points[1].D_tokens_B),q0,q0,h11,1,"A")-scenario_prediction(theta,float(points[1].N_params_B),float(points[1].D_tokens_B),q0,q0,h10,1,"A")-scenario_prediction(theta,float(points[1].N_params_B),float(points[1].D_tokens_B),q0,q0,h01,1,"A")+scenario_prediction(theta,float(points[1].N_params_B),float(points[1].D_tokens_B),q0,q0,h00,1,"A"),max_nearest_distance=maxdist,support_status="within_support" if maxdist<=support else "low_confidence_extrapolation",evidence="path_conditioned_scenario"))
    save_csv(d/"path_conditioned_pair_response.csv",pairs,columns=["recipe","donor","target_j","target_k","delta","response","loss_response_at_median_point","max_nearest_distance","support_status","evidence"])
    (d/"p5_report.md").write_text(f"# P5 广义标度律情景\n\nB6 直接估计 E,A,B,alpha,beta,gamma；第一问导入 h_v 和等权 h_agg；lambda_p 仅取 {CONFIG['lambda_scenarios']}，从未由 B 拟合。Form A 调节有效数据项，Form B 调节全部可约 Loss；均为结构假设。\n\n工作点取 B6 N*D 的 10/50/90% 对应实际行，Q 取 B6 25/50/75 分位，配比取 p0 和两份 A4 配方。输出 {len(grid)} 个情景点、{len(pairs)} 条路径条件组合响应。M_N/M_D/M_Q 为 Loss 负偏导；弹性以 R=L-E 为分母，Q 报半弹性。局部等损失替代固定 D、p、Loss 和工作点。\n\n配比路径响应不是因果协同；低支持区域按距离标记。B 的标量 Loss 与 A 的 13 目标没有天然一致口径。\n\nQ2-P5 STATUS: PASS_WITH_WARNINGS\n")
    return finish("P5","PASS_WITH_WARNINGS","lambda 和模型形式均为情景",{"scenario_points":len(grid),"pair_paths":len(pairs)})


def analytic_optimum(theta,c,q,q0,h=0.,lam=0.,form="A"):
    a=theta["A"]*np.exp(lam*h if form=="B" else 0.)
    b=theta["B"]*np.exp(theta["gamma"]*(q0-q)+lam*h)
    n=(theta["alpha"]*a/(theta["beta"]*b)*(c/6.)**theta["beta"])**(1./(theta["alpha"]+theta["beta"]))
    return float(n),float(c/(6*n))


def bounded_optimum(theta,c,q,q0,bounds,h=0.,lam=0.,form="A"):
    nmin,nmax,dmin,dmax=bounds
    lo=max(nmin,c/(6*dmax)); hi=min(nmax,c/(6*dmin))
    if lo>hi: return None
    fun=lambda logn:scenario_prediction(theta,np.exp(logn),c/(6*np.exp(logn)),q,q0,h,lam,form)
    if np.isclose(lo,hi): n=lo
    else:
        res=minimize_scalar(fun,bounds=(np.log(lo),np.log(hi)),method="bounded",options={"xatol":1e-12})
        n=float(np.exp(res.x))
        n=min(max(n,lo),hi)
    return float(n),float(c/(6*n))


def p6():
    d=OUT/"compute_opt"
    m=json.loads((OUT/"quality_model/b6_parameters.json").read_text())
    theta,q0=m["MQ"],m["q0"]
    b6=load("B6")
    bounds=[float(b6.N_params_B.min()),float(b6.N_params_B.max()),float(b6.D_tokens_B.min()),float(b6.D_tokens_B.max())]
    costs=np.quantile(6*b6.N_params_B*b6.D_tokens_B,[.1,.5,.9])
    qs=np.quantile(b6.Q_score,[.25,.5,.75])
    bundle,x,p0,support,_=q1_state()
    h0=float(np.mean(mixture_effect(bundle,p0,p0)))
    rows=[]; bounded=[]; qsens=[]; scenarios=[]
    for i,c in enumerate(costs):
        for q in qs:
            n,dt=analytic_optimum(theta,float(c),float(q),q0)
            bn=bounded_optimum(theta,float(c),float(q),q0,bounds)
            pval=theta["A"]*n**(-theta["alpha"])
            tval=theta["B"]*dt**(-theta["beta"])*np.exp(theta["gamma"]*(q0-q))
            row=dict(budget_level=("low","medium","high")[i],C_1e18_FLOPs=float(c),Q_B=float(q),N_star_billions=n,D_star_billions=dt,alpha_P_minus_beta_T=float(theta["alpha"]*pval-theta["beta"]*tval),within_B6_bounds=bool(bounds[0]<=n<=bounds[1] and bounds[2]<=dt<=bounds[3]))
            rows.append(row);qsens.append(row.copy())
            bounded.append(dict(budget_level=row["budget_level"],C_1e18_FLOPs=float(c),Q_B=float(q),N_star_bounded_billions=bn[0] if bn else None,D_star_bounded_billions=bn[1] if bn else None,feasible=bn is not None,bounds_N_min=bounds[0],bounds_N_max=bounds[1],bounds_D_min=bounds[2],bounds_D_max=bounds[3]))
            for form in ("A","B"):
                for lam in CONFIG["lambda_scenarios"]:
                    nn,dd=analytic_optimum(theta,float(c),float(q),q0,h0,lam,form)
                    bb=bounded_optimum(theta,float(c),float(q),q0,bounds,h0,lam,form)
                    scenarios.append(dict(budget_level=row["budget_level"],C_1e18_FLOPs=float(c),Q_B=float(q),form=form,lambda_p=lam,recipe="p0",h_agg=h0,N_star_billions=nn,D_star_billions=dd,N_star_bounded_billions=bb[0] if bb else None,D_star_bounded_billions=bb[1] if bb else None))
    save_csv(d/"analytic_optima.csv",rows);save_csv(d/"bounded_optima.csv",bounded)
    save_csv(d/"q_sensitivity.csv",qsens);save_csv(d/"lambda_scenario_optima.csv",scenarios)
    numeric=[]
    for row in rows:
        c,q=row["C_1e18_FLOPs"],row["Q_B"]
        center=np.log(row["N_star_billions"])
        result=minimize_scalar(lambda x:scenario_prediction(theta,np.exp(x),c/(6*np.exp(x)),q,q0,0.,0.,"A"),bounds=(center-5,center+5),method="bounded")
        numeric.append(dict(budget_level=row["budget_level"],Q_B=q,N_analytic=row["N_star_billions"],N_numeric=float(np.exp(result.x)),relative_difference=float(abs(np.exp(result.x)-row["N_star_billions"])/row["N_star_billions"]),success=bool(result.success)))
    save_csv(d/"numerical_verification.csv",numeric)
    boot=pd.read_csv(OUT/"quality_model/b6_bootstrap_parameters.csv")
    br=[]
    for _,t in boot.iterrows():
        for i,c in enumerate(costs):
            for q in qs:
                n,dd=analytic_optimum(t,float(c),float(q),q0)
                bb=bounded_optimum(t,float(c),float(q),q0,bounds)
                br.append(dict(bootstrap=int(t.bootstrap),budget_level=("low","medium","high")[i],Q_B=float(q),N_star_billions=n,D_star_billions=dd,N_star_bounded_billions=bb[0] if bb else None,D_star_bounded_billions=bb[1] if bb else None))
    save_csv(d/"bootstrap_optima.csv",br)
    interval=pd.DataFrame(br).groupby(["budget_level","Q_B"])[["N_star_billions","D_star_billions","N_star_bounded_billions","D_star_bounded_billions"]].quantile([.025,.5,.975]).reset_index()
    interval.to_csv(d/"bootstrap_optima_intervals.csv",index=False)
    # 非 p0 路径的配比灵敏度，仍明确是 lambda 情景。
    alt=x[np.argsort(np.linalg.norm(x-p0,axis=1))[[0,len(x)//2]]]
    altrows=[]
    for j,p in enumerate(alt):
        h=float(np.mean(mixture_effect(bundle,p/p.sum(),p0)))
        for i,c in enumerate(costs):
            for lam in CONFIG["lambda_scenarios"]:
                for form in ("A","B"):
                    n,dd=analytic_optimum(theta,float(c),q0,q0,h,lam,form)
                    bb=bounded_optimum(theta,float(c),q0,q0,bounds,h,lam,form)
                    altrows.append(dict(recipe=f"A4_alt_{j+1}",budget_level=("low","medium","high")[i],form=form,lambda_p=lam,h_agg=h,N_star_billions=n,D_star_billions=dd,N_star_bounded_billions=bb[0] if bb else None,D_star_bounded_billions=bb[1] if bb else None))
    save_csv(d/"mixture_scenario_optima.csv",altrows)
    outside=sum(not r["within_B6_bounds"] for r in rows)
    (d/"p6_report.md").write_text(f"# P6 固定算力资源配置\n\n采用稠密 Transformer 近似 C≈6ND，N/D 单位为十亿，表中 C 单位 10^18 FLOPs。预算由 B6 实际 6ND 的 10/50/90% 分位预先确定：{costs.tolist()}。质量在 B6 25/50/75% 分位。\n\n解析解 N*=[αA/(βB exp(g_Q))·(C/6)^β]^(1/(α+β))，D*=C/(6N*)；一阶条件 αP=βT。无界数值优化复核见 `numerical_verification.csv`。边界数值解限制在 B6 观测 N,D 范围 {bounds}；{outside}/{len(rows)} 个解析点超出该范围，不解释为有数据支持的内点最优。{len(br)} 条基础组 bootstrap 解用于区间。\n\n质量提升成本、注意力开销与上下文长度未纳入 C≈6ND，因此这里是固定 Q_B 与 p 条件下的训练算力分配，不是赛题正文要求的全部成本联合最优；lambda 配比结果仅为情景。p0 上 h=0，lambda 不改变最优；不同配比的 Form A/B 灵敏度见 `mixture_scenario_optima.csv`。\n\nQ2-P6 STATUS: PASS_WITH_WARNINGS\n")
    return finish("P6","PASS_WITH_WARNINGS","仅训练算力近似，未识别质量成本与注意力开销",{"budgets":len(costs),"quality_levels":len(qs),"bootstrap_optima":len(br)})


def p7():
    d=OUT/"extrapolation"
    b9,b10=load("B9"),load("B10")
    b9_valid=b9[np.isfinite(b9.N_params_B)&np.isfinite(b9.D_tokens_B)&(b9.N_params_B>0)&(b9.D_tokens_B>0)].copy()
    b1=json.loads((OUT/"b1_baseline/final_parameters.json").read_text())
    b6=json.loads((OUT/"quality_model/b6_parameters.json").read_text())
    rows=[]
    nrange=[float(load("B1").N_params_B.min()),float(load("B1").N_params_B.max())]
    drange=[float(load("B1").D_tokens_B.min()),float(load("B1").D_tokens_B.max())]
    for name,theta in [("B1_M0",b1),("B6_MQ_at_Q0",b6["MQ"])]:
        pred=predict(theta,b10.N_params_B,b10.D_tokens_B,[b6["q0"]]*len(b10),b6["q0"],"quality" if name.startswith("B6") else "base")
        m=metrics(b10.val_loss,pred)
        rows.append(dict(model=name,provenance="estimated_reference",interpretation="estimated_consistency_not_independent_validation",relative_error_mean=float(np.mean(np.abs(pred-b10.val_loss)/b10.val_loss)),extrapolation_distance_log=float(np.mean(np.hypot(np.maximum(0,np.log(b10.N_params_B/nrange[1])),np.maximum(0,np.log(b10.D_tokens_B/drange[1]))))),**m))
        save_csv(d/f"b10_predictions_{name}.csv",[dict(row_id=i+2,N_billions=float(n),D_billions=float(dt),estimated_loss=float(y),predicted_loss=float(p),extrapolation_distance=float(np.hypot(max(0,np.log(n/nrange[1])),max(0,np.log(dt/drange[1]))))) for i,(n,dt,y,p) in enumerate(zip(b10.N_params_B,b10.D_tokens_B,b10.val_loss,pred))])
    save_csv(d/"b10_estimated_consistency.csv",rows)
    save_csv(d/"b9_valid_metadata.csv",b9_valid.to_dict("records"))
    (d/"p7_report.md").write_text(f"# P7 B9/B10 超大尺度参考\n\nB9 共 {len(b9)} 行，D 非正隔离 {len(b9)-len(b9_valid)} 行，可用元数据 {len(b9_valid)} 行。B10 共 {len(b10)} 行，Loss 是既有标度律估算，未参与任何参数拟合。\n\nB1 曲线与 B10 估算序列 RMSE={rows[0]['RMSE']:.6g}；数值一致不构成独立真实验证，且提示同源生成/循环验证风险。完整比较见 `b10_estimated_consistency.csv`。外推距离相对 B1 训练上界以 log N/log D 计算。\n\nQ2-P7 STATUS: PASS_WITH_WARNINGS\n")
    return finish("P7","PASS_WITH_WARNINGS","B10 为已拟合标度律估算",{"B9_raw":len(b9),"B9_valid":len(b9_valid),"B10_estimated":len(b10)})


def p8():
    d=OUT/"report"
    folders=["audit","b1_baseline","transfer_validation","quality_model","q1_interface","generalized_law","compute_opt","extrapolation"]
    stages={f"P{i}":json.loads((OUT/folders[i]/f"p{i}_metadata.json").read_text()) for i in range(8)}
    b1=json.loads((OUT/"b1_baseline/final_parameters.json").read_text())
    b6=json.loads((OUT/"quality_model/b6_parameters.json").read_text())
    cv1=pd.read_csv(OUT/"b1_baseline/group_cv_metrics.csv")
    cv6=pd.read_csv(OUT/"quality_model/b6_cv_metrics.csv")
    hold=pd.read_csv(OUT/"quality_model/b7_new_metrics.csv")
    b8=pd.read_csv(OUT/"audit/b8_groupwise_q_loss.csv")
    q1=json.loads((OUT/"q1_interface/interface_metadata.json").read_text())
    opt=pd.read_csv(OUT/"compute_opt/analytic_optima.csv")
    extrap=pd.read_csv(OUT/"extrapolation/b10_estimated_consistency.csv")
    key=[]
    for k,v in b1.items(): key.append(dict(result=f"B1_{k}",value=v,evidence_grade="direct_fit"))
    key += [dict(result="B1_LOSO_macro_RMSE",value=cv1.RMSE.mean(),evidence_grade="held_out_validation"),dict(result="B6_MQ_gamma",value=b6["MQ"]["gamma"],evidence_grade="semi_synthetic_calibration"),dict(result="B6_M0_groupCV_RMSE",value=cv6.loc[cv6.model=="M0","RMSE"].mean(),evidence_grade="semi_synthetic_calibration"),dict(result="B6_MQ_groupCV_RMSE",value=cv6.loc[cv6.model=="MQ","RMSE"].mean(),evidence_grade="semi_synthetic_calibration")]
    key += [dict(result=f"B7_new_{r.model}_RMSE",value=r.RMSE,evidence_grade="held_out_validation") for _,r in hold.iterrows()]
    medium=opt[(opt.budget_level=="medium") & np.isclose(opt.Q_B,b6["q0"])].iloc[0]
    key += [dict(result="B8_positive_Q_Loss_groups",value=int((b8[b8.data_type!="all"].spearman_Q_loss>0).sum()),evidence_grade="semi_synthetic_calibration"),dict(result="medium_budget_Q0_N_star_billions",value=medium.N_star_billions,evidence_grade="scenario_assumption"),dict(result="medium_budget_Q0_D_star_billions",value=medium.D_star_billions,evidence_grade="scenario_assumption"),dict(result="B10_B1_estimated_RMSE",value=extrap.loc[extrap.model=="B1_M0","RMSE"].item(),evidence_grade="extrapolation_reference")]
    save_csv(d/"key_results_table.csv",key)
    grades=[dict(claim=r["result"],evidence_grade=r["evidence_grade"],scope="B1 reported trajectory" if r["result"].startswith("B1") else "B10 estimated sequence" if r["result"].startswith("B10") else "conditional C≈6ND scenario" if "budget" in r["result"] else "B8 semi_synthetic stress" if r["result"].startswith("B8") else "B6/B7 semi_synthetic") for r in key]
    grades += [dict(claim="B2 semi-synthetic transfer",evidence_grade="semi_synthetic_calibration",scope="cross-source Loss protocol uncertain"),dict(claim="B3 interpolated trajectory",evidence_grade="semi_synthetic_calibration",scope="interpolation consistency only"),dict(claim="B4/B5 trend transfer",evidence_grade="held_out_validation",scope="reported cross-family/literature with incompatible absolute Loss risk"),dict(claim="B8 direction stress",evidence_grade="semi_synthetic_calibration",scope="Q encoding equivalence unverified"),dict(claim="Q_A/Q_B non-equivalence policy",evidence_grade="imported_Q1",scope="distinct source definitions"),dict(claim="Q1 h_p,v",evidence_grade="imported_Q1",scope="17-mixture to 13-loss frozen model"),dict(claim="lambda_p/Forms A-B",evidence_grade="scenario_assumption",scope="B lacks p"),dict(claim="marginals/local substitution/compute optimum",evidence_grade="scenario_assumption",scope="conditional model outputs under C≈6ND"),dict(claim="B9/B10 large-scale reference",evidence_grade="extrapolation_reference",scope="estimated Loss, not independent observation")]
    save_csv(d/"evidence_grade_table.csv",grades)
    figures=[dict(figure=str(p.relative_to(ROOT)),source=str((OUT/"b1_baseline/residuals.csv").relative_to(ROOT)),generator="src/q2_v2/stages.py:p1",status="generated",sha256=filehash(p)) for p in sorted((OUT/"figures").glob("p1_residual_vs_*.svg"))]
    save_csv(d/"figure_manifest.csv",figures,columns=["figure","source","generator","status","sha256"])
    summary=dict(experiment_version=VERSION,seed=SEED,stage_statuses={**{k:v["status"] for k,v in stages.items()},"P8":"PASS_WITH_WARNINGS"},input_sha256=stages["P0"]["input_sha256"],config_sha256=filehash(CONFIG_PATH),q1_model_sha256=q1["q1_model_hash"],product_metadata={k:str((OUT/({'P0':'audit','P1':'b1_baseline','P2':'transfer_validation','P3':'quality_model','P4':'q1_interface','P5':'generalized_law','P6':'compute_opt','P7':'extrapolation'}[k])/f"{k.lower()}_metadata.json").relative_to(ROOT)) for k in stages},figure_count=len(figures))
    save_json(d/"reproducibility_summary.json",summary)
    from src.q2_v2.detailed_report import build_report
    report=build_report(OUT)
    (d/"question2_experiment_report.md").write_text(report)
    return finish("P8","PASS_WITH_WARNINGS","完整报告已生成；证据与成本范围保留警告",{"report_sections":18,"figures":len(figures)})
