"""基于 q2-fusion-rerun-v1 和正式 Q1 接口重跑问题三。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/q3_latest_20260926"
Q2 = ROOT / "results/q2_fusion_rerun_v1"
Q1 = ROOT / "results/q1_revision_v2_1/q2_interface"
VERSION = "q3-latest-20260926-v1"
SEED = 20260926

BUDGETS = [1e19, 1e22, 1e24]
CONTEXTS = [2048, 4096, 8192, 32768, 131072]
ETAS = [1e-4, 2e-4, 4e-4]
COST_TYPES = ["exp", "pow", "log"]
BOX = {"N": (0.07, 11.97), "D": (10.0, 600.0), "Q": (0.5, 1.0)}


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


def quality_cost(kind: str, q):
    q = np.asarray(q, float)
    if kind == "exp":
        return 1e7 * np.exp(6 * q)
    if kind == "pow":
        return 5e9 * q**4
    if kind == "log":
        return 2e9 * np.log1p(10 * q)
    raise ValueError(kind)


def loss(params: dict, n, d, q, h=0.0, lambda_p=0.0):
    """Q2 主模型 Form A；配比只作为显式情景作用于数据项。"""
    return (params["E"] + params["A"] * np.asarray(n) ** (-params["alpha"])
            + params["B"] * np.asarray(d) ** (-params["beta"])
            * np.exp(params["gamma"] * (params["q0"] - np.asarray(q)) + lambda_p * h))


def equivalent_legacy_parameters(params: dict):
    """给出与旧八参数计算器严格等价的参数，仅用于可复核迁移。"""
    return {
        "E": params["E"], "A": params["A"], "alpha": params["alpha"],
        "B": params["B"] * np.exp(params["gamma"] * params["q0"]),
        "beta": params["beta"], "rho_N": 0.0, "rho_D": params["gamma"], "E1": 0.0,
    }


def solve_box(params: dict, budget: float, context: int, eta: float, cost_type: str,
              h: float = 0.0, lambda_p: float = 0.0, nd: int = 260, nq: int = 220):
    nmin, nmax = BOX["N"]
    dmin, dmax = BOX["D"]
    q0, qmax = BOX["Q"]
    c18 = budget / 1e18
    k = 6 + eta * context
    if c18 < k * nmin * dmin:
        return None

    def scan(dlo, dhi, qlo, qhi, nd_, nq_):
        ds = np.linspace(dlo, dhi, nd_)
        best = None
        for q in np.linspace(qlo, qhi, nq_):
            psi = max(float(quality_cost(cost_type, q) - quality_cost(cost_type, q0)), 0.0) / 1e9
            ns = np.minimum(nmax, (c18 / ds - psi) / k)
            valid = ns >= nmin
            if not valid.any():
                continue
            values = loss(params, ns[valid], ds[valid], q, h, lambda_p)
            pos = int(np.argmin(values))
            candidate = (float(values[pos]), float(ns[valid][pos]), float(ds[valid][pos]), float(q))
            if best is None or candidate[0] < best[0]:
                best = candidate
        return best

    best = scan(dmin, dmax, q0, qmax, nd, nq)
    if best is None:
        return None
    for _ in range(2):
        dstep = (dmax - dmin) / nd * 5
        qstep = (qmax - q0) / nq * 5
        refined = scan(max(dmin, best[2] - dstep), min(dmax, best[2] + dstep),
                       max(q0, best[3] - qstep), min(qmax, best[3] + qstep), 300, 260)
        if refined and refined[0] < best[0]:
            best = refined
        nd, nq = 300, 260
    value, n, d, q = best
    psi = max(float(quality_cost(cost_type, q) - quality_cost(cost_type, q0)), 0.0) / 1e9
    used = d * (k * n + psi)
    shares = {"train": 6*n*d/c18, "attention": eta*context*n*d/c18, "quality": d*psi/c18}
    return {
        "N": n, "D": d, "Q": q, "L": value, "budget_used_fraction": used/c18,
        "idle_fraction": max(0.0, 1-used/c18), "s_train": shares["train"],
        "s_attention": shares["attention"], "s_quality": shares["quality"],
        "dominant_cost": max(shares, key=shares.get),
        "N_status": "min" if np.isclose(n, nmin, atol=1e-5) else "max" if np.isclose(n, nmax, atol=1e-5) else "interior",
        "D_status": "min" if np.isclose(d, dmin, atol=1e-4) else "max" if np.isclose(d, dmax, atol=1e-4) else "interior",
        "Q_status": "reference" if np.isclose(q, q0, atol=1e-5) else "max" if np.isclose(q, qmax, atol=1e-5) else "interior",
    }


def verify_local(params: dict, row: pd.Series):
    if not row.feasible:
        return {"checked": False, "reason": "infeasible"}
    c18 = row.C / 1e18
    k = 6 + row.eta * row.Lctx
    q0 = params["q0"]
    def objective(x):
        return float(loss(params, x[0], x[1], x[2]))
    def remaining(x):
        psi = max(float(quality_cost(row.cost_type, x[2]) - quality_cost(row.cost_type, q0)), 0.0)/1e9
        return c18 - x[1]*(k*x[0]+psi)
    result = minimize(objective, [row.N,row.D,row.Q], method="SLSQP",
                      bounds=[BOX["N"],BOX["D"],BOX["Q"]],
                      constraints=[{"type":"ineq","fun":remaining}],
                      options={"ftol":1e-11,"maxiter":2000})
    rel = (row.L-float(result.fun))/max(abs(row.L),1e-12)
    return {"checked": True, "success": bool(result.success), "local_loss": float(result.fun),
            "grid_loss": float(row.L), "relative_improvement": float(rel),
            "budget_remaining_C18": float(remaining(result.x))}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    params_raw = json.loads((Q2/"quality_model/b6_parameters.json").read_text())
    params = dict(params_raw["MQ"], q0=float(params_raw["q0"]))
    q1_manifest = json.loads((Q1/"manifest.json").read_text())
    if q1_manifest["status"] != "PASS":
        raise RuntimeError("Q1 formal interface is not PASS")
    interface = pd.read_csv(ROOT/q1_manifest["mixture_response_path"])

    transformed = equivalent_legacy_parameters(params)
    probes = pd.DataFrame({"N":[.1,1,10],"D":[10,100,600],"Q":[.5,.7,1.]})
    direct = loss(params,probes.N,probes.D,probes.Q)
    legacy = (transformed["E"] + transformed["A"]*probes.N**(-transformed["alpha"])
              + transformed["B"]*probes.D**(-transformed["beta"])*np.exp(-transformed["rho_D"]*probes.Q))
    transform_error = float(np.max(np.abs(direct-legacy)))
    if transform_error > 1e-12:
        raise RuntimeError("Q2 parameter transform is not equivalent")
    write_json(OUT/"parameter_transform.json", {"source_formula":"E+A*N^-alpha+B*D^-beta*exp(gamma*(q0-Q))",
        "optimization_formula":"E+A*N^-alpha+B_eff*D^-beta*exp(-rho_D*Q)",
        "source_parameters":params,"equivalent_parameters":transformed,"max_abs_probe_error":transform_error})

    rows=[]
    for eta in ETAS:
        for context in CONTEXTS:
            for cost_type in COST_TYPES:
                for budget in BUDGETS:
                    result=solve_box(params,budget,context,eta,cost_type)
                    record={"eta":eta,"Lctx":context,"cost_type":cost_type,"C":budget,"feasible":result is not None}
                    if result: record.update(result)
                    rows.append(record)
    grid=pd.DataFrame(rows)
    grid.to_csv(OUT/"scenario_grid_all.csv",index=False)
    grid[np.isclose(grid.eta,2e-4)].to_csv(OUT/"scenario_grid_main.csv",index=False)

    # 问题四机制腿使用的局部总 Loss 预算弹性；固定在主情景中档预算工作点。
    eps_step=.01
    low=solve_box(params,1e22*np.exp(-eps_step),32768,2e-4,"exp",nd=400,nq=350)
    high=solve_box(params,1e22*np.exp(eps_step),32768,2e-4,"exp",nd=400,nq=350)
    epsilon_total=float((np.log(high["L"])-np.log(low["L"]))/(2*eps_step))
    mechanism={"C":1e22,"Lctx":32768,"eta":2e-4,"cost_type":"exp",
        "definition":"d log(total Loss*) / d log(C)","epsilon_C":epsilon_total,
        "finite_difference_log_step":eps_step,"low":low,"high":high}
    write_json(OUT/"mechanism_elasticity.json",mechanism)

    # LightGBM 不可微，配比只在正式接口的观测/估算配方与 p0 上做离散情景比较。
    mix=interface[["split","data_role","index","nearest_train_distance","support_status","h_agg"]].copy()
    mix.insert(0,"recipe_id",[f"{s}:{int(i)}" for s,i in zip(mix.split,mix["index"])])
    mix=pd.concat([pd.DataFrame([{"recipe_id":"p0","split":"reference","data_role":"reference",
        "index":0,"nearest_train_distance":0.0,"support_status":"within_support","h_agg":0.0}]),mix],ignore_index=True)
    base=solve_box(params,1e22,32768,2e-4,"exp")
    scenario=[]
    for lam in (0,.5,1,1.5):
        for _,r in mix.iterrows():
            scenario.append({**r.to_dict(),"lambda_p":lam,"conditional_loss":float(loss(params,base["N"],base["D"],base["Q"],r.h_agg,lam))})
    mix_scen=pd.DataFrame(scenario)
    mix_scen.to_csv(OUT/"mixture_scenarios.csv",index=False)

    # 直接传播 Q2 的组 bootstrap 参数，不在 Q3 重新窥视 B7/B8。
    boot=pd.read_csv(Q2/"quality_model/b6_bootstrap_parameters.csv")
    boot_rows=[]
    for _,r in boot.iterrows():
        bp={k:float(r[k]) for k in ("E","A","B","alpha","beta","gamma")};bp["q0"]=params["q0"]
        opt=solve_box(bp,1e22,32768,2e-4,"exp",nd=140,nq=120)
        if opt: boot_rows.append({"bootstrap":int(r.bootstrap),**{k:opt[k] for k in ("N","D","Q","L")}})
    pd.DataFrame(boot_rows).to_csv(OUT/"bootstrap_optima.csv",index=False)

    checks=[]
    feasible=grid[grid.feasible].reset_index(drop=True)
    for pos in np.linspace(0,len(feasible)-1,9,dtype=int):
        row=feasible.iloc[pos]
        checks.append({"row":int(pos),"eta":row.eta,"Lctx":int(row.Lctx),"cost_type":row.cost_type,"C":row.C,**verify_local(params,row)})
    check_frame=pd.DataFrame(checks)
    check_frame.to_csv(OUT/"kkt_local_verification.csv",index=False)
    kkt_pass=bool(check_frame.success.all() and (check_frame.relative_improvement < 2e-5).all() and (check_frame.budget_remaining_C18 >= -1e-6).all())

    sources=[Q2/"quality_model/b6_parameters.json",Q2/"quality_model/b6_bootstrap_parameters.csv",Q1/"manifest.json",ROOT/q1_manifest["mixture_response_path"]]
    products=[OUT/"parameter_transform.json",OUT/"scenario_grid_all.csv",OUT/"scenario_grid_main.csv",OUT/"mechanism_elasticity.json",OUT/"mixture_scenarios.csv",OUT/"bootstrap_optima.csv",OUT/"kkt_local_verification.csv"]
    status="PASS_WITH_WARNINGS" if kkt_pass else "BLOCKED"
    metadata={"version":VERSION,"latest":True,"status":status,"seed":SEED,
        "upstream":{"q2":"q2-fusion-rerun-v1","q1_interface":q1_manifest["interface_version"]},
        "source_sha256":{str(p.relative_to(ROOT)):sha256(p) for p in sources},
        "product_sha256":{str(p.relative_to(ROOT)):sha256(p) for p in products},
        "counts":{"grid":len(grid),"feasible":int(grid.feasible.sum()),"infeasible":int((~grid.feasible).sum()),"mixture_recipes":len(mix),"bootstrap_success":len(boot_rows),"kkt_checks":len(checks)},
        "checks":{"formal_q1_interface":True,"parameter_transform_max_abs_error":transform_error,"kkt_local_pass":kkt_pass,"mechanism_epsilon_finite":bool(np.isfinite(epsilon_total))},
        "assumptions":{"quality_cost":"three uncalibrated scenarios","eta":"scenario","lambda_p":"scenario_not_identified","mixture_optimization":"discrete formal-interface candidates; not global causal optimum"}}
    write_json(OUT/"metadata.json",metadata)
    write_json(OUT/"LATEST_VERSION.json",{"latest":True,"version":VERSION,"status":status,"supersedes":"results/q3 and 团队交付包_问题三融合方案.zip"})
    main=grid[np.isclose(grid.eta,2e-4)]
    bdf=pd.DataFrame(boot_rows)
    report=["# 问题三融合方案｜最新版 2026-09-26","",f"**LATEST / 当前最新版**：`{VERSION}`；状态 `{status}`。","",
        "本版使用问题二 `q2-fusion-rerun-v1` 的 MQ 主模型和问题一正式 LightGBM 接口；不再使用 q2_v8 SET_A 或 mix_final_model_quad.csv。","",
        "## 重跑摘要","",f"- 情景网格 {len(grid)} 行：可行 {int(grid.feasible.sum())}，不可行 {int((~grid.feasible).sum())}。",f"- 主 η=2e-4 网格 {len(main)} 行。",f"- Q2 组 bootstrap 传播成功 {len(bdf)}/{len(boot)} 次。",f"- 局部数值/KKT 复核 {len(checks)} 点，验收={kkt_pass}。",f"- 参数化等价转换最大绝对误差 {transform_error:.3e}。",f"- Q4 机制腿工作点的总 Loss 预算弹性 epsilon_C={epsilon_total:.8f}。","",
        "## 配比结论","",f"正式接口中加入 p0 后候选 h_agg 范围 [{mix.h_agg.min():.6g}, {mix.h_agg.max():.6g}]。当前 LightGBM 接口下 p0 是这些离散候选的最小值；这不是连续单纯形上的全局最优证明。lambda_p 仍只取情景值。","",
        "## 证据边界","","质量成本函数、eta 与 lambda_p 均未由联合实验识别；高预算解常触及 N/D 支持边界。所有最优值是给定模型与成本情景的条件解。","",
        f"Q3 LATEST STATUS: {status}",""]
    (OUT/"report.md").write_text("\n".join(report),encoding="utf-8")
    (OUT/"README_LATEST.md").write_text("# LATEST｜问题三最新版\n\n唯一推荐入口：`PYTHONPATH=src .venv/bin/python -m src.q3_latest.pipeline`。\n\n查看 `LATEST_VERSION.json`、`metadata.json` 和 `report.md`。\n",encoding="utf-8")
    return metadata


if __name__ == "__main__":
    result=run()
    print(f"Q3_LATEST_STATUS: {result['status']}")
