"""无额外依赖的可追溯 SVG 图表。"""
from html import escape


COLORS = ["#3568a8", "#dc693e", "#3c9b72", "#9b64a9", "#a78c32", "#3a9aaa", "#c05a75"]


def _head(width,height,title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="35" y="38" font-size="22" font-family="sans-serif" font-weight="bold">{escape(title)}</text>']


def _text(x,y,value,size=12,color="#263238",anchor="start"):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-family="sans-serif">{escape(str(value))}</text>'


def _save(path,parts):
    path.write_text("\n".join([*parts,"</svg>"]) + "\n",encoding="utf-8")


def weights_chart(path,names,weights):
    width,height=1150,790
    p=_head(width,height,"22 indicator weights · A1 entropy + Spearman")
    order=sorted(range(len(names)),key=lambda j:weights[j],reverse=True)
    max_weight=max(weights)*1.12
    for position,j in enumerate(order):
        y=66+position*30
        p.append(_text(360,y+15,names[j],12,anchor="end"))
        p.append(f'<rect x="375" y="{y}" width="{weights[j]/max_weight*660:.2f}" height="19" fill="#3568a8"/>')
        p.append(_text(1045,y+15,f"{weights[j]:.4f}",12))
    p.append(_text(375,756,"Weight (sum = 1) · Source: A1 audited valid records; q1-p1-v1",11,"#546e7a"))
    _save(path,p)


def correlation_chart(path,names,matrix):
    width,height=1080,1050
    p=_head(width,height,"Spearman correlation · 22 normalized indicators · A1")
    x0,y0,cell=290,70,31
    for i,name in enumerate(names):
        p.append(f'<text transform="translate({x0+i*cell+14},65) rotate(-55)" font-size="10" font-family="sans-serif">{escape(name)}</text>')
        p.append(_text(x0-8,y0+i*cell+19,name,10,anchor="end"))
        for j,rho in enumerate(matrix[i]):
            magnitude=abs(rho)
            if rho>=0:
                red,green,blue=255,int(255-155*magnitude),int(255-155*magnitude)
            else:
                red,green,blue=int(255-160*magnitude),int(255-110*magnitude),255
            p.append(f'<rect x="{x0+j*cell}" y="{y0+i*cell}" width="{cell}" height="{cell}" fill="#{red:02x}{green:02x}{blue:02x}" stroke="#eeeeee"/>')
    p.append(_text(290,790,"Blue: negative (-1)    White: 0    Red: positive (+1)",12))
    p.append(_text(35,1010,"Source: A1 audited valid records · same fixed normalization · q1-p1-v1",11,"#546e7a"))
    _save(path,p)


def domain_chart(path,rows):
    selected=sorted((r for r in rows if r["view"]=="A1"),key=lambda r:r["domain"])
    width,height=1000,520
    p=_head(width,height,"A1 domain mean quality · 95% normal-approx. interval")
    for i,row in enumerate(selected):
        y=74+i*54
        p.append(_text(185,y+21,f"{row['domain']} (n={row['n']})",13,anchor="end"))
        p.append(f'<rect x="205" y="{y}" width="{row["mean_Q"]*6.5:.2f}" height="29" fill="{COLORS[i]}"/>')
        left=205+row["ci95_low_Q"]*6.5
        right=205+row["ci95_high_Q"]*6.5
        p.append(f'<path d="M{left:.2f} {y-5} V{y+34} M{left:.2f} {y+14.5} H{right:.2f} M{right:.2f} {y-5} V{y+34}" fill="none" stroke="#17232b" stroke-width="1.5"/>')
        p.append(_text(865,y+20,f"{row['mean_Q']:.2f} / 100",13))
    p.append(_text(205,460,"Quality score Q (0–100); intervals and exact counts: domain_quality_summary.csv",12))
    p.append(_text(35,495,"Source: A1 audited valid records · q1-p1-v1",11,"#546e7a"))
    _save(path,p)


def _max_bin_share(buckets):
    return max(count/sum(bucket["hist"]) for bucket in buckets for count in bucket["hist"])


def _poly_hist(bucket,x0,y0,width,height,max_share):
    n=sum(bucket["hist"])
    if not n:return ""
    points=[]
    for i,count in enumerate(bucket["hist"]):
        x=x0+(i+0.5)*width/20
        y=y0+height-(count/n)*height/max_share
        points.append(f"{x:.1f},{max(y0,y):.1f}")
    return " ".join(points)


def distribution_chart(path,buckets):
    width,height=1050,620
    p=_head(width,height,"A1 domain quality distributions · 5-point bins")
    x0,y0,w,h=75,85,780,430
    p.append(f'<path d="M{x0} {y0} V{y0+h} H{x0+w}" fill="none" stroke="#333"/>')
    domains=sorted(domain for view,domain in buckets if view=="A1")
    max_share=_max_bin_share([buckets[("A1",domain)] for domain in domains])
    for i,domain in enumerate(domains):
        bucket=buckets[("A1",domain)]
        p.append(f'<polyline points="{_poly_hist(bucket,x0,y0,w,h,max_share)}" fill="none" stroke="{COLORS[i]}" stroke-width="2"/>')
        p.append(_text(875,105+i*37,f"{domain} (n={len(bucket['scores'])})",12,COLORS[i]))
    for value in range(0,101,20):
        p.append(_text(x0+value*w/100,y0+h+20,value,11,anchor="middle"))
    p.append(_text(75,555,f"Q (0–100) · share per 5-point bin; top = {max_share:.1%} for all series",12))
    p.append(_text(35,590,"Source: A1 audited valid records · q1-p1-v1",11,"#546e7a"))
    _save(path,p)


def extension_chart(path,buckets):
    width,height=1050,690
    p=_head(width,height,"A1 sample versus full extension · quality distributions")
    for panel,(source,domain) in enumerate([("A2","arxiv"),("A3","github")]):
        x0,y0,w,h=75,80+panel*290,780,205
        p.append(f'<path d="M{x0} {y0} V{y0+h} H{x0+w}" fill="none" stroke="#333"/>')
        max_share=_max_bin_share([buckets[("A1",domain)],buckets[(source,domain)]])
        for bucket_name,color,label,offset in [("A1","#3568a8","A1",0),(source,"#dc693e",source,1)]:
            bucket=buckets[(bucket_name,domain)]
            p.append(f'<polyline points="{_poly_hist(bucket,x0,y0,w,h,max_share)}" fill="none" stroke="{color}" stroke-width="2"/>')
            p.append(_text(870,y0+30+offset*25,f"{label}: n={len(bucket['scores'])}",12,color))
        p.append(_text(75,y0-10,domain,14))
        p.append(_text(75,y0+h+27,f"Q (0–100); 5-point bins; top = {max_share:.1%} within panel",11))
    p.append(_text(35,660,"Source: audited A1/A2/A3; extensions include exact overlap · q1-p1-v1",11,"#546e7a"))
    _save(path,p)


def robustness_chart(path,main,equal):
    width,height=700,700
    p=_head(width,height,"A1 main versus equal-weight quality score")
    x0,y0,size=80,85,520
    p.append(f'<path d="M{x0} {y0} V{y0+size} H{x0+size}" fill="none" stroke="#333"/>')
    p.append(f'<path d="M{x0} {y0+size} L{x0+size} {y0}" stroke="#aaaaaa" stroke-dasharray="4,4"/>')
    step=max(1,len(main)//2500)
    for a,b in zip(main[::step],equal[::step]):
        p.append(f'<circle cx="{x0+a*size/100:.2f}" cy="{y0+size-b*size/100:.2f}" r="1.3" fill="#3568a8" fill-opacity="0.35"/>')
    p.append(_text(80,635,"Main Q (0–100) →    Equal-weight Q (0–100) ↑",12))
    p.append(_text(35,670,f"Source: A1 audited valid; display every {step}th record, n={len(main)} · q1-p1-v1",11,"#546e7a"))
    _save(path,p)


def make_all_figures(directory,names,weights,correlation,buckets,rows,comparisons,main,equal):
    weights_chart(directory / "p1_indicator_weights.svg",names,weights)
    correlation_chart(directory / "p1_spearman_heatmap.svg",names,correlation)
    domain_chart(directory / "p1_domain_means.svg",rows)
    distribution_chart(directory / "p1_domain_distributions.svg",buckets)
    extension_chart(directory / "p1_extension_distributions.svg",buckets)
    robustness_chart(directory / "p1_equal_weight_comparison.svg",main,equal)
