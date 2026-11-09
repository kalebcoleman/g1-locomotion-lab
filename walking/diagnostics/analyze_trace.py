"""Cross-check the captured CUDA graph (DOT dump) against the kernels seen in the CUPTI trace. Usage: analyze_trace.py <evidence_dir>"""
import re, json, sys, collections, os
d = sys.argv[1]
dot = open(os.path.join(d, "newton_graph.dot")).read()
nodes = re.findall(r'\[style="[a-z]+" shape="record" label="\{\s*(KERNEL|MEMCPY|MEMSET|[A-Z_ ]+)\n(.*?)\n\}"\]', dot, re.S)
kinds = collections.Counter(k.strip() for k, _ in nodes)
gk = []
for k, body in nodes:
    if k.strip() == "KERNEL":
        m = re.search(r"\| ([A-Za-z0-9_]+?)\\<\\<\\<", body)
        if m: gk.append(m.group(1))
strip = lambda n: re.sub(r"_[0-9a-f]{8}_cuda_kernel_(forward|backward)$", "", n)
graph_names = collections.Counter(gk)
tr = json.load(open(os.path.join(d, "traced_kernels.json")))
tr_warp = {n: v for n, v in tr.items() if "_cuda_kernel_forward" in n}
tr_other = {n: v for n, v in tr.items() if "_cuda_kernel_forward" not in n}
gset, tset = set(graph_names), set(tr_warp)
inter = gset & tset
out = dict(
  graph_node_kinds=dict(kinds), graph_kernel_nodes=len(gk), graph_distinct_kernels=len(gset),
  graph_kernels_found_in_trace=len(inter), graph_kernels_not_in_trace=sorted(gset - tset)[:30],
  trace_warp_kernels_distinct=len(tset), trace_warp_kernels_not_in_graph=sorted(tset - gset),
  trace_warp_launches=sum(v["count"] for v in tr_warp.values()), trace_warp_gpu_us=sum(v["total_us"] for v in tr_warp.values()),
  trace_nonwarp_kernels_distinct=len(tr_other), trace_nonwarp_launches=sum(v["count"] for v in tr_other.values()),
  trace_nonwarp_gpu_us=sum(v["total_us"] for v in tr_other.values()),
  graph_kernels_gpu_us_in_trace=sum(tr_warp[n]["total_us"] for n in inter), graph_kernel_launches_in_trace=sum(tr_warp[n]["count"] for n in inter),
  sample_graph_kernels=[strip(n) for n in list(graph_names)[:40]],
  top_graph_kernels_by_gpu_time=[(strip(n), tr_warp[n]["count"], round(tr_warp[n]["total_us"], 1)) for n in sorted(inter, key=lambda n: -tr_warp[n]["total_us"])[:15]],
  top_nonwarp_kernels=[(n[:90], v["count"], round(v["total_us"], 1)) for n, v in sorted(tr_other.items(), key=lambda kv: -kv[1]["total_us"])[:8]],
  trace_warp_kernels_not_in_graph_top=[(strip(n), tr_warp[n]["count"], round(tr_warp[n]["total_us"], 1)) for n in sorted(tset - gset, key=lambda n: -tr_warp[n]["total_us"])[:10]])
json.dump(out, open(os.path.join(d, "graph_vs_trace.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
