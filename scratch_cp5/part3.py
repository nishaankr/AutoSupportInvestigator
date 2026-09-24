import json, subprocess, sys

def cli(*args):
    p = subprocess.run(["uv", "run", "autosupport", *args, "--json"], capture_output=True, text=True, encoding="utf-8")
    if p.returncode: sys.exit(f"CLI failed: {p.stderr[-400:]}")
    return json.loads(p.stdout)

def show(tag, d):
    i, r = d["interrupt"], d["result"]
    print(f"[{tag}] status={d['status']} paused={(i['type'], i['confidence']) if i else None} "
          f"result={(r['status'], r['escalation']['trigger']) if r else None}")

d = cli("new", "--customer", sys.argv[1], "--subject", "MongoDB integration options",
        "--body", "Can you provide guidance and options for integrating MongoDB 4.4 into our SaaS-based project management platform? We would like practical methods and any resources to get started.")
show("new", d)
if d["interrupt"] and d["interrupt"]["type"] == "clarification":
    d = cli("resume", d["ticket_id"], "--answer",
            "We run a multi-tenant SaaS on Linux, want a scalable production integration with API connections and data synchronization; no specific error, just the recommended integration methods and resources.")
    show("answered", d)
if not (d["interrupt"] and d["interrupt"]["type"] == "confirmation"):
    sys.exit("did not reach confirm_resolution")
draft1 = d["interrupt"]["resolution"]
tid = d["ticket_id"]
d2 = cli("resume", tid, "--reject", "Please spell out the data synchronization option in more detail and keep the tone shorter.")
show("rejected", d2)
i2 = d2["interrupt"]
if i2 and i2["type"] == "confirmation":
    print("DRAFTS DIFFER:", draft1 != i2["resolution"])
    print("--- draft 1 ---\n" + draft1[:500]); print("--- draft 2 ---\n" + i2["resolution"][:500])
    d3 = cli("resume", tid, "--reject", "Still not what I need.")
    show("rejected again (max_revisions=1 -> escalate)", d3)
    r = d3["result"]
    if r: print("final:", r["status"], r["escalation"]["trigger"], "acceptance:", r["acceptance"], "revisions:", r["stats"]["revisions"])
else:
    print("no revised draft; ended:", d2["result"] and (d2["result"]["status"], d2["result"]["escalation"]["trigger"]))
