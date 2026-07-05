package main

import (
	"html/template"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// reportModel is the per-model view the template renders: the size stats plus a
// handful of concrete keep/drop decisions pulled from results.jsonl so the page
// shows the reasoning, not just the numbers.
type reportModel struct {
	Model     string
	Stats     ModelStats
	SavedPct  string
	Examples  []reportExample
	KeptTotal int
	DropTotal int
	ErrTotal  int
}

type reportExample struct {
	ClaimID string
	Skill   string
	Text    string
	Verdict string
	Reason  string
}

type reportData struct {
	JudgeModel string
	Samples    int
	Models     []reportModel
}

// writeReport renders the docs subpage from the run's stats plus the sampled
// decisions in each model's results.jsonl. It overwrites docsPath on every run
// so the page always reflects the latest completed probes.
func writeReport(docsPath string, stats []ModelStats, cfg *Config, modelsDir string) error {
	judgeModel := ""
	if len(cfg.Judges) > 0 {
		judgeModel = cfg.Judges[0].Model
	}
	data := reportData{JudgeModel: judgeModel, Samples: cfg.Settings.Samples}
	for _, ms := range stats {
		rm := reportModel{Model: ms.Model, Stats: ms, SavedPct: pct(ms.OrigBytes, ms.PrunedBytes)}
		for _, s := range ms.Skills {
			rm.KeptTotal += s.Kept
			rm.DropTotal += s.Dropped
			rm.ErrTotal += s.Errored
		}
		rm.Examples = sampleExamples(filepath.Join(modelsDir, ms.Model, "results.jsonl"))
		data.Models = append(data.Models, rm)
	}

	if err := os.MkdirAll(filepath.Dir(docsPath), 0o755); err != nil {
		return err
	}
	f, err := os.Create(docsPath)
	if err != nil {
		return err
	}
	if err := reportTmpl.Execute(f, data); err != nil {
		f.Close()
		return err
	}
	return f.Close() // report is truncated/rewritten each run, so a flush failure must surface
}

// sampleExamples reads a model's results and returns up to a few decided
// probes, preferring a mix of keep and drop so the page shows both outcomes.
func sampleExamples(resultsPath string) []reportExample {
	results := readResults(resultsPath)
	var keeps, drops []reportExample
	ids := make([]string, 0, len(results))
	for id := range results {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		r := results[id]
		if r.Err != "" {
			continue
		}
		ex := reportExample{ClaimID: r.ClaimID, Skill: r.Skill, Text: r.Text, Verdict: r.Verdict}
		if len(r.Samples) > 0 {
			ex.Reason = r.Samples[len(r.Samples)-1].Reason
		}
		if r.Keep {
			keeps = append(keeps, ex)
		} else {
			drops = append(drops, ex)
		}
	}
	out := append(take(keeps, 4), take(drops, 4)...)
	return out
}

func take(s []reportExample, n int) []reportExample {
	if len(s) > n {
		return s[:n]
	}
	return s
}

var reportTmpl = template.Must(template.New("report").Funcs(template.FuncMap{
	"upper": strings.ToUpper,
}).Parse(reportHTML))

const reportHTML = `<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>skill crafter · codehalter</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root{--green:#3DE08A;--green-dim:#2A8C5B;--mono:'JetBrains Mono',ui-monospace,monospace;}
html[data-theme="dark"]{--bg:#0A0E0D;--bg-2:#0E1311;--panel:#121916;--panel-2:#0F1513;--line:#1E2723;--line-soft:#161E1A;--text:#E6EDE8;--text-dim:#7C8B83;--text-faint:#4A554F;--code:#9FE9C2;--glow:.85;}
*{box-sizing:border-box;}html,body{margin:0;}
body{background:var(--bg);color:var(--text);font-family:var(--mono);line-height:1.6;-webkit-font-smoothing:antialiased;}
a{color:inherit;text-decoration:none;}code{color:var(--code);}
.wrap{max-width:1120px;margin:0 auto;padding:0 32px;}
nav{background:var(--bg);border-bottom:1px solid var(--line-soft);}
nav .row{display:flex;align-items:center;gap:14px;height:68px;}
nav .wordmark{font-weight:800;letter-spacing:-.045em;font-size:21px;}
nav .spacer{flex:1;}
nav .links a{font-size:13px;color:var(--text-dim);font-weight:500;}
nav .links a:hover{color:var(--text);}
.hero{padding:70px 0 40px;}
.eyebrow{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--green-dim);font-weight:700;margin-bottom:14px;}
h1{font-size:44px;font-weight:800;letter-spacing:-.045em;line-height:1.05;margin:0 0 20px;max-width:20ch;}
.lede{font-size:16px;color:var(--text-dim);max-width:70ch;margin:0 0 10px;line-height:1.7;}
.lede b{color:var(--text);font-weight:600;}
.section{padding:44px 0;border-top:1px solid var(--line-soft);}
.sec-title{font-size:24px;font-weight:800;letter-spacing:-.03em;margin:0 0 6px;}
.sec-sub{color:var(--text-dim);font-size:14px;margin:0 0 28px;}
.mcard{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:26px;margin-bottom:22px;}
.mhead{display:flex;align-items:baseline;gap:12px;margin-bottom:20px;flex-wrap:wrap;}
.mhead h3{font-size:20px;font-weight:800;margin:0;letter-spacing:-.02em;}
.mhead .tag{font-size:12px;color:var(--text-faint);}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line-soft);border:1px solid var(--line-soft);border-radius:12px;overflow:hidden;margin-bottom:22px;}
.stat{background:var(--panel-2);padding:22px 16px;text-align:center;}
.stat .num{font-size:26px;font-weight:800;letter-spacing:-.03em;color:var(--green);line-height:1;margin-bottom:8px;}
.stat .num.dim{color:var(--text);}
.stat .lbl{font-size:12px;color:var(--text-dim);}
table{width:100%;border-collapse:collapse;font-size:13px;}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line-soft);}
th{color:var(--text-dim);font-weight:600;font-size:11px;letter-spacing:.06em;text-transform:uppercase;}
td.num,th.num{text-align:right;}
.pill{display:inline-block;font-size:11px;font-weight:700;padding:2px 8px;border-radius:20px;}
.pill.keep{color:var(--green);border:1px solid var(--green-dim);}
.pill.drop{color:var(--text-faint);border:1px solid var(--line);}
.ex{border-top:1px solid var(--line-soft);padding:14px 0;}
.ex .claim{font-size:13.5px;color:var(--text);margin-bottom:6px;}
.ex .reason{font-size:12.5px;color:var(--text-dim);}
.ex .cid{color:var(--text-faint);font-size:11px;}
footer{padding:50px 0;color:var(--text-faint);font-size:12px;border-top:1px solid var(--line-soft);}
</style>
</head>
<body>
<nav><div class="wrap"><div class="row">
  <span class="wordmark">codehalter</span>
  <span class="spacer"></span>
  <span class="links"><a href="index.html#skill-crafter">about</a> <a href="index.html">← home</a></span>
</div></div></nav>

<div class="wrap">
  <header class="hero">
    <div class="eyebrow">skill crafter</div>
    <h1>Different models need different skills.</h1>
    <p class="lede">A coding SKILL file is a list of behavioral statements. Some are load-bearing for a
      given model; others tell it to do what it already does. <b>Skill crafter probes each statement per
      model</b> and keeps only the ones that change behavior.</p>
    <p class="lede">For every atomic claim, the judge model (<code>{{.JudgeModel}}</code>) authors a
      disguised test question and a rubric. The target model answers it <b>with and without</b> the claim,
      {{.Samples}}× each; the judge decides whether the claim mattered. <b>keep</b> = it did, <b>drop</b> =
      the model already behaves that way.</p>
  </header>

  <section class="section">
    <h2 class="sec-title">Results by model</h2>
    <p class="sec-sub">Pruned skill = original minus every dropped statement. Errored claims are kept
      (untested, never removed).</p>

    {{range .Models}}
    <div class="mcard">
      <div class="mhead">
        <h3>{{.Model}}</h3>
        <span class="tag">{{.Stats.OrigBytes}} → {{.Stats.PrunedBytes}} bytes</span>
      </div>
      <div class="stats">
        <div class="stat"><div class="num">{{.SavedPct}}</div><div class="lbl">size change</div></div>
        <div class="stat"><div class="num dim">{{.KeptTotal}}</div><div class="lbl">kept</div></div>
        <div class="stat"><div class="num dim">{{.DropTotal}}</div><div class="lbl">dropped</div></div>
        <div class="stat"><div class="num dim">{{.ErrTotal}}</div><div class="lbl">errored</div></div>
      </div>
      <table>
        <thead><tr><th>skill</th><th class="num">claims</th><th class="num">kept</th><th class="num">dropped</th><th class="num">errored</th><th class="num">bytes</th></tr></thead>
        <tbody>
        {{range .Stats.Skills}}
          <tr><td><code>SKILL-{{.Skill}}</code></td><td class="num">{{.Claims}}</td><td class="num">{{.Kept}}</td><td class="num">{{.Dropped}}</td><td class="num">{{.Errored}}</td><td class="num">{{.OrigBytes}} → {{.PrunedBytes}}</td></tr>
        {{end}}
        </tbody>
      </table>
      {{if .Examples}}
      <div style="margin-top:22px;">
        {{range .Examples}}
        <div class="ex">
          <div class="claim"><span class="pill {{.Verdict}}">{{upper .Verdict}}</span> {{.Text}}</div>
          <div class="reason">{{.Reason}} <span class="cid">· {{.ClaimID}}</span></div>
        </div>
        {{end}}
      </div>
      {{end}}
    </div>
    {{end}}
  </section>

  <footer>Generated by <code>crafter</code> · codehalter skill crafter.</footer>
</div>
</body>
</html>
`
