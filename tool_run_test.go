package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

// The exit code leads the result (non-zero is data) and the card keeps the live
// terminal rather than a text copy.
func TestRunCommandEndToEnd(t *testing.T) {
	h := newTerminalHarness(t)

	result, failed := runCmdExecute(context.Background(), h.agent, h.sess.ID, `{"command":"echo hi; exit 2"}`)
	if failed {
		t.Fatalf("failed = true, want false (a non-zero exit must not fail the turn)")
	}
	if !strings.HasPrefix(result, "exit 2\n") {
		t.Errorf("result = %q, want it to start with the exit code", result)
	}
	if !strings.Contains(result, "hi") {
		t.Errorf("result = %q, want the command output", result)
	}

	done := h.waitForStatus("completed")
	if done == nil {
		t.Fatal("the tool call was never completed")
	}
	if _, ok := done["content"]; ok {
		t.Errorf("completing update carries content %v, want it omitted so the terminal stays visible", done["content"])
	}
	if done["title"] != "Run: echo hi; exit 2 (exit 2)" {
		t.Errorf("title = %v, want the exit code in it", done["title"])
	}
	if !h.embeddedTerminal() {
		t.Error("no update embedded the terminal; the user would see an empty card")
	}
}

// terminal/create takes an argv, so run_command must wrap the line in bash -c.
func TestRunCommandIsAShellLine(t *testing.T) {
	h := newTerminalHarness(t)

	result, _ := runCmdExecute(context.Background(), h.agent, h.sess.ID,
		`{"command":"echo one && echo two | tr a-z A-Z"}`)
	if !strings.Contains(result, "one") || !strings.Contains(result, "TWO") {
		t.Errorf("result = %q, want both the && and the pipe to have run", result)
	}
}

// Output past cmdOutputCap comes back as head+tail with a marker, the command
// still running to completion.
func TestRunCommandCapsHugeOutput(t *testing.T) {
	h := newTerminalHarness(t)
	saved := cmdOutputCap
	cmdOutputCap = 4096
	t.Cleanup(func() { cmdOutputCap = saved })

	result, failed := runCmdExecute(context.Background(), h.agent, h.sess.ID, `{"command":"seq 1 20000"}`)
	if failed {
		t.Fatalf("failed=true: %.100s", result)
	}
	if !strings.HasPrefix(result, "exit 0") {
		t.Errorf("missing exit header: %.80s", result)
	}
	if !strings.Contains(result, "bytes omitted") {
		t.Errorf("over-cap output should carry an elision marker: %.200s", result)
	}
	if len(result) > cmdOutputCap+1024 {
		t.Errorf("captured output not bounded: %d bytes (cap %d)", len(result), cmdOutputCap)
	}
	if !strings.Contains(result, "\n1\n") {
		t.Error("head lost: want early line 1")
	}
	if !strings.Contains(result, "\n20000\n") {
		t.Error("tail lost: want final line 20000")
	}
}

// Both executors refuse an empty command; separators alone once crashed the job classifier.
func TestRunCommandRequiresCommand(t *testing.T) {
	h := newTerminalHarness(t)
	for name, run := range map[string]func(context.Context, *agent, string, string) (string, bool){"run_command": runCmdExecute, "run_background": runBackgroundExecute} {
		for _, args := range []string{`{}`, `{"command":" ; "}`, `{"command":"\n"}`} {
			if res, failed := run(context.Background(), h.agent, h.sess.ID, args); failed || !strings.Contains(res, "command is required") {
				t.Errorf("%s %s: %s (failed=%v), want command-required", name, args, res, failed)
			}
		}
	}
}

// A respond parks for a finite run, judged by the last command's tool and
// subcommand; a word in a path or a flag says nothing.
func TestFiniteRunCommands(t *testing.T) {
	for cmd, want := range map[string]bool{
		"cd rust && just test > /tmp/gate.log 2>&1":     true,
		"cd rust && cargo test 2>&1 | tee /tmp/t.log":   true,
		"xvfb-run -a cargo test":                        true,
		"npm run test":                                  true,
		"python -m pytest -q":                           true,
		"docker compose up --build":                     false,
		"cmake --build build && ./build/app":            false,
		"node build":                                    false,
		"java -jar build/libs/app.jar":                  false,
		"jest --watchAll":                               false,
		"cargo build --release && ./target/release/api": false,
		"npm run build && npm run preview":              false,
	} {
		cmds := shellSegments(cmd, false)
		last := cmds[len(cmds)-1]
		if got := finiteRunRe.MatchString(last) && !serverRunRe.MatchString(last); got != want {
			t.Errorf("%q: finite = %v, want %v", cmd, got, want)
		}
	}
}

// With no job running a sleep is ordinary, and a sleep inside other work is not
// a wait.
func TestRunCommandRefusesSleepForBackgroundJob(t *testing.T) {
	h := newTerminalHarness(t)
	res, failed := runCmdExecute(context.Background(), h.agent, h.sess.ID, `{"command":"sleep 0.01"}`)
	if failed || strings.Contains(res, "refused") {
		t.Fatalf("sleep with no job running: %s (failed=%v)", res, failed)
	}
	h.agent.registerBgJob(&backgroundJob{id: 7, sid: h.sess.ID, cmdStr: "cd rust && just test > /tmp/t.log 2>&1"})
	for _, cmd := range []string{"sleep 120", "sleep 90 && tail -20 /tmp/t.log", "cd /workspaces/x; sleep 60", "sleep"} {
		res, failed = runCmdExecute(context.Background(), h.agent, h.sess.ID, `{"command":`+strconv.Quote(cmd)+`}`)
		if !strings.Contains(res, "refused") || !strings.Contains(res, "job 7") {
			t.Errorf("%q with job 7 running: %s (failed=%v)", cmd, res, failed)
		}
	}
	for _, cmd := range []string{"tail -20 /tmp/t.log", "for i in 1; do sleep 0.01; done", "echo hi; sleep 0.01"} {
		res, failed = runCmdExecute(context.Background(), h.agent, h.sess.ID, `{"command":`+strconv.Quote(cmd)+`}`)
		if failed || strings.Contains(res, "refused") {
			t.Errorf("%q must run: %s (failed=%v)", cmd, res, failed)
		}
	}
}

// A SIGKILL comes back as 137 through pipefail, never as a zero-value 0.
func TestRunCommandSignalExit(t *testing.T) {
	h := newTerminalHarness(t)
	result, _ := runCmdExecute(context.Background(), h.agent, h.sess.ID, `{"command":"kill -9 $$"}`)
	if !strings.HasPrefix(result, "exit 137\n") {
		t.Errorf("result = %q, want exit 137 for a signal death", result)
	}
	if !h.sawMethod("terminal/release") {
		t.Errorf("terminal was never released; got %v", h.sentMethods())
	}
}

// The command is also killed and leaves no job behind.
func TestRunCommandCancelReportsPartialOutput(t *testing.T) {
	h := newTerminalHarness(t)
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(300 * time.Millisecond)
		cancel()
	}()
	result, _ := runCmdExecute(ctx, h.agent, h.sess.ID, `{"command":"echo early; sleep 30"}`)
	if !strings.Contains(result, "early") || !strings.Contains(result, "terminal error") {
		t.Errorf("result = %q, want the output produced before the cancel and the error", result)
	}
	if left := h.agent.runningBgJobs(h.sess.ID, nil); left != "" {
		t.Errorf("a cancelled command stayed tracked: %s", left)
	}
}

// The command is not killed; its exit later arrives as a note with the code and
// log tail, as for run_background.
func TestRunCommandHandsOverAfterWait(t *testing.T) {
	h := newTerminalHarness(t)
	defer h.agent.shutdownBackground()
	old := cmdHandoverWait
	cmdHandoverWait = 300 * time.Millisecond
	defer func() { cmdHandoverWait = old }()
	h.sess.ctl.held.Lock() // a turn is running, so the note queues instead of starting a turn
	defer h.sess.ctl.held.Unlock()

	result, failed := runCmdExecute(context.Background(), h.agent, h.sess.ID, `{"command":"echo started; sleep 1; echo finished; exit 3","wake_after":60}`)
	if failed {
		t.Fatal("a handover must not fail the turn")
	}
	for _, want := range []string{"still running after", "background job 1", "nothing was killed", "started", "parks the turn", "woken after 1m"} {
		if !strings.Contains(result, want) {
			t.Errorf("handover result lacks %q: %q", want, result)
		}
	}
	if jobs := h.agent.parkableJobs(h.sess.ID, time.Time{}); !strings.Contains(jobs, "job 1") {
		t.Errorf("the handed-over command is not a parkable job: %q", jobs)
	}
	if h.sawMethod("terminal/kill") {
		t.Error("the command was killed at the handover")
	}

	deadline := time.Now().Add(5 * time.Second)
	for !h.sess.hasPending() {
		if time.Now().After(deadline) {
			t.Fatal("the job never reported its exit")
		}
		time.Sleep(10 * time.Millisecond)
	}
	note := h.sess.takePending()[0].note
	logName := "codehalter-" + strconv.Itoa(os.Getpid()) + "-job-1.log"
	for _, want := range []string{"exited with code 3", "finished", logName} {
		if !strings.Contains(note.full, want) {
			t.Errorf("exit note lacks %q: %q", want, note.full)
		}
	}
	if jobs := h.agent.parkableJobs(h.sess.ID, time.Time{}); jobs != "" {
		t.Errorf("finished job still parkable: %s", jobs)
	}
	if b, err := os.ReadFile(filepath.Join(os.TempDir(), logName)); err != nil || !strings.Contains(string(b), "finished") {
		t.Errorf("the log the model is pointed at is not there or incomplete: %v %q", err, b)
	}
}

// A command writing into a redirect file is never killed, however quiet its
// terminal.
func TestRunCommandStallKillsHungJob(t *testing.T) {
	h := newTerminalHarness(t)
	defer h.agent.shutdownBackground()
	oldWait, oldStall, oldPoll := cmdHandoverWait, bgStallTimeout, bgStallPoll
	cmdHandoverWait, bgStallTimeout, bgStallPoll = 100*time.Millisecond, 400*time.Millisecond, 50*time.Millisecond
	defer func() { cmdHandoverWait, bgStallTimeout, bgStallPoll = oldWait, oldStall, oldPoll }()
	h.sess.ctl.held.Lock()
	defer h.sess.ctl.held.Unlock()
	waitNote := func() bgNote {
		deadline := time.Now().Add(5 * time.Second)
		for !h.sess.hasPending() {
			if time.Now().After(deadline) {
				t.Fatal("no note")
			}
			time.Sleep(10 * time.Millisecond)
		}
		return *h.sess.takePending()[0].note
	}

	runCmdExecute(context.Background(), h.agent, h.sess.ID, `{"command":"echo start; sleep 30"}`)
	if n := waitNote(); !strings.Contains(n.full, "killed after") || !strings.Contains(n.full, "without any output") {
		t.Errorf("hung command's note: %q", n.full)
	}

	log := filepath.Join(t.TempDir(), "suite.log")
	runCmdExecute(context.Background(), h.agent, h.sess.ID, `{"command":"(for i in 1 2 3 4 5 6 7 8 9 10; do echo tick $i; sleep 0.1; done) > `+log+` 2>&1; echo exit=$? >> `+log+`"}`)
	if n := waitNote(); !strings.Contains(n.full, "exited with code 0") {
		t.Errorf("a suite writing into its file was not left alone: %q", n.full)
	}
}

// Scripts that compute or generate, reads and builds get no note.
func TestToolHints(t *testing.T) {
	edit := "cd rust && python3 - <<'PY'\np='tests/zoom_widgets.rs'\ns=open(p).read()\nopen(p,'w').write(s.replace('a','b'))\nPY"
	if target, ok := scriptEditTarget(edit); !ok || target != "tests/zoom_widgets.rs" {
		t.Errorf("heredoc edit = %q %v", target, ok)
	}
	for _, cmd := range []string{
		"python3 - <<'PY'\nprint(open('src/a.rs').read().count('fn '))\nPY",
		"python3 - <<'PY'\nimport json\njson.dump({'a': 1}, open('fixtures/demo.toml','w'))\nPY",
		"cargo test",
		"cd rust && sed -n '1,5p' src/a.rs; grep -n 'fn a' -A 8 src/a.rs",
	} {
		if note, _ := toolHints(cmd); note != "" {
			t.Errorf("%q got a note: %q", cmd, note)
		}
	}
	first, told := toolHints(edit)
	if !strings.Contains(first, `Its replace is exactly this edit_file call: {"path": "tests/zoom_widgets.rs", "old_text": "a", "new_text": "b"}`) || !strings.Contains(told, "edit_file instead of a script on tests/zoom_widgets.rs") {
		t.Errorf("hint = %q / %q", first, told)
	}
	multi := "python3 - <<'PY'\np='src/ui/window.rs'\ns=open(p).read()\ns=s.replace(\"\"\"let zoom = 1.0;\"\"\", \"\"\"let zoom = ZOOM;\"\"\")\ns=s.replace('old()', 'new()')\nopen(p,'w').write(s)\nPY"
	if got := scriptEditPreview(multi, "src/ui/window.rs"); !strings.Contains(got, "Its 2 replaces are ONE edit_file call with an edits list") {
		t.Errorf("multi preview = %q", got)
	}
	splice := "python3 - <<'PY'\np='src/a.rs'\ns=open(p).read()\ni=s.index('fn a(')\nopen(p,'w').write(s[:i])\nPY"
	if got := scriptEditPreview(splice, "src/a.rs"); !strings.Contains(got, `"start": "<fragment of the block's first line>"`) {
		t.Errorf("splice preview = %q", got)
	}
}

func TestRunCommandNamesTheToolItWasMeantFor(t *testing.T) {
	h := newTerminalHarness(t)
	res, _ := runCmdExecute(context.Background(), h.agent, h.sess.ID, `{"reads":[{"path":"a.rs","line":1,"limit":5}]}`)
	if !strings.Contains(res, "Call read_file with them") {
		t.Errorf("reads sent to run_command = %q", res)
	}
	res, _ = runCmdExecute(context.Background(), h.agent, h.sess.ID, `{"path":"a.rs","old_text":"x","new_text":"y"}`)
	if !strings.Contains(res, "Call edit_file with them") {
		t.Errorf("an edit sent to run_command = %q", res)
	}
}

// A plain sleep, a plain background start and ordinary lines get nothing.
func TestBgSleepHint(t *testing.T) {
	cmd := `ps aux | grep -c x; cd rust && (xvfb-run -a cargo test --test deco > /tmp/d.log 2>&1; echo "exit=$?" >> /tmp/d.log) & sleep 45; tail -6 /tmp/d.log`
	note, told := bgSleepHint(cmd)
	if !strings.Contains(note, "slept 45 s") || !strings.Contains(note, `xvfb-run -a cargo test --test deco`) || !strings.Contains(told, "run_background") {
		t.Errorf("note = %q / %q", note, told)
	}
	if note, _ := bgSleepHint("cargo build & sleep 10"); !strings.Contains(note, "<the command you put in the background>") {
		t.Errorf("unparenthesised = %q", note)
	}
	for _, c := range []string{"sleep 3", "npm run dev &", "cargo test 2>&1 | tail", "echo a && sleep 1"} {
		if note, _ := bgSleepHint(c); note != "" {
			t.Errorf("%q got a note: %q", c, note)
		}
	}
}

// A read of files, a search or a print is served whole; anything that writes,
// runs or hides a command is not.
func TestOnlyReads(t *testing.T) {
	for cmd, want := range map[string]bool{
		`cd rust && awk 'NR>=90 && NR<=200 {printf "%d|%s\n", NR, $0}' src/a.rs`: true,
		`sed -n '10,40p' src/a.rs; echo "=== b ==="; sed -n '1,9p' src/b.rs`:     true,
		`grep -rn -C3 -F 'fn run' src tests | head -50`:                          true,
		`cat a.rs 2>/dev/null || echo missing`:                                   true,
		`git --no-pager log -3 --oneline && git -C rust show HEAD:src/a.rs`:      true,
		`find src -name '*.rs' | sort`:                                           true,
		`echo hi`:                                                                false,
		`sed -i 's/a/b/' src/a.rs`:                                               false,
		`cat a.rs > /tmp/copy`:                                                   false,
		`cargo test 2>&1 | tail -20`:                                             false,
		`grep -n x $(git ls-files)`:                                              false,
		`python3 - <<'EOF'` + "\nprint(1)\nEOF":                                  false,
		`git checkout src/a.rs`:                                                  false,
		`find . -name '*.tmp' -delete`:                                           false,
		`awk 'BEGIN { system("rm -rf x") }' a.rs`:                                false,
		`sed -Ei 's/a/b/' src/a.rs`:                                              false,
		`sed --in-place=.bak 's/a/b/' src/a.rs`:                                  false,
		`awk '{print > "out.txt"}' a.rs`:                                         false,
		`find . -name x -fprint list`:                                            false,
		`sort -o sorted.txt a.txt`:                                               false,
		`git diff --output=d.patch`:                                              false,
	} {
		if got := onlyReads(cmd); got != want {
			t.Errorf("onlyReads(%q) = %v, want %v", cmd, got, want)
		}
	}
}

// A shell read is served like read_file, whole up to its cap; a build is still
// clipped to head and tail.
func TestLiveToolOutputServesShellReadsWhole(t *testing.T) {
	body := strings.Repeat("let x = 1;\n", 400)
	read := `{"command":"sed -n '1,400p' src/a.rs"}`
	if got := liveToolOutput("run_command", read, "exit 0\n\n"+body); !strings.HasSuffix(got, body) {
		t.Errorf("a shell read was clipped: %d of %d chars", len(got), len(body))
	}
	build := `{"command":"cargo build"}`
	if got := liveToolOutput("run_command", build, "exit 0\n\n"+body); !strings.Contains(got, "chars omitted") {
		t.Error("a build's long output was not clipped")
	}
	// Past the allowance a read keeps its end too: a failing suite's log ends in its summary.
	log := strings.Repeat("running a test\n", 4000) + "test result: FAILED. 3 failed\n"
	got := liveToolOutput("run_command", `{"command":"cat /tmp/job.log"}`, log)
	if len(got) > liveExemptCap+1024 || !strings.HasSuffix(got, "3 failed\n") || !strings.HasPrefix(got, "running a test") {
		t.Errorf("an oversized shell read: %d chars, start or end lost", len(got))
	}
}

// An inline script or an image tool on a picture is refused with the way to
// look at it; a project's own script and a plain listing are not.
func TestRunCommandRefusesPictureScripts(t *testing.T) {
	h := newTerminalHarness(t)
	for _, cmd := range []string{
		"cd rust && python3 -c \"from PIL import Image\nim = Image.open('shots/08-produce.png')\"",
		"python3 - <<'PY'\nfrom PIL import Image\nImage.open('a.png').crop((0,0,9,9)).save('/tmp/c.png')\nPY",
		"convert shots/a.png -crop 800x400+0+0 /tmp/b.png",
	} {
		b, _ := json.Marshal(map[string]string{"command": cmd})
		res, _ := runCmdExecute(context.Background(), h.agent, h.sess.ID, string(b))
		if !strings.HasPrefix(res, "refused:") || !strings.Contains(res, `"region"`) {
			t.Errorf("%q: %q, want the refusal naming screenshot's region", cmd, res)
		}
	}
	for _, cmd := range []string{"ls -l shots/a.png", "python3 tools/icons.py assets/a.png"} {
		b, _ := json.Marshal(map[string]string{"command": cmd})
		if res, _ := runCmdExecute(context.Background(), h.agent, h.sess.ID, string(b)); strings.HasPrefix(res, "refused:") {
			t.Errorf("%q was refused: %q", cmd, res)
		}
	}
}

// What a command leaves running is swept when it exits: two green suites each
// left an Xvfb behind, and the next `xvfb-run -a` took the next display number.
func TestRunCommandSweepsWhatItLeftBehind(t *testing.T) {
	h := newTerminalHarness(t)
	defer h.agent.shutdownBackground()
	log := filepath.Join(t.TempDir(), "child.pid")
	out, _ := runCmdExecute(context.Background(), h.agent, h.sess.ID, `{"command":"sleep 30 & echo $! > `+log+`; echo left a child"}`)
	if !strings.HasPrefix(out, "exit 0\n") {
		t.Fatalf("command: %q", out)
	}
	pidText, err := os.ReadFile(log)
	if err != nil {
		t.Fatal(err)
	}
	pid, _ := strconv.Atoi(strings.TrimSpace(string(pidText)))
	if pid <= 0 {
		t.Fatalf("child pid %q", pidText)
	}
	deadline := time.Now().Add(5 * time.Second)
	for syscall.Kill(pid, 0) == nil {
		if time.Now().After(deadline) {
			_ = syscall.Kill(pid, syscall.SIGKILL)
			t.Fatalf("the child (pid %d) survived the command's exit", pid)
		}
		time.Sleep(50 * time.Millisecond)
	}
}
