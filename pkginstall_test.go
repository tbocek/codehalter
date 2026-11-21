package main

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func osInfoWith(id, idLike string) osInfo {
	return osInfo{Fields: map[string]string{"ID": id, "ID_LIKE": idLike}}
}

func TestDistroFamilies(t *testing.T) {
	cases := []struct {
		id, idLike string
		want       []string
	}{
		{"arch", "", []string{"arch"}},
		{"ubuntu", "debian", []string{"debian"}},
		{"linuxmint", "ubuntu debian", []string{"debian"}},
		{"fedora", "", []string{"fedora"}},
		{"centos", "rhel fedora", []string{"fedora"}},
		{"alpine", "", []string{"alpine"}},
		{"opensuse-leap", "suse", []string{"suse"}},
		{"wasmos", "", nil},
	}
	for _, c := range cases {
		got := distroFamilies(osInfoWith(c.id, c.idLike))
		if strings.Join(got, ",") != strings.Join(c.want, ",") {
			t.Errorf("ID=%s ID_LIKE=%q: got %v, want %v", c.id, c.idLike, got, c.want)
		}
	}
}

func TestPkgInstallCmd(t *testing.T) {
	cases := []struct {
		family string
		want   string
	}{
		{"arch", "sudo pacman -S --noconfirm --needed shfmt"},
		{"debian", "sudo apt-get install -y --no-install-recommends shfmt"},
		{"fedora", "sudo dnf -y install shfmt"},
		{"alpine", "sudo apk add shfmt"},
		{"suse", "sudo zypper -n install shfmt"},
	}
	for _, c := range cases {
		argv, ok := pkgInstallCmd([]string{c.family}, "shfmt")
		if !ok || strings.Join(argv, " ") != c.want {
			t.Errorf("%s: got %v (%v), want %q", c.family, argv, ok, c.want)
		}
	}
	if _, ok := pkgInstallCmd(nil, "shfmt"); ok {
		t.Error("an unknown distro must not produce an install command")
	}
}

// The package name differs per family for the same binary.
func TestSyntaxToolPackagePerFamily(t *testing.T) {
	for bin, want := range map[string][4]string{
		"clang-format": {"clang", "clang-format", "clang", "clang"},
		"go":           {"go", "golang-go", "golang", "go"},
		"rustfmt":      {"rust", "rustfmt", "rustfmt", "rust"},
	} {
		tool, ok := syntaxToolPackageFor(bin)
		if !ok {
			t.Fatalf("no package entry for %s", bin)
		}
		for i, fam := range []string{"arch", "debian", "fedora", "alpine"} {
			pkg, ok := tool.packageName([]string{fam})
			if !ok || pkg != want[i] {
				t.Errorf("%s on %s: got %q (%v), want %q", bin, fam, pkg, ok, want[i])
			}
		}
	}
}

const archImage = `FROM quay.io/archlinux/archlinux:base-devel

RUN pacman -Syu --noconfirm --needed \
        ca-certificates \
        curl \
        go \
    && rm -rf /var/cache/pacman/pkg/*

USER dev
`

func TestEnsureDockerfileLine(t *testing.T) {
	t.Run("already listed changes nothing", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "Dockerfile")
		if err := os.WriteFile(path, []byte(archImage), 0o644); err != nil {
			t.Fatal(err)
		}
		note, err := ensureDockerfileLine(path, []string{"arch"}, "go", "gofmt")
		if err != nil {
			t.Fatal(err)
		}
		data, _ := os.ReadFile(path)
		if !strings.Contains(note, "already listed") || string(data) != archImage {
			t.Errorf("got %q; file changed:\n%s", note, data)
		}
	})

	t.Run("inserts into the existing list", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "Dockerfile")
		if err := os.WriteFile(path, []byte(archImage), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := ensureDockerfileLine(path, []string{"arch"}, "shfmt", "shfmt"); err != nil {
			t.Fatal(err)
		}
		data, _ := os.ReadFile(path)
		want := "        go \\\n        shfmt \\\n    && rm -rf /var/cache/pacman/pkg/*"
		if !strings.Contains(string(data), want) {
			t.Errorf("the package was not added to the list:\n%s", data)
		}
		if strings.Contains(string(data), "USER dev\n\nRUN") {
			t.Error("a new layer was appended instead of extending the list")
		}
		if packageListed(archImage, "golang-go") {
			t.Error("a word inside another package name must not match")
		}
	})

	t.Run("appends a layer when the base uses no known list", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "Dockerfile")
		base := "FROM scratch\nRUN echo hi\n"
		if err := os.WriteFile(path, []byte(base), 0o644); err != nil {
			t.Fatal(err)
		}
		note, err := ensureDockerfileLine(path, []string{"debian"}, "python3-ruff", "ruff")
		if err != nil {
			t.Fatal(err)
		}
		data, _ := os.ReadFile(path)
		if !strings.HasPrefix(string(data), base) || !strings.Contains(string(data), "apt-get install -y --no-install-recommends") {
			t.Errorf("want the original kept plus a new layer, got %q:\n%s", note, data)
		}
		if !strings.Contains(string(data), "# codehalter: syntax check for ruff") {
			t.Errorf("the new layer should say why it is there:\n%s", data)
		}
	})

	t.Run("one line added to this repo's own image copy", func(t *testing.T) {
		real, err := os.ReadFile(filepath.Join(".devcontainer", "Dockerfile"))
		if err != nil {
			t.Skip("no .devcontainer/Dockerfile here")
		}
		path := filepath.Join(t.TempDir(), "Dockerfile")
		if err := os.WriteFile(path, real, 0o644); err != nil {
			t.Fatal(err)
		}
		if note, err := ensureDockerfileLine(path, []string{"arch"}, "shfmt", "shfmt"); err != nil || !strings.Contains(note, "already listed") {
			t.Fatalf("shfmt is already in the image: %q, %v", note, err)
		}
		before, _ := os.ReadFile(path)
		if _, err := ensureDockerfileLine(path, []string{"arch"}, "rust", "rustfmt"); err != nil {
			t.Fatal(err)
		}
		after, _ := os.ReadFile(path)
		bl, al := strings.Split(string(before), "\n"), strings.Split(string(after), "\n")
		if len(al) != len(bl)+1 {
			t.Fatalf("want exactly one line added: %d before, %d after", len(bl), len(al))
		}
		// One inserted line shifts everything after it: the only real difference is where
		// the new line appears, every other line must still be there in order.
		var changed []string
		for i, j := 0, 0; i < len(bl); i, j = i+1, j+1 {
			if bl[i] == al[j] {
				continue
			}
			changed = append(changed, al[j])
			j++
			if bl[i] != al[j] {
				t.Errorf("line %d changed from %q to %q", i, bl[i], al[j])
			}
		}
		if len(changed) != 1 || !strings.Contains(changed[0], "rust") {
			t.Errorf("want exactly one new line naming rust, got %q", changed)
		}
		t.Logf("the only added line: %q", changed[0])
	})

	t.Run("unknown distro is refused, not guessed", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "Dockerfile")
		if err := os.WriteFile(path, []byte(archImage), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := ensureDockerfileLine(path, nil, "shfmt", "shfmt"); err == nil {
			t.Error("want an error for a distro with no package manager")
		}
	})
}

// A notice is said, not offered: no card, and above all no prompt turn behind it.
// A bare test agent has no transport, so any offer would fail on the connection
// rather than stay quiet; reaching that failure is what this test rules out.
func TestProposeFixNoticeOnlySays(t *testing.T) {
	a, s := newTestAgent(t)
	before := s.messageCount()
	a.proposeFix(context.Background(), s.ID, fixProblem{desc: "🟡 nothing to accept", notice: true})
	if got := s.messageCount(); got != before {
		t.Errorf("a notice added %d messages, want none", got-before)
	}
}

func TestSyntaxToolProblemWithoutAnyoneToAsk(t *testing.T) {
	a, s := newTestAgent(t)
	gaps := []syntaxGap{{bin: "rustfmt", langs: []string{"Rust"}, pkg: "rust", cmd: []string{"sudo", "pacman", "-S", "--noconfirm", "--needed", "rust"}}}
	p := syntaxToolProblem(a, s.Cwd, gaps)
	if !p.notice || p.install != nil || p.prompt != "" {
		t.Errorf("with no elicitation and no terminal the problem must be a plain notice: %+v", p)
	}
	if !strings.Contains(p.desc, "rustfmt") || !strings.Contains(p.desc, "pacman") {
		t.Errorf("the notice must name the tool and the command: %q", p.desc)
	}
}

// With someone to ask, the problem is a question whose accepted branch installs; the
// card itself defaults to No (askNoYesWithCard), so only an explicit yes installs.
func TestSyntaxToolProblemAsksWhenSomeoneIsThere(t *testing.T) {
	a, s := newTestAgent(t)
	a.clientCaps.Elicitation = &struct {
		Form *struct{} `json:"form"`
		URL  *struct{} `json:"url"`
	}{Form: &struct{}{}}
	gaps := []syntaxGap{{bin: "rustfmt", langs: []string{"Rust"}, pkg: "rust", cmd: []string{"sudo", "pacman", "-S", "--noconfirm", "--needed", "rust"}}}
	p := syntaxToolProblem(a, s.Cwd, gaps)
	if p.notice || p.install == nil {
		t.Errorf("with elicitation the card must be answerable and install on yes: %+v", p)
	}
	if !strings.Contains(p.desc, "install rustfmt") {
		t.Errorf("the card must name what it would install: %q", p.desc)
	}
}

func writeTestFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestDetectSyntaxGaps(t *testing.T) {
	t.Run("a tree of languages every base image can already parse asks for nothing", func(t *testing.T) {
		dir := t.TempDir()
		writeTestFile(t, filepath.Join(dir, "main.go"), "package main\n")
		writeTestFile(t, filepath.Join(dir, "a.py"), "x = 1\n")
		writeTestFile(t, filepath.Join(dir, "run.sh"), "echo hi\n")
		if gaps := detectSyntaxGaps(detectStacks(dir), dir); len(gaps) != 0 {
			t.Errorf("go, python3 and bash are in every base image, so nothing is missing: %+v", gaps)
		}
	})

	t.Run("a Go project without gofmt names the toolchain, not gofmt", func(t *testing.T) {
		dir := t.TempDir()
		writeTestFile(t, filepath.Join(dir, "main.go"), "package main\n")
		t.Setenv("PATH", dir) // nothing on PATH: every probe is missing
		gaps := detectSyntaxGaps([]string{"go"}, dir)
		if len(gaps) != 1 || gaps[0].bin != "gofmt" {
			t.Fatalf("want one gap for gofmt, got %+v", gaps)
		}
	})

	t.Run("go project with gofmt present has no gap", func(t *testing.T) {
		if _, err := exec.LookPath("gofmt"); err != nil {
			t.Skip("gofmt not available")
		}
		dir := t.TempDir()
		writeTestFile(t, filepath.Join(dir, "main.go"), "package main\n")
		if gaps := detectSyntaxGaps([]string{"go"}, dir); len(gaps) != 0 {
			t.Errorf("gofmt answers, so there is no gap, got %+v", gaps)
		}
	})

	t.Run("a tree with no source files asks for nothing", func(t *testing.T) {
		if gaps := detectSyntaxGaps(nil, t.TempDir()); len(gaps) != 0 {
			t.Errorf("empty tree produced gaps: %+v", gaps)
		}
	})

	t.Run("every gap names an installable command", func(t *testing.T) {
		dir := t.TempDir()
		writeTestFile(t, filepath.Join(dir, "x.rs"), "fn main() {}\n")
		writeTestFile(t, filepath.Join(dir, "y.sh"), "echo hi\n")
		for _, g := range detectSyntaxGaps([]string{"rust"}, dir) {
			if g.pkg == "" || len(g.cmd) == 0 {
				t.Errorf("gap %+v has no package or command", g)
			}
		}
	})
}
