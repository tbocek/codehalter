package main

import (
	"strings"
	"testing"
)

// Not 2>&1, /dev/null, variables or quoted names.
func TestRedirectTargets(t *testing.T) {
	got := redirectTargets(`cd rust && (just test > /tmp/a.log 2>&1; echo "exit=$?" >> /tmp/a.log); cargo check 2> err.txt &> all.txt; x > $OUT; y > "q q"; z > /dev/null`, "/w")
	want := []string{"/tmp/a.log", "/tmp/a.log", "/w/err.txt", "/w/all.txt"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("targets = %v, want %v", got, want)
	}
}

// A here-document's body is another program's text: the Python `while step>=1:`
// that made a job "write /workspaces/compress/=1" and refused the next one.
func TestRedirectTargetsSkipHeredocBodies(t *testing.T) {
	for _, tc := range []struct{ cmd, want string }{
		{"cd /w && python3 - <<'EOF' > /tmp/loc14.log 2>&1\nwhile step>=1:\n    x = a > b\nEOF\ncat /tmp/loc14.log > /tmp/copy.log", "/tmp/loc14.log,/tmp/copy.log"},
		{"cat << \"eof\" > a.txt\nx > b.txt\neof", "/w/a.txt"},
		{"cat <<-END >> a.txt\n\tx > b.txt\n\tEND\necho y > c.txt", "/w/a.txt,/w/c.txt"},
		{"cat <<A <<B > a.txt\nx > b\nA\ny > c\nB\necho > d.txt", "/w/a.txt,/w/d.txt"},
		{"cat <<EOF > a.txt\nnever closed > b.txt", "/w/a.txt"},
		// No here-documents: a here-string, shifts.
		{"grep x <<< \"$v\" > a.txt\necho > b.txt", "/w/a.txt,/w/b.txt"},
		{"python3 -c \"print(1 << shift)\" > a.txt\necho $((1<<3)) > b.txt", "/w/a.txt,/w/b.txt"},
	} {
		if got := strings.Join(redirectTargets(tc.cmd, "/w"), ","); got != tc.want {
			t.Errorf("%q: targets = %s, want %s", tc.cmd, got, tc.want)
		}
	}
}

// tee writes its files as a redirect does.
func TestRedirectTargetsTee(t *testing.T) {
	got := strings.Join(redirectTargets("just test 2>&1 | tee -a /tmp/t.log out.txt | grep FAIL; echo tee > x.txt; ls | tee /dev/stderr", "/w"), ",")
	if want := "/w/x.txt,/tmp/t.log,/w/out.txt"; got != want {
		t.Errorf("targets = %s, want %s", got, want)
	}
}
