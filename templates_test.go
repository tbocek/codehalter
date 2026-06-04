package main

import (
	"strings"
	"testing"
)

func TestRenderMacro(t *testing.T) {
	// {{}} + args → substituted.
	if got, msg := renderMacro("grill", "do {{}} now", "the thing"); got != "do the thing now" || msg != "" {
		t.Errorf("substitute: got %q msg %q", got, msg)
	}
	// {{}} + no args → user-facing stop message, nothing rendered.
	if got, msg := renderMacro("grill", "do {{}} now", "   "); got != "" || msg == "" {
		t.Errorf("missing-arg: got %q msg %q (want empty render + a message)", got, msg)
	}
	// no {{}} + args → appended.
	if got, _ := renderMacro("grill", "fixed body", "extra"); got != "fixed body\n\nextra" {
		t.Errorf("append: got %q", got)
	}
	// no {{}} + no args → as-is.
	if got, _ := renderMacro("grill", "fixed body", ""); got != "fixed body" {
		t.Errorf("as-is: got %q", got)
	}
}

func TestExpandMacroNonCommand(t *testing.T) {
	for _, s := range []string{"hello world", "/nope-not-a-template here", "", "no slash here"} {
		if _, _, handled := expandMacro(s); handled {
			t.Errorf("expandMacro(%q) handled=true, want false", s)
		}
	}
}

func TestTemplateNamesIncludesGrillMe(t *testing.T) {
	if !contains(templateNames(), "grill-me") {
		t.Errorf("templateNames() = %v, want it to include grill-me (docs/TEMPLATE-grill-me.md)", templateNames())
	}
}

// expandMacro on the real /grill-me template: it carries a {{}}, so no args
// must stop with a message and args must land in the rendered prompt.
func TestExpandMacroGrillMe(t *testing.T) {
	if _, stopMsg, handled := expandMacro("/grill-me"); !handled || stopMsg == "" {
		t.Errorf("/grill-me with no args: handled=%v stopMsg=%q (want handled + a stop message)", handled, stopMsg)
	}
	rendered, stopMsg, handled := expandMacro("/grill-me the auth design")
	if !handled || stopMsg != "" || !strings.Contains(rendered, "the auth design") {
		t.Errorf("/grill-me <args>: handled=%v stopMsg=%q rendered=%q", handled, stopMsg, rendered)
	}
}
