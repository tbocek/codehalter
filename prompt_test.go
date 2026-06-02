package main

import "testing"

// TestResourcePath pins the URI → read_file path mapping used when an embedded
// "resource" / "resource_link" block is folded into the prompt: file:// URIs
// collapse to their percent-decoded path with the fragment stripped, non-file
// URIs pass through verbatim.
func TestResourcePath(t *testing.T) {
	cases := map[string]string{
		"file:///workspaces/codehalter/llm.go":          "/workspaces/codehalter/llm.go",
		"file:///workspaces/codehalter/llm.go#L801-836": "/workspaces/codehalter/llm.go",
		"file:///a%20b/c.go":                            "/a b/c.go",
		"/plain/path.go":                                "/plain/path.go",
		"https://example.com/x":                         "https://example.com/x",
		"":                                              "",
	}
	for uri, want := range cases {
		if got := resourcePath(uri); got != want {
			t.Errorf("resourcePath(%q) = %q, want %q", uri, got, want)
		}
	}
}

// TestResourceLabel checks the human-readable header: the path plus the URI
// fragment (an editor line range) in parentheses when present.
func TestResourceLabel(t *testing.T) {
	cases := map[string]string{
		"file:///workspaces/codehalter/llm.go":          "/workspaces/codehalter/llm.go",
		"file:///workspaces/codehalter/llm.go#L801-836": "/workspaces/codehalter/llm.go (L801-836)",
		"":                                              "attachment",
	}
	for uri, want := range cases {
		if got := resourceLabel(uri); got != want {
			t.Errorf("resourceLabel(%q) = %q, want %q", uri, got, want)
		}
	}
}
