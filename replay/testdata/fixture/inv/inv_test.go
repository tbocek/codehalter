package inv

import "testing"

func TestParse(t *testing.T) {
	items, err := Parse("a,1,100\n\nb,2,50\n")
	if err != nil || len(items) != 2 || items[1].Qty != 2 {
		t.Fatalf("items %v, err %v", items, err)
	}
	if _, err := Parse("a,1\n"); err == nil {
		t.Fatal("a short line must fail")
	}
}

func TestTotal(t *testing.T) {
	items := []Item{{"a", 2, 100}, {"b", 1, 50}}
	if got := Total(items); got != 250 {
		t.Fatalf("Total = %d, want 250", got)
	}
}

func TestFormat(t *testing.T) {
	if got := Format(1234); got != "12.34 €" {
		t.Fatalf("Format = %q", got)
	}
}
