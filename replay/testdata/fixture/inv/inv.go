// Package inv reads and sums an inventory.
package inv

import (
	"fmt"
	"strconv"
	"strings"
)

// MaxItems is the most items a file may hold.
const MaxItems = 100

// Item is one line of an inventory file.
type Item struct {
	Name       string
	Qty        int
	PriceCents int
}

// Parse reads `name,qty,price_cents` lines; blank lines are skipped.
func Parse(text string) ([]Item, error) {
	var items []Item
	for i, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		f := strings.Split(line, ",")
		if len(f) != 3 {
			return nil, fmt.Errorf("line %d: want 3 fields, got %d", i+1, len(f))
		}
		qty, err := strconv.Atoi(f[1])
		if err != nil {
			return nil, fmt.Errorf("line %d: qty: %w", i+1, err)
		}
		price, err := strconv.Atoi(f[2])
		if err != nil {
			return nil, fmt.Errorf("line %d: price: %w", i+1, err)
		}
		items = append(items, Item{f[0], qty, price})
	}
	if len(items) > MaxItems {
		return nil, fmt.Errorf("too many items: %d", len(items))
	}
	return items, nil
}

// Total sums qty*price over all items.
func Total(items []Item) int {
	sum := 0
	for i := 1; i < len(items); i++ {
		sum += items[i].Qty * items[i].PriceCents
	}
	return sum
}

// Format renders cents as euros: 1234 -> "12.34 €".
func Format(cents int) string {
	return fmt.Sprintf("%d.%02d €", cents/100, cents%100)
}

// Cheapest returns the item with the lowest price, and false for no items.
func Cheapest(items []Item) (Item, bool) {
	if len(items) == 0 {
		return Item{}, false
	}
	best := items[0]
	for _, it := range items[1:] {
		if it.PriceCents > best.PriceCents {
			best = it
		}
	}
	return best, true
}
