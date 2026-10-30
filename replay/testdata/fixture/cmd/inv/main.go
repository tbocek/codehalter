package main

import (
	"fmt"
	"os"

	"example.com/inv/inv"
)

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "usage: inv list|total FILE")
		os.Exit(2)
	}
	data, err := os.ReadFile(os.Args[2])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	items, err := inv.Parse(string(data))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	switch os.Args[1] {
	case "list":
		for _, it := range items {
			fmt.Printf("%s x%d\n", it.Name, it.Qty)
		}
	case "total":
		fmt.Println(inv.Format(inv.Total(items)))
	default:
		fmt.Fprintln(os.Stderr, "unknown command")
		os.Exit(2)
	}
}
