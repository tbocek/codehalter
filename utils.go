package main

import "sync"

// parallel runs fn for each index [0, n) with up to `cap` concurrent
// goroutines. Callers pass an explicit upper bound matched to the work-list
// (e.g. launch_subagent's SubagentPinOrder length, probeAllLLMs's len(conns))
// so excess work queues instead of contending for slots.
func parallel(n, cap int, fn func(i int)) {
	if cap > n {
		cap = n
	}
	var wg sync.WaitGroup
	sem := make(chan struct{}, cap)
	for i := range n {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			fn(i)
		}(i)
	}
	wg.Wait()
}
