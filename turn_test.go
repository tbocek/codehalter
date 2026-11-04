package main

import (
	"context"
	"testing"
	"time"
)

// A second holder waits for the gate; the turn in flight is NOT cancelled.
func TestHoldTurnWaitsItsTurn(t *testing.T) {
	a, s := newTestAgent(t)
	firstCtx, firstRelease, ok := a.holdTurn(context.Background(), s, true)
	if !ok {
		t.Fatal("first holdTurn refused")
	}
	if !s.turnRunning() {
		t.Error("turnRunning should report the held turn")
	}

	got := make(chan error, 1) // the second turn's ctx.Err(), read before it releases
	go func() {
		ctx, release, _ := a.holdTurn(context.Background(), s, true)
		got <- ctx.Err()
		release()
	}()

	select {
	case <-got:
		t.Fatal("the second turn ran while the first still held the gate")
	case <-time.After(50 * time.Millisecond):
	}
	if firstCtx.Err() != nil {
		t.Error("waiting for the gate cancelled the turn in flight")
	}
	firstRelease()
	select {
	case err := <-got:
		if err != nil {
			t.Error("the second turn started with a cancelled ctx")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("the second turn never ran")
	}
}

// !wait is refused while a turn runs, and the running turn is untouched.
func TestHoldTurnNeverInterrupts(t *testing.T) {
	a, s := newTestAgent(t)
	ctx, release, _ := a.holdTurn(context.Background(), s, true)
	if _, _, ok := a.holdTurn(context.Background(), s, false); ok {
		t.Fatal("a background turn got in while a turn was running")
	}
	if ctx.Err() != nil {
		t.Error("the running turn was cancelled by a background attempt")
	}
	release()
	_, release2, ok := a.holdTurn(context.Background(), s, false)
	if !ok {
		t.Fatal("a background turn was refused on an idle session")
	}
	release2()
}

// release closes the phase row and frees the turn; a queued note is left for the
// drain, which runs it as a turn rather than storing it silently.
func TestHoldTurnReleaseFinishes(t *testing.T) {
	a, s := newTestAgent(t)
	_, release, _ := a.holdTurn(context.Background(), s, true)
	s.phaseMu.Lock()
	s.phaseActive = true
	s.phaseMu.Unlock()
	s.addBgNote(bgNote{line: "job 1 done", full: "background job 1 exited with code 0"})

	release()

	s.phaseMu.Lock()
	active := s.phaseActive
	s.phaseMu.Unlock()
	if active {
		t.Error("the phase row is still open after release")
	}
	if !s.hasPending() || len(s.Messages) != 0 {
		t.Errorf("release consumed the note instead of leaving it for the drain: %+v", s.Messages)
	}
	if _, r, ok := a.holdTurn(context.Background(), s, false); !ok {
		t.Error("the turn is still held after release")
	} else {
		r()
	}
}
