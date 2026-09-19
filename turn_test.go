package main

import (
	"context"
	"strings"
	"testing"
	"time"
)

// TestHoldTurnSupersedes pins what a typed prompt does to a turn in flight: it
// cancels it, flags it as replaced (so its cancel handler stays quiet), waits
// for it to release, and only then runs.
func TestHoldTurnSupersedes(t *testing.T) {
	a, s := newTestAgent(t)
	oldCtx, oldRelease, ok := a.holdTurn(context.Background(), s, true)
	if !ok {
		t.Fatal("first holdTurn refused")
	}

	got := make(chan error, 1) // the new turn's ctx.Err(), read before it releases
	go func() {
		ctx, release, _ := a.holdTurn(context.Background(), s, true)
		got <- ctx.Err()
		release()
	}()

	select {
	case <-oldCtx.Done():
	case <-time.After(2 * time.Second):
		t.Fatal("the turn in flight was not cancelled")
	}
	if !s.superseded() {
		t.Error("the replaced turn does not know it is being replaced")
	}
	select {
	case <-got:
		t.Fatal("the new turn ran before the old one released")
	case <-time.After(50 * time.Millisecond):
	}
	oldRelease()
	select {
	case err := <-got:
		if err != nil {
			t.Error("the new turn started with a cancelled ctx")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("the new turn never ran")
	}
	if s.superseded() {
		t.Error("superseded still set after the new turn took over")
	}
}

// TestHoldTurnNeverInterrupts pins the rule for work codehalter starts on its
// own: while a turn runs, it is refused and the running turn is untouched.
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

// TestHoldTurnReleaseFinishes pins what every turn does on its way out, however
// it started: the phase row is closed and finished background jobs are
// reported (stored for the model), then the turn is free again.
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
	if s.hasBgNotes() {
		t.Error("a finished job's note was not delivered at release")
	}
	if n := len(s.Messages); n == 0 || !strings.Contains(s.Messages[n-1].Content, "exited with code 0") {
		t.Errorf("the note was not stored for the model: %+v", s.Messages)
	}
	if _, r, ok := a.holdTurn(context.Background(), s, false); !ok {
		t.Error("the turn is still held after release")
	} else {
		r()
	}
}
