package main

import (
	"context"
	"log/slog"
	"strings"
	"sync"
)

// One turn per session. Every way a turn starts goes through holdTurn, so
// locking, cancellation and end-of-turn cleanup live in one place:
//
//   - a typed prompt and the fix cards at session open wait for the gate;
//   - work codehalter starts itself (a background job's report) never waits,
//     it runs only when no turn is running.
//
// Nothing replaces a running turn: typing steers it (Session.addSteer) and the
// stop button cancels it. Inside a held turn, runPromptTurn runs each further
// model turn: an accepted fix card, a /spec round, a queued follow-up.

// turnControl is a session's turn gate. held is taken for the whole turn; the
// rest is guarded by mu.
type turnControl struct {
	held    sync.Mutex
	mu      sync.Mutex
	cancel  context.CancelFunc
	running bool
	// warmStop ends the keep-alive that runs between turns (keepWarm).
	warmStop func()
}

// turnRunning reports whether a turn holds the gate right now. A prompt typed
// then is steering, not a replacement, so Prompt queues it instead of waiting
// for the gate.
func (s *Session) turnRunning() bool {
	s.ctl.mu.Lock()
	defer s.ctl.mu.Unlock()
	return s.ctl.running
}

// cancelTurn stops the turn in flight, if any. The caller does not wait for it
// to unwind; holdTurn does.
func (s *Session) cancelTurn() {
	s.ctl.mu.Lock()
	c := s.ctl.cancel
	s.ctl.mu.Unlock()
	if c != nil {
		c()
	}
}

// holdTurn makes the caller's turn the active one on sess and returns its ctx
// and the release every exit path must call.
//
// wait blocks until the gate is free: two turns must never run side by side,
// or they send divergent snapshots of the same session (one compacting while
// the other re-sends the pre-compaction context). !wait returns ok=false when
// a turn is running, and the caller tries again later.
//
// release closes the phase row, reports background jobs that finished during
// the turn (at its end, never in the middle), cancels the ctx and frees the
// turn, in that order. Background ctx for the reporting: the turn's own may be
// cancelled already.
func (a *agent) holdTurn(parent context.Context, sess *Session, wait bool) (ctx context.Context, release func(), ok bool) {
	if wait {
		sess.ctl.held.Lock()
	} else if !sess.ctl.held.TryLock() {
		return nil, nil, false
	}
	ctx, cancel := context.WithCancel(parent)
	sess.ctl.mu.Lock()
	sess.ctl.cancel = cancel
	sess.ctl.running = true
	if sess.ctl.warmStop != nil {
		sess.ctl.warmStop() // this turn calls the model itself from here on
		sess.ctl.warmStop = nil
	}
	sess.ctl.mu.Unlock()
	release = func() {
		a.finalizePlan(sess.ID)
		a.flushBgNotes(context.Background(), sess)
		// Between turns the conversation's prefix sits unused in the server's
		// KV cache, where an idle slot is reclaimed and the next turn pays to
		// re-read the whole prompt. Refresh it until the next turn starts, or
		// until keepWarmFor says the session is over rather than idle.
		warmConn := a.connFor("execute")
		stop := a.keepWarm(sess, warmConn, func() []llmMessage { return a.buildLLMContext(sess) })
		sess.ctl.mu.Lock()
		sess.ctl.running = false
		sess.ctl.warmStop = stop
		sess.ctl.mu.Unlock()
		cancel()
		sess.ctl.held.Unlock()
	}
	return ctx, release, true
}

// maxSteerTurns bounds drainSteer: each turn it runs can leave more queued
// behind it, and a user typing during those is steering THAT turn, not asking
// for another. Three is enough for the race this covers (typed as the last
// round ended) without turning a fast typist into an unbounded chain.
const maxSteerTurns = 3

// drainSteer runs what the user typed too late for any round to pick up. The
// turn is over by then, so it becomes a turn of its own rather than being
// dropped or silently stored where nothing would answer it.
func (a *agent) drainSteer(ctx context.Context, sess *Session) {
	for range maxSteerTurns {
		queued := sess.takeSteer()
		if len(queued) == 0 {
			return
		}
		if err := a.runPromptTurn(ctx, sess, strings.Join(queued, "\n\n")); err != nil {
			slog.Debug("drainSteer: the follow-up turn failed", "sid", sess.ID, "err", err)
			return
		}
	}
}

// runPromptTurn stores text as a user message and runs one full turn on it,
// inside a turn the caller already holds. It is the turn for everything that is
// not a prompt the user typed: an accepted fix card, a /spec round, a
// background job's report.
func (a *agent) runPromptTurn(ctx context.Context, sess *Session, text string) error {
	sess.AddUser(text)
	sess.saveOrLog()
	return a.runTurn(ctx, sess.ID)
}
