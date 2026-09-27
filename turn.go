package main

import (
	"context"
	"log/slog"
	"strings"
	"sync"
)

// One turn per session, always started via holdTurn. Nothing replaces a running
// turn: typing steers it (Session.addSteer) and the stop button cancels it.

// held is taken for the whole turn; the other fields are guarded by mu.
type turnControl struct {
	held     sync.Mutex
	mu       sync.Mutex
	cancel   context.CancelFunc
	running  bool
	warmStop func()
	stopped  bool // the last turn ended by the user's Stop; queued input waits for the next prompt
}

func (s *Session) turnRunning() bool {
	s.ctl.mu.Lock()
	defer s.ctl.mu.Unlock()
	return s.ctl.running
}

// Does not wait for the turn to unwind; holdTurn's release does.
func (s *Session) cancelTurn() {
	s.ctl.mu.Lock()
	c := s.ctl.cancel
	if s.ctl.running {
		s.ctl.stopped = true
	}
	s.ctl.mu.Unlock()
	if c != nil {
		c()
	}
}

// Two turns must never run side by side: they would send divergent snapshots of
// one session. !wait returns ok=false instead of blocking. Every exit must call release.
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
	sess.ctl.stopped = false
	if sess.ctl.warmStop != nil {
		sess.ctl.warmStop() // this turn calls the model itself from here on
		sess.ctl.warmStop = nil
	}
	sess.ctl.mu.Unlock()
	release = func() {
		// Background ctx: the turn's own may be cancelled already.
		sess.phaseMu.Lock()
		active, phase := sess.phaseActive, sess.phaseCurrent
		sess.phaseActive = false
		sess.phaseMu.Unlock()
		if active {
			a.sendUpdate(context.Background(), sess.ID, planUpdate{Kind: "plan", Entries: phaseEntries(phase, true, "")})
		}
		a.sayRunningBgJobs(sess)
		// An idle server slot gets reclaimed and the next turn re-reads the whole
		// prompt, so keep the prefix warm until the next turn starts.
		warmConn := a.connFor("execute")
		stop := a.keepWarm(sess, warmConn, func() []llmMessage { return a.buildLLMContext(sess) })
		pending := sess.hasPending()
		sess.ctl.mu.Lock()
		sess.ctl.running = false
		sess.ctl.warmStop = stop
		if !pending {
			sess.ctl.stopped = false // nothing was cut off; later notes report as usual
		}
		sess.ctl.mu.Unlock()
		cancel()
		sess.ctl.held.Unlock()
	}
	return ctx, release, true
}

// Text or a note arriving during a drained turn joins that turn, so a small cap
// covers the arrived-as-the-round-ended race without an unbounded chain.
const maxSteerTurns = 3

// Runs what arrived too late for any round to pick up (typed text, job notes) as
// a turn of its own.
func (a *agent) drainSteer(ctx context.Context, sess *Session) {
	for range maxSteerTurns {
		items := sess.takePending()
		if len(items) == 0 {
			return
		}
		prompt, hasNote := a.sayPending(ctx, sess.ID, items, false)
		if hasNote {
			prompt += "\n\nContinue the work that was waiting on this result, if any was; otherwise tell the user what the result means. Do not start new work the user did not ask for."
		}
		err := a.runPromptTurn(ctx, sess, prompt)
		if err == nil {
			continue
		}
		if hasNote && !isCancelled(err) {
			slog.Warn("background job report turn failed", "sid", sess.ID, "err", err)
			a.say(context.Background(), sess.ID, "⚠ Could not report on the finished background job: "+err.Error()+"\n")
		} else {
			slog.Debug("drainSteer: the follow-up turn failed", "sid", sess.ID, "err", err)
		}
		return
	}
}

// sayPending shows the user what was taken off the queue and returns it for the
// model as one message, in arrival order. midTurn: the items join a running turn
// rather than start one.
func (a *agent) sayPending(ctx context.Context, sid string, items []pendingInput, midTurn bool) (text string, hasNote bool) {
	noteEnd := "\n\n"
	if midTurn {
		noteEnd = "\n"
	}
	saidText := !midTurn // a turn started by the text itself needs no picked-up line
	var parts []string
	for _, it := range items {
		if it.note != nil {
			a.say(ctx, sid, "\n🔔 "+it.note.line+noteEnd)
			parts = append(parts, it.note.full)
			hasNote = true
			continue
		}
		if !saidText {
			a.say(ctx, sid, "\n↪ picked up: "+firstLine(it.text)+"\n")
			saidText = true
		}
		parts = append(parts, it.text)
	}
	return strings.Join(parts, "\n\n"), hasNote
}

// The caller must already hold the turn.
func (a *agent) runPromptTurn(ctx context.Context, sess *Session, text string) error {
	sess.AddUser(text)
	sess.saveOrLog()
	return a.runTurn(ctx, sess.ID)
}

// stoppedIdle: the last turn was stopped with input still queued, which waits
// for the next prompt.
func (s *Session) stoppedIdle() bool {
	s.ctl.mu.Lock()
	defer s.ctl.mu.Unlock()
	return s.ctl.stopped && !s.ctl.running
}
