package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"sync"
	"testing"
	"time"
)

// Reverse order and half numeric ids: a routing mix-up hands a caller someone else's result.
func TestRPCPeerRoutesConcurrentResponses(t *testing.T) {
	peerR, w := io.Pipe()
	r, peerW := io.Pipe()
	t.Cleanup(func() { peerW.Close(); w.Close() })
	p := newRPCPeer(w, "", false)
	go p.serve(r, func(*jsonrpcRequest) { t.Error("a response was dispatched as a request") })

	const n = 20
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			raw, err := p.sendRequest(ctx, "echo", map[string]int{"n": i})
			if err != nil {
				t.Errorf("request %d: %v", i, err)
				return
			}
			if string(raw) != fmt.Sprint(i) {
				t.Errorf("request %d got result %s", i, raw)
			}
		}()
	}

	br := bufio.NewReader(peerR)
	reqs := make([]jsonrpcRequest, n)
	for i := range reqs {
		line, err := br.ReadString('\n')
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal([]byte(line), &reqs[i]); err != nil {
			t.Fatal(err)
		}
	}
	for i := n - 1; i >= 0; i-- {
		var params struct{ N int }
		if err := json.Unmarshal(reqs[i].Params, &params); err != nil {
			t.Fatal(err)
		}
		id := string(*reqs[i].ID)
		if i%2 == 0 {
			id = id[1 : len(id)-1]
		}
		if _, err := fmt.Fprintf(peerW, `{"jsonrpc":"2.0","id":%s,"result":%d}`+"\n", id, params.N); err != nil {
			t.Fatal(err)
		}
	}
	wg.Wait()
}

type writerFunc func([]byte) (int, error)

func (f writerFunc) Write(b []byte) (int, error) { return f(b) }

// The write blocks until serve has routed the response and hit EOF, so both select cases are ready.
func TestRPCPeerDeliversResponseThatBeatsClose(t *testing.T) {
	for i := 0; i < 20; i++ {
		r, peerW := io.Pipe()
		var p *rpcPeer
		p = newRPCPeer(writerFunc(func(b []byte) (int, error) {
			go func() {
				fmt.Fprint(peerW, `{"jsonrpc":"2.0","id":"1","result":7}`+"\n")
				peerW.Close()
			}()
			<-p.done
			return len(b), nil
		}), "", false)
		go p.serve(r, func(*jsonrpcRequest) {})

		raw, err := p.sendRequest(context.Background(), "m", nil)
		if err != nil || string(raw) != "7" {
			t.Fatalf("run %d: got %s, %v; want the result that arrived before EOF", i, raw, err)
		}
	}
}
