// Line-delimited JSON framing shared by the ACP server and the MCP client.
// JSON-RPC semantics (ids, pending-response demux, method dispatch) live in
// the callers — this layer only frames.
package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"strings"
	"sync"
)

// lineProtocol moves JSON messages over a line-delimited stdio pair. Each
// writeMessage emits one JSON object followed by '\n'; serve reads one line
// at a time and hands the raw bytes to onLine. Writes are serialised so two
// concurrent writers can't interleave halves of a JSON object on the wire.
type lineProtocol struct {
	w       io.Writer
	r       io.Reader
	writeMu sync.Mutex
}

func newLineProtocol(w io.Writer, r io.Reader) *lineProtocol {
	return &lineProtocol{w: w, r: r}
}

func (p *lineProtocol) writeMessage(msg any) error {
	b, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	slog.Debug("writing", "msg", string(b))
	b = append(b, '\n')
	p.writeMu.Lock()
	defer p.writeMu.Unlock()
	_, err = p.w.Write(b)
	return err
}

// serve reads one JSON object per line and forwards the bytes to onLine.
// Returns when the reader hits EOF or an unrecoverable error. ReadString is
// used (not bufio.Scanner) so a large MCP tool response can't trip the
// scanner's MaxScanTokenSize cap.
func (p *lineProtocol) serve(onLine func([]byte)) {
	br := bufio.NewReader(p.r)
	for {
		line, err := br.ReadString('\n')
		if err != nil {
			if !errors.Is(err, io.EOF) {
				slog.Debug("read error", "error", err)
			}
			return
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			continue
		}
		onLine([]byte(line))
	}
}
