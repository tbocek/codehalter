package main

import (
	"log"
	"net/http"
	"os"

	"github.com/tbocek/codehalter/improve"
)

func main() {
	dataDir := os.Getenv("IMPROVE_DATA_DIR")
	if dataDir == "" {
		dataDir = "/improve"
	}

	addr := os.Getenv("IMPROVE_ADDR")
	if addr == "" {
		addr = ":8080"
	}

	handler := improve.NewServer(dataDir)
	log.Printf("listening on %s (data: %s)", addr, dataDir)
	if err := http.ListenAndServe(addr, handler); err != nil {
		log.Fatalf("server: %v", err)
	}
}
