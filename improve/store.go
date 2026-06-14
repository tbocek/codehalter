package improve

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"time"

	"github.com/BurntSushi/toml"
)

// Store writes each improvement entry as a separate TOML file into dataDir.
// Files are named YYYY-MM-DDTHHMMSSZ.toml. Returns the number of entries
// stored, any redaction notes (passed through), and an error.
func Store(dataDir string, payload ImprovementPayload, redactionNotes []string) (int, []string, error) {
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return 0, nil, fmt.Errorf("create data dir: %w", err)
	}

	for i, entry := range payload.Improvements {
		timestamp := time.Now().UTC().Format(time.RFC3339)
		filename := timestamp + fmt.Sprintf("_%03d", i+1) + ".toml"
		path := filepath.Join(dataDir, filename)

		f, err := os.Create(path)
		if err != nil {
			return i, redactionNotes, fmt.Errorf("create %s: %w", filename, err)
		}
		defer f.Close()

		if err := toml.NewEncoder(f).Encode(entry); err != nil {
			return i, redactionNotes, fmt.Errorf("write %s: %w", filename, err)
		}
	}

	return len(payload.Improvements), redactionNotes, nil
}

// LoadAll reads all TOML files in dataDir and returns the combined entries.
func LoadAll(dataDir string) ([]ImprovementEntry, error) {
	var entries []ImprovementEntry

	err := filepath.WalkDir(dataDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || filepath.Ext(path) != ".toml" {
			return nil
		}

		var entry ImprovementEntry
		if _, err := toml.DecodeFile(path, &entry); err != nil {
			return fmt.Errorf("decode %s: %w", path, err)
		}
		entries = append(entries, entry)
		return nil
	})
	if err != nil {
		return nil, err
	}

	return entries, nil
}
