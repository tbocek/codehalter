package improve

type ImprovementEntry struct {
	Title     string `json:"title" toml:"title"`
	File      string `json:"file" toml:"file"`
	Type      string `json:"type" toml:"type"`
	Original  string `json:"original" toml:"original"`
	New       string `json:"new" toml:"new"`
	Reasoning string `json:"reasoning" toml:"reasoning"`
	Ip        string `json:"ip" toml:"ip"`
	Model     string `json:"model" toml:"model"`
	License   string `json:"license" toml:"license"`
}

type ImprovementPayload struct {
	Improvements []ImprovementEntry `json:"improvements" toml:"improvements"`
}
