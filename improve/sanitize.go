package improve

import (
	"regexp"
)

var sensitivePatterns = []*regexp.Regexp{
	// api_key= "value" or api_key: "value"
	regexp.MustCompile(`(?i)(api_key\s*[=:])\s*"?([^"\s,;}{]+)"?`),
	// token= "value" or token: "value"
	regexp.MustCompile(`(?i)(token\s*[=:])\s*"?([^"\s,;}{]+)"?`),
	// password= "value" or password: "value"
	regexp.MustCompile(`(?i)(password\s*[=:])\s*"?([^"\s,;}{]+)"?`),
	// secret= "value" or secret: "value"
	regexp.MustCompile(`(?i)(secret\s*[=:])\s*"?([^"\s,;}{]+)"?`),
	// Bearer <token>
	regexp.MustCompile(`(?i)(Bearer\s+)([A-Za-z0-9_\-\.]+)`),
	// Authorization: Bearer <token>
	regexp.MustCompile(`(?i)(Authorization:\s*Bearer\s+)([A-Za-z0-9_\-\.]+)`),
}

// Sanitize scans the Original and New fields for sensitive patterns and
// redacts matched values to [REDACTED]. Returns the sanitized entry and
// a list of redaction notes describing what was found.
func Sanitize(entry ImprovementEntry) (ImprovementEntry, []string) {
	notes := []string{}

	sanitizeField := func(field string, val string) string {
		for _, re := range sensitivePatterns {
			matches := re.FindStringSubmatch(val)
			if matches != nil {
				notes = append(notes, field+": redacted "+matches[1])
				val = re.ReplaceAllString(val, matches[1]+"[REDACTED]")
			}
		}
		return val
	}

	entry.Original = sanitizeField("original", entry.Original)
	entry.New = sanitizeField("new", entry.New)

	return entry, notes
}
