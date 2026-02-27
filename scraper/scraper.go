package scraper

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// ReplaySearchResult is the response format from the showdown search API
// Examples:
// [{"id":"gen9nationaldex-2252278474","p1":"Ash Ketchum","p2":"Gary Oak","format":"gen9nationaldex","log":"","inputlog":null,"uploadtime":1707907530,"views":0,"p1id":"ashketchum","p2id":"garyoak","formatid":"gen9nationaldex","rating":1500,"private":0,"password":null}]
type ReplaySearchResult struct {
	ID         string   `json:"id"`
	P1         string   `json:"p1"`
	P2         string   `json:"p2"`
	Players    []string `json:"players"` // Sometimes names are nested here for standard searches
	Format     string   `json:"format"`
	UploadTime int64    `json:"uploadtime"`
	Rating     int      `json:"rating"`
}

// ScrapeConfig holds the configuration for scraping replays
type ScrapeConfig struct {
	Format      string
	NumGames    int
	OutputDir   string
	PageLimit   int
	Concurrency int
}

// ScrapeReplays orchestrates fetching replay IDs and downloading logs
func ScrapeReplays(cfg ScrapeConfig) error {
	fmt.Printf("Scraping %d %s replays to %s...\n", cfg.NumGames, cfg.Format, cfg.OutputDir)

	// Make sure output dir exists
	if err := os.MkdirAll(cfg.OutputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output dir: %w", err)
	}

	successCount := 0

	// 1. Fetch replay IDs and stream downloads
	replayIDs, err := fetchReplayIDs(cfg, func(batch []string) {
		fmt.Printf("\nFound %d replay IDs. Starting download for batch...\n", len(batch))
		count := downloadLogs(batch, cfg)
		successCount += count
	})
	if err != nil {
		return fmt.Errorf("failed fetching replay IDs: %w", err)
	}

	fmt.Printf("\nSuccessfully downloaded %d/%d replays.\n", successCount, len(replayIDs))
	return nil
}

// fetchReplayIDs paginates through the search API and pivots by Username to bypass limits
func fetchReplayIDs(cfg ScrapeConfig, onBatch func([]string)) ([]string, error) {
	var allIDs []string
	var batch []string
	seenIDs := make(map[string]bool)

	userQueue := make([]string, 0)
	seenUsers := make(map[string]bool)

	client := &http.Client{Timeout: 10 * time.Second}

	// Helper to fetch and parse a specific query URL
	fetchAPI := func(url string) ([]ReplaySearchResult, error) {
		req, err := http.NewRequest("GET", url, nil)
		if err != nil {
			return nil, err
		}
		req.Header.Set("User-Agent", "Pokemon Showdown Engine Parser GoBot/1.0")

		resp, err := client.Do(req)
		if err != nil {
			return nil, err
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("API returned status %d", resp.StatusCode)
		}

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, err
		}

		var results []ReplaySearchResult
		if err := json.Unmarshal(body, &results); err != nil {
			if strings.Contains(string(body), "<html") {
				return nil, fmt.Errorf("API returned HTML instead of JSON")
			}
			return nil, err
		}
		return results, nil
	}

	// 1. Initial Timeline Scrape (Pages 1 to 5) to seed the User Queue
	fmt.Println("Phase 1: Seeding Username queue from recent matches...")
	for page := 1; page <= 5; page++ {
		url := ""
		if cfg.Format == "all-singles" {
			url = fmt.Sprintf("https://replay.pokemonshowdown.com/search.json?page=%d", page)
		} else {
			url = fmt.Sprintf("https://replay.pokemonshowdown.com/search.json?format=%s&page=%d", cfg.Format, page)
		}

		results, err := fetchAPI(url)
		if err != nil || len(results) == 0 {
			break
		}

		for _, result := range results {
			if len(allIDs) >= cfg.NumGames {
				break
			}
			if cfg.Format == "all-singles" && !isSinglesFormat(result.Format) {
				continue
			}
			if result.Rating < 1600 {
				continue
			}
			if !seenIDs[result.ID] {
				allIDs = append(allIDs, result.ID)
				batch = append(batch, result.ID)
				seenIDs[result.ID] = true

				// Seed users
				if len(result.Players) >= 2 {
					p1, p2 := result.Players[0], result.Players[1]
					if p1 != "" && !seenUsers[p1] {
						seenUsers[p1] = true
						userQueue = append(userQueue, p1)
					}
					if p2 != "" && !seenUsers[p2] {
						seenUsers[p2] = true
						userQueue = append(userQueue, p2)
					}
				} else {
					if result.P1 != "" && !seenUsers[result.P1] {
						seenUsers[result.P1] = true
						userQueue = append(userQueue, result.P1)
					}
					if result.P2 != "" && !seenUsers[result.P2] {
						seenUsers[result.P2] = true
						userQueue = append(userQueue, result.P2)
					}
				}
			}
		}

		if len(batch) >= 1000 {
			onBatch(batch)
			batch = make([]string, 0, 1000)
		}

		time.Sleep(500 * time.Millisecond) // Polite delay
	}

	// 2. Username Pivot Crawling (Bypassing 100-page limit)
	fmt.Printf("Phase 2: Crawling through %d seeded players to find %d total matches...\n", len(userQueue), cfg.NumGames)

	for len(userQueue) > 0 && len(allIDs) < cfg.NumGames {
		// Pop user
		user := userQueue[0]
		userQueue = userQueue[1:]

		// Fetch their history in this specific format
		url := ""
		if cfg.Format == "all-singles" {
			url = fmt.Sprintf("https://replay.pokemonshowdown.com/search.json?user=%s", user)
		} else {
			url = fmt.Sprintf("https://replay.pokemonshowdown.com/search.json?user=%s&format=%s", user, cfg.Format)
		}

		results, err := fetchAPI(url)

		if err == nil && len(results) > 0 {
			for _, result := range results {
				if len(allIDs) >= cfg.NumGames {
					break
				}
				if cfg.Format == "all-singles" && !isSinglesFormat(result.Format) {
					continue
				}
				if result.Rating < 1600 {
					continue
				}
				if !seenIDs[result.ID] {
					allIDs = append(allIDs, result.ID)
					batch = append(batch, result.ID)
					seenIDs[result.ID] = true

					// Grab opponents to add to our infinite crawler web
					if len(result.Players) >= 2 {
						p1, p2 := result.Players[0], result.Players[1]
						if p1 != "" && p1 != user && !seenUsers[p1] {
							seenUsers[p1] = true
							userQueue = append(userQueue, p1)
						}
						if p2 != "" && p2 != user && !seenUsers[p2] {
							seenUsers[p2] = true
							userQueue = append(userQueue, p2)
						}
					} else {
						if result.P1 != "" && result.P1 != user && !seenUsers[result.P1] {
							seenUsers[result.P1] = true
							userQueue = append(userQueue, result.P1)
						}
						if result.P2 != "" && result.P2 != user && !seenUsers[result.P2] {
							seenUsers[result.P2] = true
							userQueue = append(userQueue, result.P2)
						}
					}
				}
			}

			if len(batch) >= 1000 {
				onBatch(batch)
				batch = make([]string, 0, 1000)
			} else if len(allIDs)%500 < 50 {
				fmt.Printf("\r...Found %d/%d unique matches (Queue: %d)...", len(allIDs), cfg.NumGames, len(userQueue))
			}
		}

		time.Sleep(500 * time.Millisecond) // Don't hammer the user search API
	}
	if len(batch) > 0 {
		onBatch(batch)
	}

	fmt.Printf("\nFinished crawling. Total unique IDs: %d\n", len(allIDs))

	return allIDs, nil
}

// downloadLogs handles the concurrent downloading of replay logs and saves them to disk
func downloadLogs(replayIDs []string, cfg ScrapeConfig) int {
	// Setup worker pool
	jobs := make(chan string, len(replayIDs))
	results := make(chan bool, len(replayIDs))

	for w := 1; w <= cfg.Concurrency; w++ {
		go downloadWorker(jobs, results, cfg.OutputDir)
	}

	// Send jobs
	for _, id := range replayIDs {
		jobs <- id
	}
	close(jobs)

	// Collect results
	successCount := 0
	for i := 0; i < len(replayIDs); i++ {
		if <-results {
			successCount++
		}

		fmt.Printf("\rDownloaded: %d/%d", i+1, len(replayIDs))
	}

	return successCount
}

func downloadWorker(jobs <-chan string, results chan<- bool, outputDir string) {
	client := &http.Client{Timeout: 15 * time.Second}

	for id := range jobs {
		// e.g. https://replay.pokemonshowdown.com/gen9nationaldex-2252278474.log
		url := fmt.Sprintf("https://replay.pokemonshowdown.com/%s.log", id)

		req, err := http.NewRequest("GET", url, nil)
		if err != nil {
			results <- false
			continue
		}

		// Let's be polite to Showdown servers
		req.Header.Set("User-Agent", "Pokemon Showdown Engine Parser GoBot/1.0")

		resp, err := client.Do(req)
		if err != nil {
			results <- false
			continue
		}

		if resp.StatusCode != http.StatusOK {
			resp.Body.Close()
			results <- false
			continue
		}

		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			results <- false
			continue
		}

		// Ensure the match has at least 10 turns
		if !strings.Contains(string(body), "|turn|10") {
			results <- false
			continue
		}

		// Save the file
		filename := filepath.Join(outputDir, fmt.Sprintf("%s.log", id))
		err = os.WriteFile(filename, body, 0644)
		if err != nil {
			results <- false
			continue
		}

		results <- true

		// Rate limit within the worker to avoid hammering the server
		time.Sleep(200 * time.Millisecond)
	}
}

// isSinglesFormat checks if a given format string looks like a Singles format
func isSinglesFormat(format string) bool {
	formatLower := strings.ToLower(format)
	if strings.Contains(formatLower, "double") ||
		strings.Contains(formatLower, "vgc") ||
		strings.Contains(formatLower, "triple") ||
		strings.Contains(formatLower, "free-for-all") ||
		strings.Contains(formatLower, "ffa") ||
		strings.Contains(formatLower, "multi") ||
		strings.Contains(formatLower, "custom") {
		return false
	}
	return true
}
