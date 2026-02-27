package client

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
)

// Login handles the Pokemon Showdown authentication flow
func Login(username, password, challstr string) (string, error) {
	// POST to action.php to get assertion
	data := url.Values{}
	data.Set("act", "login")
	data.Set("name", username)
	data.Set("pass", password)
	data.Set("challstr", challstr)

	resp, err := http.PostForm("https://play.pokemonshowdown.com/action.php", data)
	if err != nil {
		return "", fmt.Errorf("login POST failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read login response: %w", err)
	}

	// Response starts with "]" followed by JSON
	bodyStr := string(body)
	if len(bodyStr) > 0 && bodyStr[0] == ']' {
		bodyStr = bodyStr[1:]
	}

	var result struct {
		ActionsSuccess bool   `json:"actionsuccess"`
		Assertion      string `json:"assertion"`
		CurUser        struct {
			LoggedIn bool   `json:"loggedin"`
			Username string `json:"username"`
		} `json:"curuser"`
	}

	if err := json.Unmarshal([]byte(bodyStr), &result); err != nil {
		return "", fmt.Errorf("failed to parse login response: %w (body: %s)", err, bodyStr)
	}

	if result.Assertion == "" {
		return "", fmt.Errorf("login failed: no assertion returned (body: %s)", bodyStr)
	}

	// If assertion starts with ";;" it's an error message
	if strings.HasPrefix(result.Assertion, ";;") {
		return "", fmt.Errorf("login rejected: %s", result.Assertion)
	}

	return result.Assertion, nil
}

// LoginAsGuest logs in without a password (guest account)
func LoginAsGuest(username, challstr string) (string, error) {
	data := url.Values{}
	data.Set("act", "getassertion")
	data.Set("userid", strings.ToLower(username))
	data.Set("challstr", challstr)

	resp, err := http.PostForm("https://play.pokemonshowdown.com/action.php", data)
	if err != nil {
		return "", fmt.Errorf("guest login POST failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read guest login response: %w", err)
	}

	assertion := strings.TrimSpace(string(body))
	if assertion == "" || strings.HasPrefix(assertion, ";;") {
		return "", fmt.Errorf("guest login failed: %s", assertion)
	}

	return assertion, nil
}
