package client

import (
	"encoding/json"
	"fmt"
	"log"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/gorilla/websocket"

	"github.com/pokemon-engine/bot"
	"github.com/pokemon-engine/evaluator"
	"github.com/pokemon-engine/gamedata"
	"github.com/pokemon-engine/simulator"
)

const (
	ShowdownWSURL  = "wss://sim3.psim.us/showdown/websocket"
	SearchMoveTime = 1500 * time.Millisecond
)

// ShowdownBot manages the WebSocket lifecycle
type ShowdownBot struct {
	conn     *websocket.Conn
	username string
	password string
	battles  map[string]*BattleContext // roomID -> context
	moveTime time.Duration
}

// RunBot is the main entry point for the live bot
func RunBot(username, password string, moveTime time.Duration) error {
	// Load game data
	if err := gamedata.LoadPokedex("data/pokedex.json"); err != nil {
		log.Printf("Warning: Pokedex not loaded: %v", err)
	}
	if err := gamedata.LoadMovedex("data/moves.json"); err != nil {
		log.Printf("Warning: Movedex not loaded: %v", err)
	}
	evaluator.InitEvaluator()

	bot := &ShowdownBot{
		username: username,
		password: password,
		battles:  make(map[string]*BattleContext),
		moveTime: moveTime,
	}

	log.Printf("Connecting to Pokemon Showdown (%s)...", ShowdownWSURL)

	dialer := websocket.Dialer{
		HandshakeTimeout: 10 * time.Second,
	}
	conn, _, err := dialer.Dial(ShowdownWSURL, nil)
	if err != nil {
		return fmt.Errorf("failed to connect: %w", err)
	}
	defer conn.Close()
	bot.conn = conn

	log.Println("Connected! Waiting for challstr...")

	// Read loop
	for {
		_, message, err := conn.ReadMessage()
		if err != nil {
			return fmt.Errorf("read error: %w", err)
		}

		bot.handleMessage(string(message))
	}
}

func (b *ShowdownBot) handleMessage(raw string) {
	// Messages can contain multiple lines
	// Lines starting with > indicate a room
	lines := strings.Split(raw, "\n")

	roomID := ""
	if len(lines) > 0 && strings.HasPrefix(lines[0], ">") {
		roomID = lines[0][1:]
		lines = lines[1:]
	}

	for _, line := range lines {
		b.handleLine(roomID, line)
	}
}

func (b *ShowdownBot) handleLine(roomID, line string) {
	if line == "" {
		return
	}

	parts := strings.SplitN(line, "|", 3)
	if len(parts) < 2 {
		return
	}

	// parts[0] is empty (before first |), parts[1] is message type
	msgType := parts[1]
	var msgData string
	if len(parts) > 2 {
		msgData = parts[2]
	}

	switch msgType {
	case "challstr":
		b.onChallstr(msgData)
	case "updateuser":
		b.onUpdateUser(msgData)
	case "request":
		b.onRequest(roomID, msgData)
	case "error":
		b.onError(roomID, msgData)
	case "win":
		b.onWin(roomID, msgData)
	case "tie":
		b.onTie(roomID)
	case "updatesearch":
		// Matchmaking status update - can log it
	case "pm":
		b.onPM(msgData)
	case "init":
		if msgData == "battle" {
			b.onBattleInit(roomID)
		}
	case "deinit":
		delete(b.battles, roomID)
	case "player":
		b.onPlayer(roomID, msgData)
	case "poke":
		b.onPoke(roomID, msgData)
	case "teampreview":
		b.onTeamPreview(roomID)
	case "turn":
		log.Printf("[%s] Turn %s", roomID, msgData)
	case "weather":
		b.onFieldUpdate(roomID, msgData, "weather")
	case "-fieldstart":
		b.onFieldUpdate(roomID, msgData, "fieldstart")
	case "-fieldend":
		b.onFieldUpdate(roomID, msgData, "fieldend")
	case "-sidestart":
		b.onSideUpdate(roomID, msgData, "sidestart")
	case "-sideend":
		b.onSideUpdate(roomID, msgData, "sideend")
	case "-start":
		b.onOpponentUpdate(roomID, msgData, "start")
	case "-end":
		b.onOpponentUpdate(roomID, msgData, "end")
	case "move":
		b.onOpponentAction(roomID, msgData, false)
	case "switch", "drag":
		b.onOpponentAction(roomID, msgData, true)
	case "faint":
		b.onOpponentUpdate(roomID, msgData, "faint")
	case "-damage", "-heal":
		b.onOpponentUpdate(roomID, msgData, "hp")
	case "-status":
		b.onOpponentUpdate(roomID, msgData, "status")
	case "-curestatus":
		b.onOpponentUpdate(roomID, msgData, "curestatus")
	case "-boost", "-unboost", "-setboost":
		b.onOpponentUpdate(roomID, msgData, "boost")
	case "-clearallboost":
		b.onOpponentUpdate(roomID, msgData, "clearallboost")
	case "-item":
		b.onOpponentUpdate(roomID, msgData, "item")
	case "-enditem":
		b.onOpponentUpdate(roomID, msgData, "enditem")
	case "-ability":
		b.onOpponentUpdate(roomID, msgData, "ability")
	case "-terastallize":
		b.onOpponentUpdate(roomID, msgData, "terastallize")
	}
}

// onOpponentAction logs the opponent's moves and switches
func (b *ShowdownBot) onOpponentAction(roomID, data string, isSwitch bool) {
	ctx, exists := b.battles[roomID]
	if !exists {
		return
	}

	// Ensure State exists before processing
	if ctx.State == nil {
		ctx.State = &simulator.BattleState{}
		teamSize := 6
		if ctx.Request != nil && len(ctx.Request.Side.Pokemon) > 0 {
			teamSize = len(ctx.Request.Side.Pokemon)
		}
		ctx.State.P2 = simulator.PlayerState{
			ID:              ctx.OpponentID,
			ActiveIdx:       -1,
			TeamSize:        teamSize,
			CanTerastallize: true,
		}
		for i := 0; i < 6; i++ {
			ctx.State.P2.Team[i] = simulator.PokemonState{
				Name:    "Unknown",
				Species: "Unknown",
				HP:      100,
				MaxHP:   100,
				Boosts:  114420174,
			}
		}
	} else if ctx.State.P2.TeamSize < 6 {
		ctx.State.P2.TeamSize = 6
	}

	parts := strings.Split(data, "|")
	if len(parts) < 2 {
		return
	}

	target := strings.TrimSpace(parts[0]) // e.g. "p2a: Gengar"

	// If the target doesn't start with the opponent's ID, ignore it
	if ctx.OpponentID != "" && !strings.HasPrefix(target, ctx.OpponentID) {
		return
	}

	if !isSwitch {
		// It's a move. Data is like "p2a: Gengar|Shadow Ball"
		moveName := strings.TrimSpace(parts[1])
		moveID := strings.ToLower(strings.ReplaceAll(moveName, " ", ""))

		active := ctx.State.P2.GetActive()
		if active != nil {
			// Add to known moves if not already there
			found := false
			for i := 0; i < active.NumMoves; i++ {
				if active.Moves[i] == moveID {
					found = true
					break
				}
			}
			if !found && active.NumMoves < 4 {
				active.Moves[active.NumMoves] = moveID
				active.NumMoves++
			}
		}
		return
	}

	// data is formatted as "p2a: Species|Details|Condition" or similar
	details := parts[1]
	species, level, gender := ParseDetails(details)

	// Update the active Pokemon
	idx := -1
	for i := 0; i < ctx.State.P2.TeamSize; i++ {
		if ctx.State.P2.Team[i].Species == species {
			idx = i
			break
		}
	}

	// If not found by species, replace the first "Unknown" slot
	if idx == -1 {
		for i := 0; i < ctx.State.P2.TeamSize; i++ {
			if ctx.State.P2.Team[i].Species == "Unknown" {
				idx = i
				break
			}
		}
	}

	// Fallback to slot 0 if all slots are somehow filled and name not found
	if idx == -1 {
		idx = 0
	}

	// Deactivate previous active
	if ctx.State.P2.ActiveIdx != -1 {
		ctx.State.P2.Team[ctx.State.P2.ActiveIdx].IsActive = false
	}

	ctx.State.P2.Team[idx].Species = species
	ctx.State.P2.Team[idx].Name = species
	ctx.State.P2.Team[idx].IsActive = true
	ctx.State.P2.Team[idx].Level = level
	ctx.State.P2.Team[idx].Gender = gender
	ctx.State.P2.ActiveIdx = idx

	if len(parts) > 2 {
		condition := parts[2]
		hp, maxHp, fainted := ParseCondition(condition)
		status := ParseStatus(condition)
		ctx.State.P2.Team[idx].HP = hp
		ctx.State.P2.Team[idx].MaxHP = maxHp
		ctx.State.P2.Team[idx].Fainted = fainted
		ctx.State.P2.Team[idx].Status = status
	}

	log.Printf("[%s] Opponent switched to: %s (L%d %s)", roomID, species, level, gender)
}

func (b *ShowdownBot) onOpponentUpdate(roomID, data, updateType string) {
	ctx, exists := b.battles[roomID]
	if !exists || ctx.State == nil {
		return
	}

	parts := strings.Split(data, "|")
	if len(parts) == 0 {
		return
	}

	target := strings.TrimSpace(parts[0])
	if ctx.OpponentID != "" && !strings.HasPrefix(target, ctx.OpponentID) {
		return
	}

	active := ctx.State.P2.GetActive()
	if active == nil {
		return
	}

	switch updateType {
	case "faint":
		active.Fainted = true
		active.HP = 0
		active.IsActive = false
		ctx.State.P2.ActiveIdx = -1
	case "hp":
		if len(parts) >= 2 {
			hp, maxHp, fainted := ParseCondition(parts[1])
			active.HP = hp
			active.MaxHP = maxHp
			active.Fainted = fainted
		}
	case "status":
		if len(parts) >= 2 {
			active.Status = parts[1]
		}
	case "curestatus":
		active.Status = ""
	case "boost":
		if len(parts) >= 3 {
			stat := parts[1]
			amount, _ := strconv.Atoi(parts[2])
			if shift, ok := simulator.GetStatShift(stat); ok {
				cur := active.GetBoost(shift)
				active.SetBoost(shift, cur+amount)
			}
		}
	case "unboost":
		if len(parts) >= 3 {
			stat := parts[1]
			amount, _ := strconv.Atoi(parts[2])
			if shift, ok := simulator.GetStatShift(stat); ok {
				cur := active.GetBoost(shift)
				active.SetBoost(shift, cur-amount)
			}
		}
	case "clearallboost":
		active.Boosts = 114420174
	case "item":
		if len(parts) >= 2 {
			active.Item = strings.ToLower(strings.ReplaceAll(parts[1], " ", ""))
		}
	case "enditem":
		active.Item = ""
	case "ability":
		if len(parts) >= 2 {
			active.Ability = strings.ToLower(strings.ReplaceAll(parts[1], " ", ""))
		}
	case "terastallize":
		if len(parts) >= 2 {
			active.Terastallized = true
			active.TeraType = normalizeTypeName(parts[1])
			ctx.State.P2.CanTerastallize = false
		}
	case "start":
		if len(parts) >= 2 {
			effect := parts[1]
			if strings.HasPrefix(effect, "move: ") {
				effect = strings.TrimPrefix(effect, "move: ")
			}
			bit := simulator.MapVolatileToBit(effect)
			active.ToggleVolatile(bit, true)
		}
	case "end":
		if len(parts) >= 2 {
			effect := parts[1]
			if strings.HasPrefix(effect, "move: ") {
				effect = strings.TrimPrefix(effect, "move: ")
			}
			bit := simulator.MapVolatileToBit(effect)
			active.ToggleVolatile(bit, false)
		}
	}
}

func (b *ShowdownBot) onFieldUpdate(roomID, data, updateType string) {
	ctx, exists := b.battles[roomID]
	if !exists || ctx.State == nil {
		return
	}

	parts := strings.Split(data, "|")
	if len(parts) == 0 {
		return
	}

	val := parts[0]
	if strings.HasPrefix(val, "move: ") {
		val = strings.TrimPrefix(val, "move: ")
	}

	switch updateType {
	case "weather":
		if val == "none" {
			ctx.State.Field.Weather = ""
		} else {
			ctx.State.Field.Weather = val
		}
	case "fieldstart":
		switch {
		case strings.Contains(val, "Electric Terrain"):
			ctx.State.Field.Terrain = "Electric Terrain"
		case strings.Contains(val, "Grassy Terrain"):
			ctx.State.Field.Terrain = "Grassy Terrain"
		case strings.Contains(val, "Psychic Terrain"):
			ctx.State.Field.Terrain = "Psychic Terrain"
		case strings.Contains(val, "Misty Terrain"):
			ctx.State.Field.Terrain = "Misty Terrain"
		case strings.Contains(val, "Trick Room"):
			ctx.State.Field.TrickRoom = true
		case strings.Contains(val, "Gravity"):
			ctx.State.Field.Gravity = true
		}
	case "fieldend":
		switch {
		case strings.Contains(val, "Terrain"):
			ctx.State.Field.Terrain = ""
		case strings.Contains(val, "Trick Room"):
			ctx.State.Field.TrickRoom = false
		case strings.Contains(val, "Gravity"):
			ctx.State.Field.Gravity = false
		}
	}
}

func (b *ShowdownBot) onSideUpdate(roomID, data, updateType string) {
	ctx, exists := b.battles[roomID]
	if !exists || ctx.State == nil {
		return
	}

	parts := strings.Split(data, "|")
	if len(parts) < 2 {
		return
	}

	playerID := parts[0][:2]
	val := parts[1]
	if strings.HasPrefix(val, "move: ") {
		val = strings.TrimPrefix(val, "move: ")
	}

	var side *simulator.SideConditions
	if playerID == ctx.PlayerID {
		side = &ctx.State.P1.Side
	} else {
		side = &ctx.State.P2.Side
	}

	if updateType == "sidestart" {
		switch {
		case strings.Contains(val, "Stealth Rock"):
			side.StealthRock = true
		case val == "Spikes":
			if side.Spikes < 3 {
				side.Spikes++
			}
		case strings.Contains(val, "Toxic Spikes"):
			if side.ToxicSpikes < 2 {
				side.ToxicSpikes++
			}
		case strings.Contains(val, "Sticky Web"):
			side.StickyWeb = true
		case strings.Contains(val, "Reflect"):
			side.Reflect = true
		case strings.Contains(val, "Light Screen"):
			side.LightScreen = true
		case strings.Contains(val, "Aurora Veil"):
			side.AuroraVeil = true
		case strings.Contains(val, "Tailwind"):
			side.Tailwind = true
		case strings.Contains(val, "Safeguard"):
			side.Safeguard = true
		case strings.Contains(val, "Mist"):
			side.Mist = true
		}
	} else {
		switch {
		case strings.Contains(val, "Stealth Rock"):
			side.StealthRock = false
		case val == "Spikes":
			side.Spikes = 0
		case strings.Contains(val, "Toxic Spikes"):
			side.ToxicSpikes = 0
		case strings.Contains(val, "Sticky Web"):
			side.StickyWeb = false
		case strings.Contains(val, "Reflect"):
			side.Reflect = false
		case strings.Contains(val, "Light Screen"):
			side.LightScreen = false
		case strings.Contains(val, "Aurora Veil"):
			side.AuroraVeil = false
		case strings.Contains(val, "Tailwind"):
			side.Tailwind = false
		case strings.Contains(val, "Safeguard"):
			side.Safeguard = false
		case strings.Contains(val, "Mist"):
			side.Mist = false
		}
	}
}

func (b *ShowdownBot) onChallstr(challstr string) {
	log.Println("Received challstr, logging in...")

	var assertion string
	var err error

	if b.password != "" {
		assertion, err = Login(b.username, b.password, challstr)
	} else {
		assertion, err = LoginAsGuest(b.username, challstr)
	}

	if err != nil {
		log.Printf("Login failed: %v", err)
		return
	}

	// Send login command
	cmd := fmt.Sprintf("|/trn %s,0,%s", b.username, assertion)
	b.send("", cmd)
	log.Printf("Sent login command for user: %s", b.username)
}

func (b *ShowdownBot) onUpdateUser(data string) {
	parts := strings.SplitN(data, "|", 4)
	if len(parts) < 2 {
		return
	}
	username := strings.TrimSpace(parts[0])
	isGuest := parts[1] == "0"

	if isGuest {
		log.Printf("Logged in as guest: %s", username)
	} else {
		log.Printf("Successfully logged in as: %s", username)
	}

	// Queue for random battle
	log.Println("Searching for a random battle...")
	b.send("", "|/search randombattle")
}

func (b *ShowdownBot) onBattleInit(roomID string) {
	log.Printf("[%s] Battle started!", roomID)
	b.battles[roomID] = &BattleContext{
		RoomID: roomID,
	}
	// Join the battle room to receive messages
	b.send(roomID, "/join "+roomID)
	// Auto-enable battle timer
	b.send(roomID, "/timer on")
}

func (b *ShowdownBot) onPoke(roomID, data string) {
	// |poke|p1|Species, L80, M|item
	parts := strings.Split(data, "|")
	if len(parts) < 2 {
		return
	}

	playerID := parts[0]
	speciesInfo := parts[1]
	species, level, gender := ParseDetails(speciesInfo)

	ctx, exists := b.battles[roomID]
	if !exists {
		return
	}

	// Only care about opponent's team
	if ctx.PlayerID != "" && playerID == ctx.PlayerID {
		return
	}

	if ctx.OpponentID == "" {
		ctx.OpponentID = playerID
	}

	if ctx.State == nil {
		ctx.State = &simulator.BattleState{}
		ctx.State.P2.ID = playerID
		ctx.State.P2.TeamSize = 6 // Assume 6 for random battles
		ctx.State.P2.CanTerastallize = true
		for i := 0; i < 6; i++ {
			ctx.State.P2.Team[i] = simulator.PokemonState{
				Name:    "Unknown",
				Species: "Unknown",
				HP:      100,
				MaxHP:   100,
				Boosts:  114420174,
			}
		}
	}

	// Add to team if not already there
	found := false
	idx := -1
	for i := 0; i < ctx.State.P2.TeamSize; i++ {
		if ctx.State.P2.Team[i].Species == species {
			found = true
			break
		}
		if idx == -1 && ctx.State.P2.Team[i].Species == "Unknown" {
			idx = i
		}
	}

	if !found {
		if idx == -1 && ctx.State.P2.TeamSize < 6 {
			idx = ctx.State.P2.TeamSize
			ctx.State.P2.TeamSize++
		}
		if idx != -1 {
			ctx.State.P2.Team[idx].Species = species
			ctx.State.P2.Team[idx].Name = species
			ctx.State.P2.Team[idx].Level = level
			ctx.State.P2.Team[idx].Gender = gender
		}
	}
}

func (b *ShowdownBot) onPlayer(roomID, data string) {
	// |player|p1|Username|avatar|rating
	parts := strings.SplitN(data, "|", 4)
	if len(parts) < 2 {
		return
	}
	playerID := parts[0]
	playerName := parts[1]

	// Check if this player is us
	if strings.EqualFold(strings.TrimSpace(playerName), strings.TrimSpace(b.username)) {
		ctx, exists := b.battles[roomID]
		if exists {
			ctx.PlayerID = playerID
			if playerID == "p1" {
				ctx.OpponentID = "p2"
			} else {
				ctx.OpponentID = "p1"
			}
			log.Printf("[%s] We are %s (opponent is %s)", roomID, playerID, ctx.OpponentID)
		}
	}
}

func (b *ShowdownBot) onRequest(roomID, data string) {
	if data == "" {
		return
	}

	var req ShowdownRequest
	if err := json.Unmarshal([]byte(data), &req); err != nil {
		log.Printf("[%s] Failed to parse request: %v", roomID, err)
		return
	}

	// Store the request
	ctx, exists := b.battles[roomID]
	if !exists {
		ctx = &BattleContext{RoomID: roomID}
		b.battles[roomID] = ctx
	}
	ctx.Request = &req
	ctx.PlayerID = req.Side.ID

	// Don't act if we're waiting for the opponent
	if req.Wait {
		return
	}

	state := RequestToBattleState(&req, ctx.State)
	// Update context state with the latest synced version
	ctx.State = state
	DebugPrintState(roomID, state)
	logAttentionWeights(roomID, state)

	choice, actionIdx, searchResult := ChooseBestAction(&req, b.moveTime, ctx.State)
	if choice == "" {
		return
	}

	// Log eval info for all candidate actions
	if actionIdx >= 0 && len(searchResult.ActionScores) > 0 {
		log.Printf("[%s] === Eval (depth %d, %d nodes) ===", roomID, searchResult.Depth, searchResult.NodesSearched)

		// Sort actions for consistent display
		actions := make([]int, 0, len(searchResult.ActionScores))
		for a := range searchResult.ActionScores {
			actions = append(actions, a)
		}
		sort.Ints(actions)

		for _, a := range actions {
			label := bot.ActionToString(state, &state.P1, a)
			// Enrich move labels with actual move names from the request
			moveIdx := simulator.BaseMoveIndex(a)
			if simulator.IsAttackAction(a) && len(req.Active) > 0 && moveIdx >= 0 && moveIdx < len(req.Active[0].Moves) {
				if simulator.IsTeraAction(a) {
					label = fmt.Sprintf("move %d (%s) [tera]", moveIdx+1, req.Active[0].Moves[moveIdx].Move)
				} else {
					label = fmt.Sprintf("move %d (%s)", moveIdx+1, req.Active[0].Moves[moveIdx].Move)
				}
			}
			marker := "  "
			if a == searchResult.BestAction {
				marker = ">>"
			}
			log.Printf("[%s]   %s %-30s : %.4f", roomID, marker, label, searchResult.ActionScores[a])
		}

		bestLabel := bot.ActionToString(state, &state.P1, searchResult.BestAction)
		bestMoveIdx := simulator.BaseMoveIndex(searchResult.BestAction)
		if simulator.IsAttackAction(searchResult.BestAction) && len(req.Active) > 0 && bestMoveIdx >= 0 && bestMoveIdx < len(req.Active[0].Moves) {
			if simulator.IsTeraAction(searchResult.BestAction) {
				bestLabel = fmt.Sprintf("move %d (%s) [tera]", bestMoveIdx+1, req.Active[0].Moves[bestMoveIdx].Move)
			} else {
				bestLabel = fmt.Sprintf("move %d (%s)", bestMoveIdx+1, req.Active[0].Moves[bestMoveIdx].Move)
			}
		}
		log.Printf("[%s]   >> Best: %s = %.4f", roomID, bestLabel, searchResult.Score)
	}

	// Send the choice with rqid
	cmd := fmt.Sprintf("/choose %s|%d", choice, req.Rqid)
	b.send(roomID, cmd)
	log.Printf("[%s] Sent: %s", roomID, cmd)
}

func (b *ShowdownBot) onTeamPreview(roomID string) {
	ctx, exists := b.battles[roomID]
	if !exists || ctx.Request == nil {
		// Send default order
		b.send(roomID, "/choose default")
		return
	}

	ctx.Request.TeamPreview = true
	choice, _, _ := ChooseBestAction(ctx.Request, b.moveTime, ctx.State)
	cmd := fmt.Sprintf("/choose %s|%d", choice, ctx.Request.Rqid)
	b.send(roomID, cmd)
	log.Printf("[%s] Team Preview: %s", roomID, cmd)
}

func (b *ShowdownBot) onError(roomID, data string) {
	log.Printf("[%s] Error: %s", roomID, data)

	// If it's an invalid choice, try default
	if strings.Contains(data, "[Invalid choice]") || strings.Contains(data, "[Unavailable choice]") {
		b.send(roomID, "/choose default")
		log.Printf("[%s] Sent fallback: /choose default", roomID)
	}
}

func (b *ShowdownBot) onWin(roomID, winner string) {
	log.Printf("[%s] Game over! Winner: %s", roomID, winner)

	delete(b.battles, roomID)

	// Leave the room and re-queue
	b.send("", "|/leave "+roomID)

	log.Println("Re-queuing for next battle...")
	time.Sleep(2 * time.Second)
	b.send("", "|/search randombattle")
}

func (b *ShowdownBot) onTie(roomID string) {
	log.Printf("[%s] Game ended in a tie!", roomID)

	delete(b.battles, roomID)
	b.send("", "|/leave "+roomID)

	log.Println("Re-queuing for next battle...")
	time.Sleep(2 * time.Second)
	b.send("", "|/search randombattle")
}

func (b *ShowdownBot) onPM(data string) {
	parts := strings.SplitN(data, "|", 3)
	if len(parts) < 3 {
		return
	}
	sender := strings.TrimSpace(parts[0])
	message := parts[2]

	log.Printf("PM from %s: %s", sender, message)

	// Auto-accept challenges
	if strings.HasPrefix(message, "/challenge") {
		log.Printf("Accepting challenge from %s", sender)
		b.send("", fmt.Sprintf("|/accept %s", sender))
	}
}

func (b *ShowdownBot) send(roomID, message string) {
	var payload string
	if roomID != "" {
		payload = roomID + "|" + message
	} else {
		payload = message
	}

	if err := b.conn.WriteMessage(websocket.TextMessage, []byte(payload)); err != nil {
		log.Printf("Send error: %v", err)
	}
}

func logAttentionWeights(roomID string, state *simulator.BattleState) {
	_, attentionCache := evaluator.GetCaches()
	weights, ok := evaluator.AttentionWeights(state, attentionCache)
	if !ok {
		return
	}

	p1Labels := attentionSlotLabels(&state.P1)
	p2Labels := attentionSlotLabels(&state.P2)
	log.Printf("[%s] --- Attention Weights ---", roomID)
	for i := 0; i < 6; i++ {
		if p1Labels[i] == "None" {
			continue
		}
		log.Printf("[%s]   P1 %-24s : %.4f", roomID, p1Labels[i], weights[i])
	}
	for i := 0; i < 6; i++ {
		if p2Labels[i] == "None" {
			continue
		}
		log.Printf("[%s]   P2 %-24s : %.4f", roomID, p2Labels[i], weights[6+i])
	}
}

func attentionSlotLabels(player *simulator.PlayerState) [6]string {
	var labels [6]string
	idx := 0

	active := player.GetActive()
	if active != nil {
		labels[idx] = formatAttentionPokemonLabel(active)
		idx++
	}

	remaining := make([]*simulator.PokemonState, 0, player.TeamSize)
	for i := 0; i < player.TeamSize; i++ {
		poke := &player.Team[i]
		if poke.IsActive {
			continue
		}
		remaining = append(remaining, poke)
	}
	sort.Slice(remaining, func(i, j int) bool {
		return remaining[i].Species < remaining[j].Species
	})

	for _, poke := range remaining {
		if idx >= 6 {
			break
		}
		labels[idx] = formatAttentionPokemonLabel(poke)
		idx++
	}

	for idx < 6 {
		labels[idx] = "None"
		idx++
	}
	return labels
}

func formatAttentionPokemonLabel(poke *simulator.PokemonState) string {
	if poke == nil || poke.Species == "" {
		return "Unknown"
	}
	label := poke.Species
	if poke.Fainted {
		label += " (fnt)"
	}
	return label
}
