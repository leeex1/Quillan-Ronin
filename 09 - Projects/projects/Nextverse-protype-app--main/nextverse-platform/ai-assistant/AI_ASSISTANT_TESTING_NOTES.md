# AI Assistant Manual Testing Notes

## Test: Verify AI Chat Boost Indication

**Objective:** To confirm that the AI Assistant's response includes the `"[QVM-Enhanced by distilgpt2]"` fragment when using the configured `QLRC-6B_simulated` LLM variant. This fragment serves as a basic "boost active" indicator for the MVP.

**Status:** Prepared. Execution pending manual verification.

**Prerequisites:**
1.  Ensure you have Python installed in your environment.
2.  Navigate to the `nextverse-platform/ai-assistant/` directory in your terminal.
3.  Install necessary Python dependencies: `pip install -r requirements.txt`. (This was updated to include `python-socketio[client]`).

**Steps to Execute:**

1.  **Start the AI Assistant Service:**
    *   In your terminal, from the `nextverse-platform/ai-assistant/` directory, run:
        ```bash
        python main.py
        ```
    *   Observe the output to ensure the Flask-SocketIO server starts correctly (e.g., running on `http://0.0.0.0:5000/`).

2.  **Run the AI Assistant Test Client:**
    *   Open a **new** terminal window or tab.
    *   Navigate to the `nextverse-platform/ai-assistant/` directory.
    *   Run the client script created for this test:
        ```bash
        python ai_assistant_test_client.py
        ```

3.  **Observe Client Output:**
    *   The client will attempt to connect to `ws://localhost:5000`.
    *   It will send a predefined test message.
    *   The client script is designed to print a specific success message if the boost fragment is found:
        `SUCCESS: Confirmed '[QVM-Enhanced by distilgpt2]' fragment!`
    *   It will also print the full JSON response from the AI. Manually inspect the `ai_response` field within this JSON for the presence of the `[QVM-Enhanced by distilgpt2]` string.

4.  **Cleanup:**
    *   The `ai_assistant_test_client.py` script should disconnect automatically.
    *   Stop the AI service (`python main.py`) in the first terminal by pressing `Ctrl+C`.

**Troubleshooting Notes:**
*   If the client cannot connect, ensure `main.py` is running and accessible on `localhost:5000`.
*   If the boost fragment is missing, verify that `SELECTED_LLM_VARIANT_KEY` in `nextverse-platform/ai-assistant/utils/config.py` is set to `"QLRC-6B_simulated"`.

---

## Test: AI to Game Engine Communication (Integration)

**Objective:** To confirm that the AI Assistant can send a "move_to" command to the Game Engine, and the Game Engine's `AICommandServer` receives, parses, and processes this command for a designated AI-controlled avatar.

**Status:** Prepared. Execution pending manual verification.

**Prerequisites:**
1.  **Game Engine Compiled:** Ensure the C++ Game Engine (`nextverse-platform/game_engine/`) has been compiled with the latest changes including the `AICommandServer` and AI avatar management. The Game Engine should be listening on port 8081 (or as configured in `GAME_ENGINE_COMMAND_PORT`).
2.  **AI Assistant Dependencies:** `pip install -r requirements.txt` has been run for the AI assistant.
3.  **AI Assistant Configuration:**
    *   `nextverse-platform/ai-assistant/utils/config.py` should have `GAME_ENGINE_HOST` set to `"localhost"` (or the correct host if testing across different machines/containers not covered by Docker Compose service names).
    *   `GAME_ENGINE_COMMAND_PORT` should be set to `8081` (matching the Game Engine).

**Temporary Modification for Test Trigger:**
For this manual integration test, a temporary modification to `nextverse-platform/ai-assistant/core/ai_brain.py` is required to easily trigger the command sending.
*   In the `AIBrain.generate_response` method, after processing any user input (e.g., before the final `return response_data`), add the following line to send a command for "ai_1":
    ```python
    # ---- TEMPORARY TEST CODE ----
    if hasattr(self, '_control_ai_avatar_placeholder'):
        print("AI_BRAIN_TEST_TRIGGER: Calling _control_ai_avatar_placeholder for ai_1")
        # This example sends commands for ai_1 and ai_2. Adjust if only one is needed for this specific test.
        self._control_ai_avatar_placeholder(avatar_id="ai_1", target_x=10.0, target_y=15.0, target_z=1.0)
        # self._control_ai_avatar_placeholder(avatar_id="ai_2", target_x=-10.0, target_y=18.0, target_z=1.0) # Example for a second avatar
    # ---- END TEMPORARY TEST CODE ----
    ```
*   **Remember to remove this temporary code after testing.** (The actual temporary code in `ai_brain.py` sends for "ai_1" and "ai_2").

**Steps to Execute:**

1.  **Start the Game Engine:**
    *   Navigate to your Game Engine's build directory (e.g., `nextverse-platform/game_engine/build/` or wherever the executable `NextVerseGameEngineApp` is).
    *   Run the game engine executable:
        ```bash
        ./NextVerseGameEngineApp
        ```
    *   Observe its console output. You should see messages indicating the `AICommandServer` has started and is listening on its configured port (e.g., 8081).

2.  **Start the AI Assistant Service:**
    *   In a new terminal, navigate to `nextverse-platform/ai-assistant/`.
    *   Run:
        ```bash
        python main.py
        ```
    *   Ensure it starts without errors.

3.  **Trigger the AI Command (via AI Test Client):**
    *   Open a **third** terminal window.
    *   Navigate to `nextverse-platform/ai-assistant/`.
    *   Run the `ai_assistant_test_client.py`:
        ```bash
        python ai_assistant_test_client.py
        ```
    *   This will send a message to the AI Assistant. Due to the temporary modification in `AIBrain`, receiving this message will trigger the `_control_ai_avatar_placeholder` call(s).

4.  **Observe Console Outputs:**
    *   **AI Assistant Console:**
        *   Look for `AI_BRAIN_INTEGRATION_TEST: Attempting to send AI avatar control commands.`.
        *   Look for `AIBrain: Sending command to Game Engine: {'avatar_id': 'ai_1', ...}` (and for 'ai_2' if enabled).
        *   Look for `AIBrain: Game Engine command send status: True` (or `False` if there was an issue).
    *   **Game Engine Console:**
        *   Look for `AICommandServer: Accepted connection from ...`.
        *   Look for `AICommandServer: Received data: {"avatar_id": "ai_1", ...}`.
        *   Look for `AICommandServer: Pushed command to queue for avatar_id: ai_1, action: move_to`.
        *   Look for `Game: Processing AI Command for avatar 'ai_1' action 'move_to'.`
        *   Look for `Game: Moved AI avatar 'ai_1' to (coordinates).` (and similar for 'ai_2').

5.  **Cleanup:**
    *   The `ai_assistant_test_client.py` should disconnect automatically.
    *   Stop the AI service (`python main.py`) using `Ctrl+C`.
    *   Stop the Game Engine (`NextVerseGameEngineApp`) using `Ctrl+C` or by closing its window.
    *   **Remove the temporary trigger code from `AIBrain.generate_response`**.

**Troubleshooting Notes:**
*   **Connection Issues:**
    *   If the AI Assistant logs "Game Engine command send status: False" or connection errors:
        *   Ensure the Game Engine is running and its `AICommandServer` has started on the correct port (e.g., 8081).
        *   Verify `GAME_ENGINE_HOST` (`localhost`) and `GAME_ENGINE_COMMAND_PORT` (`8081`) in the AI Assistant's `config.py` match where the Game Engine is listening.
        *   Check for firewall issues if running not on localhost or in complex network setups (especially if Game Engine is local and AI is in Docker trying to reach localhost on host).
*   **Game Engine Doesn't Process Command:**
    *   If the Game Engine console doesn't show the expected "Received data" or "Processing AI Command" messages:
        *   Verify the `AICommandServer` started correctly.
        *   Double-check the port numbers.
        *   Ensure the JSON format sent by `GameEngineClient` exactly matches what `parse_json_command_mvp` in `AICommandServer.cpp` expects (it's very basic and sensitive to format).

### Future Unit Tests for Game Engine Server

The `AICommandServer` in the C++ Game Engine, particularly its JSON parsing (`parse_json_command_mvp`) and command handling logic, would benefit from dedicated C++ unit tests (e.g., in a new `TestAICommandServer.cpp` within `game_engine/tests/`) in the future to ensure robustness independently of live network tests. These tests could mock incoming socket data with various valid and invalid JSON strings to verify the parser and command queuing.

---

## Test: Interactive Avatar Scene (1 Human, 2 AI)

**Objective**: To verify that the Game Engine renders a scene with one human-controlled avatar and two AI-controlled avatars, where the AI avatars' positions are updated based on commands sent by the AI Assistant, **and to verify basic contextual AI queries from the Game Engine to the AI Assistant.**

**Status**: Prepared. Execution pending manual verification.

**Prerequisites**:
*   All prerequisites from "Test: AI to Game Engine Communication (Integration)" (Python dependencies installed, services runnable).
*   Ensure Game Engine is built with the latest changes (including `AICommandServer`, human avatar control - specifically the loading of "ai_1" and "ai_2" avatars, the 'W'/ 'Esc' key controls for the human avatar, and the 'I' key for contextual query).
*   Ensure AI Assistant is using the latest code (including the temporary trigger in `AIBrain.generate_response` that sends `move_to` commands for both "ai_1" and "ai_2", and the `/contextual_query` endpoint).

**Steps to Execute**:

1.  **Start the Game Engine.**
    *   Navigate to the Game Engine's build directory (e.g., `nextverse-platform/game_engine/build/` or wherever the executable `NextVerseGameEngineApp` is located).
    *   Run the `NextVerseGameEngineApp` executable.
    *   **Expected**: The Game Engine window appears, showing the scene. You should be able to identify:
        *   The human-controlled avatar (e.g., a "panda" model).
        *   Two AI-controlled avatars: "ai_1" (e.g., an rgbCube) and "ai_2" (e.g., a box model) at their initial positions.
        *   The Game Engine console should show that the `AICommandServer` is listening (e.g., on port 8081).

2.  **Start the AI Assistant Service.**
    *   In a separate terminal, navigate to `nextverse-platform/ai-assistant/`.
    *   Run `python main.py`.
    *   Observe its console for successful startup and that the `/contextual_query` route is available.

3.  **Trigger AI Avatar Commands (and verify initial scene).**
    *   In a third terminal, navigate to `nextverse-platform/ai-assistant/`.
    *   Run `python ai_assistant_test_client.py`. This sends a message to the AI Assistant.
    *   The AI Assistant (due to the temporary code in `AIBrain.generate_response`) will then send two `move_to` commands for "ai_1" (to X=5, Y=10, Z=0) and "ai_2" (to X=-5, Y=12, Z=0).

4.  **Observe the Game Engine for AI Avatar Movement.**
    *   **Primary Observation (Visual)**: In the Game Engine window, the two AI avatars ("ai_1" and "ai_2") should visually change their positions from their initial startup locations to the new coordinates sent by the AI Assistant. This change will be instantaneous.
    *   **Secondary Observation (Console Logs for AI Movement)**:
        *   **Game Engine Console**: Confirm logs for `AICommandServer` receiving commands and `Game` processing them for "ai_1" and "ai_2".
        *   **AI Assistant Console**: Confirm logs for `AIBrain` sending commands.

5.  **Test Contextual AI Query.**
    *   Focus the Game Engine window.
    *   Press the 'I' key. This is configured to send a query about "crystal_A".
    *   **Observe Game Engine Console**:
        *   `Game: Sending contextual query to AI: {"event_type": "inspect_object", "object_id": "crystal_A", ...}`
        *   `Game: AI Contextual Response (Raw): {"status":"success","ai_generated_message":"This is the Crystal of Alacrity. It hums with a faint energy.","original_query":{...}}` (or similar)
        *   `Game (AI Says - Contextual): This is the Crystal of Alacrity. It hums with a faint energy.`
    *   **Observe AI Assistant Console**:
        *   `/contextual_query: Received data: {'event_type': 'inspect_object', 'object_id': 'crystal_A', ...}`
        *   `AIBrain: Processing game context query: event_type='inspect_object', object_id='crystal_A', ...`
        *   `/contextual_query: Sending response: {'status': 'success', 'ai_generated_message': 'This is the Crystal of Alacrity. It hums with a faint energy.', ...}`

6.  **Test Human Avatar Control.**
    *   Focus the Game Engine window.
    *   Press and hold the 'W' key. The human-controlled avatar should move forward.
    *   Press the 'Esc' key. The Game Engine window should close, and the application should terminate gracefully.

**Expected Outcome Summary**:
*   The Game Engine window correctly displays the initial scene with one human avatar and two AI avatars.
*   AI avatars "ai_1" and "ai_2" visually move to their new target positions after the AI Assistant is triggered.
*   Pressing 'I' in the Game Engine triggers a contextual query, and the AI's predefined response for "crystal_A" is printed to the Game Engine console.
*   Relevant log messages appear in Game Engine and AI Assistant consoles confirming all command and query flows.
*   The human-controlled avatar responds to 'W' key presses for movement.
*   The Game Engine application quits cleanly when the 'Esc' key is pressed.

**Cleanup**:
*   Stop the AI Assistant service (`Ctrl+C` in its terminal).
*   The Game Engine should have exited via the 'Esc' key. If not, close its window or use `Ctrl+C` in its terminal.
*   **Crucially, remember to remove the temporary command-sending code from `AIBrain.generate_response` in `ai_brain.py` after this test.**

---

## Known Issues, MVP Limitations, and Next Steps (Post-Manual-Testing)

*   **"Boost Mechanism is a Placeholder:"**
    *   The current 'boost' integration (indicated by `[QVM-Enhanced by...]` in AI responses) is a placeholder. True performance impact or calculation is not yet implemented. This was for testing the communication pathway and configuration selection.

*   **"Game Engine HTTP Client for Contextual AI:"**
    *   The C++ `HTTPClient` usage in `Game::sendContextualQueryToAI` is a simplified synchronous-style implementation for MVP. For more robust, non-blocking requests that don't stall the game loop, Panda3D's asynchronous task management features with `HTTPClient` should be utilized in a future iteration.

*   **"Game Engine AICommandServer JSON Parsing:"**
    *   The `AICommandServer` in the Game Engine uses a very basic manual string parser for incoming JSON commands from the AI Assistant. This approach is fragile and not suitable for complex JSON or production use. It should be replaced with a proper C++ JSON library (e.g., nlohmann/json) if command structures evolve or for increased robustness.

*   **"AI Avatar Movement is Basic:"**
    *   AI-controlled avatars currently have their positions set directly by the `Game` class upon receiving a 'move_to' command. This results in instantaneous teleportation. For smoother and more natural movement, techniques like interpolation towards a target, pathfinding, and integration with an animation system would be necessary.

*   **"Contextual AI Responses are Hardcoded:"**
    *   The AI's responses to contextual queries from the Game Engine (e.g., when the player inspects 'crystal_A') are currently predefined strings within `AIBrain.process_game_context_query`. A more dynamic and intelligent system for contextual understanding and response generation (likely involving the LLM and NLU components) is a key area for future development.

*   **"Temporary Test Code in AIBrain and Game Engine:"**
    *   The logic in `AIBrain.generate_response` that unconditionally sends `move_to` commands for "ai_1" and "ai_2" is temporary and was implemented specifically for testing the AI-to-Game-Engine communication pipeline. This must be removed and replaced with intentional AI-driven avatar control logic.
    *   Similarly, the 'I' key in the Game Engine triggering a query for a hardcoded 'crystal_A' object ID is for MVP demonstration purposes and should be replaced with a more dynamic object selection/interaction mechanism.

*   **"Further Unit Testing for AICommandServer:"**
    *   The C++ `AICommandServer`'s internal logic, especially the `parse_json_command_mvp` function and the command dispatching/queuing mechanism, would significantly benefit from dedicated C++ unit tests (e.g., in a new `TestAICommandServer.cpp`). This was noted previously and remains an important next step for reliability.
