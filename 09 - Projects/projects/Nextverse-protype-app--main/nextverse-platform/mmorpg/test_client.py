import zmq
import time
import threading

def run_test_client():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)

    server_address = "tcp://localhost:5555"
    print(f"REQ/REP Client: Connecting to server at {server_address}...")
    try:
        socket.connect(server_address)
    except zmq.error.ZMQError as e:
        print(f"REQ/REP Client: Could not connect to server: {e}")
        return

    commands = [
        "CREATE_ACCOUNT testuser testpass test@example.com",
        "LOGIN testuser testpass",
        "LIST_CHARS 1", # Assuming AccountID 1 from LOGIN
        "CREATE_CHAR 1 TestCharNet",
        "ENTER_WORLD 1", # Assuming CharID 1 from CREATE_CHAR
        "GET_STATE 1",
        "INVENTORY 1",
        "PICKUP 1 1 5",  # Iron Ore
        "PICKUP 1 2 3",  # Wood Log
        "PICKUP 1 7 5",  # Quantum Shard
        "PICKUP 1 8 1",  # Unstable Matrix
        "INVENTORY 1",
        "CRAFT 1 101",   # Smelt Iron Ingot
        "CRAFT 1 901",   # Quantum Infuse Proto-Blade (needs Proto-Blade ID 9, which is not explicitly picked up here yet - assuming recipe gives it or it's a starting item)
                         # For a full test, would need to ensure Proto-Blade (ID 9) is acquired first.
                         # For now, this tests if the QVM path is taken.
        "INVENTORY 1",
        "SAY 1 Hello from ZMQ!",
        "MOVE 1 north",
        "LOOK 1",
        "HAIL 1 1",      # Hail Guard Thompson (NPC Instance ID 1)
        "HAIL 1 3",      # Hail Forest Goblin Alpha (NPC Instance ID 3)
        "QUESTS_AVAILABLE 1 1",
        "QUEST_ACCEPT 1 1",
        "QUEST_LOG 1",
        # To complete Quest 1 (Collect 3 Iron Ore), we already picked up 5.
        "QUEST_COMPLETE 1 1 1",
        "QUEST_LOG 1",
        "GET_STATE 1",           # Check XP
        # Attacking NPC (Goblin ID 3, 50HP, 3DEF. Player Atk 10. Dmg = 7. Needs ~8 hits)
        "ATTACK 1 npc_3", "ATTACK 1 npc_3", "ATTACK 1 npc_3", "ATTACK 1 npc_3",
        "ATTACK 1 npc_3", "ATTACK 1 npc_3", "ATTACK 1 npc_3", "ATTACK 1 npc_3",
        "MOVE 1 starting_zone_forest_edge",
        "SAY 1 Goodbye from the forest edge!",
        "DISCONNECT 1"
    ]

    for cmd in commands:
        print(f"\nREQ/REP Client: Sending: {cmd}")
        try:
            socket.send_string(cmd)
            if socket.poll(7000): # Increased timeout for potentially slower QVM or DB ops
                reply = socket.recv_string()
                print(f"REQ/REP Client: Received: {reply}")
            else:
                print("REQ/REP Client: ERROR: No response from server within timeout period.")
        except zmq.error.ZMQError as e:
            print(f"REQ/REP Client: ZMQ Error: {e}")
            break
        except Exception as e:
            print(f"REQ/REP Client: An unexpected error occurred: {e}")
            break

    print("\nREQ/REP Client: Test finished.")
    socket.close()
    context.term()

def subscriber_thread(pub_address="tcp://localhost:5556"):
    sub_context = zmq.Context()
    sub_socket = sub_context.socket(zmq.SUB)
    print(f"PUB/SUB Client: Connecting to PUB server at {pub_address}...")
    try:
        sub_socket.connect(pub_address)
        sub_socket.subscribe("") # Subscribe to ALL topics
        print("PUB/SUB Client: Connected and subscribed to ALL topics.")
    except zmq.error.ZMQError as e:
        print(f"PUB/SUB Client: Could not connect: {e}")
        return

    print("PUB/SUB Client: Waiting for messages...")
    run_duration = 45 # Extended duration to catch more events
    end_time = time.time() + run_duration

    while time.time() < end_time:
        try:
            if sub_socket.poll(1000, zmq.POLLIN):
                topic = sub_socket.recv_string()
                message = sub_socket.recv_string()
                print(f"[PUB/SUB] Topic: {topic}, Message: {message}")
        except zmq.error.ZMQError as e:
            if e.errno == zmq.ETERM:
                print("[PUB/SUB] Context terminated.")
                break
            print(f"[PUB/SUB] ZMQ Error: {e}")
            break
        except Exception as e:
            print(f"[PUB/SUB] An unexpected error: {e}")
            break

    print("PUB/SUB Client: Subscriber thread finishing.")
    try:
        if not sub_socket.closed: sub_socket.close()
        if not sub_context.closed: sub_context.term()
    except Exception as e:
        print(f"PUB/SUB Client: Error during cleanup: {e}")

if __name__ == "__main__":
    print("Test client starting in 3 seconds...")
    time.sleep(3)

    pub_sub_thread = threading.Thread(target=subscriber_thread, daemon=True)
    pub_sub_thread.start()

    run_test_client()

    print("REQ/REP tests finished. Waiting for final PUB/SUB messages (approx 15s)...")
    pub_sub_thread.join(timeout=15)

    print("Main test script finished.")
