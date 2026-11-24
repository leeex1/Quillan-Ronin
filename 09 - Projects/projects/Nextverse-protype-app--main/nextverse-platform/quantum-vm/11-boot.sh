#!/bin/bash
# Quantum VM 11-Copy Boot Script
# Implements JQLD formula with 11 mirrored instances
# This script initializes and manages a cluster of simulated quantum core instances
# and a master coordination service.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration Section ---
# These environment variables can be overridden externally.
# QUANTUM_COPIES: Defines how many instances of the quantum-core-binary to start.
#                 Defaults to 11, aligning with the "11 Copy Principle".
QUANTUM_COPIES=${QUANTUM_COPIES:-11}
# JQLD_BOOST: A target value for the J-Quantum Algorithmic Logic Density (JQLD) boost.
#             This is a simulated performance metric. Defaults to 420.
JQLD_BOOST=${JQLD_BOOST:-420}
# TURBO_MODE: A flag that might enable/disable certain performance features (currently unused in this script).
TURBO_MODE=${TURBO_MODE:-true}

echo "⚡ Starting Quantum VM with ${QUANTUM_COPIES} instances..."
echo "🔧 JQLD Boost Target: ${JQLD_BOOST}x"

# --- Instance Setup Loop ---
# This loop creates, configures, and starts each quantum core instance.
for i in $(seq 1 $QUANTUM_COPIES); do
    # Create a dedicated directory for each instance.
    mkdir -p "instance-$i"
    
    # Copy the quantum core binary into the instance directory.
    # Assumes 'quantum-core-binary' is present in the current directory where 11-boot.sh is run.
    cp quantum-core-binary "instance-$i/"
    
    # Generate an instance-specific configuration file (config.json).
    # This uses a 'here document' (cat << EOF ... EOF) to write multi-line content.
    # The configuration includes:
    #   - instance_id: A unique ID for this instance.
    #   - total_copies: The total number of instances being run.
    #   - jqld_params: Parameters for the JQLD formula.
    #     - Njq, psi_parallel: Specific parameters for the JQLD calculation within the core binary.
    #     - boost_factor: The portion of the total JQLD_BOOST allocated to this instance.
    #                     Calculated using 'bc' for floating-point arithmetic.
    #                       `scale=2` sets the precision for the division.
    #   - ipc_port: A unique port number for this instance's Inter-Process Communication (IPC) or web server.
    #               Starts from 7001 and increments for each instance.
    #   - shared_memory: A path for a conceptual shared memory segment for this instance.
    cat > "instance-$i/config.json" << EOF
{
    "instance_id": $i,
    "total_copies": $QUANTUM_COPIES,
    "jqld_params": {
        "Njq": 5,
        "psi_parallel": 0.8,
        "boost_factor": $(echo "scale=2; $JQLD_BOOST / $QUANTUM_COPIES" | bc)
    },
    "ipc_port": $((7000 + $i)),
    "shared_memory": "/shared/quantum-mem-$i"
}
EOF

    # Start the quantum instance.
    echo "🚀 Starting Quantum Instance $i..."
    cd "instance-$i" # Change directory into the instance's folder.
    # Execute the quantum-core-binary in the background (`&`).
    # It's passed its config file and a '--daemon' flag (assuming the binary supports this).
    ./quantum-core-binary --config=config.json --daemon &
    # Capture the Process ID (PID) of the last backgrounded command ($!)
    # and store it in a .pid file in the parent directory.
    echo $! > "../quantum-$i.pid"
    cd .. # Return to the main quantum-vm directory.
    
    # Stagger the startup of instances slightly to avoid resource contention.
    sleep 0.5
done

# --- Instance Synchronization and Verification ---
# Wait for all instances to initialize and be ready.
echo "⏳ Waiting for Quantum instances to synchronize..."
sleep 5 # A simple fixed delay; more robust solutions might involve polling health endpoints.

# Verify JQLD boost activation across all instances.
# This section assumes each quantum-core-binary instance exposes an HTTP endpoint
# at `/boost` that returns JSON containing its `current_boost` value.
echo "🧮 Verifying JQLD boost activation..."
TOTAL_BOOST=0
for i in $(seq 1 $QUANTUM_COPIES); do
    # Construct the URL for the instance's /boost endpoint.
    # `curl -s` fetches the URL silently.
    # `jq -r '.current_boost'` parses the JSON response and extracts the raw string value of 'current_boost'.
    INSTANCE_BOOST=$(curl -s "http://localhost:$((7000 + $i))/boost" | jq -r '.current_boost')

    # Add robustness: Check if INSTANCE_BOOST is a valid number.
    if ! [[ "$INSTANCE_BOOST" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        echo "   WARNING: Instance $i did not return a valid boost number. Received: '$INSTANCE_BOOST'. Assuming 0 boost for this instance."
        INSTANCE_BOOST=0
    fi

    # Accumulate the total boost using 'bc' for floating-point addition.
    TOTAL_BOOST=$(echo "$TOTAL_BOOST + $INSTANCE_BOOST" | bc)
    echo "   Instance $i: ${INSTANCE_BOOST}x boost"
done

echo "🎯 Total Quantum Boost: ${TOTAL_BOOST}x (Target: ${JQLD_BOOST}x)"

# Compare the achieved total boost with the target JQLD_BOOST.
# `bc -l` is used for the comparison, treating the expression as a boolean (1 for true, 0 for false).
if (( $(echo "$TOTAL_BOOST >= $JQLD_BOOST" | bc -l) )); then
    echo "✅ JQLD Formula validation PASSED"
else
    echo "⚠️  JQLD Formula validation WARNING: Below target"
fi

# --- Master Coordination Service Setup ---
# This section creates and starts the quantum-master.py script.
# The script is embedded directly into this shell script using a 'here document'.
# This is an unconventional way to deploy a Python script; typically, it would be a separate file.
echo "🔗 Starting Quantum Coordination Master..."
# ==============================================================================
# BEGIN EMBEDDED PYTHON SCRIPT: quantum-master.py
# Note: This Python script is embedded within the shell script using a heredoc.
# For easier maintenance and development, consider managing this as a separate
# Python file in future versions.
# ==============================================================================
cat > quantum-master.py << 'EOF'
#!/usr/bin/env python3
# This Python script is generated and launched by the 11-boot.sh script.
# It acts as a master coordinator for the quantum core instances.

import asyncio
import json
import aiohttp
from aiohttp import web
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- QuantumMaster Class ---
# Handles task coordination, distribution to instances, and result aggregation.
class QuantumMaster:
    def __init__(self, copies=11):
        # `copies` should ideally be dynamically set from an environment variable
        # or argument passed by 11-boot.sh to match QUANTUM_COPIES.
        # Here, it's hardcoded but matches the default QUANTUM_COPIES.
        self.copies = copies
        self.instances = {} # This dictionary is initialized but not used in the current script.
        
    async def coordinate_task(self, task):
        """
        Distributes a task across the quantum instances.
        This is a simplified simulation of task distribution.
        It conceptually divides task data among instances.
        """
        if not task.get('data'): # Handles empty list or if 'data' key is missing
            logging.warning("coordinate_task received task with no data.")
            return {'status': 'success', 'total_boost': 0, 'results': [], 'message': 'No data to process.'}
        if self.copies == 0:
            logging.error("coordinate_task error: self.copies is 0, cannot distribute tasks.")
            return {'status': 'error', 'total_boost': 0, 'results': [], 'message': 'No instances configured.'}

        tasks = []
        
        # Data distribution logic:
        # If len(data) < self.copies, chunk_size will be 0.
        # Slicing with chunk_size=0: task['data'][0:0] is empty.
        # This means if there are fewer data items than copies,
        # some initial instances will get one item each, and subsequent instances will get empty chunks.
        # For example, 3 items, 11 copies:
        # instance 0 gets item 0 (chunk [data[0]])
        # instance 1 gets item 1 (chunk [data[1]])
        # instance 2 gets item 2 (chunk [data[2]])
        # instances 3-10 get empty chunks.
        # This behavior is kept for this refinement, but more sophisticated distribution might be needed for optimal load balancing.
        if len(task.get('data', [])) < self.copies and len(task.get('data', [])) > 0 :
            chunk_size = 1 # Assign one item per instance until data runs out
        elif len(task.get('data',[])) == 0:
            chunk_size = 0 # Should be caught by earlier check
        else:
            chunk_size = len(task.get('data', [])) // self.copies

        for i in range(self.copies):
            # Determine the URL for each quantum core instance.
            # Note: Port calculation is 7000 + i + 1, which matches instance startup (7001, 7002, ...).
            instance_url = f"http://localhost:{7000 + i + 1}"
            
            # Divide data into chunks for each instance.
            if len(task.get('data', [])) < self.copies and len(task.get('data',[])) > 0: # Small data set
                if i < len(task.get('data', [])):
                    chunk = [task['data'][i]] # Give one item to this instance
                else:
                    chunk = [] # No more data for this instance
            elif chunk_size > 0 : # Normal case or data evenly divisible
                chunk = task['data'][i*chunk_size:(i+1)*chunk_size if (i+1)*chunk_size <= len(task.get('data',[])) else len(task.get('data',[])) ]
                # Ensure last chunk takes all remaining if not perfectly divisible
                if i == self.copies - 1 and (i+1)*chunk_size > len(task.get('data',[])): # Check if it's the last instance and if there's a remainder
                     chunk = task['data'][i*chunk_size:]
            else: # No data or chunk_size is 0 for other reasons (should be minimal with above checks)
                chunk = []

            # Prepare data payload for the individual instance.
            task_data = {
                'chunk_id': i,
                'total_chunks': self.copies,
                'data': chunk,
                'operation': task.get('operation', 'compute') # Default operation is 'compute'.
            }
            
            # Append the asynchronous call to self.send_to_instance to a list of tasks.
            tasks.append(self.send_to_instance(instance_url, task_data))
        
        # Execute all instance calls concurrently using asyncio.gather.
        results = await asyncio.gather(*tasks)
        # Merge the results obtained from all instances.
        return self.merge_results(results)
    
    async def send_to_instance(self, url, data):
        """
        Sends a piece of the task (data) to a specific quantum core instance.
        Assumes the instance has an /execute endpoint that accepts POST requests with JSON.
        Includes timeout and error handling for HTTP and JSON parsing issues.
        """
        # Default error response
        error_response = {
            'status': 'error',
            'message': 'Failed to get valid response from instance',
            'instance_url': url,
            'http_status': None,
            'data': [] # Ensure data field for consistent merging
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{url}/execute", json=data, timeout=10) as resp: # Added timeout
                    error_response['http_status'] = resp.status
                    if resp.status == 200:
                        try:
                            return await resp.json()
                        except aiohttp.ContentTypeError as e:
                            logging.error(f"Instance {url} returned non-JSON response: {e}")
                            error_response['message'] = f"Instance returned non-JSON response: {await resp.text()}"
                            return error_response
                    else:
                        logging.error(f"Instance {url} returned HTTP status {resp.status}: {await resp.text()}")
                        error_response['message'] = f"Instance returned HTTP {resp.status}: {await resp.text()}"
                        return error_response
        except aiohttp.ClientError as e: # Catches connection errors, timeouts, etc.
            logging.error(f"ClientError connecting to instance {url}: {e}")
            error_response['message'] = f"ClientError connecting to instance: {e}"
            return error_response
        except asyncio.TimeoutError:
            logging.error(f"Timeout connecting to instance {url}")
            error_response['message'] = "Timeout connecting to instance"
            return error_response
    
    def merge_results(self, results):
        """
        Merges results from all quantum instances.
        Aggregates 'boost_applied', collects 'data', and lists any errors.
        Determines overall status based on individual instance results.
        """
        merged = {'status': 'success', 'total_boost': 0, 'aggregated_results': [], 'errors': []}
        all_successful = True
        for result in results:
            if result.get('status') == 'success':
                merged['total_boost'] += result.get('boost_applied', 0)
                # Assuming result['data'] is the actual result from an instance, not list of items
                merged['aggregated_results'].append(result.get('data'))
            else:
                all_successful = False
                merged['errors'].append(result) # Append the whole error dict from send_to_instance

        if not all_successful and not merged['aggregated_results']: # No successful results at all
            merged['status'] = 'error'
        elif not all_successful: # Partial success
            merged['status'] = 'partial_success'
        
        # Rename 'results' to 'aggregated_results' to avoid confusion if instance result also has a 'results' key
        return merged

    # --- Web API Endpoints ---
    async def health_check(self, request):
        """
        Handles GET requests to /health.
        Returns a JSON response indicating the master is healthy and the number of instances it manages.
        """
        return web.json_response({'status': 'healthy', 'instances': self.copies})
    
    async def execute(self, request):
        """
        Handles POST requests to /execute.
        Expects a JSON payload representing the task, coordinates it, and returns the merged result.
        """
        data = await request.json()
        result = await self.coordinate_task(data)
        return web.json_response(result)

# --- Main Execution Block for quantum-master.py ---
if __name__ == '__main__':
    # Create an instance of the QuantumMaster.
    # Consider passing QUANTUM_COPIES from boot script here.
    master = QuantumMaster()

    # Create an aiohttp web application.
    app = web.Application()
    # Register routes for the health check and task execution endpoints.
    app.router.add_get('/health', master.health_check)
    app.router.add_post('/execute', master.execute)
    
    print("🔗 Quantum Master running on http://localhost:7777")
    # Run the aiohttp application on localhost, port 7777.
    web.run_app(app, host='localhost', port=7777)
# ==============================================================================
# END EMBEDDED PYTHON SCRIPT: quantum-master.py
# ==============================================================================
EOF

# Make the generated quantum-master.py executable (though it's run with python3 directly).
# chmod +x quantum-master.py # This line is not strictly necessary as it's called with `python3 quantum-master.py`

# Start the quantum-master.py script in the background.
python3 quantum-master.py &
# Capture its PID and store it in quantum-master.pid.
echo $! > quantum-master.pid

echo "✅ Quantum VM Boot Complete!"
echo "🔗 Master Coordinator: http://localhost:7777"
echo "📊 Instance Status: http://localhost:7777/health"
