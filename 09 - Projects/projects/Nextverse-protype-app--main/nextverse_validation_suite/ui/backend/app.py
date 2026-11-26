# NextVerse UI (verified 2026-11-26): formula_engine.py now computes deterministic
# math. The random fallback below is only reachable if the engine import fails;
# it is not validation evidence and should be removed once the import is guaranteed.
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import os
import sys
import time  # Should be in formula_engine, but guide has it here too
import threading  # For background tasks

# Adjust path to import NextVerseFormulaEngine
# Assuming 'core' is a sibling directory to 'ui' inside 'nextverse_validation_suite'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Corrected path to be .../nextverse_validation_suite/
# So core is BASE_DIR/core
sys.path.append(BASE_DIR)  # This should add 'nextverse_validation_suite' to path

# Conditional import for formula_engine
try:
    from core.formula_engine import NextVerseFormulaEngine, FormulaResult
except ImportError:
    print("Warning: NextVerseFormulaEngine not found. Using dummy engine.")

    # Define dummy classes if the real one isn't available yet
    # This helps in setting up the Flask app structure first.
    class FormulaResult:
        def __init__(
            self,
            formula_name,
            theoretical_value,
            measured_performance,
            speedup_factor,
            execution_time,
            parameters,
            success,
            error_message=None,
        ):
            self.formula_name = formula_name
            self.theoretical_value = theoretical_value
            self.measured_performance = measured_performance
            self.speedup_factor = speedup_factor
            self.execution_time = execution_time
            self.parameters = parameters
            self.success = success
            self.error_message = error_message

    class NextVerseFormulaEngine:
        def __init__(self):
            self.formulas = [
                "JQLD",
                "LVVM",
                "DESS",
                "JRRN",
                "LRPP",
                "DVVE",
                "JSSC",
                "LSSS",
                "DNNL",
                "JHFR",
                "LMCB",
            ]
            print("Using Dummy Formula Engine")

        def _get_dummy_result(self, formula_name):
            import random

            time.sleep(random.uniform(0.1, 0.2))  # Simulate work
            is_success = random.choice(
                [True, True, True, False]
            )  # Higher chance of success for dummy
            return FormulaResult(
                formula_name=formula_name,
                theoretical_value=random.uniform(100, 1000),
                measured_performance=random.uniform(50, 500),
                speedup_factor=random.uniform(1, 25) if is_success else 0,
                execution_time=random.uniform(0.05, 0.5),
                parameters={"param1": "dummy_val", "param2": random.randint(1, 100)},
                success=is_success,
                error_message="Dummy error message" if not is_success else None,
            )

        def calculate_jqld(
            self, C=1.0, Njq=5, psi_task=0.6, psi_parallel=0.8, theta_q=0.02
        ):
            # print(f"Dummy JQLD called with: C={C}, Njq={Njq}, psi_task={psi_task}, psi_parallel={psi_parallel}, theta_q={theta_q}")
            return self._get_dummy_result("JQLD")

        def calculate_lvvm(
            self,
            Q_from_jqld=100.0,
            R_vm=0.98,
            psi_vm=0.25,
            theta_vm=0.02,
            theta_sync=0.01,
        ):
            # print(f"Dummy LVVM called with: Q_from_jqld={Q_from_jqld}, R_vm={R_vm}, psi_vm={psi_vm}, etc.")
            return self._get_dummy_result("LVVM")

        def __getattr__(self, name):
            if name.startswith("calculate_"):
                formula_name_upper = name.split("_", 1)[1].upper()
                # Ensure that it's one of the known formulas to avoid infinite recursion on other attributes
                if formula_name_upper in self.formulas:
                    # print(f"Dummy method for {formula_name_upper} called via __getattr__")
                    return lambda **kwargs: self._get_dummy_result(formula_name_upper)
            # Important: raise AttributeError for other attributes to ensure normal behavior
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        def run_complete_validation(self):
            results = {}
            q_val_for_lvvm = 0  # Default Q value if JQLD fails or not run first

            jqld_res = self.calculate_jqld()  # JQLD is usually first
            results["JQLD"] = jqld_res
            if jqld_res.success:
                q_val_for_lvvm = jqld_res.theoretical_value

            # Simulate running other formulas
            for formula_name in self.formulas:
                if formula_name == "JQLD":
                    continue  # Already done

                if formula_name == "LVVM":
                    res = self.calculate_lvvm(Q_from_jqld=q_val_for_lvvm)
                else:
                    # Generic call using __getattr__ for other dummy formulas
                    method_to_call = getattr(self, f"calculate_{formula_name.lower()}")
                    res = method_to_call()
                results[formula_name] = res

            compound_speedup = 1.0
            successful_tests_count = 0
            for res in results.values():
                if res.success:
                    successful_tests_count += 1
                    if (
                        res.speedup_factor > 0
                    ):  # Avoid multiplying by zero if speedup is 0 for a successful test
                        compound_speedup *= res.speedup_factor

            compound_speedup = min(
                compound_speedup, 10**12
            )  # Cap compound speedup for dummy data
            # print(f"Dummy run_complete_validation: {len(results)} results, compound_speedup: {compound_speedup}")
            return results, compound_speedup


app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static",
)
# Fail closed: no predictable fallback key is compiled in. The runner
# (nextverse_validation_suite/main.py) refuses to start without it.
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "")
# Explicit CORS allowlist only; wildcard origins are never permitted.
_cors_origins = [
    origin.strip()
    for origin in os.environ.get(
        "NEXTVERSE_CORS_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000"
    ).split(",")
    if origin.strip()
]
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins=_cors_origins)

engine = NextVerseFormulaEngine()  # Initialize the engine (dummy or real)


@app.route("/")
def index_route():  # Renamed from 'index' to avoid conflict if 'index' is a blueprint name or common var
    return render_template("index.html")


@app.route("/api/validate-formula/<formula_name>", methods=["GET"])
def validate_single_formula_route(formula_name):
    formula_name_upper = formula_name.upper()
    # print(f"API call: /api/validate-formula/{formula_name_upper}")
    try:
        result = None
        method_name = f"calculate_{formula_name_upper.lower()}"

        if hasattr(engine, method_name):
            method_to_call = getattr(engine, method_name)
            if formula_name_upper == "LVVM":
                # Ensure JQLD is run first to get a Q value for LVVM, even if dummy
                jqld_temp_res = engine.calculate_jqld()
                q_val = (
                    jqld_temp_res.theoretical_value
                    if jqld_temp_res and jqld_temp_res.success
                    else 100.0
                )  # Default Q
                result = method_to_call(Q_from_jqld=q_val)
            else:
                result = method_to_call()  # Call with default params for other formulas
        else:
            # print(f"Error: Formula method {method_name} not found in engine.")
            return jsonify(
                {"error": f"Formula method {method_name} not implemented in engine"}
            ), 400

        if result:
            return jsonify(
                {
                    "formula": result.formula_name,
                    "theoretical": result.theoretical_value,
                    "measured": result.measured_performance,
                    "speedup": result.speedup_factor,
                    "execution_time": result.execution_time,
                    "success": result.success,
                    "parameters": result.parameters,
                    "error_message": result.error_message,
                }
            )
        else:
            # print(f"Error: Could not obtain result for {formula_name_upper} from method {method_name}")
            return jsonify(
                {"error": f"Could not obtain result for {formula_name_upper}"}
            ), 500

    except Exception as e:
        # print(f"Exception in validate_single_formula_route for {formula_name_upper}: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/validate-all", methods=["GET"])
def validate_all_formulas_route():
    # print("API call: /api/validate-all")
    try:
        results, compound_speedup = engine.run_complete_validation()

        json_results = {}
        for name, result_obj in results.items():
            json_results[name] = {
                "formula": result_obj.formula_name,
                "theoretical": result_obj.theoretical_value,
                "measured": result_obj.measured_performance,
                "speedup": result_obj.speedup_factor,
                "execution_time": result_obj.execution_time,
                "success": result_obj.success,
                "parameters": result_obj.parameters,
                "error_message": result_obj.error_message,
            }

        # print(f"Returning {len(json_results)} results, compound_speedup: {compound_speedup}")
        return jsonify(
            {
                "results": json_results,
                "compound_speedup": compound_speedup,
                "total_formulas": len(results),
                "successful_tests": sum(1 for r in results.values() if r.success),
            }
        )
    except Exception as e:
        # print(f"Exception in validate_all_formulas_route: {str(e)}")
        return jsonify({"error": str(e)}), 500


@socketio.on("start_live_test")
def handle_live_test_event(data):  # Renamed from handle_live_test to avoid confusion
    print(f"SocketIO event 'start_live_test' received with data: {data}")

    def run_test_background_thread():  # Renamed for clarity
        formulas_to_test = engine.formulas  # Get from engine instance
        total_formulas = len(formulas_to_test)
        q_value_from_jqld = 0  # Initialize Q value

        for i, formula_name_live in enumerate(formulas_to_test):  # Renamed variable
            # print(f"Live test: Testing {formula_name_live} ({i+1}/{total_formulas})")
            socketio.emit(
                "test_progress",
                {
                    "current": i + 1,
                    "total": total_formulas,
                    "formula": formula_name_live,
                    "status": "testing",
                },
            )

            result_live = None  # Renamed variable
            try:
                method_name_live = f"calculate_{formula_name_live.lower()}"
                if hasattr(engine, method_name_live):
                    method_to_call_live = getattr(engine, method_name_live)
                    if formula_name_live == "JQLD":
                        result_live = method_to_call_live()
                        if result_live and result_live.success:
                            q_value_from_jqld = result_live.theoretical_value
                    elif formula_name_live == "LVVM":
                        # Use Q value from JQLD if available, otherwise dummy engine handles default
                        result_live = method_to_call_live(
                            Q_from_jqld=q_value_from_jqld
                            if q_value_from_jqld
                            else 100.0
                        )
                    else:
                        result_live = method_to_call_live()
                else:  # Should not happen with __getattr__ in dummy engine
                    raise AttributeError(
                        f"Method {method_name_live} not found in engine for live test."
                    )

                if result_live:
                    socketio.emit(
                        "formula_result",
                        {
                            "formula": formula_name_live,
                            "result": {
                                "theoretical": result_live.theoretical_value,
                                "measured": result_live.measured_performance,
                                "speedup": result_live.speedup_factor,
                                "execution_time": result_live.execution_time,
                                "success": result_live.success,
                                "parameters": result_live.parameters,
                                "error_message": result_live.error_message,
                            },
                        },
                    )
                else:
                    socketio.emit(
                        "formula_result",
                        {
                            "formula": formula_name_live,
                            "result": {
                                "success": False,
                                "error_message": "No result returned from engine during live test",
                            },
                        },
                    )

            except Exception as e_live:  # Renamed variable
                # print(f"Exception during live test of {formula_name_live}: {str(e_live)}")
                socketio.emit(
                    "formula_result",
                    {
                        "formula": formula_name_live,
                        "result": {"success": False, "error_message": str(e_live)},
                    },
                )

            time.sleep(0.05)  # Shorter delay for live test UI updates

        # print("Live test: All formulas tested.")
        socketio.emit("test_complete", {"message": "All formula tests completed."})

    # print("Starting live test thread.")
    thread = threading.Thread(target=run_test_background_thread)
    thread.daemon = True  # Allow main program to exit even if thread is running
    thread.start()


# The main execution block should be in the top-level main.py
# if __name__ == '__main__':
#     print("Starting Flask app directly from app.py (debugging disabled by default)")
#     socketio.run(app, host='127.0.0.1', port=5000, debug=False, use_reloader=False)
