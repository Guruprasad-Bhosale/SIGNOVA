"""
SIGNOVA — Interactive Python Console & Platform Launcher

Run this script directly in a Python terminal or command window:
    python run_signova.py
"""

import sys
import os
import argparse
import subprocess
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))


def print_banner():
    banner = r"""
================================================================================
   ____ ___ ____ _   _  _____     ___
  / ___|_ _/ ___| \ | |/ _ \ \   / / \   
  \___ \| | |  _|  \| | | | \ \ / / _ \  
   ___) | | |_| | |\  | |_| |\ V / ___ \ 
  |____/___\____|_| \_|\___/  \_/_/   \_\
  Continuous Indian Sign Language (ISL) -> English Translation System
================================================================================
  Status: Phase 20 Live Camera & Streaming Verified | Protected Baseline Checked
================================================================================
"""
    print(banner)


def start_annotation_server(port: int = 8000, open_browser: bool = True):
    """Starts the SIGNOVA Human Annotation Platform."""
    print(f"\n[+] Launching SIGNOVA Human Annotation Platform at http://127.0.0.1:{port} ...")
    if open_browser:
        try:
            webbrowser.open(f"http://127.0.0.1:{port}")
        except Exception:
            pass
    import uvicorn
    uvicorn.run("apps.annotation.main:app", host="127.0.0.1", port=port, reload=False)


def start_api_server(port: int = 8000):
    """Starts the SIGNOVA Core Backend API & Live Streaming Gateway."""
    print(f"\n[+] Launching SIGNOVA Backend API & Live WebSocket Gateway at http://127.0.0.1:{port} ...")
    print(f"[+] Swagger UI available at http://127.0.0.1:{port}/docs")
    print(f"[+] Live Streaming WebSocket at ws://127.0.0.1:{port}/ws/live")
    import uvicorn
    uvicorn.run("apps.api.main:app", host="127.0.0.1", port=port, reload=False)


def run_standalone_live_camera():
    """Runs the standalone OpenCV live camera runner."""
    print("\n[+] Launching Standalone OpenCV Live Camera Runner...\n")
    from scripts.run_live_camera import run_live_camera
    run_live_camera()


def run_live_runtime_check():
    """Runs the Phase 20 live runtime diagnostic check."""
    print("\n[+] Running Phase 20 Live Runtime Diagnostic Check...\n")
    from scripts.check_live_runtime import check_live_runtime
    check_live_runtime()


def run_live_smoke_test():
    """Runs the Phase 20 live runtime integration smoke test."""
    print("\n[+] Running Phase 20 Live Runtime Integration Smoke Test...\n")
    from scripts.test_live_runtime import run_live_runtime_smoke_test
    run_live_runtime_smoke_test()


def run_live_profiler():
    """Runs the Phase 20 live runtime latency and FPS profiler."""
    print("\n[+] Running Phase 20 Live Runtime Latency & FPS Profiler...\n")
    from scripts.profile_live_runtime import profile_live_runtime
    profile_live_runtime()


def run_phase19_verification():
    """Runs the complete Phase 19 verification report."""
    print("\n[+] Running Phase 19 Verification Workflow...\n")
    from scripts.run_phase19_verification import run_verification
    run_verification()


def run_phase19_check():
    """Runs the lightweight Phase 19 diagnostic check."""
    print("\n[+] Running Phase 19 Diagnostic Quick Check...\n")
    from scripts.check_phase19 import main as check_main
    check_main()


def run_phase21_check():
    """Runs the lightweight Phase 21 diagnostic check."""
    print("\n[+] Running Phase 21 Diagnostic Quick Check...\n")
    from scripts.check_phase21 import main as check21_main
    check21_main()


def run_phase21_verification():
    """Runs the complete Phase 21 verification workflow."""
    print("\n[+] Running Phase 21 Verification Workflow...\n")
    from scripts.run_phase21_verification import run_phase21_verification as verify21
    verify21()


def run_smoke_test():
    """Runs the dynamic real data pipeline smoke test."""
    print("\n[+] Running Phase 19 Real Data Pipeline Smoke Test...\n")
    from scripts.smoke_test_phase19_real_data_pipeline import smoke_test_phase19_pipeline
    smoke_test_phase19_pipeline()


def run_unit_tests():
    """Runs pytest across the SIGNOVA unit test suite."""
    print("\n[+] Running Unit Test Suite (pytest)...\n")
    cmd = [sys.executable, "-m", "pytest", "tests/unit", "-v"]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT))


def inspect_pilot_assignments():
    """Displays current pilot assignment summary."""
    from signova.operations.assignment import Phase19PilotAssignmentManager
    mgr = Phase19PilotAssignmentManager()
    manifest_path = mgr.manifest_path
    if not manifest_path.exists():
        print("[!] Generating Phase 19 pilot assignments (20 diagnostic videos)...")
        mgr.generate_pilot_assignments()
    
    assignments = mgr.load_assignments()
    stats = mgr.get_assignment_stats()
    print(f"\n--- Phase 19 Pilot Assignment Manifest ({manifest_path}) ---")
    print(f"Total Assigned Slots: {stats.get('total_assigned', len(assignments))}")
    print(f"Unique Video Clips:   {stats.get('unique_videos', 0)}")
    print(f"Double Annotated:     {stats.get('double_annotated', 0)}")
    print(f"Pending Annotations:  {stats.get('pending_annotations', 0)}")
    print(f"Completed:            {stats.get('completed_annotations', 0)}")
    print("\nSample Assignment Entries:")
    for a in assignments[:5]:
        print(f"  * {a.assignment_id} -> Video: {a.video_id} | Annotator: {a.annotator_id} | Status: {a.annotation_status}")
    if len(assignments) > 5:
        print(f"  ... and {len(assignments) - 5} more assignments.")


def interactive_menu():
    """Interactive CLI menu for the Python window."""
    while True:
        print_banner()
        print("Select an option:")
        print("  [1] Launch Core Backend API & Live WebSocket Gateway (FastAPI at http://127.0.0.1:8000)")
        print("  [2] Run Standalone OpenCV Live Camera Runner (Direct Python Camera)")
        print("  [3] Run Phase 20 Live Runtime Diagnostic Check")
        print("  [4] Run Phase 20 Live Runtime Integration Smoke Test")
        print("  [5] Run Phase 20 Live Runtime Latency & FPS Profiler")
        print("  [6] Launch Human Annotation Web Platform (FastAPI + UI at http://127.0.0.1:8001)")
        print("  [7] Run Phase 19 Verification & CTC Readiness Report")
        print("  [8] Run Phase 19 Real Data Pipeline Smoke Test")
        print("  [9] Inspect Pilot Assignments & Annotation Statistics")
        print("  [21] Run Phase 21 Genuine CTC Diagnostic Check")
        print("  [V21] Run Phase 21 Verification & Status Summary")
        print("  [T] Run Automated Unit Tests (pytest)")
        print("  [0] Exit")
        print("-" * 80)
        
        try:
            choice = input("Enter choice: ").strip().upper()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting SIGNOVA.")
            break

        if choice == "1":
            try:
                start_api_server(8000)
            except KeyboardInterrupt:
                print("\nServer stopped.")
        elif choice == "2":
            try:
                run_standalone_live_camera()
            except KeyboardInterrupt:
                print("\nCamera stopped.")
        elif choice == "3":
            run_live_runtime_check()
            input("\nPress Enter to return to menu...")
        elif choice == "4":
            run_live_smoke_test()
            input("\nPress Enter to return to menu...")
        elif choice == "5":
            run_live_profiler()
            input("\nPress Enter to return to menu...")
        elif choice == "6":
            try:
                start_annotation_server(8001)
            except KeyboardInterrupt:
                print("\nServer stopped.")
        elif choice == "7":
            run_phase19_verification()
            input("\nPress Enter to return to menu...")
        elif choice == "8":
            run_smoke_test()
            input("\nPress Enter to return to menu...")
        elif choice == "9":
            inspect_pilot_assignments()
            input("\nPress Enter to return to menu...")
        elif choice in ("21", "P21"):
            run_phase21_check()
            input("\nPress Enter to return to menu...")
        elif choice in ("V21", "VERIFY21"):
            run_phase21_verification()
            input("\nPress Enter to return to menu...")
        elif choice in ("T", "TEST"):
            run_unit_tests()
            input("\nPress Enter to return to menu...")
        elif choice in ("0", "Q", "EXIT"):
            print("\nExiting SIGNOVA. Goodbye!")
            break
        else:
            print("\n[!] Invalid choice. Please select from the menu.")


def main():
    parser = argparse.ArgumentParser(description="SIGNOVA Interactive Platform Runner")
    parser.add_argument("--api", action="store_true", help="Start Backend API & Live WebSocket Gateway")
    parser.add_argument("--camera", action="store_true", help="Start Standalone OpenCV Live Camera")
    parser.add_argument("--live-check", action="store_true", help="Run Phase 20 Live Runtime Diagnostic Check")
    parser.add_argument("--live-smoke", action="store_true", help="Run Phase 20 Live Runtime Integration Smoke Test")
    parser.add_argument("--profile", action="store_true", help="Run Phase 20 Latency & FPS Profiler")
    parser.add_argument("--server", action="store_true", help="Start Human Annotation Platform")
    parser.add_argument("--verify", action="store_true", help="Run Phase 19 verification report")
    parser.add_argument("--phase21-check", action="store_true", help="Run Phase 21 Diagnostic Check")
    parser.add_argument("--phase21-verify", action="store_true", help="Run Phase 21 Verification Workflow")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind server (default: 8000)")
    args = parser.parse_args()

    if args.api:
        start_api_server(port=args.port)
    elif args.camera:
        run_standalone_live_camera()
    elif args.live_check:
        run_live_runtime_check()
    elif args.live_smoke:
        run_live_smoke_test()
    elif args.profile:
        run_live_profiler()
    elif args.server:
        start_annotation_server(port=args.port)
    elif args.verify:
        run_phase19_verification()
    elif args.phase21_check:
        run_phase21_check()
    elif args.phase21_verify:
        run_phase21_verification()
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
