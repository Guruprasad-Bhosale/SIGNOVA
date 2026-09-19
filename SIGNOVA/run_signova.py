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


def run_phase22_check():
    """Runs the lightweight Phase 22 diagnostic check."""
    print("\n[+] Running Phase 22 Diagnostic Quick Check...\n")
    from scripts.check_phase22 import main as check22_main
    check22_main()


def run_phase22_verification():
    """Runs the complete Phase 22 verification workflow."""
    print("\n[+] Running Phase 22 Verification Workflow...\n")
    from scripts.run_phase22_verification import run_phase22_verification as verify22
    verify22()


def run_phase23_check():
    """Runs the lightweight Phase 23 diagnostic check."""
    print("\n[+] Running Phase 23 Diagnostic Quick Check...\n")
    from scripts.check_phase23 import main as check23_main
    check23_main()


def run_phase23_verification():
    """Runs the complete Phase 23 verification workflow."""
    print("\n[+] Running Phase 23 Verification Workflow...\n")
    from scripts.run_phase23_verification import run_phase23_verification as verify23
    verify23()


def run_phase23_training(train: bool = False):
    """Runs the Phase 23 gated CTC training orchestrator."""
    print("\n[+] Running Phase 23 Gated Training Orchestrator...\n")
    from scripts.run_phase23_training import main as train23_main
    if train:
        sys.argv = ["run_phase23_training.py", "--train"]
    else:
        sys.argv = ["run_phase23_training.py"]
    train23_main()


def run_phase24_check():
    """Runs the lightweight Phase 24 diagnostic check."""
    print("\n[+] Running Phase 24 Diagnostic Quick Check...\n")
    from scripts.check_phase24 import main as check24_main
    sys.argv = ["check_phase24.py"]
    check24_main()


def run_phase24_verification():
    """Runs the complete Phase 24 verification workflow."""
    print("\n[+] Running Phase 24 Verification Workflow...\n")
    from scripts.run_phase24_verification import run_phase24_verification as verify24
    verify24()


def run_phase24_training(train: bool = False):
    """Runs the Phase 24 gated CTC training execution."""
    print("\n[+] Running Phase 24 Gated CTC Training Execution...\n")
    from scripts.run_phase24_training import main as train24_main
    if train:
        sys.argv = ["run_phase24_training.py", "--train"]
    else:
        sys.argv = ["run_phase24_training.py"]
    train24_main()


def run_phase24_evaluation():
    """Runs the Phase 24 held-out evaluation & error analysis."""
    print("\n[+] Running Phase 24 Held-Out Evaluation...\n")
    from scripts.run_phase24_evaluation import main as eval24_main
    sys.argv = ["run_phase24_evaluation.py"]
    eval24_main()


def run_phase25_check():
    """Runs the Phase 25 readiness and human acquisition check."""
    print("\n[+] Running Phase 25 Diagnostic Quick Check...\n")
    from scripts.check_phase25 import main as check25_main
    sys.argv = ["check_phase25.py"]
    check25_main()


def run_phase25_verification():
    """Runs the complete Phase 25 verification workflow."""
    print("\n[+] Running Phase 25 Verification Workflow...\n")
    from scripts.run_phase25_verification import run_phase25_verification as verify25
    verify25()


def run_phase25_acquisition():
    """Runs the Phase 25 human acquisition runner."""
    print("\n[+] Running Phase 25 Human Acquisition Runner...\n")
    from scripts.run_phase25_acquisition import main as acq25_main
    sys.argv = ["run_phase25_acquisition.py"]
    acq25_main()


def run_phase25_build_dataset():
    """Runs the Phase 25 dataset builder."""
    print("\n[+] Running Phase 25 Dataset Builder...\n")
    from scripts.run_phase25_build_dataset import main as build25_main
    sys.argv = ["run_phase25_build_dataset.py"]
    build25_main()


def run_phase25_freeze_dataset():
    """Runs the Phase 25 dataset freeze tool."""
    print("\n[+] Running Phase 25 Dataset Freeze Tool...\n")
    from scripts.run_phase25_freeze_dataset import main as freeze25_main
    sys.argv = ["run_phase25_freeze_dataset.py"]
    freeze25_main()


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
        print("  [22] Run Phase 22 Human Annotation & Dataset Formation Diagnostic Check")
        print("  [V22] Run Phase 22 Verification & Status Dashboard")
        print("  [23] Run Phase 23 Execution & Qualification Diagnostic Check")
        print("  [V23] Run Phase 23 Verification & Multi-Dimensional Dashboard")
        print("  [24] Run Phase 24 First Real CTC Experiment Diagnostic Check")
        print("  [V24] Run Phase 24 Verification & Status Dashboard")
        print("  [25] Run Phase 25 Human Annotation & Acquisition Diagnostic Check")
        print("  [V25] Run Phase 25 Verification & Status Dashboard")
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
        elif choice in ("22", "P22"):
            run_phase22_check()
            input("\nPress Enter to return to menu...")
        elif choice in ("V22", "VERIFY22"):
            run_phase22_verification()
            input("\nPress Enter to return to menu...")
        elif choice in ("23", "P23"):
            run_phase23_check()
            input("\nPress Enter to return to menu...")
        elif choice in ("V23", "VERIFY23"):
            run_phase23_verification()
            input("\nPress Enter to return to menu...")
        elif choice in ("24", "P24"):
            run_phase24_check()
            input("\nPress Enter to return to menu...")
        elif choice in ("V24", "VERIFY24"):
            run_phase24_verification()
            input("\nPress Enter to return to menu...")
        elif choice in ("25", "P25"):
            run_phase25_check()
            input("\nPress Enter to return to menu...")
        elif choice in ("V25", "VERIFY25"):
            run_phase25_verification()
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
    parser.add_argument("--phase22-check", action="store_true", help="Run Phase 22 Diagnostic Check")
    parser.add_argument("--phase22-verify", action="store_true", help="Run Phase 22 Verification Dashboard")
    parser.add_argument("--phase23-check", action="store_true", help="Run Phase 23 Diagnostic Check")
    parser.add_argument("--phase23-verify", action="store_true", help="Run Phase 23 Verification Dashboard")
    parser.add_argument("--phase23-train", action="store_true", help="Run Phase 23 Gated Real CTC Training")
    parser.add_argument("--phase24-check", action="store_true", help="Run Phase 24 Diagnostic Check")
    parser.add_argument("--phase24-verify", action="store_true", help="Run Phase 24 Verification Dashboard")
    parser.add_argument("--phase24-train", action="store_true", help="Run Phase 24 Gated Real CTC Training")
    parser.add_argument("--phase24-eval", action="store_true", help="Run Phase 24 Held-Out Evaluation")
    parser.add_argument("--phase25-check", action="store_true", help="Run Phase 25 Diagnostic Check")
    parser.add_argument("--phase25-verify", action="store_true", help="Run Phase 25 Verification Dashboard")
    parser.add_argument("--phase25-acquire", action="store_true", help="Run Phase 25 Human Acquisition Runner")
    parser.add_argument("--phase25-build", action="store_true", help="Run Phase 25 Dataset Builder")
    parser.add_argument("--phase25-freeze", action="store_true", help="Run Phase 25 Dataset Freeze Tool")
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
    elif args.phase22_check:
        run_phase22_check()
    elif args.phase22_verify:
        run_phase22_verification()
    elif args.phase23_check:
        run_phase23_check()
    elif args.phase23_verify:
        run_phase23_verification()
    elif args.phase23_train:
        run_phase23_training(train=True)
    elif args.phase24_check:
        run_phase24_check()
    elif args.phase24_verify:
        run_phase24_verification()
    elif args.phase24_train:
        run_phase24_training(train=True)
    elif args.phase24_eval:
        run_phase24_evaluation()
    elif args.phase25_check:
        run_phase25_check()
    elif args.phase25_verify:
        run_phase25_verification()
    elif args.phase25_acquire:
        run_phase25_acquisition()
    elif args.phase25_build:
        run_phase25_build_dataset()
    elif args.phase25_freeze:
        run_phase25_freeze_dataset()
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
