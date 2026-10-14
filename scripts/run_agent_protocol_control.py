"""Exercise the local transport with a public-message-only scripted controller."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from probebench.agent_protocol import PROTOCOL, encode_message
from probebench.run import digest


def choose_action(message):
    """Transport control only; no language model or hidden configuration is consulted."""
    action = {"protocol":PROTOCOL,"request_id":message["request_id"],"action":"submit",
              "command_opd_nm":[0]*6}
    if message["latest_observation"]["index"] == 0 and "expose" in message["available_actions"]:
        action.update(action="expose",command_opd_nm=[min(80,message["instrument"]["command_limit_nm"]),0,0,0,0,0])
    elif message["track"] == "tool_assisted":
        if message["tool_result"] is not None:
            action["command_opd_nm"] = message["tool_result"]["estimated_opd_nm"]
        elif "estimate" in message["available_actions"]:
            action["action"] = "estimate"
            del action["command_opd_nm"]
    return action


def run(config,output,resume=False):
    command = [sys.executable,"-m","probebench.agent_session","--config",str(config),"--output",str(output)]
    if resume:
        command.append("--resume")
    child = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    try:
        while True:
            line = child.stdout.readline()
            if not line:
                raise RuntimeError("Protocol process ended without a final response")
            message = json.loads(line)
            if message["protocol"] != PROTOCOL:
                raise ValueError("Unexpected protocol version")
            if message["kind"] == "final":
                break
            child.stdin.write(encode_message(choose_action(message))+"\n")
            child.stdin.flush()
        if child.wait(timeout=30) != 0:
            raise RuntimeError("Protocol process failed")
        return {"controller":"scripted_protocol_control_not_a_language_model",
            "controller_source_sha256":digest(Path(__file__)),
            "session_manifest_sha256":digest(Path(output)/"manifest.json"),
            "track":message["track"],"status":message["status"],"resources":message["resources"]}
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)
        child.stdin.close()
        child.stdout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()
    result = run(args.config,args.output,args.resume)
    print(encode_message(result))
    if result["status"] != "submitted":
        raise SystemExit(1)
