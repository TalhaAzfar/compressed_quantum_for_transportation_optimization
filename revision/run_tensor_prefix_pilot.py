"""Run the already authorized, bounded pilot through saved hardware stages."""
import json
import time
from .tensor_prefix_prepare import settings
from .tensor_prefix_hardware import screen_plan,submit,collect,refinement_plan,final_plans


def wait_for(stage):
    _,out=settings()
    while not (out/"raw_counts"/(stage+".json")).exists():
        collect()
        if (out/(stage+"_failure.json")).exists():
            raise RuntimeError((out/(stage+"_failure.json")).read_text())
        if not (out/"raw_counts"/(stage+".json")).exists():
            time.sleep(30)


def run():
    config,out=settings()
    assert all((out/"cases"/(spec["id"]+".json")).exists() for spec in config["cases"])
    screen_plan();submit("screen");wait_for("screen")
    for step in range(config["spsa_steps"]):
        refinement_plan(step);submit(f"refine_{step}");wait_for(f"refine_{step}")
    final_plans()
    for block in range(config["final_blocks"]):submit(f"final_{block}")
    for block in range(config["final_blocks"]):wait_for(f"final_{block}")
    from .analyze_tensor_prefix import analyze
    analyze()


if __name__=="__main__":run()
