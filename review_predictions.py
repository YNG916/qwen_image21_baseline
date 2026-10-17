"""Post-generation GT comparisons; never called by the model's inference path."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from qwen_mvwd.common import read_json, within, write_json
from qwen_mvwd.data import Level1Data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True)
    args = p.parse_args()
    directory = Path(args.predictions).resolve()
    meta = read_json(directory / "metadata.json")
    if not meta["complete"] or meta["ground_truth_loaded"]:
        raise ValueError("Require completed GT-free generation")
    data = Level1Data(meta["config"], meta["dataset"]["split"])
    count = 0
    for line in (directory / "predictions.jsonl").read_text().splitlines():
        row = json.loads(line)
        path = within(directory, row["path"])
        target = data.target({"episode_id": row["episode_id"], "frame_index": row["frame_index"], "robots": [row["robot_id"]]})
        with Image.open(path) as image:
            predicted = image.convert("RGB")
        canvas = Image.new("RGB", (target.width * 2, target.height + 24), "white")
        canvas.paste(target, (0, 24)); canvas.paste(predicted, (target.width, 24))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), "GT", fill="black")
        draw.text((target.width + 4, 4), "Qwen prediction (diagnostic)", fill="black")
        destination = path.with_suffix(".comparison.png")
        if destination.exists():
            raise FileExistsError(destination)
        canvas.save(destination)
        count += 1
    write_json(directory / "comparison_receipt.json", {"images": count, "gt_read_only_after_generation": True,
        "dataset_release": data.meta["release_id"]})
    print("Exported", count, "adjacent GT comparisons")


if __name__ == "__main__":
    main()
