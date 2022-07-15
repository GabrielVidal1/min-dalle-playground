import argparse
import base64
import os
from pathlib import Path
from io import BytesIO
import time

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from min_dalle import MinDalle
import torch


app = Flask(__name__)
CORS(app)
print("--> Starting DALL-E Server. This might take up to two minutes.")

dalle_model = None

DEFAULT_IMG_OUTPUT_DIR = "generations"

parser = argparse.ArgumentParser(description = "A DALL-E app to turn your textual prompts into visionary delights")
parser.add_argument("--port", type=int, default=8000, help = "backend port")
parser.add_argument("--img_format", type = str.lower, default = "JPEG", help = "Generated images format", choices=['jpeg', 'png'])
parser.add_argument("--output_dir", type = str, default = DEFAULT_IMG_OUTPUT_DIR, help = "Customer directory for generated images")
parser.add_argument("--dtype", type = str, choices=["float32", "float16", "bfloat16"], default="float16")
parser.add_argument("--models_root", type = str, default='./pretrained')
args = parser.parse_args()

@app.route("/dalle", methods=["POST"])
@cross_origin()
def generate_images_api():
    json_data = request.get_json(force=True)

    prompt = json_data["text"]
    image_count = json_data["num_images"]

    generated_imgs = dalle_model.generate_images(
        text=prompt,
        seed=-1,
        image_count=image_count,
        temperature=1,
        top_k=256,
        supercondition_factor=16,
        is_verbose=True
    )

    returned_generated_images = []

    dir_name = os.path.join(args.output_dir,f"{time.strftime('%Y-%m-%d_%H:%M:%S')}_{prompt}")
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    for idx, img in enumerate(generated_imgs):
        img.save(os.path.join(dir_name, f'{idx}.{args.img_format}'), format=args.img_format)

        buffered = BytesIO()
        img.save(buffered, format=args.img_format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        returned_generated_images.append(img_str)

    print(f"Created {image_count} images from text prompt [{prompt}]")
    
    response = {
        'generatedImgs': returned_generated_images,
        'generatedImgsFormat': args.img_format
    }
    return jsonify(response)


@app.route("/", methods=["GET"])
@cross_origin()
def health_check():
    return jsonify(success=True)


with app.app_context():
    dalle_model = MinDalle(
        models_root=args.models_root,
        dtype=getattr(torch, args.dtype),
        is_mega=True, 
        is_reusable=True
    )

    dalle_model.generate_images("warm-up", 1)
    print("--> DALL-E Server is up and running!")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port, debug=False)