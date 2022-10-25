from torch import autocast
from flask import request,jsonify,Flask,send_file
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler

app = Flask(__name__)


lms = LMSDiscreteScheduler(
            beta_start=0.00085, 
                beta_end=0.012, 
                    beta_schedule="scaled_linear"
                    )

pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", 
                scheduler=lms,
                    use_auth_token="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"
                    ).to("cuda")

@app.route("/diffuse",methods=['POST'])
def diffuse():
    user_data = request.get_json()
    prompt = user_data['prompt']
    with autocast("cuda"):
        image = pipe(prompt)["sample"][0]  
    print(prompt)
    prompt = prompt.replace(' ','_')
    fname = "images/%s.png"%(prompt) 
    image.save(fname)
    return send_file(fname, mimetype='image/png')



