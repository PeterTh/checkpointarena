import json
from itertools import product
import os
import time
from urllib import request, parse

output_dir = "C:/dev/ai/ComfyUI/output"

with open("highres_fix.json", 'r') as file:
    prompt_text = file.read()

def queue_prompt(prompt):
    p = {"prompt": prompt}
    data = json.dumps(p).encode('utf-8')
    req =  request.Request("http://127.0.0.1:8188/prompt", data=data)
    handler = request.urlopen(req)
    print(handler.status)

json_prompt = json.loads(prompt_text)

def set_pos_prompt(text):
    json_prompt["16"]["inputs"]["text"] = text

def set_neg_prompt(text):
    json_prompt["7"]["inputs"]["text"] = text

def set_checkpoint(ckpt):
    json_prompt["4"]["inputs"]["ckpt_name"] = ckpt

def set_output_prefix(prefix):
    json_prompt["26"]["inputs"]["filename_prefix"] = prefix

# set_pos_prompt("armored knight")
# set_output_prefix("absolutereality_v10")
# set_checkpoint("absolutereality_v10.safetensors")
# queue_prompt(json_prompt)

prompts = [
    ("flying raven", 'raven'),
    ("armored knight", 'knight'),
    ("medieval castle", 'castle'),
    ("spaceship in space", 'spaceship'),
    ("busy street", 'street'),
    ("portrait", 'portrait'),
    ("portrait of a man", 'portrait_m'),
    ("portrait of a woman", 'portrait_w'),
    ("fantasy forest", 'forest'),
    ("running horse", 'horse'),
    ("snowy mountaintop", 'mountain'),
    ("bowl of salad", 'salad'),
]
checkpoints = [
    ('absolutereality_v10', 'absrealv10', 'photo', ''),
    ('aZovyaPhotoreal_v1Ultra', 'azovyaphotov1u', 'photo', ''),
    ('cosplaymix_v20', 'cosplayv20', 'photo', ''),
    ('cyberrealistic_v30', 'cyberrealv30', 'photo', ''),
    ('edgeOfRealism_eorV20Fp16BakedVAE', 'eorv20', 'photo', ''),
    ('henmixReal_v40', 'henmixrealv40', 'photo', ''),
    ('hrl32_hrl32', 'hrl32', 'photo', ''),
    ('icbinpICantBelieveIts_v8', 'icbinpv8', 'photo', 'https://civitai.com/models/28059?modelVersionId=83527'),
    ('majicmixRealistic_v5', 'majicrealv5', 'photo', 'https://civitai.com/models/43331?modelVersionId=82446'),
    ('realisticVision_v20', 'realvisv20', 'photo', ''),
    ('reliberate_v10', 'reliberatev10', 'photo', 'https://civitai.com/models/79754/reliberate'),
    ('urpmv13', 'urpmv13', 'photo', ''),

    ('aZovyaRPGArtistTools_v3VAE', 'azovyarpgv3', 'other', 'https://civitai.com/models/8124?modelVersionId=87886'),
    ('beenyou_r11', 'beenyour11', 'other', ''),
    ('ddosmix_V2', 'ddosmixv2', 'other', ''),
    ('deliberate_v2', 'deliberatev2', 'other', ''),
    ('dreamshaper_4BakedVaeFp16', 'dreamshaper4', 'other', 'https://civitai.com/models/4384?modelVersionId=24365'),
    ('dreamshaper_6BakedVae', 'dreamshaper6', 'other', 'https://civitai.com/models/4384?modelVersionId=78164'),
    ('ghostmix_v20Bakedvae', 'ghostmixv20', 'other', 'https://civitai.com/models/36520?modelVersionId=76907'),
    ('lyriel_v16', 'lyrielv16', 'other', ''),
    ('mdjrny-v4', 'mdjrnyv4', 'other', ''),
    ('neverendingDreamNED_bakedVae', 'ned', 'other', ''),
    ('protogenV22Anime_22', 'protogenv22a', 'other', ''),
    ('revAnimated_v121', 'revanimv121', 'other', ''),
    ('rpg_V4', 'rpgv4', 'other', ''),

    ('ambientgrapemix_v10', 'ambgrapev10', 'anime', ''),
    ('anyloraCheckpoint_bakedvaeFtmseFp16NOT', 'anylora', 'anime', ''),
    ('anythingV5PrtRE', 'anythingv5', 'anime', ''),
    ('AOM3', 'aom3', 'anime', ''),
    ('AOM3A1', 'aom3a1', 'anime', ''),
    ('AOM3A1B', 'aom3a1b', 'anime', ''),
    ('AOM3A2', 'aom3a2', 'anime', ''),
    ('AOM3A3', 'aom3a3', 'anime', ''),
    ('cetusMix_whalefall', 'cetuswhale', 'anime', ''),
    ('CounterfeitV30_v30', 'counterfv30', 'anime', ''),
    ('hassaku_v11', 'hassakuv11', 'anime', 'https://civitai.com/models/2583?modelVersionId=37521'),
    ('hassakuHentaiModel_v12', 'hassakuv12', 'anime', 'https://civitai.com/models/2583?modelVersionId=62528'),
    ('hm_grapefruitv41', 'grapefruitv41', 'anime', ''),
    ('meinamix_meinaV9', 'meinav9', 'anime', ''),
    ('nyanMix_230303Absurd2', 'nyan230303a2', 'anime', ''),
    ('nyanMix_230303Normal', 'nyan230303n', 'anime', ''),
    ('tmndMixPlusPruned', 'tmndmixplus', 'anime', ''),
]
extra_prompt_mapping = {
    "anime": ", high quality, highly detailed, masterpiece",
    "other": ", high quality",
    "photo": ", photo, high quality, 4k",
}

results = []

for checkpoint_data, prompt_data in product(checkpoints, prompts):
    checkpoint, checkpoint_id, category, url = checkpoint_data
    prompt, prompt_id = prompt_data

    pos_prompt_string = "({}:1.1){}".format(prompt, extra_prompt_mapping[category])
    print(pos_prompt_string)
    set_pos_prompt(pos_prompt_string)
    
    checkpoint_fn = "{}.safetensors".format(checkpoint)
    print(checkpoint_fn)
    set_checkpoint(checkpoint_fn)

    output_prefix = "{}_{}".format(prompt_id, checkpoint_id)
    print(output_prefix)
    set_output_prefix(output_prefix)


    expected_fn = "{}_{:05d}_.png".format(output_prefix, 6)
    expected_path = os.path.join(output_dir, expected_fn)
    print(expected_path)

    if not os.path.exists(expected_path):
        queue_prompt(json_prompt)
        while not os.path.exists(expected_path):
            time.sleep(0.1)
    else:
        print("exists, skipping")

    result_entry = {
        'checkpoint': checkpoint,
        'checkpoint_id': checkpoint_id,
        'category': category,
        'prompt': prompt,
        'prompt_id': prompt_id,
        'output_prefix': output_prefix,
    }
    results.append(result_entry)

    print("-----")

# Storing the results in a JSON file
with open('results.json', 'w') as file:
    json.dump(results, file)
