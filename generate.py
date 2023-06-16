import glob
import json
import os
import shutil
import time
from itertools import product
from urllib import request, parse
from subprocess import Popen

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

def get_pos_prompt():
    return json_prompt["16"]["inputs"]["text"]

def set_pos_prompt(text):
    json_prompt["16"]["inputs"]["text"] = text

def get_neg_prompt():
    return json_prompt["7"]["inputs"]["text"]

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
    ('absolutereality_v10', 'absrealv10', 'photo', 'https://civitai.com/models/81458?modelVersionId=86437'),
    ('aZovyaPhotoreal_v1Ultra', 'azovyaphotov1u', 'photo', 'https://civitai.com/models/57319?modelVersionId=61735'),
    ('cosplaymix_v20', 'cosplayv20', 'photo', 'https://civitai.com/models/34502?modelVersionId=48334'),
    ('cyberrealistic_v30', 'cyberrealv30', 'photo', 'https://civitai.com/models/15003?modelVersionId=79754'),
    ('edgeOfRealism_eorV20Fp16BakedVAE', 'eorv20', 'photo', 'https://civitai.com/models/21813?modelVersionId=51913'),
    ('henmixReal_v40', 'henmixrealv40', 'photo', 'https://civitai.com/models/20282?modelVersionId=70458'),
    ('hrl32_hrl32', 'hrl32', 'photo', 'https://civitai.com/models/8616?modelVersionId=16369'),
    ('icbinpICantBelieveIts_v8', 'icbinpv8', 'photo', 'https://civitai.com/models/28059?modelVersionId=83527'),
    ('majicmixRealistic_v5', 'majicrealv5', 'photo', 'https://civitai.com/models/43331?modelVersionId=82446'),
    ('majicmixRealistic_v6', 'majicrealv6', 'photo', 'https://civitai.com/models/43331?modelVersionId=94640'),
    ('realisticVision_v20', 'realvisv20', 'photo', 'https://civitai.com/models/4201?modelVersionId=29460'),
    ('reliberate_v10', 'reliberatev10', 'photo', 'https://civitai.com/models/79754/reliberate'),
    ('urpmv13', 'urpmv13', 'photo', 'https://civitai.com/models/2661?modelVersionId=15640'),

    ('aZovyaRPGArtistTools_v3VAE', 'azovyarpgv3', 'other', 'https://civitai.com/models/8124?modelVersionId=87886'),
    ('beenyou_r11', 'beenyour11', 'other', 'https://civitai.com/models/27688?modelVersionId=64889'),
    ('ddosmix_V2', 'ddosmixv2', 'other', 'https://civitai.com/models/8437?modelVersionId=12183'),
    ('deliberate_v2', 'deliberatev2', 'other', 'https://civitai.com/models/4823?modelVersionId=15236'),
    ('dreamshaper_4BakedVaeFp16', 'dreamshaper4', 'other', 'https://civitai.com/models/4384?modelVersionId=24365'),
    ('dreamshaper_6BakedVae', 'dreamshaper6', 'other', 'https://civitai.com/models/4384?modelVersionId=78164'),
    ('ghostmix_v20Bakedvae', 'ghostmixv20', 'other', 'https://civitai.com/models/36520?modelVersionId=76907'),
    ('lyriel_v16', 'lyrielv16', 'other', 'https://civitai.com/models/22922?modelVersionId=72396'),
    ('mdjrny-v4', 'mdjrnyv4', 'other', 'https://huggingface.co/prompthero/openjourney/blob/main/mdjrny-v4.safetensors'),
    ('neverendingDreamNED_bakedVae', 'ned', 'other', 'https://civitai.com/models/10028?modelVersionId=11925'),
    ('protogenV22Anime_22', 'protogenv22a', 'other', 'https://civitai.com/models/3627/protogen-v22-anime-official-release'),
    ('revAnimated_v121', 'revanimv121', 'other', 'https://civitai.com/models/7371?modelVersionId=40248'),
    ('rpg_V4', 'rpgv4', 'other', 'https://civitai.com/models/1116?modelVersionId=7133'),

    ('ambientgrapemix_v10', 'ambgrapev10', 'anime', 'https://civitai.com/models/30671?modelVersionId=37023'),
    ('anyloraCheckpoint_bakedvaeFtmseFp16NOT', 'anylora', 'anime', 'https://civitai.com/models/23900?modelVersionId=29792'),
    ('anythingV5PrtRE', 'anythingv5', 'anime', 'https://civitai.com/models/9409?modelVersionId=30163'),
    ('AOM3', 'aom3', 'anime', 'https://civitai.com/models/9942?modelVersionId=11814'),
    ('AOM3A1', 'aom3a1', 'anime', 'https://civitai.com/models/9942?modelVersionId=11813'),
    ('AOM3A1B', 'aom3a1b', 'anime', 'https://civitai.com/models/9942?modelVersionId=17233'),
    ('AOM3A2', 'aom3a2', 'anime', 'https://civitai.com/models/9942?modelVersionId=11812'),
    ('AOM3A3', 'aom3a3', 'anime', 'https://civitai.com/models/9942?modelVersionId=11811'),
    ('breakdomain_I2428','breakdomaini2428','anime','https://civitai.com/models/50520?modelVersionId=96424'),
    ('cetusMix_whalefall', 'cetuswhale', 'anime', 'https://civitai.com/models/6755?modelVersionId=36936'),
    ('CounterfeitV30_v30', 'counterfv30', 'anime', 'https://civitai.com/models/4468?modelVersionId=57618'),
    ('darkSushiMixMix_225D', 'darksushi225d', 'anime', 'https://civitai.com/models/24779?modelVersionId=93208'),
    ('hassaku_v11', 'hassakuv11', 'anime', 'https://civitai.com/models/2583?modelVersionId=37521'),
    ('hassakuHentaiModel_v12', 'hassakuv12', 'anime', 'https://civitai.com/models/2583?modelVersionId=62528'),
    ('hm_grapefruitv41', 'grapefruitv41', 'anime', 'https://civitai.com/models/24383?modelVersionId=29179'),
    ('meinamix_meinaV9', 'meinav9', 'anime', 'https://civitai.com/models/7240?modelVersionId=46137'),
    ('meinamix_meinaV10', 'meinav10', 'anime', 'https://civitai.com/models/7240?modelVersionId=80511'),
    ('nyanMix_230303Absurd2', 'nyan230303a2', 'anime', 'https://civitai.com/models/14373?modelVersionId=18151'),
    ('nyanMix_230303Normal', 'nyan230303n', 'anime', 'https://civitai.com/models/14373?modelVersionId=17818'),
    ('tmndMix_tmndMixIVPruned', 'tmndmixiv', 'anime', 'https://civitai.com/models/27259?modelVersionId=88859'),
]
extra_prompt_mapping = {
    "anime": ", high quality, highly detailed, masterpiece",
    "other": ", high quality",
    "photo": ", photo, high quality, 4k",
}
sfw_options = [True, False]
sfw_extra_neg_prompt = ", nsfw, naked, nude"

results = []
default_neg_prompt = get_neg_prompt()

for checkpoint_data, prompt_data, sfw in product(checkpoints, prompts, sfw_options):
    checkpoint, checkpoint_id, category, url = checkpoint_data
    prompt, prompt_id = prompt_data

    pos_prompt_string = "({}:1.1){}".format(prompt, extra_prompt_mapping[category])
    print("+ : " + pos_prompt_string)
    set_pos_prompt(pos_prompt_string)

    neg_prompt_string = default_neg_prompt
    if sfw:
        neg_prompt_string += sfw_extra_neg_prompt
    print("- : " + neg_prompt_string)
    set_neg_prompt(neg_prompt_string)
    
    checkpoint_fn = "{}.safetensors".format(checkpoint)
    print(checkpoint_fn)
    set_checkpoint(checkpoint_fn)

    output_prefix = "{}_{}".format(prompt_id, checkpoint_id)
    if sfw:
        output_prefix += "_sfw"
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
        'sfw': sfw,
        'output_prefix': output_prefix,
    }
    results.append(result_entry)

    print("-----")

# store the results in a JSON file
with open('results.json', 'w') as file:
    json.dump(results, file)

repo_dir = os.getcwd()

def convert_to_avif_and_copy(quality, target_path):
    print("avif conversion ...")
    os.system('npx avif --input="*.png" --quality {} --effort 5'.format(quality))
    print("copy ...")    
    os.makedirs(target_path, exist_ok=True)
    for aviffn in glob.glob("*.avif"):
        shutil.copy(aviffn, target_path)

# convert any new full-size pngs to avif and copy to repo
FULL_AVIF_QUALITY=85
os.chdir(output_dir)
avif_repo_dir = os.path.join(repo_dir, "avif")
convert_to_avif_and_copy(FULL_AVIF_QUALITY, avif_repo_dir)

# generate thumbnails
THUMBNAIL_AVIF_QUALITY=65
THUMBNAIL_SIZE=256
THUMBNAIL_DIR='thumb'
PARALLELISM=16
print("thumbnails:", end='')
# this would be 3 lines in Ruby
pnglist = glob.glob("*.png")
for i in range(0, len(pnglist), PARALLELISM):
    chunk = pnglist[i:i+PARALLELISM]
    procs = []
    for pngfn in chunk:
        target = os.path.join(THUMBNAIL_DIR, pngfn)
        if not os.path.exists(target):
            procs.append(Popen('magick {0} -resize {1}x{1} {2}'.format(pngfn, THUMBNAIL_SIZE, target)))
    for p in procs:
        p.wait()
    print(".", end='', flush=True)
print("")

# convert any new thumbnail pngs to avif and copy to repo
os.chdir(THUMBNAIL_DIR)
repo_thumb_dir = os.path.join(avif_repo_dir, THUMBNAIL_DIR)
convert_to_avif_and_copy(THUMBNAIL_AVIF_QUALITY, repo_thumb_dir)
