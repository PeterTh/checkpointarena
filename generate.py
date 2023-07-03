import glob
import json
import os
import shutil
import time
from itertools import product
from urllib import request
from subprocess import Popen
from datetime import datetime

output_dir = "C:/dev/ai/ComfyUI/output"

starting_seed = 42
imgs_per_prompt = 6
# the number per batch has a huge performance impact, should probably be tuned for each target HW
imgs_per_batch = 2

with open("highres_fix.json", 'r') as file:
    prompt_text = file.read()

def queue_prompt(prompt):
    p = {"prompt": prompt}
    data = json.dumps(p).encode('utf-8')
    req =  request.Request("http://127.0.0.1:8188/prompt", data=data)
    handler = request.urlopen(req)
    print(handler.status, end=' ', flush=True)

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

def set_batch_size(sz):
    json_prompt["5"]["inputs"]["batch_size"] = sz

def set_seed(seed):
    json_prompt["3"]["inputs"]["seed"] = seed
    json_prompt["22"]["inputs"]["seed"] = seed


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
    ("couple in living room", 'couple'),
]
checkpoints = [
    ('absolutereality_v10'             , 'absrealv10'    , 'photo', 'AbsoluteReality'                                     , 'https://civitai.com/models/81458?modelVersionId=86437'),
    ('analogMadness_v50'               , 'analogmadv50'  , 'photo', 'Analog Madness - Realistic model v5.0'               , 'https://civitai.com/models/8030?modelVersionId=101080'),
    ('architecturerealmix_v10'         , 'archrealv10'   , 'photo', 'ArchitectureRealMix'                                 , 'https://civitai.com/models/84958/architecturerealmix'),
    ('aZovyaPhotoreal_v1Ultra'         , 'azovyaphotov1u', 'photo', 'A-Zovya Photoreal V1 Ultra'                          , 'https://civitai.com/models/57319?modelVersionId=61735'),
    ('aZovyaPhotoreal_v2'              , 'azovyaphotov2' , 'photo', 'A-Zovya Photoreal V2'                                , 'https://civitai.com/models/57319?modelVersionId=99805'),
    ('cosplaymix_v20'                  , 'cosplayv20'    , 'photo', 'CosplayMix v2.0'                                     , 'https://civitai.com/models/34502?modelVersionId=48334'),
    ('cosplaymix_v41'                  , 'cosplayv41'    , 'photo', 'CosplayMix v4.1'                                     , 'https://civitai.com/models/34502?modelVersionId=98501'),
    ('cyberrealistic_v30'              , 'cyberrealv30'  , 'photo', 'CyberRealistic v3.0'                                 , 'https://civitai.com/models/15003?modelVersionId=79754'),
    ('edgeOfRealism_eorV20Fp16BakedVAE', 'eorv20'        , 'photo', 'Edge Of Realism v2.0'                                , 'https://civitai.com/models/21813?modelVersionId=51913'),
    ('henmixReal_v40'                  , 'henmixrealv40' , 'photo', 'Henmix_Real v4.0'                                    , 'https://civitai.com/models/20282?modelVersionId=70458'),
    ('hrl32_hrl32'                     , 'hrl32'         , 'photo', 'HRL v3.2'                                            , 'https://civitai.com/models/8616?modelVersionId=16369'),
    ('icbinpICantBelieveIts_v8'        , 'icbinpv8'      , 'photo', 'ICBINP - "I Can\'t Believe It\'s Not Photography" v8', 'https://civitai.com/models/28059?modelVersionId=83527'),
    ('majicmixRealistic_v5'            , 'majicrealv5'   , 'photo', 'majicMIX realistic v5'                               , 'https://civitai.com/models/43331?modelVersionId=82446'),
    ('majicmixRealistic_v6'            , 'majicrealv6'   , 'photo', 'majicMIX realistic v6'                               , 'https://civitai.com/models/43331?modelVersionId=94640'),
    ('realisticVision_v20'             , 'realvisv20'    , 'photo', 'Realistic Vision v2.0'                               , 'https://civitai.com/models/4201?modelVersionId=29460'),
    ('reliberate_v10'                  , 'reliberatev10' , 'photo', 'Reliberate v1.0'                                     , 'https://civitai.com/models/79754/reliberate'),
    ('rundiffusionFX_v10'              , 'rundifffxv10'  , 'photo', 'RunDiffusion FX Photorealistic'                      , 'https://civitai.com/models/82972/rundiffusion-fx-photorealistic'),
    ('urpmv13'                         , 'urpmv13'       , 'photo', 'Uber Realistic Porn Merge v1.3'                      , 'https://civitai.com/models/2661?modelVersionId=15640'),

    ('aZovyaRPGArtistTools_v3VAE'  , 'azovyarpgv3' , 'other', 'A-Zovya RPG Artist Tools v3'  , 'https://civitai.com/models/8124?modelVersionId=87886'),
    ('beenyou_r11'                 , 'beenyour11'  , 'other', 'BeenYou R11'                  , 'https://civitai.com/models/27688?modelVersionId=64889'),
    ('colorful_v31'                , 'colorfulv31' , 'other', 'Colorful v3.1'                , 'https://civitai.com/models/7279?modelVersionId=90599'),
    ('ddosmix_V2'                  , 'ddosmixv2'   , 'other', 'DDosMix v2'                   , 'https://civitai.com/models/8437?modelVersionId=12183'),
    ('deliberate_v2'               , 'deliberatev2', 'other', 'Deliberate v2'                , 'https://civitai.com/models/4823?modelVersionId=15236'),
    ('dreamshaper_4BakedVaeFp16'   , 'dreamshaper4', 'other', 'DreamShaper v4'               , 'https://civitai.com/models/4384?modelVersionId=24365'),
    ('dreamshaper_6BakedVae'       , 'dreamshaper6', 'other', 'DreamShaper v6'               , 'https://civitai.com/models/4384?modelVersionId=78164'),
    ('ghostmix_v20Bakedvae'        , 'ghostmixv20' , 'other', 'GhostMix v2.0'                , 'https://civitai.com/models/36520?modelVersionId=76907'),
    ('gameIconInstitute_v30'       , 'gameicov30'  , 'other', 'Game Icon Institute_mode v3.0', 'https://civitai.com/models/47800?modelVersionId=76533'),
    ('lyriel_v16'                  , 'lyrielv16'   , 'other', 'Lyriel v1.6'                  , 'https://civitai.com/models/22922?modelVersionId=72396'),
    ('mdjrny-v4'                   , 'mdjrnyv4'    , 'other', 'OpenJourney v4'               , 'https://huggingface.co/prompthero/openjourney/blob/main/mdjrny-v4.safetensors'),
    ('neverendingDreamNED_bakedVae', 'ned'         , 'other', 'NeverEnding Dream v1'         , 'https://civitai.com/models/10028?modelVersionId=11925'),
    ('protogenV22Anime_22'         , 'protogenv22a', 'other', 'Protogen v2.2 (Anime)'        , 'https://civitai.com/models/3627/protogen-v22-anime-official-release'),
    ('revAnimated_v121'            , 'revanimv121' , 'other', 'ReV Animated v1.2.1'          , 'https://civitai.com/models/7371?modelVersionId=40248'),
    ('rpg_V4'                      , 'rpgv4'       , 'other', 'RPG v4'                       , 'https://civitai.com/models/1116?modelVersionId=7133'),

    ('ambientgrapemix_v10'                   , 'ambgrapev10'     , 'anime', 'AmbientGrapeMix v1.0'   , 'https://civitai.com/models/30671?modelVersionId=37023'),
    ('anyloraCheckpoint_bakedvaeFtmseFp16NOT', 'anylora'         , 'anime', 'AnyLoRA - Checkpoint'   , 'https://civitai.com/models/23900?modelVersionId=29792'),
    ('anythingV5PrtRE'                       , 'anythingv5'      , 'anime', 'Anything v5'            , 'https://civitai.com/models/9409?modelVersionId=30163'),
    ('AnythingV5Ink_v32Ink'                  , 'anythingv32ink'  , 'anime', 'Anything v3.2 [ink]'    , 'https://civitai.com/models/9409?modelVersionId=90854'),
    ('AOM3'                                  , 'aom3'            , 'anime', 'AbyssOrangeMix3'        , 'https://civitai.com/models/9942?modelVersionId=11814'),
    ('AOM3A1'                                , 'aom3a1'          , 'anime', 'AbyssOrangeMix3 A1'     , 'https://civitai.com/models/9942?modelVersionId=11813'),
    ('AOM3A1B'                               , 'aom3a1b'         , 'anime', 'AbyssOrangeMix3 A1B'    , 'https://civitai.com/models/9942?modelVersionId=17233'),
    ('AOM3A2'                                , 'aom3a2'          , 'anime', 'AbyssOrangeMix3 A2'     , 'https://civitai.com/models/9942?modelVersionId=11812'),
    ('AOM3A3'                                , 'aom3a3'          , 'anime', 'AbyssOrangeMix3 A3'     , 'https://civitai.com/models/9942?modelVersionId=11811'),
    ('breakdomain_I2428'                     , 'breakdomaini2428', 'anime', 'BreakDomain i2428'      , 'https://civitai.com/models/50520?modelVersionId=96424'),
    ('cetusMix_whalefall'                    , 'cetuswhale'      , 'anime', 'Cetus-Mix WhaleFall'    , 'https://civitai.com/models/6755?modelVersionId=36936'),
    ('cetusMix_v4'                           , 'cetusv4'         , 'anime', 'Cetus-Mix v4'           , 'https://civitai.com/models/6755?modelVersionId=78676'),
    ('CounterfeitV30_v30'                    , 'counterfv30'     , 'anime', 'Counterfeit v3.0'       , 'https://civitai.com/models/4468?modelVersionId=57618'),
    ('darkSushiMixMix_225D'                  , 'darksushi225d'   , 'anime', 'Dark Sushi Mix 2.25D'   , 'https://civitai.com/models/24779?modelVersionId=93208'),
    ('hassaku_v11'                           , 'hassakuv11'      , 'anime', 'Hassaku v1.1'           , 'https://civitai.com/models/2583?modelVersionId=37521'),
    ('hassakuHentaiModel_v12'                , 'hassakuv12'      , 'anime', 'Hassaku v1.2'           , 'https://civitai.com/models/2583?modelVersionId=62528'),
    ('hm_grapefruitv41'                      , 'grapefruitv41'   , 'anime', 'Grapefruit v4.1'        , 'https://civitai.com/models/24383?modelVersionId=29179'),
    ('meinamix_meinaV9'                      , 'meinav9'         , 'anime', 'MeinaMix v9'            , 'https://civitai.com/models/7240?modelVersionId=46137'),
    ('meinamix_meinaV10'                     , 'meinav10'        , 'anime', 'MeinaMix v10'           , 'https://civitai.com/models/7240?modelVersionId=80511'),
    ('nyanMix_230303Absurd2'                 , 'nyan230303a2'    , 'anime', 'Nyan Mix 230303 Absurd2', 'https://civitai.com/models/14373?modelVersionId=18151'),
    ('nyanMix_230303Normal'                  , 'nyan230303n'     , 'anime', 'Nyan Mix 230303 Normal' , 'https://civitai.com/models/14373?modelVersionId=17818'),
    ('tmndMix_tmndMixIVPruned'               , 'tmndmixiv'       , 'anime', 'TMND-Mix IV'            , 'https://civitai.com/models/27259?modelVersionId=88859'),
]
extra_prompt_mapping = {
    "anime": ", high quality, highly detailed, masterpiece",
    "other": ", high quality",
    "photo": ", photo, high quality, 4k",
}
neg_prompt_options = ["none", "default", "sfw"]
sfw_extra_neg_prompt = ", nsfw, naked, nude"
default_neg_prompt = get_neg_prompt()

checkpoint_map = {}
prompt_map = {}
category_map = { "anime": [], "other": [], "photo": [] }
gen_img_count = 0

for checkpoint_data, prompt_data, neg_prompt_setting in product(checkpoints, prompts, neg_prompt_options):
    checkpoint, checkpoint_id, category, checkpoint_name, url = checkpoint_data
    prompt, prompt_id = prompt_data

    sfw = neg_prompt_setting == "sfw"
    no_neg = neg_prompt_setting == "none"

    pos_prompt_string = "({}:1.1){}".format(prompt, extra_prompt_mapping[category])
    print("+ : " + pos_prompt_string)
    set_pos_prompt(pos_prompt_string)

    neg_prompt_string = ""
    if not no_neg:
        neg_prompt_string = default_neg_prompt
    if sfw:
        neg_prompt_string += sfw_extra_neg_prompt
    print("- : " + neg_prompt_string)
    set_neg_prompt(neg_prompt_string)
    
    checkpoint_fn = "{}.safetensors".format(checkpoint)
    print(checkpoint_fn)
    set_checkpoint(checkpoint_fn)

    output_prefix = "{}_{}".format(prompt_id, checkpoint_id)
    if no_neg:
        output_prefix += "_noneg"
    if sfw:
        output_prefix += "_sfw"
    print(output_prefix)
    set_output_prefix(output_prefix)

    expected_fn = "{}_{:05d}_.png".format(output_prefix, 6)
    expected_path = os.path.join(output_dir, expected_fn)
    print(expected_path)

    if not os.path.exists(expected_path):
        imgs_generated = 0
        set_batch_size(imgs_per_batch)
        while imgs_generated < imgs_per_prompt:
            set_seed(starting_seed + imgs_generated)
            queue_prompt(json_prompt)
            imgs_generated += imgs_per_batch
        while not os.path.exists(expected_path):
            time.sleep(0.1)
        print("")
    else:
        print("exists, skipping")

    checkpoint_map[checkpoint_id] = { 
        'checkpoint': checkpoint,
        'checkpoint_name': checkpoint_name,
        'category': category,
    }
    prompt_map[prompt_id] = prompt
    if checkpoint_id not in category_map[category]:
        category_map[category].append(checkpoint_id)

    gen_img_count += imgs_per_prompt
    print("-----")

# store the results in JSON files
with open('checkpoint_map.json', 'w') as file:
    json.dump(checkpoint_map, file, indent=2)
with open('prompt_map.json', 'w') as file:
    json.dump(prompt_map, file, indent=2)
with open('category_map.json', 'w') as file:
    json.dump(category_map, file, indent=2)
with open('meta_info.json', 'w') as file:
    json.dump({ "gen_time": datetime.now().isoformat(), "total_imgs": gen_img_count }, file, indent=2)

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
