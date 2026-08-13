import os
import json
import argparse
from multiprocessing import Pool
from tqdm import tqdm
import random
import pandas as pd
random.seed(27)
from evaluate.utils import get_response_mllm_api



def get_prompt(task_name, training_mode):
    task_indicator_map = {
        "gdp": "average annual GDP (the unit is PPP 2005 international dollars)",
        "pop": "total residential population",
        "acc2health": "average walking-only travel time to healthcare",
        "carbon": "total carbon emissions monthly",
        "build_height": "average of the net building height in meters",
        "mental": "the crude prevalence of mental health not good for ≥ 14 days among adults",
        "drive": "percentage of population (>16) commute by driving alone",
        "public": "percentage of population (>16) commute by public transit",
        "violent": "the occurrence of violent crime",
        "bachelor": "bachelor ratio",
        "life_exp": "the male life expectancy",
        "house_price": "the average market value of the home"
    }
    indicator = task_indicator_map[task_name]

    if task_name in ["gdp", "pop", "build_height", "acc2health", "carbon", "house_price"]:
        prompt_part1 = f"""Suppose you are a professional socioeconomic data analyst. Based on the provided satellite image and 10 street view photos taken within the same area covered by the satellite image, please estimate '{indicator}' for this area."""
    elif task_name in ["mental", "public", "bachelor", "drive", "violent"]:
        prompt_part1 = f"""Suppose you are a professional socioeconomic data analyst. Based on the provided satellite imagery and several street view photos, please estimate '{indicator}' in the census tract where the images are taken. """
    elif task_name in ["life_exp"]:
        prompt_part1 = f"""Suppose you are a professional socioeconomic data analyst. Based on the provided satellite imagery and multiple street view photos, please estimate '{indicator}' in the MSOA area where the images were taken."""
    if training_mode == "rl":
        prompt_part2 = f"""Analyze visual features relevant to '{indicator}' — including building, vehicle, road infrastructure, street furniture, person, greenery, and other environmental characteristics.

Infer the likely geographic location based on visual clues such as architectural styles, signage, urban layout, and vegetation, and apply your background knowledge of that region's typical socioeconomic profile.
    
Estimate this area's relative standing in the global distribution of '{indicator}'.
Assign a score from 1 to 10, where:
1 = Bottom 10% globally (worst condition)
10 = Top 10% globally (best condition)
The score must be a whole number (integer) between 1 and 10. Justify your score with evidence from your visual analysis and geolocation inference.

Please reason step-by-step and write your reasoning process inside the <think> </think> tags. Then, format the final answer as <answer>NUMBER</answer>."""
    elif training_mode == "sft":
        prompt_part2 = f"""Analyze visual features relevant to '{indicator}' — including building, vehicle, road infrastructure, street furniture, person, greenery, and other environmental characteristics.

Infer the likely geographic location based on visual clues such as architectural styles, signage, urban layout, and vegetation, and apply your background knowledge of that region's typical socioeconomic profile.
    
Estimate this area's relative standing in the global distribution of '{indicator}'.
Assign a score from 1 to 10, where:
1 = Bottom 10% globally (worst condition)
10 = Top 10% globally (best condition)
The score must be a whole number (integer) between 1 and 10."""
    return prompt_part1, prompt_part2


def generate_session_simple(data, task_name, model_name, training_mode):
    url_file = ""
    df_url = pd.read_csv(url_file)
    url_dict = dict(zip(df_url['image_name'], df_url['image_url']))
    
    
    basic_prompt, prompt_part2 = get_prompt(task_name, training_mode)
    
    content = []
    
    content.append({
        "type": "text",
        "text": basic_prompt
    })
    images = data.get("images", [])
    
    for i, image_path in enumerate(images):
        image_name = os.path.basename(image_path)
        image_url = url_dict.get(image_name)
        if not image_url:
            raise ValueError(f"❌ no url for {image_name}")
        
        content.append({
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        })
            
    
    content.append({
        "type": "text",
        "text": prompt_part2
    })


    session = [{
        "role": "user",
        "content": content
    }]
    return session



def single_eval_task(args):
    model_name, d, task_name, port, training_mode = args
    session = generate_session_simple(d, task_name, model_name, training_mode)

    reference = d["reference"]
    img_path = d["images"]
    city = d["city"]

    if task_name in ["gdp", "pop", "build_height", "acc2health", "house_price"]:
        area = d["area"]

    elif task_name in ["mental", "public", "bachelor", "drive", "violent"]:
        area = d["ct"]
    elif task_name in ["life_exp"]:
        area = d["msoa"]


    ret = get_response_mllm_api(session, model_name, temperature=0, max_tokens=2000, infer_server=None,json_mode=False, port=port)

    return {
        "area": area,
        "city": city,
        "images": img_path,
        "session": session,
        "reference": reference,
        "task": task_name,
        "response": ret,
        "rank_group": d["rank_group"]
    }



def eval_task(model_name, task_name, domain, training_mode, port, num_process, eval_version):
 
    if domain == "in":
        task_path = f"data/training_middle/train_9_city/test_data/{task_name}-{eval_version}-10.json"

    else:
        if task_name in ["gdp", "pop", "mental", "public", "bachelor"]:
            task_path = f"data/training_middle/test_4_city/4cities_{task_name}-{eval_version}-10.json"
        elif task_name in ["life_exp"]:
            task_path = f"data/training_middle/out-domain-indicator/out-domain-city/{task_name}-{eval_version}-10.json"
        else:
            task_path = f"data/training_middle/out-domain-indicator/in-domain-city/{task_name}-{eval_version}-10.json"

    
    print("task_path:", task_path)
    model_name_full = model_name.replace("/", "_")
    response_path = f'results/training/{training_mode}/{domain}/{model_name_full}_{task_name}_response-{eval_version}-10.json'
    
    
    output_dir = os.path.dirname(response_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(task_path, "r") as f:
        data = json.load(f)

    if len(data) > 200:
        data = random.sample(data, 200)

    data = random.shuffle(data)

    args_list = [(model_name, d, task_name, port, training_mode) for d in data]
    response = []
    with Pool(num_process) as pool:
        print("Processing tasks in parallel...")
        for result in tqdm(pool.imap(single_eval_task, args_list), total=len(data)):
            if result:
                response.append(result)

    with open(response_path, "w") as f:
        json.dump(response, f, indent=4, ensure_ascii=False)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='qwen2_5_vl_7b_comparison_cot_addr_relation-v2.1-epoch1-sft-20250916', help='model name')
    parser.add_argument('--task_name', type=str, default='gdp', help='task name')
    parser.add_argument('--training_mode', type=str, default='sft', help='sft or rl')
    parser.add_argument('--domain', type=str, default='in', help='in or out')
    parser.add_argument('--port', type=str, default='8190', help='port number')
    parser.add_argument('--num_process', type=int, default=20, help='number of processes for evaluation')
    parser.add_argument('--eval_version', type=str, default=20, help='evaluate data version')
    parser.add_argument('--number', type=int)
    args = parser.parse_args() 
    print(args)
    
    eval_task(args.model_name, args.task_name, args.domain, args.training_mode, args.port, args.num_process, args.eval_version)
    
if __name__ == "__main__":
    main()
