import argparse
import logging
import torch
import pandas as pd
import json
import deepspeed

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from transformers.generation.utils import GenerationConfig

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_name_or_path_baseline",
        type=str,
        help="Path to baseline model",
        required=True,
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--penalty_alpha",
        type=float,
        default=0.6,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help='Specify num of return sequences',
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help='Specify num of return sequences',
    )
    parser.add_argument("--language",
                        type=str,
                        default="English",
                        choices=["English", "Chinese", "Japanese"])
    parser.add_argument("--model_type",
                        type=str,
                        default="llama2",
                        choices=["alpaca", "open_llama", "llama2", "vicuna", "orca", "falcon","baichuan_base", "baichuan_chat","stablebeluga", "platypus", "mistral"],
                        required=True)
    parser.add_argument("--input_prompts",
                        type=str,
                        help="Path to input prompts",
                        required=True,
    )
    parser.add_argument("--output_response",
                        type=str,
                        help="Path to out response",
                        required=True,
    )
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus"
    )
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.',
    )
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).',
    )

    args = parser.parse_args()

    return args


def get_model(config, model_path, tokenizer):

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        from_tf=bool(".ckpt" in model_path),
        config=config,
        ignore_mismatched_sizes=True,
        trust_remote_code=True
    )
    model.resize_token_embeddings(len(tokenizer))

    # prepare the tokenizer and model config
    tokenizer.pad_token = tokenizer.eos_token
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    return model


def generate(model,
             tokenizer,
             inputs,
             num_beams=1,
             num_beam_groups=1,
             do_sample=False,
             num_return_sequences=1,
             max_new_tokens=1024,
             temperature=0.0,
             top_k=50,
             top_p=1.0):

    generate_ids = model.generate(inputs.input_ids,
                                  num_beams=num_beams,
                                  num_beam_groups=num_beam_groups,
                                  do_sample=do_sample,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens,
                                  temperature=0.0,
                                  top_k=50,
                                  top_p=1.0)

    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    return result


def get_zero_ds_config(offload, stage=2):

    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False
    }
    return {"zero": zero_opt_dict,}


def print_utils(gen_output):
    for i in range(len(gen_output)):
        print()
        print(gen_output[i])
        print()

def prompt_eval(args, model_baseline, tokenizer_baseline, device, prompts,outputfile):
    responses_dict_list = []  
    for i, prompt in enumerate(prompts, start=1):
        #inputs_baseline = tokenizer_baseline(prompt, return_tensors="pt").to(device)
        print("==========prompt begin=========")
        if args.model_type == "baichuan_chat":
            message = "user:" + prompt + "\nassistant:"
            message_baseline = tokenizer_baseline(message, return_tensors="pt").to(device)
            r_base = generate(model_baseline,
                              tokenizer_baseline,
                              message_baseline,
                              num_beams=1,
                              num_return_sequences=args.num_return_sequences,
                              max_new_tokens=args.max_new_tokens,
                              temperature=0.0,
                              top_k=50,
                              top_p=1.0)
        else:
            inputs_baseline = tokenizer_baseline(prompt, return_tensors="pt").to(device)
            r_base = generate(model_baseline,
                          tokenizer_baseline,
                          inputs_baseline,
                          num_beams=1,
                          num_return_sequences=args.num_return_sequences,
                          max_new_tokens=args.max_new_tokens,
                          temperature=0.0,
                          top_k=50,
                          top_p=1.0)
        responses_dict = {"prompt{}".format(i): r_base}

        responses_dict_list.append(responses_dict)
        print_utils(r_base)
        print("====================prompt end=============================")
        print()
        print()

    with open(outputfile, 'w', encoding='utf-8') as json_file:
        json.dump(responses_dict_list, json_file, ensure_ascii=False, indent=4)

def main():
    args = parse_args()
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        deepspeed.init_distributed()
    print("get args")
    ds_config = get_zero_ds_config(offload=args.offload, stage=args.zero_stage)

    if args.model_type == "baichuan_base":
        print("baichuan_base tokenizer loading")
        tokenizer_baseline = AutoTokenizer.from_pretrained(args.model_name_or_path_baseline, trust_remote_code=True)
        print("baichuan_base tokenizer loaded")
        model_baseline = AutoModelForCausalLM.from_pretrained(args.model_name_or_path_baseline, trust_remote_code=True)
    elif args.model_type == "baichuan_chat":
        print("baichuan_chat tokenizer loading")
        tokenizer_baseline = AutoTokenizer.from_pretrained(args.model_name_or_path_baseline, use_fast=False, trust_remote_code=True)
        print("baichuan_chat tokenizer loaded")
        model_baseline = AutoModelForCausalLM.from_pretrained(args.model_name_or_path_baseline, trust_remote_code=True)
        model_baseline.generation_config = GenerationConfig.from_pretrained(args.model_name_or_path_baseline)
    else:
        config_baseline = AutoConfig.from_pretrained(args.model_name_or_path_baseline, trust_remote_code=True)
        print("config loaded")
        if args.model_type == "alpaca":
            tokenizer_baseline = AutoTokenizer.from_pretrained(args.model_name_or_path_baseline,fast_tokenizer=True, unk_token="<unk>",
                                                    bos_token="<s>",
                                                    eos_token="</s>")
        else:
            tokenizer_baseline = AutoTokenizer.from_pretrained(args.model_name_or_path_baseline,fast_tokenizer=True)
        print("tokenizer loaded")
        model_baseline = get_model(config_baseline, args.model_name_or_path_baseline,
                               tokenizer_baseline)
        print("model loaded")

    if args.model_type in ["baichuan_base", "baichuan_chat", "falcon"]:
        ds_type = torch.bfloat16
    else:
        ds_type = torch.float16

    ds_model = deepspeed.init_inference(
        model=model_baseline,     
        mp_size=2,        # GPU numbers
        dtype=ds_type, 
        replace_method="auto", 
        replace_with_kernel_inject=True,
        config=ds_config, 
        max_out_tokens=args.max_new_tokens,
        )
    
    print(f"Model load at: {ds_model.module.device}\n")
    df = pd.read_csv(args.input_prompts)  
    prompts = df["Prompt"].apply(lambda x: x.strip('"')).tolist()
    if args.model_type == "alpaca":
        prompts = [
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:".format(
                prompt) for prompt in prompts]
    elif args.model_type == "open_llama":
        prompts = ["Q:{}\nA:".format(prompt) for prompt in prompts]
    elif args.model_type == "llama2":
        prompts = [
            "User:{}\nAssistant:".format(
                prompt) for prompt in prompts]
    elif args.model_type == "vicuna":
        prompts = ["USER: {}\nASSISTANT:".format(prompt) for prompt in prompts]
    elif args.model_type == "orca":
        prompts = [
            "### System:\nYou are an AI assistant that follows instruction extremely well. Help as much as you can.\n\n### User:{}\n\n### Assistant:".format(
                prompt) for prompt in prompts]
    elif args.model_type == "falcon":
        prompts = [
            "System:You are an AI assistant that follows instruction extremely well. Help as much as you can.\nUser:{}\nFalcon:".format(
                prompt) for prompt in prompts]
    elif args.model_type == "baichuan_base":
        prompts = ["{}\n".format(prompt) for prompt in prompts]
    elif args.model_type == "stablebeluga":
        prompts = ["### System:\nYou are StableBeluga, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.\n\n### User:{}\n\n### Assistant:\n".format(prompt) for prompt in prompts]
    elif args.model_type == "mistral":
        prompts = ["[INST]{}[/INST]\n".format(prompt) for prompt in prompts]
    elif args.model_type == "platypus":
        prompts = ["### Instruction:\n{}\n\n### Response:".format(prompt) for prompt in prompts]
    
    prompt_eval(args, ds_model, tokenizer_baseline, device, prompts, args.output_response)

if __name__ == "__main__":
    main()