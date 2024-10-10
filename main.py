
import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

from sampling import autoregressive_sampling, speculative_sampling, speculative_sampling_v2
from globals import Decoder


# my local models
MODELZOO = {
    "llama7b": "/home/huangzl/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590/",
    "llama1b": "/home/huangzl/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/"
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    # parser.add_argument('--input', type=str, default="Any recommendations for my holidays in Abu Dhabi?")
    parser.add_argument('--dataset', type=str, default="oasst")
    parser.add_argument('--prefill_len', type=int, default="256")
    parser.add_argument('--max_tokens', '-M', type=int, default=32, help='max token number generated.')
    parser.add_argument('--n_test', type=int, default=10)
    parser.add_argument('--approx_model_name', type=str, default="llama7b")
    parser.add_argument('--target_model_name', type=str, default="llama1b")
    parser.add_argument('--gamma', '-g', type=int, default=4, help='guess time.')
    parser.add_argument('--seed', '-s', type=int, default=0, help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='enable verbose mode')
    parser.add_argument('--benchmark', '-b', action='store_true', default=False, help='show benchmark results.')
    parser.add_argument('--profiling', '-p', action='store_true', default=False, help='collect torch profiler results.')
    
    args = parser.parse_args()
    return args

def get_inputs(tokenizer, dataset_name, prefill_len, n_test):
    dataset_list = []

    if dataset_name == "oasst":
        file_path = f"data/oasst_prompts.json"
        with open(file_path, "r") as f:
            dataset = json.load(f)
        dataset = [x[1] for x in dataset]
        count = 0
        idx = 0
        
        while True:
            tokens = tokenizer(dataset[idx], return_tensors='pt')
            if tokens.input_ids.shape[1] >= prefill_len:
                dataset_list.append(tokens.input_ids[:, :prefill_len].to('cuda:0'))
                count += 1
            idx += 1
            if count >= n_test:
                break
    else:
        raise ValueError(f"dataset {dataset_name} not supported")
    
    return dataset_list


def generate(dataset, prefill_len, num_tokens, n_test, approx_model_name, target_model_name, gamma = 4,
             random_seed = 0, verbose = False, use_benchmark = False, use_profiling = False):
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    
    assert torch.cuda.is_available()
    torch.manual_seed(random_seed)
    
    tokenizer = AutoTokenizer.from_pretrained(approx_model_name)
    Decoder().set_tokenizer(tokenizer)
    
    print(f"begin loading models: \n {approx_model_name} \n {target_model_name}")
    small_model = AutoModelForCausalLM.from_pretrained(approx_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="cuda:0")
    large_model = AutoModelForCausalLM.from_pretrained(target_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="cuda:0")
    print("finish loading models")
    
    dataset_list = get_inputs(tokenizer, dataset, prefill_len, n_test)

    top_k = 20
    top_p = 0.9
    temperature = 1.0
    acc_list = []

    for i in range(n_test):

        input_ids = dataset_list[i]
        output, acc = speculative_sampling(prefix=input_ids, 
                             approx_model=small_model, 
                             target_model=large_model, 
                             max_len=num_tokens, 
                             gamma=gamma, 
                             temperature=temperature, 
                             top_k=top_k, 
                             top_p=top_p, 
                             random_seed=random_seed, 
                             verbose=verbose)
        acc_list.append(acc)
        prefill_text = tokenizer.decode(output[0, :prefill_len], skip_special_tokens=True)
        decode_text = tokenizer.decode(output[0, prefill_len:], skip_special_tokens=True)
        print(f"\n========= {i}th TEST ==========")
        print(f"\nprefill text: {prefill_text}")
        print(f"\ndecode text: {decode_text}")

    print(f"accuracy list: {acc_list}")
    print(f"average accuracy: {sum(acc_list) / len(acc_list)}")
        

if __name__ == "__main__":
    args = parse_arguments()
    
    generate(dataset=args.dataset, 
             prefill_len=args.prefill_len, 
             num_tokens=args.max_tokens, 
             n_test=args.n_test,
             approx_model_name=MODELZOO[args.approx_model_name], 
             target_model_name=MODELZOO[args.target_model_name], 
             gamma=args.gamma,
             random_seed=args.seed, 
             verbose=args.verbose, 
             use_benchmark=args.benchmark, 
             use_profiling=args.profiling)