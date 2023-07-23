import argparse
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test-HF-Llama-7B')
    parser.add_argument('--version-name', type=str, default='huggyllama/llama-7b', help='version-name')
    parser.add_argument('--prompt-seq-length', type=int, default=1920, help='How many tokens in prefill?')
    parser.add_argument('--generate-seq-length', type=int, default=128, help='How many tokens in token generation?')
    parser.add_argument('--compute-device', type=str, default='cuda:0', metavar='S',
                        help='compute device (cpu or cuda)')
    args = parser.parse_args()
    # init
    tokenizer = AutoTokenizer.from_pretrained(args.version_name)    
    
    with torch.no_grad():
        model = AutoModelForCausalLM.from_pretrained(args.version_name, torch_dtype=torch.float16)
        model = model.to(args.compute_device)
        # infer
        with open('./raw_wiki_text.txt', 'r') as file: 
            prompt = file.read()
            
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        input_length = inputs.input_ids.shape[1]
        print(f"Raw input context length: {input_length}")
        
        if input_length > args.prompt_seq_length:
            inputs.input_ids= inputs.input_ids[:, 0:args.prompt_seq_length]
        
        input_length = inputs.input_ids.shape[1]
        print(f"Cut input context length: {input_length}")
        
        print(f"Input IDs: {inputs.input_ids}")
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)  
        
        start.record()
        outputs = model.generate(
            inputs.input_ids, max_new_tokens=args.generate_seq_length, do_sample=True, temperature=0.7, top_p=0.7, top_k=50, return_dict_in_generate=True
        )
        end.record()
        torch.cuda.synchronize()
        
        token = outputs.sequences[0, input_length:]
        print(f"Output IDs: {token}")
        output_str = tokenizer.decode(token)
        print(output_str)
        print(f"<====<Redpajama-{args.version_name}-HF> Input length: {input_length}, output length: {args.generate_seq_length} time: {start.elapsed_time(end)} milliseconds====>")
