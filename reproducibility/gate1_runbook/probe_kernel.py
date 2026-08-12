import os, sys

def main():
    import torch
    from vllm import LLM, SamplingParams
    cap = torch.cuda.get_device_capability()
    print(f"[PROBE] device={torch.cuda.get_device_name(0)} capability={cap[0]}{cap[1]}", flush=True)
    print(f"[PROBE] VLLM_DISABLED_KERNELS={os.environ.get('VLLM_DISABLED_KERNELS','<unset>')}", flush=True)
    print(f"[PROBE] model={sys.argv[1]}", flush=True)
    try:
        llm = LLM(model=sys.argv[1], enforce_eager=True, max_model_len=512,
                  gpu_memory_utilization=0.30, disable_log_stats=True)
        out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=8, temperature=0))
        print("[PROBE] GENERATED:", repr(out[0].outputs[0].text), flush=True)
        print("[PROBE] RESULT=FORWARD_PASS_OK", flush=True)
    except Exception as e:
        print(f"[PROBE] RESULT=FAILED type={type(e).__name__}", flush=True)
        print(f"[PROBE] ERRMSG={str(e)[:400]}", flush=True)
        sys.exit(9)

if __name__ == '__main__':
    main()
