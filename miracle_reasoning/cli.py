from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from .config import FAMILIES, ModelConfig
from .model import MiracleModel
from .opencl import OpenCLBackend, OpenCLError, list_devices


def _tokenizer(path):
    path = Path(path)
    if path.suffix == ".model":
        import sentencepiece
        return sentencepiece.SentencePieceProcessor(model_file=str(path))
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(path))
    class Adapter:
        def encode(self, text):
            return tokenizer.encode(text, add_special_tokens=False).ids

        def decode(self, ids):
            return tokenizer.decode(ids)
    return Adapter()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Miracle non-Neural OpenCL CPU inference")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("devices", help="list OpenCL device types and memory limits")
    inspect = commands.add_parser("inspect", help="inspect a model configuration without OpenCL")
    inspect.add_argument("model_dir", type=Path)
    inspect.add_argument("--family", choices=FAMILIES)
    export = commands.add_parser("export", help="export a native checkpoint to FP32 safetensors")
    export.add_argument("--checkpoint", required=True, type=Path)
    export.add_argument("--config", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--family", choices=FAMILIES)
    generate = commands.add_parser("generate", help="generate tokens using an OpenCL CPU device")
    generate.add_argument("model_dir", type=Path)
    inputs = generate.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--ids", help="comma/space separated token IDs")
    inputs.add_argument("--prompt")
    generate.add_argument("--tokenizer", type=Path)
    generate.add_argument("--family", choices=FAMILIES)
    generate.add_argument("--context-length", type=int, help="default: min(2048, model context limit)")
    generate.add_argument("--weight-cache-mib", type=int, default=256)
    generate.add_argument("--platform-index", type=int)
    generate.add_argument("--device-index", type=int)
    generate.add_argument("--max-new-tokens", type=int, default=32)
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--top-k", type=int, default=0)
    generate.add_argument("--top-p", type=float, default=1.0)
    generate.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    try:
        if args.command == "devices":
            print(json.dumps([d.to_dict() for d in list_devices()], indent=2, ensure_ascii=False))
        elif args.command == "inspect":
            from .weights import expected_shapes
            config = ModelConfig.from_dict(json.loads((args.model_dir/"config.json").read_text(encoding="utf-8")), args.family)
            total = sum(math.prod(shape) for shape in expected_shapes(config).values())
            print(json.dumps({"family": config.family, "layers": config.layers,
                              "hidden_size": config.hidden_size, "max_positions": config.max_positions,
                              "inference_parameters": total, "fp32_weight_bytes": total*4}, indent=2))
        elif args.command == "export":
            from .export import export_checkpoint
            config = export_checkpoint(args.checkpoint, args.config, args.output, family=args.family)
            print(json.dumps({"output": str(args.output.resolve()), "family": config.family}))
        else:
            tokenizer = None
            if args.prompt is not None:
                if args.tokenizer is None:
                    raise ValueError("--prompt requires --tokenizer (SentencePiece .model or tokenizers .json)")
                tokenizer = _tokenizer(args.tokenizer)
                ids = tokenizer.encode(args.prompt)
            else:
                ids = [int(t) for t in args.ids.replace(",", " ").split()]
            backend = OpenCLBackend(platform_index=args.platform_index, device_index=args.device_index)
            model = MiracleModel.from_pretrained(args.model_dir, family=args.family, backend=backend,
                                                 context_length=args.context_length,
                                                 weight_cache_bytes=args.weight_cache_mib*1024*1024)
            generated = model.generate(ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                       top_k=args.top_k, top_p=args.top_p, seed=args.seed)
            result = {"device": backend.device.to_dict(), "token_ids": generated}
            if tokenizer:
                result["text"] = tokenizer.decode(generated)
            print(json.dumps(result, ensure_ascii=False))
        return 0
    except (ValueError, OSError, KeyError, ImportError, OpenCLError, FloatingPointError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
