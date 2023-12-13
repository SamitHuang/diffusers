
# Usgae

## Convert MS checkpoint to PT

```
python convert_lora_ms2pt.py {path to ms_ckpt} 
```

The converted checkpoint will be saved in the same folder of {path to ms_ckpt}.

## Run infer in diffusers

```
python infer_lora.py --model_path {path to converted ckpt}
```
