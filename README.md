# synthetic-text-watermarking

## Setup

A **watermarking_config.json** file should be specified at the root of the repository that contains the configuration for the SynthID watermarking model. This should include the watermarking keys `keys` used for watermarking/detecting (a list of 20-30 random integers that serve as your private digital signature) and the `ngram_len`.

For example:
```
{
    "ngram_len": 5,  # This corresponds to H=4 context window size in the paper
    "keys": [
        654,
        400,
        836,
        123,
        340,
        443,
        597,
        160,
        57,
        29,
        590,
        639,
        13,
        715,
        468,
        990,
        966,
        226,
        324,
        585,
        118,
        504,
        421,
        521,
        129,
        669,
        732,
        225,
        90,
        960,
    ],
    "sampling_table_size": 65536,  # 2**16
    "sampling_table_seed": 0,
    "context_history_size": 1024,
}
```


## Train model

Training can be ran using the following script:

```
python detector_training.py --model_name=google/gemma-7b-it --watermarking_config=watermarking_config.json
```

Check the script for more parameters are are tunable and check out paper at link https://www.nature.com/articles/s41586-024-08025-4 for more information on these parameters.


## Running inference

Run generation with watermarking of output text can be run using the following script:

```
python text_watermarker.py --watermark --watermarking_config=watermarking_config.json --input_text="This is a test input"
```

Run watermark detection on a given text sequence:

```
python text_watermarker.py --detect --watermarking_config=watermarking_config.json --input_text="This is a test input"
```
