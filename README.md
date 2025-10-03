# synthetic-text-watermarking

### Config

A **credentials.json** file should be specified at the root of the repository that enumerates the SynthID watermarking key used for watermarking/detecting. The "watermarking keys" is a list of 20-30 random integers that serve as your private digital signature.

For example:
```
{
    "synthid_watermarking_keys": [634, 300, 846, 15, 310, ...]
}
```
