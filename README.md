<h1 align="center">
  GetKGC
</h1>

### Dependencies

- Compatible with PyTorch 1.11.0+cu113 and Python 3.x.
- Dependencies can be installed using `requirements.txt`.


### Training and testing:

##### WN18RR
```shell
python3 main.py -dataset WN18RR \
                -batch_size 128 \
                -pretrained_model bert-large-uncased \
                -desc_max_length 40 \
                -lr 5e-4 \
                -prompt_length 10 \
                -alpha 0.1 \
                -n_lar 8 \
                -label_smoothing 0.1 \
                -embed_dim 144 \
                -k_w 12 \
                -k_h 12 \
                -alpha_step 0.00001


# evaluation commandline:
python3 main.py -dataset WN18RR \
                -batch_size 128 \
                -pretrained_model bert-large-uncased \
                -desc_max_length 40 \
                -lr 5e-4 \
                -prompt_length 10 \
                -alpha 0.1 \
                -n_lar 8 \
                -label_smoothing 0.1 \
                -embed_dim 144 \
                -k_w 12 \
                -k_h 12 \
                -alpha_step 0.00001 \
                -model_path path/to/trained/model
                
```
##### FB15k-237
```shell
python3 main.py -dataset FB15k-237 \
                -batch_size 128 \
                -pretrained_model bert-base-uncased \
                -epoch 60 \
                -desc_max_length 40 \
                -lr 5e-4 \
                -prompt_length 10 \
                -alpha 0.1 \
                -n_lar 8 \
                -label_smoothing 0.1 \
                -embed_dim 156 \
                -k_w 12 \
                -k_h 13 \
                -alpha_step 0.00001 

# evaluation commandline:
python3 main.py -dataset FB15k-237 \
                -batch_size 128 \
                -pretrained_model bert-base-uncased \
                -desc_max_length 40 \
                -lr 5e-4 \
                -prompt_length 10 \
                -alpha 0.1 \
                -n_lar 8 \
                -label_smoothing 0.1 \
                -embed_dim 156 \
                -k_w 12 \
                -k_h 13 \
                -alpha_step 0.00001 \
                -model_path path/to/trained/model
```