python run.py \
    --function evaluate \
    --outputs_path ./output/predictions_pretrain.txt \
    --pretrain_corpus_path ./dataset/pretrain/wiki.txt \
    --eval_corpus_path ./dataset/finetune/birth_places_dev.tsv \
    --reading_params_path ./output/pretrain.pt