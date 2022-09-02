for index in {0..4}; do
  CUDA_VISIBLE_DEVICES=1, python latent.py msr_output/origin/saved_models_"$index"/checkpoint-best/model.bin ../msr_dataset/origin/data_split_"$index"/train.jsonl ../msr_dataset/origin/data_split_"$index"/test.jsonl msr_output/latent/data_split_"$index"
done