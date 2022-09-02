
N=5
(
for thing in {10..19}; do
    ((i=i%N)); ((i++==0)) && wait
    python backbone.py  --model_state_dir msr_result/ggnn_model/origin/"$thing"/Model_ep_49.bin --dataset msr --input_dir reveal_model_data/msr_data/origin/data_split_"$thing"  --node_tag node_features --graph_tag graph --label_tag targets --feature_size 100 --data_split "$thing"  --model_type ggnn &
done
)
