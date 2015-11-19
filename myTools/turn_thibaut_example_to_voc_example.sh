cls_arr=("dog" "cat" "motorbike" "boat" "aeroplane" "horse" "cow" "sofa" "diningtable" "bicycle")
typ_arr=("train" "val" "trainval")
for cls in ${cls_arr[@]}
do
	for typ in ${typ_arr[@]}
	do
		cat /local/wangxin/Data/ferrari_gaze/example_files/100/"$cls"_"$typ"_scale_100_matconvnet_m_2048_layer_20.txt |awk -F ' ' '{print $1" "$2}' |grep _> /local/wangxin/Data/ferrari_gaze/voc_example_file_10_categories/"$cls"_"$typ".txt
done
done
