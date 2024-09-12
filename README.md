# AttM_INTERSPEECH24
Official release code for paper Attentive Merging of Hidden Embeddings from Pre-trained Speech Model for Anti-spoofing Detection

Before training/evaluation the model with ASVspoof2019 LA or other dataset, you need to have the meta files like, for example:

wav_nosox.scp
file_name  path/to/filename.flac
LA_T_1002656 /data/asvspoof2019/LA/ASVspoof2019_LA_train/flac/LA_T_1002656.flac
LA_T_1003665 /data/asvspoof2019/LA/ASVspoof2019_LA_train/flac/LA_T_1003665.flac

train.tsv, validation_tsv, and evaluation.tsv
file_name  label
LA_T_1002656	spoof
LA_T_1003665	bonafide

Then run the code as:
python -u main_antispoof.py --config ./antispoof.conf \
  --meta_dir /meta/path/to/save/tsv_files --feat_file /feat/path/to/save/wav_nosox.scp --output_dir output/path --seed 2048 

  
