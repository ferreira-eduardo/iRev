############3
# treinado com fold 0 mcpn imdb
# A3CNF  ALFM CARL ConvMF DAML DEEPCON MPCN TARMF
##########



# Protocolo de seeds (sem folds): split fixo, uma execução por seed; mesmas seeds do ghc2f.
# Resultados em results/new_protocol/<dataset>/ (summary.csv = média e desvio entre seeds).
SEEDS="[23,43]"

for dataset in All_Beauty_data
do

#    python3 main.py train --model=A3NCF --dataset=$dataset --num_fea=1 --topics=True  --direct_output=True --batch_size=4096
#    python3 main.py train --model=ALFM --dataset=$dataset --num_fea=1 --topics=True --direct_output=True --batch_size=256 --word_dim=300 --fc_dim=500
#    python3 main.py train --model=CARL --dataset=$dataset --num-fea=3 --emb_opt=word2vec --batch_size=1024
#    python3 main.py train --model=ConvMF --dataset=$dataset --emb_opt=word2vec --num-fea=1 --batch_size=1024
#    python3 main.py train --model=DAML --dataset=$dataset --emb_opt=word2vec --num-fea=2 --batch_size=1024
#    python3 main.py train --model=DeepCoNN --dataset=$dataset --emb_opt=word2vec --num-fea=1 --word_dim=300
    python main.py train --dataset=$dataset --model=Letter --batch_size=64 --r_max_len=100 --i_max_r=30 --u_max_r=30 --word_dim=300 --seeds=$SEEDS
#    python3 main.py train --model=TARMF --dataset=$dataset --emb_opt=word2vec --num-fea=2 --batch_size=2048



    # this model is very slow
#    python3 main.py train --model=NARRE --dataset=$dataset --emb_opt=word2vec --num-fea=2

  # too slow 1.43hrs per epoch
#    python3 main.py train --model=D_ATTN --dataset=$dataset --emb_opt=word2vec --num-fea=1 --batch_size=1096


    # descobrir  'checkpoints/user_features_pmf_Imdb_data_word2vec.npy'
    #48 min por epoca
#    python3 main.py train --model=TRANSNET --dataset=$dataset --emb_opt=word2vec --num-fea=1 --output=fm --transnet=True
    # erro  IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
#    python3 main.py train --model=ANR --dataset=$dataset --emb_opt=word2vec --num-fea=1 --id_emb_size=500

    #[Errno 2] No such file or directory: 'checkpoints/user_features_pmf_Imdb_data_word2vec.npy'
#    python3 main.py train --model=HRDR --dataset=$dataset --emb_opt=word2vec --num-fea=2 --filters_num=32 --batch_size=32
#[Errno 2] No such file or directory: 'checkpoints/user_features_pmf_Imdb_data_word2vec.npy'
#    python3 main.py train --model=CARP --dataset=$dataset --output=lfm --emb_opt=word2vec
#    python3 main.py train --model=CARM --dataset=$dataset --emb_opt=word2vec
#
#    python3 main.py train --model=MAN --dataset=$dataset --batch_size=32 --man=True --emb_opt=word2vec --output=nfm
#    python3 main.py train --model=NRPA --dataset=$dataset --emb_opt=word2vec --num-fea=1

done
