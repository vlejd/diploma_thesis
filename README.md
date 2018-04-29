# Diploma thesis
Code related to my diploma thesis: **Improving LSA word weights for document classification**

We run experiments on google cloud.

# Instalation guide
These sommands are designed to be run in home directory (you may need to change some paths).

```
yes | sudo apt install htop ;
yes | sudo apt install unzip ;
yes | sudo apt install git ;
git clone https://github.com/facebookresearch/SentEval.git ;
yes | sudo apt install python3-pip  ;
yes | sudo apt-get install python3-tk ;
export LC_ALL=C ;
cd ;
cd SentEval/data/downstream ;
./get_transfer_data.bash ;

git clone https://github.com/vlejd/diploma_thesis.git ;
cd diploma_thesis/ ;
yes | pip3 install -r requirements.txt ;
cd ;
cd diploma_thesis/lsa_backprop ;
mkdir dumps_new ;
```


Beacause we need to run a lot of experiments, our code is designed to be run on google cloud compute engine.
Experiment reds environsment variables that determin, which part of the experiment should be computed on specific machine.

To run experiments, do 
```
screen -S dip -dm bash -c "
cd ;
cd diploma_thesis/lsa_backprop ; 
yes | git pull; 
export SENTEVAL_DATA_BASE=/home/vlejd/SentEval/data/downstream/ ; # <SET THIS PATH>
export SHARDING=20;
export OFFSET=10;
export THREADS=1;
export INSCRIPT=1;
python3 experimentsbatch.py;
";
```


# Google cloud

