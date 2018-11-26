# Stochastic Answer Networks for Machine Reading Comprehension

Three ways have been provided to run this program: (each is explained separatley.)
1. Using the codes directly from GitHub repository.
2. Using Docker container.
3. Using Singularity Hub for HPC systems.

### 1. GetHub Clone

For this method the following packages should be installed (using pip) :(you can use the setup environment provided)
+ Python (version 3.6)
+ PyTorch (0.4.0)
+ spaCy (2.0.12)
+ pandas==0.22.0
+ tqm
+ colorlog
+ allennlp

In addition you have to have following packages (install using apt-get and be sure your apt-get is uptodate : `apt-get update` ):
+ wget
+ unzip

#### Setup Environment
1. python3.6
2. Clone the GitHub repository:
   > git clone https://github.com/sinaehsani6/SQuAD2
3. install all requirements:
   > pip install -r requirements.txt
4. download all the data/word2vec 
   > sh download.sh

### 2. Docker Container

If you wan't to use the docker containers, you can download the docker containers from [here](https://www.docker.com/community-edition#/download).

1. Pull the docker file:
   > docker pull sinaehsani/sansrc_new:1st

2. Run the docker image:
   > docker run -it sinaehsani/sansrc_new:1st

3. Then you need to download the data/word2vec:
   > sh download.sh
  
### 3. Singularity Hub

For an easier run on HPC systems (HPC systems do not support Docker containers), a singularity hub is established for an easier use, to run the singularity hub, use the following codes (it is likley that you may face some issues running the program this way. To solve this problem you sould use the singularity hub to trigger a container first, please look at https://github.com/sinaehsani6/dockertosing to see a singularity recepie example):

1. Pull the singularity file:
   > singularity pull shub://sinaehsani6/dockertosing

2. Run the singularity image:
   > singularity shell sinaehsani6-dockertosing-master-latest.simg 
   
3. Clone the GitHub repository:
   > git clone https://github.com/sinaehsani6/SQuAD2
   
4. Then you need to download the data/word2vec:
   > sh download.sh
 

### Train the Model on SQuAD v2.0
1. preprocess data
   > python prepro.py --v2_on
2. train a Model
   > python train.py --v2_on

### Use of ELMo
train a Model with ELMo
   > python train.py --elmo_on --v2_on

### Train the Model on the diffrent preprocessed data (see section 4.1 of the paper)
1. preprocess data
   > python prepro2.py --v2_on
2. train a Model
   > python train2.py --v2_on

## Notes and Acknowledgments
Most defentions were imported from: https://github.com/kevinduh/san_mrc
Some of code are adapted from: https://github.com/hitvoice/DrQA <br/>
ELMo is from: https://allennlp.org


