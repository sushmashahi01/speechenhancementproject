# Speech Enhachment Project 

Download: [Valentini Noisy Speech
Dataset](http://datashare.ed.ac.uk/handle/10283/2791) and unzip it to dataset folder then also unzip 
- clean trainset wav: Clean training speech consists of 11,572 samples
- noisy trainset wav: Corresponding noisy speech for training, consists of 11,572 samples.
- clean testset wav: Clean test speech samples, consisting of 824 samples
-  noisy testset wav: contains noisy speech for testing, consisting of 824 sample

### Now install required dependency and run the code

```bash
pip install -r requirements.txt

python3 svm_audio_classifiier.py
python3 cnn_audio_classifiier.py
